"""TruthNote Benchmark — SSE 看板路由。

挂在主 app 的 /api/benchmark/* 下，与 verify/dashboard 共享同一端口。
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from pathlib import Path

from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse, StreamingResponse

ROOT = Path(__file__).resolve().parent.parent.parent

# 判定分类（与 run_benchmark_fast.py 一致）
PROBLEMATIC = {"谣言", "大部分不实", "误导性信息"}
OK = {"属实", "部分属实"}
UNCERTAIN = {"无法核实"}

router = APIRouter(prefix="/api/benchmark", tags=["benchmark"])

_sse_queue: queue.Queue = queue.Queue(maxsize=5000)
_benchmark_running = False
_benchmark_lock = threading.Lock()


def _push_event(event: str, data: dict) -> None:
    with contextlib.suppress(queue.Full):
        _sse_queue.put_nowait({"event": event, "data": data})


def _load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line.strip()))
    return rows


def _run_benchmark(
    true_only: bool,
    false_only: bool,
    sample: int,
    timeout: int,
    skip: int,
    dataset: str,
) -> None:
    global _benchmark_running

    from .agents import _pipeline_progress
    from .pipeline import verify_message

    _pipeline_progress.queue = _sse_queue

    try:
        all_cases: list[dict] = []

        if dataset == "demo":
            demo_path = ROOT / "data" / "eval" / "demo_13.jsonl"
            if demo_path.exists():
                for c in _load_jsonl(demo_path):
                    expected = (c.get("expected") or "").upper()
                    c["_type"] = "TRUE" if expected == "TRUE" else "FALSE"
                    c.setdefault("topic", c.get("category", "?"))
                    all_cases.append(c)
        else:
            if not false_only:
                true_path = ROOT / "data" / "true_cases.jsonl"
                if true_path.exists():
                    for c in _load_jsonl(true_path):
                        c["_type"] = "TRUE"
                        all_cases.append(c)

            if not true_only:
                false_path = ROOT / "data" / "external_testset.jsonl"
                if false_path.exists():
                    for c in _load_jsonl(false_path):
                        c["_type"] = "FALSE"
                        c.setdefault("expected", "FALSE")
                        c.setdefault("topic", c.get("category", "?"))
                        all_cases.append(c)

        if sample > 0:
            all_cases = all_cases[:sample]

        if skip > 0:
            all_cases = all_cases[skip:]

        total = len(all_cases)
        _push_event("benchmark_start", {"total": total})

        results: list[dict] = []

        for i, case in enumerate(all_cases):
            text = case["text"]
            expected = case.get("expected", "")
            case_type = case["_type"]
            topic = case.get("topic", "")

            _push_event(
                "case_start",
                {
                    "num": i + 1,
                    "total": total,
                    "text": text[:120],
                    "type": case_type,
                    "topic": topic,
                    "expected": expected,
                },
            )

            _pipeline_progress.start_case(i + 1, total, text)

            t0 = time.time()
            actual = ""
            tier = ""
            error_msg = ""
            full_detail = None
            elapsed = 0.0

            try:
                pool = ThreadPoolExecutor(1)
                future = pool.submit(verify_message, text, use_memory=False)
                try:
                    result = future.result(timeout=timeout)
                    elapsed = time.time() - t0
                    actual = result.overall_verdict.value
                    full_detail = result.model_dump(mode="json")
                except FutureTimeout:
                    elapsed = time.time() - t0
                    actual = "TIMEOUT"
                    error_msg = f"{elapsed:.0f}s 超时"
                    pool.shutdown(wait=False, cancel_futures=True)
                finally:
                    pool.shutdown(wait=False)
            except Exception as e:
                elapsed = time.time() - t0
                actual = "ERROR"
                error_msg = str(e)[:200]

            if actual in ("TIMEOUT", "ERROR"):
                tier = actual.lower()
            elif case_type == "TRUE":
                tier = (
                    "ok"
                    if actual in OK
                    else ("problematic" if actual in PROBLEMATIC else "uncertain")
                )
            else:
                is_caught = actual in PROBLEMATIC
                tier = "caught" if is_caught else ("uncertain" if actual in UNCERTAIN else "missed")

            row = {
                "num": i + 1,
                "text": text[:120],
                "full_text": text,
                "expected": expected,
                "actual": actual,
                "type": case_type,
                "topic": topic,
                "tier": tier,
                "elapsed": round(elapsed, 1),
                "error": error_msg,
                "detail": full_detail,
            }
            results.append(row)

            _push_event(
                "case_done",
                {
                    "num": i + 1,
                    "text": text[:120],
                    "full_text": text,
                    "verdict": actual,
                    "expected": expected,
                    "elapsed": round(elapsed, 1),
                    "tier": tier,
                    "type": case_type,
                    "topic": topic,
                    "error": error_msg,
                    "detail": full_detail,
                },
            )

        true_cases = [r for r in results if r["type"] == "TRUE"]
        false_cases = [r for r in results if r["type"] == "FALSE"]

        summary: dict = {"total": len(results)}

        if true_cases:
            t = len(true_cases)
            ok = sum(1 for r in true_cases if r["tier"] == "ok")
            unc = sum(1 for r in true_cases if r["tier"] == "uncertain")
            prob = sum(1 for r in true_cases if r["tier"] == "problematic")
            to = sum(1 for r in true_cases if r["actual"] == "TIMEOUT")
            err = sum(1 for r in true_cases if r["actual"] == "ERROR")
            summary["true_total"] = t
            summary["true_ok"] = ok
            summary["true_uncertain"] = unc
            summary["true_problematic"] = prob
            summary["true_timeout"] = to
            summary["true_error"] = err
            summary["false_positive_rate"] = round(prob / t, 3) if t else 0

        if false_cases:
            t = len(false_cases)
            caught = sum(1 for r in false_cases if r["tier"] == "caught")
            unc = sum(1 for r in false_cases if r["tier"] == "uncertain")
            missed = sum(1 for r in false_cases if r["tier"] == "missed")
            to = sum(1 for r in false_cases if r["actual"] == "TIMEOUT")
            err = sum(1 for r in false_cases if r["actual"] == "ERROR")
            valid = t - to - err
            summary["false_total"] = t
            summary["false_caught"] = caught
            summary["false_uncertain"] = unc
            summary["false_missed"] = missed
            summary["false_timeout"] = to
            summary["false_error"] = err
            summary["recall"] = round(caught / valid, 3) if valid else 0

        avg_time = sum(r["elapsed"] for r in results) / len(results) if results else 0
        summary["avg_time"] = round(avg_time, 1)

        _push_event("benchmark_done", summary)

        out = ROOT / "data" / "benchmark_dashboard_results.jsonl"
        with open(out, "w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    finally:
        _pipeline_progress.queue = None
        with _benchmark_lock:
            _benchmark_running = False


@router.get("/status")
async def status():
    return {"running": _benchmark_running}


@router.post("/start")
async def start_benchmark(
    true_only: bool = Query(False),
    false_only: bool = Query(False),
    sample: int = Query(0, description="限制用例数，0=全部"),
    timeout: int = Query(90, description="单条超时秒数"),
    skip: int = Query(0, description="跳过前N条"),
    dataset: str = Query("default", description="default | demo"),
):
    global _benchmark_running

    with _benchmark_lock:
        if _benchmark_running:
            return JSONResponse(
                {"ok": False, "error": "benchmark 正在运行中"},
                status_code=409,
            )
        _benchmark_running = True

    while not _sse_queue.empty():
        try:
            _sse_queue.get_nowait()
        except queue.Empty:
            break

    threading.Thread(
        target=_run_benchmark,
        args=(true_only, false_only, sample, timeout, skip, dataset),
        daemon=True,
    ).start()

    return {"ok": True}


@router.get("/events")
async def events():
    async def event_generator():
        while True:
            try:
                item = _sse_queue.get_nowait()
                event_name = item["event"]
                data_str = json.dumps(item["data"], ensure_ascii=False)
                yield f"event: {event_name}\ndata: {data_str}\n\n"
                if event_name == "benchmark_done":
                    break
            except queue.Empty:
                yield ": heartbeat\n\n"
                await asyncio.sleep(0.15)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )

#!/usr/bin/env python3
"""快速 benchmark：TRUE + FALSE 混合测试，带超时保护。"""

from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

PROBLEMATIC = {"谣言", "大部分不实", "误导性信息"}
OK = {"属实", "部分属实"}
UNCERTAIN = {"无法核实"}
VERDICT_MAP = {
    "FALSE": "谣言",
    "MOSTLY_FALSE": "大部分不实",
    "MISLEADING": "误导性信息",
    "PARTLY_TRUE": "部分属实",
    "TRUE": "属实",
    "UNVERIFIABLE": "无法核实",
}


def load_jsonl(path: Path) -> list[dict]:
    cases = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                cases.append(json.loads(line.strip()))
    return cases


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--true-only", action="store_true")
    parser.add_argument("--false-only", action="store_true")
    parser.add_argument("--sample", type=int, default=0)
    parser.add_argument("--timeout", type=int, default=90)
    args = parser.parse_args()

    from truthnote.pipeline import verify_message

    all_cases = []
    if not args.false_only:
        true_path = ROOT / "data" / "true_cases.jsonl"
        for c in load_jsonl(true_path):
            c["_type"] = "TRUE"
            all_cases.append(c)

    if not args.true_only:
        false_path = ROOT / "data" / "external_testset.jsonl"
        for c in load_jsonl(false_path):
            c["_type"] = "FALSE"
            c.setdefault("expected", "FALSE")
            c.setdefault("topic", c.get("category", "?"))
            all_cases.append(c)

    if args.sample > 0:
        all_cases = all_cases[: args.sample]

    print(f"共 {len(all_cases)} 条（timeout={args.timeout}s/条）\n")

    results = []
    for i, case in enumerate(all_cases):
        text = case["text"]
        expected = case["expected"]
        case_type = case["_type"]
        topic = case.get("topic", "")

        print(f"\n{'=' * 60}", flush=True)
        print(f"[{i + 1}/{len(all_cases)}] ({case_type}/{topic}) {text[:60]}...", flush=True)
        print(f"{'=' * 60}", flush=True)

        from truthnote.agents import _pipeline_progress

        _pipeline_progress.start_case(i + 1, len(all_cases), text)

        t0 = time.time()
        try:
            pool = ThreadPoolExecutor(1)
            future = pool.submit(verify_message, text, use_memory=False)
            try:
                result = future.result(timeout=args.timeout)
                elapsed = time.time() - t0
                actual = result.overall_verdict.value
            except FutureTimeout:
                elapsed = time.time() - t0
                actual = "TIMEOUT"
                print(f"⏰ {elapsed:.0f}s TIMEOUT")
                pool.shutdown(wait=False, cancel_futures=True)
                results.append(
                    {
                        "text": text,
                        "expected": expected,
                        "actual": actual,
                        "type": case_type,
                        "topic": topic,
                        "elapsed": round(elapsed, 1),
                    }
                )
                continue
            finally:
                pool.shutdown(wait=False)
        except Exception as e:
            elapsed = time.time() - t0
            actual = "ERROR"
            print(f"💥 {elapsed:.0f}s {e}")
            results.append(
                {
                    "text": text,
                    "expected": expected,
                    "actual": actual,
                    "type": case_type,
                    "topic": topic,
                    "elapsed": round(elapsed, 1),
                }
            )
            continue

        if case_type == "TRUE":
            tier = (
                "ok" if actual in OK else ("problematic" if actual in PROBLEMATIC else "uncertain")
            )
            sym = "✓" if tier == "ok" else ("~" if tier == "uncertain" else "✗")
        else:
            is_caught = actual in PROBLEMATIC
            tier = "caught" if is_caught else ("uncertain" if actual in UNCERTAIN else "missed")
            sym = "✓" if is_caught else ("~" if tier == "uncertain" else "✗")

        print(f"  📊 结果: {sym} {actual} ({elapsed:.0f}s)", flush=True)
        results.append(
            {
                "text": text,
                "expected": expected,
                "actual": actual,
                "type": case_type,
                "topic": topic,
                "tier": tier,
                "elapsed": round(elapsed, 1),
            }
        )

    # 汇总
    print("\n" + "=" * 60)
    true_cases = [r for r in results if r["type"] == "TRUE"]
    false_cases = [r for r in results if r["type"] == "FALSE"]

    if true_cases:
        ok = sum(1 for r in true_cases if r.get("tier") == "ok")
        unc = sum(1 for r in true_cases if r.get("tier") == "uncertain")
        prob = sum(1 for r in true_cases if r.get("tier") == "problematic")
        to = sum(1 for r in true_cases if r.get("actual") == "TIMEOUT")
        err = sum(1 for r in true_cases if r.get("actual") == "ERROR")
        t = len(true_cases)
        print(
            f"TRUE cases ({t}): ✓ 没问题 {ok} | ~ 不确定 {unc} | ✗ 误判 {prob} | ⏰ 超时 {to} | 💥 错误 {err}"  # noqa: E501
        )
        print(f"  误伤率: {prob}/{t} = {prob / t:.1%}")

    if false_cases:
        caught = sum(1 for r in false_cases if r.get("tier") == "caught")
        unc = sum(1 for r in false_cases if r.get("tier") == "uncertain")
        missed = sum(1 for r in false_cases if r.get("tier") == "missed")
        to = sum(1 for r in false_cases if r.get("actual") == "TIMEOUT")
        err = sum(1 for r in false_cases if r.get("actual") == "ERROR")
        t = len(false_cases)
        print(
            f"FALSE cases ({t}): ✓ 抓到 {caught} | ~ 不确定 {unc} | ✗ 漏掉 {missed} | ⏰ 超时 {to} | 💥 错误 {err}"  # noqa: E501
        )
        valid = t - to - err
        if valid > 0:
            print(f"  召回率: {caught}/{valid} = {caught / valid:.1%}（去超时/错误）")

    avg_time = sum(r["elapsed"] for r in results) / len(results) if results else 0
    print(f"\n平均耗时: {avg_time:.1f}s/条")

    # 保存
    out = ROOT / "data" / "benchmark_fast_results.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"结果保存: {out}")


if __name__ == "__main__":
    main()

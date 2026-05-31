#!/usr/bin/env python3
"""外部验证基准评测脚本。

读取 data/external_testset.jsonl（来自中国互联网联合辟谣平台 piyao.org.cn
等权威平台的真实辟谣样本），跑 TruthNote 流水线，计算：
- 召回率（recall）——系统识别真实谣言为 false-leaning 的比例
- 各主题分项召回率
- 平均耗时 / LLM 调用次数

与 run_benchmark.py 的关键差异：
- 输入是 JSONL，不是 JSON 数组
- 所有 expected 都是 false-leaning（FALSE/MOSTLY_FALSE/MISLEADING）
- 主要指标是 recall（区别于 internal 的 6 类 accuracy）

用法：
    python scripts/run_external_benchmark.py --output data/external_benchmark.json
    python scripts/run_external_benchmark.py --sample 5   # 只跑前 5 条
    python scripts/run_external_benchmark.py --cache-ttl 720
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

# 任何 false-leaning verdict 都算"识别为谣言"
FALSE_LEANING = {"谣言", "大部分不实", "误导性信息"}


def load_jsonl(path: Path) -> list[dict]:
    cases = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            cases.append(json.loads(line))
    return cases


def run_eval(
    cases: list[dict],
    verify_fn,
    *,
    per_case_timeout: int = 90,
    incremental_path: Path | None = None,
) -> tuple[list[dict], list[dict]]:
    """对每条 case 跑核查，per_case_timeout 秒超时则跳过。

    每条完成后立即 append 到 incremental_path（jsonl）防止脚本中途死所有数据丢失。
    """
    results = []
    errors = []

    verdict_map = {
        "FALSE": "谣言",
        "MOSTLY_FALSE": "大部分不实",
        "MISLEADING": "误导性信息",
        "PARTLY_TRUE": "部分属实",
        "TRUE": "属实",
        "UNVERIFIABLE": "无法核实",
    }

    def _append_incremental(entry: dict):
        if incremental_path:
            with open(incremental_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    for i, case in enumerate(cases):
        text = case["text"]
        expected = case["expected"]
        acceptable = set(case.get("acceptable", [expected]))
        acceptable_zh = {verdict_map.get(v, v) for v in acceptable}

        print(f"  [{i + 1}/{len(cases)}] {text[:40]}...", end=" ", flush=True)

        t0 = time.time()
        try:
            # ThreadPoolExecutor 提供 cooperative timeout
            with ThreadPoolExecutor(max_workers=1) as ex:
                future = ex.submit(verify_fn, text, use_memory=True)
                try:
                    result = future.result(timeout=per_case_timeout)
                except FutureTimeout:
                    elapsed = time.time() - t0
                    print(f"⏱  TIMEOUT ({elapsed:.1f}s)")
                    err_entry = {
                        "text": text[:60],
                        "error": f"timeout after {per_case_timeout}s",
                    }
                    errors.append(err_entry)
                    _append_incremental({"status": "timeout", **err_entry, "index": i + 1})
                    continue

            elapsed = time.time() - t0
            actual = result.overall_verdict.value
            ok = actual in acceptable_zh
            false_leaning_ok = actual in FALSE_LEANING
            llm_calls = result.trace.total_llm_calls if result.trace else 0

            status = "✓" if false_leaning_ok else "✗"
            print(f"{status} {elapsed:.1f}s | {actual} | {llm_calls} calls")

            entry = {
                "text": text,
                "topic": case.get("topic", ""),
                "source": case.get("source", ""),
                "expected": expected,
                "actual": actual,
                "acceptable_ok": ok,
                "false_leaning_ok": false_leaning_ok,
                "elapsed_sec": round(elapsed, 2),
                "llm_calls": llm_calls,
            }
            results.append(entry)
            _append_incremental({"status": "ok", "index": i + 1, **entry})
        except Exception as e:
            elapsed = time.time() - t0
            print(f"💥 ERROR ({elapsed:.1f}s): {e}")
            err_entry = {"text": text[:60], "error": str(e)}
            errors.append(err_entry)
            _append_incremental({"status": "error", **err_entry, "index": i + 1})

    return results, errors


def compute_metrics(results: list[dict]) -> dict:
    if not results:
        return {}

    total = len(results)
    strict_correct = sum(1 for r in results if r["acceptable_ok"])
    recall_correct = sum(1 for r in results if r["false_leaning_ok"])

    # 按主题分项
    by_topic: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        by_topic[r["topic"]].append(r)
    topic_metrics = {}
    for topic, rs in by_topic.items():
        topic_metrics[topic] = {
            "total": len(rs),
            "recall": sum(1 for r in rs if r["false_leaning_ok"]) / len(rs),
            "strict_accuracy": sum(1 for r in rs if r["acceptable_ok"]) / len(rs),
        }

    # 按预测分布
    pred_distribution: dict[str, int] = defaultdict(int)
    for r in results:
        pred_distribution[r["actual"]] += 1

    avg_elapsed = sum(r["elapsed_sec"] for r in results) / total
    avg_llm = sum(r["llm_calls"] for r in results) / total

    return {
        "total": total,
        "recall": recall_correct / total,
        "strict_accuracy": strict_correct / total,
        "recall_correct": recall_correct,
        "strict_correct": strict_correct,
        "by_topic": topic_metrics,
        "pred_distribution": dict(pred_distribution),
        "avg_elapsed_sec": round(avg_elapsed, 2),
        "avg_llm_calls": round(avg_llm, 2),
    }


def print_report(metrics: dict, errors: list[dict]):
    print("\n" + "=" * 70)
    print("外部验证集 Benchmark 报告（中国互联网联合辟谣平台真实样本）")
    print("=" * 70)
    print(f"总数：{metrics['total']}")
    print(
        f"召回率（识别为 false-leaning）："
        f"{metrics['recall_correct']}/{metrics['total']} ({metrics['recall']:.1%})"
    )
    print(
        f"严格准确率（命中 acceptable_verdicts）："
        f"{metrics['strict_correct']}/{metrics['total']} ({metrics['strict_accuracy']:.1%})"
    )
    print(f"平均耗时：{metrics['avg_elapsed_sec']}s  平均 LLM 调用：{metrics['avg_llm_calls']}")

    print("\n按主题分项：")
    for topic, m in sorted(metrics["by_topic"].items()):
        print(
            f"  {topic}: 召回 {m['recall']:.1%} / 严格 {m['strict_accuracy']:.1%} ({m['total']} 条)"
        )

    print("\n预测分布：")
    for verdict, count in sorted(metrics["pred_distribution"].items(), key=lambda x: -x[1]):
        print(f"  {verdict}: {count}")

    if errors:
        print(f"\n错误数：{len(errors)}")
        for e in errors[:5]:
            print(f"  - {e['text']}: {e['error'][:60]}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--testset", default="data/external_testset.jsonl", help="外部 testset jsonl 路径"
    )
    parser.add_argument("--sample", type=int, default=None, help="只跑前 N 条")
    parser.add_argument("--start-index", type=int, default=0, help="从第 N 条开始（0 起始）")
    parser.add_argument("--output", default="data/external_benchmark.json", help="结果输出路径")
    parser.add_argument("--cache-ttl", type=int, default=720, help="搜索缓存 TTL（小时）")
    parser.add_argument("--per-case-timeout", type=int, default=90, help="单条 case 超时秒数")
    args = parser.parse_args()

    import truthnote.pipeline as pipeline  # noqa: F401
    from truthnote.pipeline import verify_message

    if args.cache_ttl != 24:
        from truthnote import pipeline as _pipeline_mod
        from truthnote import search as _search_mod

        _orig_get = _search_mod.get_search_provider

        def _patched_get(**kwargs):
            provider = _orig_get(**kwargs)
            if hasattr(provider, "ttl_seconds"):
                provider.ttl_seconds = args.cache_ttl * 3600
            return provider

        _search_mod.get_search_provider = _patched_get
        _pipeline_mod.get_search_provider = _patched_get

    cases = load_jsonl(Path(args.testset))
    if args.start_index:
        cases = cases[args.start_index :]
    if args.sample:
        cases = cases[: args.sample]
    print(f"加载 {len(cases)} 条外部验证样本（{args.testset}，从 #{args.start_index + 1} 起）")

    # 中间结果增量落盘——脚本中途死也能拿到部分数据
    incremental_path = Path(args.output).with_suffix(".incremental.jsonl")
    if incremental_path.exists():
        incremental_path.unlink()
    print(f"中间结果增量写入：{incremental_path}")

    results, errors = run_eval(
        cases,
        verify_message,
        per_case_timeout=args.per_case_timeout,
        incremental_path=incremental_path,
    )
    metrics = compute_metrics(results)
    print_report(metrics, errors)

    output_data = {
        "metrics": metrics,
        "results": results,
        "errors": errors,
        "testset": args.testset,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    print(f"\n结果写入：{args.output}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""TRUE case benchmark：验证系统不会误伤真消息。

读取 data/true_cases.jsonl（17 条已确认为真的消息），
跑 TruthNote 流水线，核心指标：
- not_false_rate: 系统没有把真消息判为谣言的比例（保守门）
- true_leaning_rate: 正确判为 TRUE/MOSTLY_TRUE 的比例

用法：
    python scripts/run_true_case_benchmark.py
    python scripts/run_true_case_benchmark.py --no-memory   # 不用缓存
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

# 三档评分
PROBLEMATIC = {"谣言", "大部分不实", "误导性信息"}  # 有问题
OK = {"属实", "部分属实"}  # 没问题
UNCERTAIN = {"无法核实"}  # 不确定

VERDICT_MAP = {
    "FALSE": "谣言",
    "MOSTLY_FALSE": "大部分不实",
    "MISLEADING": "误导性信息",
    "PARTLY_TRUE": "部分属实",
    "TRUE": "属实",
    "UNVERIFIABLE": "无法核实",
}


def load_cases(path: Path) -> list[dict]:
    cases = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                cases.append(json.loads(line))
    return cases


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-memory", action="store_true", help="禁用记忆缓存")
    parser.add_argument("--per-case-timeout", type=int, default=120, help="单条超时秒数")
    args = parser.parse_args()

    from truthnote.pipeline import verify_message

    cases = load_cases(ROOT / "data" / "true_cases.jsonl")
    print(f"加载 {len(cases)} 条 TRUE case")

    output_path = ROOT / "data" / "true_case_benchmark_glm.jsonl"
    if output_path.exists():
        output_path.unlink()

    results = []
    for i, case in enumerate(cases):
        text = case["text"]
        expected = case["expected"]
        topic = case.get("topic", "")
        acceptable = set(case.get("acceptable", [expected]))
        acceptable_zh = {VERDICT_MAP.get(v, v) for v in acceptable}

        print(f"\n[{i + 1}/{len(cases)}] ({topic}) {text[:50]}...")

        t0 = time.time()
        try:
            from concurrent.futures import ThreadPoolExecutor

            with ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(verify_message, text, use_memory=not args.no_memory)
                result = future.result(timeout=args.per_case_timeout)
            elapsed = time.time() - t0
            actual = result.overall_verdict.value
            tier = (
                "ok" if actual in OK else ("problematic" if actual in PROBLEMATIC else "uncertain")
            )
            acceptable_ok = actual in acceptable_zh

            status = "✓" if tier == "ok" else ("~" if tier == "uncertain" else "✗")
            print(f"  {status} {elapsed:.1f}s | 实际: {actual} ({tier}) | 预期: {expected}")

            entry = {
                "text": text,
                "topic": topic,
                "expected": expected,
                "actual": actual,
                "tier": tier,
                "acceptable_ok": acceptable_ok,
                "elapsed": round(elapsed, 2),
                "summary": result.summary[:200] if result.summary else "",
            }
        except Exception as e:
            elapsed = time.time() - t0
            print(f"  ✗ ERROR {elapsed:.1f}s | {e}")
            entry = {
                "text": text,
                "topic": topic,
                "expected": expected,
                "actual": "ERROR",
                "tier": "error",
                "acceptable_ok": False,
                "elapsed": round(elapsed, 2),
                "error": str(e),
            }

        results.append(entry)
        with open(output_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    # 汇总（三档评分）
    total = len(results)
    ok_count = sum(1 for r in results if r.get("tier") == "ok")
    uncertain_count = sum(1 for r in results if r.get("tier") == "uncertain")
    problematic_count = sum(1 for r in results if r.get("tier") == "problematic")
    error_count = sum(1 for r in results if r.get("tier") == "error")

    print("\n" + "=" * 60)
    print("TRUE Case Benchmark 报告（三档评分）")
    print("=" * 60)
    print(f"总条数: {total}")
    print(f"  ✓ 没问题（属实/部分属实）: {ok_count}/{total} ({ok_count / total:.1%})")
    print(
        f"  ~ 不确定（无法核实）:       {uncertain_count}/{total} ({uncertain_count / total:.1%})"
    )
    print(
        f"  ✗ 有问题（误判为假）:       {problematic_count}/{total} ({problematic_count / total:.1%})"  # noqa: E501
    )
    if error_count:
        print(f"  ! 错误:                     {error_count}/{total}")

    # 按 topic 分项
    from collections import defaultdict

    by_topic = defaultdict(list)
    for r in results:
        by_topic[r["topic"]].append(r)

    print(f"\n{'主题':<8} {'没问题':>6} {'不确定':>6} {'有问题':>6} {'总数':>4}")
    print("-" * 38)
    for topic, rs in sorted(by_topic.items()):
        t_ok = sum(1 for r in rs if r.get("tier") == "ok")
        t_unc = sum(1 for r in rs if r.get("tier") == "uncertain")
        t_prob = sum(1 for r in rs if r.get("tier") == "problematic")
        print(f"{topic:<8} {t_ok:>6} {t_unc:>6} {t_prob:>6} {len(rs):>4}")

    # 误判详情（有问题的）
    misses = [r for r in results if r.get("tier") == "problematic"]
    if misses:
        print(f"\n误判为有问题的 ({len(misses)} 条):")
        for r in misses:
            print(f"  [{r['topic']}] {r['text'][:40]}... → {r['actual']}")

    print(f"\n结果写入: {output_path}")


if __name__ == "__main__":
    main()

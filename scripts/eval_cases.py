"""TruthNote 评测脚本。

加载 scenarios/rumor_testset.json，对每条谣言跑完整流水线，
比对预期判定和实际判定，输出分类别准确率报告。

用法：
    python scripts/eval_cases.py                       # 跑全量
    python scripts/eval_cases.py --index 0             # 跑单条
    python scripts/eval_cases.py --category 灾难恐慌    # 跑某类别
    python scripts/eval_cases.py --dry-run              # 只统计条目不跑
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

VERDICT_ALIASES = {
    "谣言": "谣言",
    "大部分不实": "大部分不实",
    "误导性信息": "误导性信息",
    "部分属实": "部分属实",
    "属实": "属实",
    "无法核实": "无法核实",
}


def load_testset(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def verdict_match(expected: str, actual: str) -> bool:
    return VERDICT_ALIASES.get(expected, expected) == VERDICT_ALIASES.get(actual, actual)


def run_single(case: dict, idx: int, total: int, verify_fn=None) -> dict:
    case_id = case["id"]
    msg = case["message"]
    expected = case["expected_verdict"]
    category = case["category"]

    print(f"\n[{idx + 1}/{total}] {case_id} ({category})")
    print(f"  消息：{msg[:60]}...")

    t0 = time.perf_counter()
    try:
        result = verify_fn(msg)
        actual = result.overall_verdict.value
        elapsed = time.perf_counter() - t0
        match = verdict_match(expected, actual)
        print(f"  预期：{expected} | 实际：{actual} | {'✓' if match else '✗'} | {elapsed:.1f}s")
        return {
            "id": case_id,
            "category": category,
            "expected": expected,
            "actual": actual,
            "match": match,
            "elapsed_s": round(elapsed, 2),
            "summary": result.summary,
            "friendly_reply": result.friendly_reply,
            "error": None,
        }
    except Exception as e:
        elapsed = time.perf_counter() - t0
        print(f"  错误：{e} | {elapsed:.1f}s")
        return {
            "id": case_id,
            "category": category,
            "expected": expected,
            "actual": "ERROR",
            "match": False,
            "elapsed_s": round(elapsed, 2),
            "summary": "",
            "friendly_reply": "",
            "error": str(e),
        }


def print_report(results: list[dict]) -> None:
    print("\n" + "=" * 70)
    print("评测报告")
    print("=" * 70)

    total = len(results)
    correct = sum(1 for r in results if r["match"])
    errors = sum(1 for r in results if r["error"])
    total_time = sum(r["elapsed_s"] for r in results)

    print(f"\n总体准确率：{correct}/{total} ({correct / total * 100:.1f}%)")
    print(f"错误数：{errors}")
    print(f"总耗时：{total_time:.1f}s（平均 {total_time / total:.1f}s/条）")

    by_category: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        by_category[r["category"]].append(r)

    print(f"\n{'类别':<12} {'正确':>4} {'总数':>4} {'准确率':>8} {'平均耗时':>10}")
    print("-" * 45)
    for cat, cat_results in sorted(by_category.items()):
        cat_correct = sum(1 for r in cat_results if r["match"])
        cat_total = len(cat_results)
        cat_time = sum(r["elapsed_s"] for r in cat_results)
        avg_time = cat_time / cat_total
        pct = cat_correct / cat_total * 100
        print(f"{cat:<12} {cat_correct:>4} {cat_total:>4} {pct:>7.1f}% {avg_time:>9.1f}s")

    mismatches = [r for r in results if not r["match"] and not r["error"]]
    if mismatches:
        print(f"\n误判详情（{len(mismatches)} 条）：")
        for r in mismatches:
            print(f"  [{r['id']}] 预期 {r['expected']} → 实际 {r['actual']}")

    if errors:
        error_cases = [r for r in results if r["error"]]
        print(f"\n错误详情（{len(error_cases)} 条）：")
        for r in error_cases:
            print(f"  [{r['id']}] {r['error'][:80]}")


def save_report(results: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n详细结果已保存：{path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="TruthNote 评测脚本")
    parser.add_argument("--index", type=int, default=None, help="只跑指定索引的用例")
    parser.add_argument("--category", type=str, default=None, help="只跑指定类别")
    parser.add_argument("--dry-run", action="store_true", help="只统计条目不实际跑")
    parser.add_argument("--sample", type=int, default=None, help="每类别取N条（跑代表性子集）")
    parser.add_argument("--model", type=str, default=None, help="覆盖 LLM 模型")
    parser.add_argument(
        "--provider", type=str, default=None, help="覆盖 LLM 提供商（claude_cli/openai/anthropic）"
    )
    parser.add_argument(
        "--testset",
        type=str,
        default=str(ROOT / "scenarios" / "rumor_testset.json"),
        help="测试集路径",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(ROOT / "eval_results" / "latest.json"),
        help="结果输出路径",
    )
    args = parser.parse_args()

    if args.model:
        os.environ["DEFAULT_MODEL"] = args.model
    if args.provider:
        os.environ["LLM_PROVIDER"] = args.provider
        if args.provider == "claude_cli" and not args.model:
            os.environ["DEFAULT_MODEL"] = ""

    from truthnote.pipeline import verify_message  # noqa: E402 — 延迟导入，确保 env 覆盖生效

    cases = load_testset(Path(args.testset))
    print(f"加载测试集：{len(cases)} 条用例")

    if args.index is not None:
        if 0 <= args.index < len(cases):
            cases = [cases[args.index]]
        else:
            print(f"索引 {args.index} 超出范围 [0, {len(cases) - 1}]")
            sys.exit(1)

    if args.category:
        cases = [c for c in cases if c["category"] == args.category]
        if not cases:
            print(f"未找到类别 '{args.category}' 的用例")
            sys.exit(1)

    if args.sample:
        sampled = []
        by_cat: dict[str, list[dict]] = defaultdict(list)
        for c in cases:
            by_cat[c["category"]].append(c)
        for cat_cases in by_cat.values():
            sampled.extend(cat_cases[: args.sample])
        cases = sampled
        print(f"采样模式：每类别取 {args.sample} 条")

    print(f"待评测：{len(cases)} 条")

    by_cat = defaultdict(int)
    for c in cases:
        by_cat[c["category"]] += 1
    for cat, cnt in sorted(by_cat.items()):
        print(f"  {cat}: {cnt} 条")

    if args.dry_run:
        print("\n[dry-run] 不实际执行评测")
        return

    results = []
    for i, case in enumerate(cases):
        result = run_single(case, i, len(cases), verify_fn=verify_message)
        results.append(result)

    print_report(results)
    save_report(results, Path(args.output))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""TruthNote 消融实验脚本。

逐层去掉组件，证明每一层都在起作用：
  A. 纯 LLM（无搜索、无规则、无保守门）
  B. 搜索 + LLM（无规则、无保守门）
  C. 搜索 + LLM + 规则引擎（无保守门）
  D. 完整流水线（搜索 + LLM + 规则 + 保守门）

用法：
    python scripts/run_ablation.py                    # 跑 9 条 sample
    python scripts/run_ablation.py --sample 1         # 每品类 1 条
    python scripts/run_ablation.py --output data/ablation_report.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

VERDICT_ORDER = ["谣言", "大部分不实", "误导性信息", "部分属实", "属实", "无法核实"]
VERDICT_ACCEPTABLE = {
    "谣言": {"谣言"},
    "大部分不实": {"大部分不实", "谣言"},
    "误导性信息": {"误导性信息", "大部分不实"},
    "部分属实": {"部分属实"},
    "属实": {"属实"},
    "无法核实": {"无法核实"},
}
FALSE_VERDICTS = {"谣言", "大部分不实", "误导性信息"}

ABLATION_VARIANTS = {
    "A_llm_only": {"use_search": False, "use_rules": False, "use_gates": False},
    "B_search_llm": {"use_search": True, "use_rules": False, "use_gates": False},
    "C_search_llm_rules": {"use_search": True, "use_rules": True, "use_gates": False},
    "D_full_pipeline": {"use_search": True, "use_rules": True, "use_gates": True},
}


def binary_label(verdict: str) -> str:
    return "FALSE" if verdict in FALSE_VERDICTS else "TRUE"


def load_testset(path: Path, category: str | None = None) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        cases = json.load(f)
    if category:
        cases = [c for c in cases if c["category"] == category]
    return cases


def run_variant(cases: list[dict], variant_name: str, flags: dict) -> list[dict]:
    from truthnote.pipeline import verify_message

    results = []
    print(f"\n{'=' * 60}")
    print(f"消融变体: {variant_name}")
    print(f"  搜索={flags['use_search']}  规则={flags['use_rules']}  保守门={flags['use_gates']}")
    print(f"{'=' * 60}")

    for i, case in enumerate(cases):
        cid = case["id"]
        msg = case["message"]
        expected = case["expected_verdict"]

        print(f"  [{i + 1}/{len(cases)}] {cid}: {msg[:35]}...", end=" ", flush=True)

        t0 = time.time()
        try:
            result = verify_message(
                msg,
                use_memory=False,
                use_rules=flags["use_rules"],
                use_gates=flags["use_gates"],
                use_search=flags["use_search"],
            )
            elapsed = time.time() - t0
            actual = result.overall_verdict.value

            if "acceptable_verdicts" in case:
                acceptable = set(case["acceptable_verdicts"])
            else:
                acceptable = VERDICT_ACCEPTABLE.get(expected, {expected})
            ok = actual in acceptable
            binary_ok = binary_label(expected) == binary_label(actual)

            status = "✓" if ok else "✗"
            print(f"{status} {elapsed:.1f}s | {actual} (期望 {expected})")

            results.append(
                {
                    "id": cid,
                    "category": case["category"],
                    "expected": expected,
                    "actual": actual,
                    "ok": ok,
                    "binary_ok": binary_ok,
                    "elapsed_sec": round(elapsed, 2),
                }
            )
        except Exception as e:
            elapsed = time.time() - t0
            print(f"ERROR ({elapsed:.1f}s): {e}")
            results.append(
                {
                    "id": cid,
                    "category": case["category"],
                    "expected": expected,
                    "actual": "ERROR",
                    "ok": False,
                    "binary_ok": False,
                    "elapsed_sec": round(elapsed, 2),
                }
            )

    return results


def compute_summary(results: list[dict]) -> dict:
    total = len(results)
    correct = sum(1 for r in results if r["ok"])
    binary_correct = sum(1 for r in results if r["binary_ok"])
    return {
        "total": total,
        "correct": correct,
        "accuracy": correct / total if total else 0,
        "binary_correct": binary_correct,
        "binary_accuracy": binary_correct / total if total else 0,
    }


def print_ablation_table(all_results: dict[str, list[dict]], cases: list[dict]):
    print("\n" + "=" * 70)
    print("消融实验结果对比表")
    print("=" * 70)

    summaries = {}
    for variant, results in all_results.items():
        summaries[variant] = compute_summary(results)

    print(f"\n{'变体':<25} {'六分类':>10} {'二分类':>10} {'正确/总':>10}")
    print("-" * 58)
    for variant in ABLATION_VARIANTS:
        s = summaries[variant]
        print(
            f"{variant:<25} {s['accuracy']:>9.1%} {s['binary_accuracy']:>9.1%}"
            f" {s['correct']}/{s['total']:>8}"
        )

    # 逐 case 对比
    print(f"\n{'Case':<20}", end="")
    for variant in ABLATION_VARIANTS:
        print(f" {variant[:12]:>14}", end="")
    print(f" {'金标':>8}")
    print("-" * (20 + 14 * len(ABLATION_VARIANTS) + 10))

    for case in cases:
        cid = case["id"]
        print(f"{cid:<20}", end="")
        for variant in ABLATION_VARIANTS:
            r = next((r for r in all_results[variant] if r["id"] == cid), None)
            if r:
                mark = "✓" if r["ok"] else " "
                print(f" {mark}{r['actual'][:6]:>13}", end="")
            else:
                print(f" {'N/A':>14}", end="")
        print(f" {case['expected_verdict'][:6]:>8}")

    # 增量贡献
    print("\n增量贡献分析:")
    variants = list(ABLATION_VARIANTS.keys())
    for i in range(1, len(variants)):
        prev = summaries[variants[i - 1]]
        curr = summaries[variants[i]]
        delta_6 = curr["accuracy"] - prev["accuracy"]
        delta_2 = curr["binary_accuracy"] - prev["binary_accuracy"]
        component = variants[i].split("_", 1)[1]
        print(f"  +{component}: 六分类 {delta_6:+.1%}, 二分类 {delta_2:+.1%}")


def main():
    parser = argparse.ArgumentParser(description="TruthNote 消融实验")
    parser.add_argument("--sample", type=int, help="每品类取 N 条")
    parser.add_argument("--category", type=str, help="只跑某品类")
    parser.add_argument("--output", type=str, default="data/ablation_report.json")
    parser.add_argument(
        "--testset", type=str, default=str(ROOT / "scenarios" / "rumor_testset.json")
    )
    args = parser.parse_args()

    cases = load_testset(Path(args.testset), args.category)

    if args.sample:
        sampled = []
        by_cat: dict[str, list[dict]] = defaultdict(list)
        for c in cases:
            by_cat[c["category"]].append(c)
        for cat_cases in by_cat.values():
            sampled.extend(cat_cases[: args.sample])
        cases = sampled

    print(f"TruthNote 消融实验 — {len(cases)} 条 × {len(ABLATION_VARIANTS)} 变体")

    all_results: dict[str, list[dict]] = {}

    for variant_name, flags in ABLATION_VARIANTS.items():
        all_results[variant_name] = run_variant(cases, variant_name, flags)

    print_ablation_table(all_results, cases)

    # 保存报告
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "total_cases": len(cases),
        "variants": {
            name: {
                "flags": flags,
                "summary": compute_summary(all_results[name]),
                "results": all_results[name],
            }
            for name, flags in ABLATION_VARIANTS.items()
        },
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n报告已保存: {output_path}")


if __name__ == "__main__":
    main()

"""TruthNote 国产模型对比评测。

对比不同 LLM 在核查任务上的表现：
- 准确率
- JSON 解析成功率
- 平均耗时
- 失败类型分布

用法：
    python scripts/eval_model_comparison.py                      # 用当前模型跑全量
    python scripts/eval_model_comparison.py --models qwen-max,deepseek-chat  # 对比多个模型
    python scripts/eval_model_comparison.py --count 10           # 只跑前 10 条
    python scripts/eval_model_comparison.py --report             # 只输出报告（读取已有结果）
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

RESULTS_DIR = ROOT / "data" / "eval_results"
TESTSET_PATH = ROOT / "scenarios" / "rumor_testset.json"


def load_testset() -> list[dict]:
    with open(TESTSET_PATH, encoding="utf-8") as f:
        return json.load(f)


def run_eval(model: str, cases: list[dict]) -> dict:
    """对指定模型跑评测。"""
    os.environ["DEFAULT_MODEL"] = model
    os.environ["LLM_PROVIDER"] = "openai"

    from truthnote.pipeline import verify_message

    results = {
        "model": model,
        "total": len(cases),
        "correct": 0,
        "json_success": 0,
        "json_fail": 0,
        "errors": [],
        "avg_duration_ms": 0,
        "by_category": defaultdict(lambda: {"total": 0, "correct": 0}),
        "cases": [],
    }

    total_ms = 0
    for i, case in enumerate(cases):
        msg = case["message"]
        expected = case["expected_verdict"]
        category = case["category"]
        t0 = time.perf_counter()

        try:
            result = verify_message(msg, use_memory=False)
            duration_ms = int((time.perf_counter() - t0) * 1000)
            actual = result.overall_verdict.value
            is_correct = actual == expected
            results["json_success"] += 1

            if is_correct:
                results["correct"] += 1
                results["by_category"][category]["correct"] += 1

            results["cases"].append(
                {
                    "id": case["id"],
                    "message": msg[:50],
                    "expected": expected,
                    "actual": actual,
                    "correct": is_correct,
                    "duration_ms": duration_ms,
                    "category": category,
                }
            )
        except json.JSONDecodeError:
            duration_ms = int((time.perf_counter() - t0) * 1000)
            results["json_fail"] += 1
            results["errors"].append({"id": case["id"], "error": "JSON解析失败"})
            results["cases"].append(
                {
                    "id": case["id"],
                    "message": msg[:50],
                    "expected": expected,
                    "actual": "ERROR",
                    "correct": False,
                    "duration_ms": duration_ms,
                    "category": category,
                }
            )
        except Exception as e:
            duration_ms = int((time.perf_counter() - t0) * 1000)
            results["errors"].append({"id": case["id"], "error": str(e)[:100]})
            results["cases"].append(
                {
                    "id": case["id"],
                    "message": msg[:50],
                    "expected": expected,
                    "actual": "ERROR",
                    "correct": False,
                    "duration_ms": duration_ms,
                    "category": category,
                }
            )

        total_ms += duration_ms
        results["by_category"][category]["total"] += 1

        actual = results["cases"][-1]["actual"]
        print(f"  [{i + 1}/{len(cases)}] {msg[:30]}... → {actual} ({duration_ms}ms)")

    results["avg_duration_ms"] = total_ms // max(len(cases), 1)
    results["accuracy"] = results["correct"] / max(results["total"], 1)
    results["json_success_rate"] = results["json_success"] / max(results["total"], 1)
    return results


def print_report(all_results: list[dict]) -> None:
    """输出对比报告表格。"""
    print("\n" + "=" * 70)
    print("TruthNote 国产模型评测对比报告")
    print("=" * 70)

    # Summary table
    print(f"\n{'模型':<25} {'准确率':>8} {'JSON成功率':>10} {'平均耗时':>10} {'失败数':>6}")
    print("-" * 65)
    for r in all_results:
        print(
            f"{r['model']:<25} "
            f"{r['accuracy'] * 100:>6.1f}% "
            f"{r['json_success_rate'] * 100:>8.1f}% "
            f"{r['avg_duration_ms']:>7d}ms "
            f"{len(r['errors']):>5d}"
        )
    print("-" * 65)

    # Category breakdown for best model
    if all_results:
        best = max(all_results, key=lambda r: r["accuracy"])
        print(f"\n最佳模型 [{best['model']}] 按类别准确率：")
        for cat, data in sorted(best["by_category"].items()):
            acc = data["correct"] / max(data["total"], 1) * 100
            bar = "█" * int(acc / 10) + "░" * (10 - int(acc / 10))
            print(f"  {cat:<10} {bar} {acc:.0f}% ({data['correct']}/{data['total']})")

    # Error types
    for r in all_results:
        if r["errors"]:
            print(f"\n[{r['model']}] 错误详情（前5条）：")
            for err in r["errors"][:5]:
                print(f"  - #{err['id']}: {err['error']}")


def save_results(all_results: list[dict]) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    path = RESULTS_DIR / f"eval_{timestamp}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n结果已保存：{path}")


def main():
    parser = argparse.ArgumentParser(description="TruthNote 国产模型对比评测")
    parser.add_argument("--models", default="", help="逗号分隔的模型列表")
    parser.add_argument("--count", type=int, default=0, help="只跑前 N 条")
    parser.add_argument("--report", action="store_true", help="读取最新结果输出报告")
    args = parser.parse_args()

    if args.report:
        if not RESULTS_DIR.exists():
            print("无评测结果")
            return
        latest = sorted(RESULTS_DIR.glob("eval_*.json"))[-1]
        with open(latest, encoding="utf-8") as f:
            all_results = json.load(f)
        print_report(all_results)
        return

    cases = load_testset()
    if args.count > 0:
        cases = cases[: args.count]

    models = (
        [m.strip() for m in args.models.split(",") if m.strip()] if args.models else ["qwen-max"]
    )

    all_results = []
    for model in models:
        print(f"\n{'=' * 50}")
        print(f"评测模型：{model}（{len(cases)} 条用例）")
        print(f"{'=' * 50}")
        result = run_eval(model, cases)
        all_results.append(result)

    print_report(all_results)
    save_results(all_results)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""TruthNote 基准评测脚本。

在 eval_baseline.py 基础上增加 OpenFactCheck 风格指标：
- P/R/F1（宏平均 + 加权平均 + 每类）
- 混淆矩阵
- 二分类折叠（谣言/大部分不实/误导性信息 → FALSE，其余 → TRUE）
- 耗时 + token 统计

用法：
    python scripts/run_benchmark.py                    # 跑全部 46 条
    python scripts/run_benchmark.py --sample 1         # 每品类 1 条（快速验证）
    python scripts/run_benchmark.py --category 政策法规  # 只跑某类
    python scripts/run_benchmark.py --output data/benchmark_report.json
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from collections import Counter, defaultdict
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


def binary_label(verdict: str) -> str:
    return "FALSE" if verdict in FALSE_VERDICTS else "TRUE"


def load_testset(path: Path, category: str | None = None) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        cases = json.load(f)
    if category:
        cases = [c for c in cases if c["category"] == category]
    return cases


def run_eval(cases: list[dict], verify_fn) -> tuple[list[dict], list[dict]]:
    results = []
    errors = []

    for i, case in enumerate(cases):
        cid = case["id"]
        msg = case["message"]
        expected = case["expected_verdict"]

        print(f"  [{i + 1}/{len(cases)}] {cid}: {msg[:40]}...", end=" ", flush=True)

        t0 = time.time()
        try:
            result = verify_fn(msg, use_memory=True)
            elapsed = time.time() - t0
            actual = result.overall_verdict.value

            if "acceptable_verdicts" in case:
                acceptable = set(case["acceptable_verdicts"])
            else:
                acceptable = VERDICT_ACCEPTABLE.get(expected, {expected})
            ok = actual in acceptable

            token_total = result.trace.total_tokens_used if result.trace else 0
            llm_calls = result.trace.total_llm_calls if result.trace else 0

            status = "✓" if ok else "✗"
            print(f"{status} {elapsed:.1f}s | {actual} (期望 {expected}) | {llm_calls} calls")

            # 收集诊断信息
            diagnostics_data = []
            if result.trace and result.trace.diagnostics:
                for diag in result.trace.diagnostics:
                    diagnostics_data.append(diag.model_dump())

            result_entry = {
                "id": cid,
                "category": case["category"],
                "expected": expected,
                "actual": actual,
                "ok": ok,
                "expected_binary": binary_label(expected),
                "actual_binary": binary_label(actual),
                "binary_ok": binary_label(expected) == binary_label(actual),
                "elapsed_sec": round(elapsed, 2),
                "llm_calls": llm_calls,
                "tokens": token_total,
                "claims_count": len(result.claims),
            }
            if diagnostics_data:
                result_entry["diagnostics"] = diagnostics_data
            results.append(result_entry)

            # 误判时打印诊断摘要
            if not ok and diagnostics_data:
                print("    ╰─ 诊断链路:")
                for diag in diagnostics_data:
                    for dp in diag.get("decisions", []):
                        marker = "→" if dp.get("fired") else "·"
                        vb = dp.get("verdict_before", "")
                        va = dp.get("verdict_after", "")
                        change = (
                            f" {vb}→{va}" if vb and va and vb != va else f" ={va}" if va else ""
                        )
                        print(f"       {marker} [{dp['stage']}]{change}: {dp['detail'][:80]}")
        except Exception as e:
            elapsed = time.time() - t0
            print(f"💥 ERROR ({elapsed:.1f}s): {e}")
            errors.append({"id": cid, "category": case["category"], "error": str(e)})

    return results, errors


def compute_metrics(results: list[dict]) -> dict:
    """计算完整指标：accuracy, P/R/F1, 混淆矩阵, 二分类。"""
    if not results:
        return {}

    y_true = [r["expected"] for r in results]
    y_pred = [r["actual"] for r in results]
    y_true_bin = [r["expected_binary"] for r in results]
    y_pred_bin = [r["actual_binary"] for r in results]

    total = len(results)
    correct = sum(1 for r in results if r["ok"])
    binary_correct = sum(1 for r in results if r["binary_ok"])

    # 每类 P/R/F1
    per_class = {}
    all_labels = sorted(
        set(y_true) | set(y_pred),
        key=lambda x: VERDICT_ORDER.index(x) if x in VERDICT_ORDER else 99,
    )
    for label in all_labels:
        tp = sum(1 for t, p in zip(y_true, y_pred, strict=False) if t == label and p == label)
        fp = sum(1 for t, p in zip(y_true, y_pred, strict=False) if t != label and p == label)
        fn = sum(1 for t, p in zip(y_true, y_pred, strict=False) if t == label and p != label)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        support = sum(1 for t in y_true if t == label)
        per_class[label] = {"precision": precision, "recall": recall, "f1": f1, "support": support}

    # 宏平均 / 加权平均
    labels_with_support = [lb for lb in all_labels if per_class[lb]["support"] > 0]
    macro_p = sum(per_class[lb]["precision"] for lb in labels_with_support) / max(
        len(labels_with_support), 1
    )
    macro_r = sum(per_class[lb]["recall"] for lb in labels_with_support) / max(
        len(labels_with_support), 1
    )
    macro_f1 = sum(per_class[lb]["f1"] for lb in labels_with_support) / max(
        len(labels_with_support), 1
    )

    total_support = sum(per_class[lb]["support"] for lb in labels_with_support)
    weighted_p = sum(
        per_class[lb]["precision"] * per_class[lb]["support"] for lb in labels_with_support
    ) / max(total_support, 1)
    weighted_r = sum(
        per_class[lb]["recall"] * per_class[lb]["support"] for lb in labels_with_support
    ) / max(total_support, 1)
    weighted_f1 = sum(
        per_class[lb]["f1"] * per_class[lb]["support"] for lb in labels_with_support
    ) / max(total_support, 1)

    # 混淆矩阵
    confusion = {}
    for true_label in all_labels:
        confusion[true_label] = {}
        for pred_label in all_labels:
            confusion[true_label][pred_label] = sum(
                1
                for t, p in zip(y_true, y_pred, strict=False)
                if t == true_label and p == pred_label
            )

    # 二分类指标
    bin_tp = sum(
        1 for t, p in zip(y_true_bin, y_pred_bin, strict=False) if t == "FALSE" and p == "FALSE"
    )
    bin_fp = sum(
        1 for t, p in zip(y_true_bin, y_pred_bin, strict=False) if t == "TRUE" and p == "FALSE"
    )
    bin_fn = sum(
        1 for t, p in zip(y_true_bin, y_pred_bin, strict=False) if t == "FALSE" and p == "TRUE"
    )
    bin_p = bin_tp / (bin_tp + bin_fp) if (bin_tp + bin_fp) > 0 else 0.0
    bin_r = bin_tp / (bin_tp + bin_fn) if (bin_tp + bin_fn) > 0 else 0.0
    bin_f1 = 2 * bin_p * bin_r / (bin_p + bin_r) if (bin_p + bin_r) > 0 else 0.0

    # 按品类
    by_category = defaultdict(lambda: {"total": 0, "correct": 0})
    for r in results:
        by_category[r["category"]]["total"] += 1
        if r["ok"]:
            by_category[r["category"]]["correct"] += 1

    # 耗时
    timings = [r["elapsed_sec"] for r in results]
    tokens = [r["tokens"] for r in results]

    return {
        "accuracy": correct / total,
        "total": total,
        "correct": correct,
        "binary_accuracy": binary_correct / total,
        "per_class": per_class,
        "macro": {"precision": macro_p, "recall": macro_r, "f1": macro_f1},
        "weighted": {"precision": weighted_p, "recall": weighted_r, "f1": weighted_f1},
        "binary": {"precision": bin_p, "recall": bin_r, "f1": bin_f1},
        "confusion_matrix": confusion,
        "by_category": dict(by_category),
        "timing": {
            "avg_sec": sum(timings) / len(timings),
            "min_sec": min(timings),
            "max_sec": max(timings),
            "total_sec": sum(timings),
        },
        "tokens": {
            "total": sum(tokens),
            "avg": sum(tokens) / len(tokens) if tokens else 0,
        },
    }


def print_report(metrics: dict, results: list[dict], errors: list[dict]):
    print("\n" + "=" * 70)
    print("TruthNote 基准评测报告")
    print("=" * 70)

    print(f"\n总计: {metrics['total']} 条 | 错误: {len(errors)} 条")
    print(f"六分类准确率: {metrics['correct']}/{metrics['total']} ({metrics['accuracy']:.1%})")
    print(f"二分类准确率: ({metrics['binary_accuracy']:.1%})")

    # 宏平均 / 加权平均
    m = metrics["macro"]
    w = metrics["weighted"]
    b = metrics["binary"]
    print(f"\n{'指标':<12} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print("-" * 45)
    print(f"{'宏平均':<12} {m['precision']:>10.3f} {m['recall']:>10.3f} {m['f1']:>10.3f}")
    print(f"{'加权平均':<12} {w['precision']:>10.3f} {w['recall']:>10.3f} {w['f1']:>10.3f}")
    print(f"{'二分类(FALSE)':<12} {b['precision']:>10.3f} {b['recall']:>10.3f} {b['f1']:>10.3f}")

    # 每类 P/R/F1
    print(f"\n{'判定类别':<12} {'P':>8} {'R':>8} {'F1':>8} {'样本数':>6}")
    print("-" * 45)
    for label, stats in metrics["per_class"].items():
        print(
            f"{label:<12} {stats['precision']:>8.3f} {stats['recall']:>8.3f}"
            f" {stats['f1']:>8.3f} {stats['support']:>6}"
        )

    # 混淆矩阵
    labels = list(metrics["confusion_matrix"].keys())
    print("\n混淆矩阵（行=期望，列=实际）:")
    header = f"{'':>12}" + "".join(f"{lb[:4]:>8}" for lb in labels)
    print(header)
    for true_label in labels:
        row = f"{true_label[:10]:>12}"
        for pred_label in labels:
            count = metrics["confusion_matrix"][true_label][pred_label]
            row += f"{count:>8}"
        print(row)

    # 按品类
    print(f"\n{'品类':<12} {'正确':>6} {'总数':>6} {'准确率':>8}")
    print("-" * 35)
    for cat, stats in sorted(metrics["by_category"].items()):
        pct = stats["correct"] / stats["total"] * 100
        print(f"{cat:<12} {stats['correct']:>6} {stats['total']:>6} {pct:>7.1f}%")

    # 耗时
    t = metrics["timing"]
    print(
        f"\n耗时: 平均 {t['avg_sec']:.1f}s | 最快 {t['min_sec']:.1f}s"
        f" | 最慢 {t['max_sec']:.1f}s | 总计 {t['total_sec']:.0f}s"
    )

    tk = metrics["tokens"]
    print(f"Token: 总计 {tk['total']:,} | 平均 {tk['avg']:.0f}/条")

    # 基线对比
    total = metrics["total"]
    correct = metrics["correct"]
    print("\n基线对比:")
    y_true = [r["expected"] for r in results]
    majority_label = max(set(y_true), key=y_true.count)
    majority_correct = sum(1 for t in y_true if t == majority_label)
    print(
        f"  多数投票基线（永远猜 {majority_label}）: "
        f"{majority_correct}/{total} ({majority_correct / total:.1%})"
    )

    cat_majority: dict[str, str] = {}
    for r in results:
        cat = r["category"]
        if cat not in cat_majority:
            cat_labels = [rr["expected"] for rr in results if rr["category"] == cat]
            cat_majority[cat] = max(set(cat_labels), key=cat_labels.count)
    cat_majority_correct = sum(
        1 for r in results if cat_majority.get(r["category"]) == r["expected"]
    )
    print(
        f"  品类多数投票基线: {cat_majority_correct}/{total} ({cat_majority_correct / total:.1%})"
    )
    print(f"  本系统: {correct}/{total} ({correct / total:.1%})")
    delta = correct / total - cat_majority_correct / total
    print(f"  vs 品类基线: {delta:+.1%}")

    # 误判详情（含诊断链路）
    wrong = [r for r in results if not r["ok"]]
    if wrong:
        print(f"\n{'=' * 70}")
        print(f"误判诊断 ({len(wrong)} 条)")
        print(f"{'=' * 70}")
        for r in wrong:
            print(f"\n  [{r['id']}] {r['category']} | 期望={r['expected']} → 实际={r['actual']}")
            if "diagnostics" in r:
                for diag in r["diagnostics"]:
                    print(f"    证据: {diag.get('evidence_summary', {})}")
                    for dp in diag.get("decisions", []):
                        marker = "→" if dp.get("fired") else "·"
                        vb = dp.get("verdict_before", "")
                        va = dp.get("verdict_after", "")
                        change = (
                            f" {vb}→{va}" if vb and va and vb != va else f" ={va}" if va else ""
                        )
                        print(f"    {marker} [{dp['stage']}]{change}")
                        print(f"      {dp['detail'][:120]}")
                        if dp.get("signals"):
                            print(f"      信号: {dp['signals']}")

    if errors:
        print(f"\n运行错误 ({len(errors)} 条):")
        for e in errors:
            print(f"  [{e['id']}] {e['error'][:80]}")


def main():
    parser = argparse.ArgumentParser(description="TruthNote 基准评测")
    parser.add_argument("--sample", type=int, help="每品类取 N 条")
    parser.add_argument("--category", type=str, help="只跑某品类")
    parser.add_argument(
        "--output", type=str, default="data/benchmark_report.json", help="输出 JSON"
    )
    parser.add_argument(
        "--testset", type=str, default=str(ROOT / "scenarios" / "rumor_testset.json")
    )
    parser.add_argument(
        "--cache-ttl", type=int, default=24, help="搜索缓存过期时间（小时），延长可复用旧缓存"
    )
    args = parser.parse_args()

    import truthnote.pipeline as pipeline
    from truthnote.memory import MemoryStore
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

    cases = load_testset(Path(args.testset), args.category)

    if args.sample:
        sampled = []
        by_cat: dict[str, list[dict]] = defaultdict(list)
        for c in cases:
            by_cat[c["category"]].append(c)
        for cat_cases in by_cat.values():
            sampled.extend(cat_cases[: args.sample])
        cases = sampled

    print(f"TruthNote 基准评测 — {len(cases)} 条")
    by_cat_count = Counter(c["category"] for c in cases)
    for cat, cnt in sorted(by_cat_count.items()):
        print(f"  {cat}: {cnt}")
    print()

    tmpdir = tempfile.mkdtemp()
    db_path = str(Path(tmpdir) / "benchmark_eval.db")
    pipeline._memory_store = MemoryStore(db_path)

    results, errors = run_eval(cases, verify_message)

    try:
        conn = pipeline._memory_store._conn()
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        conn.close()
    except Exception:
        pass
    pipeline._memory_store = None

    metrics = compute_metrics(results)
    print_report(metrics, results, errors)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "total_cases": len(cases),
        "metrics": metrics,
        "results": results,
        "errors": errors,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n报告已保存: {output_path}")


if __name__ == "__main__":
    main()

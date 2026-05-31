#!/usr/bin/env python3
"""基准集端到端评测：跑 46 条，出准确率 + 风险等级 + 干预策略 + 生命周期报告。

用法：
    python scripts/eval_baseline.py                  # 跑全部
    python scripts/eval_baseline.py --sample 5       # 跑前 5 条
    python scripts/eval_baseline.py --category 诈骗套路  # 只跑某类
    python scripts/eval_baseline.py --repeat          # 跑完后重发第一条，验证重复拦截
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

from truthnote.memory import MemoryStore  # noqa: E402
from truthnote.pipeline import verify_message  # noqa: E402

VERDICT_MATCH = {
    "谣言": ["谣言"],
    "大部分不实": ["大部分不实", "谣言"],
    "误导性信息": ["误导性信息", "大部分不实"],
    "部分属实": ["部分属实"],
    "属实": ["属实"],
    "无法核实": ["无法核实"],
}


def load_testset(path: Path, category: str | None = None) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        cases = json.load(f)
    if category:
        cases = [c for c in cases if c["category"] == category]
    return cases


def run_eval(cases: list[dict], db_path: str, test_repeat: bool = False):
    results = []
    errors = []
    timings = []

    for i, case in enumerate(cases):
        cid = case["id"]
        msg = case["message"]
        expected_verdict = case["expected_verdict"]
        expected_action = case.get("expected_action", "")
        expected_risk = case.get("risk_level", "")

        print(f"  [{i + 1}/{len(cases)}] {cid}: {msg[:40]}...", end=" ", flush=True)

        t0 = time.time()
        try:
            result = verify_message(msg, use_memory=True)
            elapsed = time.time() - t0
            timings.append(elapsed)

            actual_verdict = result.overall_verdict.value
            actual_risk = result.lifecycle.risk_level.value if result.lifecycle else ""
            actual_intervention = result.lifecycle.intervention_type if result.lifecycle else ""
            actual_state = result.lifecycle.current_state.value if result.lifecycle else ""

            acceptable = VERDICT_MATCH.get(expected_verdict, [expected_verdict])
            verdict_ok = actual_verdict in acceptable
            risk_ok = actual_risk == expected_risk if expected_risk else True
            action_ok = actual_intervention == expected_action if expected_action else True

            status = "✓" if verdict_ok else "✗"
            detail = f"风险={actual_risk} 干预={actual_intervention} 状态={actual_state}"
            print(
                f"{status} {elapsed:.1f}s | {actual_verdict} (期望 {expected_verdict}) | {detail}"
            )

            results.append(
                {
                    "id": cid,
                    "category": case["category"],
                    "expected_verdict": expected_verdict,
                    "actual_verdict": actual_verdict,
                    "verdict_ok": verdict_ok,
                    "expected_risk": expected_risk,
                    "actual_risk": actual_risk,
                    "risk_ok": risk_ok,
                    "expected_action": expected_action,
                    "actual_intervention": actual_intervention,
                    "action_ok": action_ok,
                    "lifecycle_state": actual_state,
                    "claims_count": len(result.claims),
                    "llm_calls": result.trace.total_llm_calls,
                    "elapsed_sec": round(elapsed, 2),
                    "summary": result.summary[:80],
                }
            )
        except Exception as e:
            elapsed = time.time() - t0
            print(f"💥 ERROR ({elapsed:.1f}s): {e}")
            errors.append({"id": cid, "error": str(e), "elapsed_sec": round(elapsed, 2)})

    repeat_result = None
    if test_repeat and cases:
        print("\n  [重复拦截测试] 重发第一条消息...", end=" ", flush=True)
        t0 = time.time()
        try:
            r2 = verify_message(cases[0]["message"], use_memory=True)
            elapsed = time.time() - t0
            is_blocked = r2.lifecycle.repeat_blocked if r2.lifecycle else False
            state = r2.lifecycle.current_state.value if r2.lifecycle else ""
            status = "✓ 秒拦" if is_blocked else "✗ 未拦截"
            print(f"{status} ({elapsed:.1f}s) 状态={state}")
            repeat_result = {
                "blocked": is_blocked,
                "state": state,
                "elapsed_sec": round(elapsed, 2),
                "summary": r2.summary[:80],
            }
        except Exception as e:
            print(f"💥 ERROR: {e}")

    return results, errors, timings, repeat_result


def print_report(results, errors, timings, repeat_result):
    total = len(results)
    if total == 0:
        print("\n无结果")
        return

    verdict_ok = sum(1 for r in results if r["verdict_ok"])
    risk_ok = sum(1 for r in results if r["risk_ok"])
    action_ok = sum(1 for r in results if r["action_ok"])

    print("\n" + "=" * 70)
    print("评测报告")
    print("=" * 70)

    print(f"\n总计: {total} 条 | 错误: {len(errors)} 条")
    print(f"判定准确率: {verdict_ok}/{total} ({verdict_ok / total * 100:.1f}%)")
    print(f"风险等级准确率: {risk_ok}/{total} ({risk_ok / total * 100:.1f}%)")
    print(f"干预策略准确率: {action_ok}/{total} ({action_ok / total * 100:.1f}%)")

    if timings:
        avg_t = sum(timings) / len(timings)
        print(f"平均耗时: {avg_t:.1f}s | 最快: {min(timings):.1f}s | 最慢: {max(timings):.1f}s")

    # 按类别统计
    cat_stats = defaultdict(lambda: {"total": 0, "verdict_ok": 0, "risk_ok": 0, "action_ok": 0})
    for r in results:
        cat = r["category"]
        cat_stats[cat]["total"] += 1
        if r["verdict_ok"]:
            cat_stats[cat]["verdict_ok"] += 1
        if r["risk_ok"]:
            cat_stats[cat]["risk_ok"] += 1
        if r["action_ok"]:
            cat_stats[cat]["action_ok"] += 1

    print(f"\n{'类别':<12} {'判定':>8} {'风险':>8} {'干预':>8}")
    print("-" * 40)
    for cat, s in sorted(cat_stats.items()):
        t = s["total"]
        print(f"{cat:<12} {s['verdict_ok']}/{t:>4}  {s['risk_ok']}/{t:>4}  {s['action_ok']}/{t:>4}")

    # 生命周期状态分布
    state_counts = Counter(r["lifecycle_state"] for r in results)
    print("\n生命周期状态分布:")
    for state, count in state_counts.most_common():
        print(f"  {state}: {count}")

    # 判定错误详情
    wrong = [r for r in results if not r["verdict_ok"]]
    if wrong:
        print(f"\n判定错误 ({len(wrong)} 条):")
        for r in wrong:
            print(f"  {r['id']}: 期望={r['expected_verdict']} 实际={r['actual_verdict']}")

    # 风险等级错误详情
    risk_wrong = [r for r in results if not r["risk_ok"]]
    if risk_wrong:
        print(f"\n风险等级错误 ({len(risk_wrong)} 条):")
        for r in risk_wrong:
            print(f"  {r['id']}: 期望={r['expected_risk']} 实际={r['actual_risk']}")

    # 重复拦截
    if repeat_result:
        tag = "✓ 成功" if repeat_result["blocked"] else "✗ 失败"
        print(f"\n重复拦截: {tag} ({repeat_result['elapsed_sec']:.1f}s)")

    if errors:
        print(f"\n运行错误 ({len(errors)} 条):")
        for e in errors:
            print(f"  {e['id']}: {e['error']}")


def main():
    parser = argparse.ArgumentParser(description="TruthNote 基准集评测")
    parser.add_argument("--sample", type=int, help="只跑前 N 条")
    parser.add_argument("--category", type=str, help="只跑某类别")
    parser.add_argument("--repeat", action="store_true", help="跑完后测重复拦截")
    parser.add_argument("--output", type=str, help="输出 JSON 报告路径")
    args = parser.parse_args()

    testset_path = ROOT / "scenarios" / "rumor_testset.json"
    cases = load_testset(testset_path, args.category)
    if args.sample:
        cases = cases[: args.sample]

    print("TruthNote 基准集评测")
    print(f"测试集: {len(cases)} 条")
    if args.category:
        print(f"过滤类别: {args.category}")
    print()

    # 用临时 DB 避免污染正式数据
    tmpdir = tempfile.mkdtemp()
    db_path = str(Path(tmpdir) / "eval.db")
    import truthnote.pipeline as pipeline

    pipeline._memory_store = MemoryStore(db_path)

    results, errors, timings, repeat_result = run_eval(cases, db_path, test_repeat=args.repeat)

    # 关闭 DB：WAL checkpoint + 释放锁
    try:
        closing_conn = pipeline._memory_store._conn()
        closing_conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        closing_conn.close()
    except Exception:
        pass
    pipeline._memory_store = None

    print_report(results, errors, timings, repeat_result)

    if args.output:
        report = {
            "total": len(results),
            "errors": len(errors),
            "verdict_accuracy": sum(1 for r in results if r["verdict_ok"]) / max(len(results), 1),
            "risk_accuracy": sum(1 for r in results if r["risk_ok"]) / max(len(results), 1),
            "action_accuracy": sum(1 for r in results if r["action_ok"]) / max(len(results), 1),
            "avg_elapsed_sec": sum(timings) / max(len(timings), 1),
            "results": results,
            "errors_detail": errors,
            "repeat_test": repeat_result,
        }
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"\n报告已保存: {args.output}")


if __name__ == "__main__":
    main()

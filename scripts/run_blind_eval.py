"""阶段 0.1 — 盲测评测脚本：喂一批 CANDY 消息 → 跑完整 orchestrator → 和 label 比对。

主指标（预注册 docs/preregistration.md §E）：全覆盖可执行准确率（UNVERIFIABLE 算错）。

用法：
  # dev 调参集，可反复跑（先小规模验证管道）
  python scripts/run_blind_eval.py --split dev --n 40 --out data/eval/baseline_dev_metrics.json

  # lockbox 封箱终评，只跑一次——需显式终评标志，否则拒绝
  python scripts/run_blind_eval.py --split lockbox --n 500 --i-am-running-final-lockbox-eval

铁律：
- UNVERIFIABLE 一律算错，单独报覆盖率（堵沉默刷分）。
- lockbox 需显式标志（防误读，预注册 §H）。
- 读 jsonl 一律 utf-8-sig；label 0=真/1=谣；字段 'gold evidence' 带空格。
- 前台跑 + 大 timeout（端到端单条 ≈ 125s，别后台会被回收）。
"""

import argparse
import glob
import json
import random
import time
from collections import Counter

ROOT = r"d:/VIBE CODING/A-hackthon/02-wenke-song"

# ── verdict → 二分动作（预注册 §D，冻结）──
TRUE_VERDICT = "属实"
ABSTAIN_VERDICT = "无法核实"
WARNING_VERDICTS = {"部分属实", "误导性信息", "大部分不实", "谣言"}


def verdict_to_correct(gold_label: int, verdict: str) -> bool:
    """gold_label: 0=真 / 1=谣。正确判据见预注册 §D。UNVERIFIABLE 一律错。"""
    if gold_label == 0:
        return verdict == TRUE_VERDICT
    return verdict in WARNING_VERDICTS


def compute_metrics(rows: list[dict]) -> dict:
    """纯函数：rows 每条含 gold_label(0/1) + verdict(中文值)。返回指标 dict。

    主指标 accuracy = 正确动作 / 全部（UNVERIFIABLE 算错）。
    """
    n = len(rows)
    n_true = sum(1 for r in rows if r["gold_label"] == 0)
    n_rumor = sum(1 for r in rows if r["gold_label"] == 1)

    if n == 0:
        return {
            "n": 0,
            "n_true": 0,
            "n_rumor": 0,
            "accuracy": 0.0,
            "coverage": 0.0,
            "covered_accuracy": 0.0,
            "true_false_positive_rate": 0.0,
            "true_abstain_rate": 0.0,
            "rumor_recall": 0.0,
            "warning_precision": 0.0,
            "balanced_accuracy": 0.0,
            "confusion": {},
        }

    correct = sum(1 for r in rows if verdict_to_correct(r["gold_label"], r["verdict"]))
    covered = [r for r in rows if r["verdict"] != ABSTAIN_VERDICT]
    n_covered = len(covered)
    correct_covered = sum(1 for r in covered if verdict_to_correct(r["gold_label"], r["verdict"]))

    true_rows = [r for r in rows if r["gold_label"] == 0]
    rumor_rows = [r for r in rows if r["gold_label"] == 1]
    warning_rows = [r for r in rows if r["verdict"] in WARNING_VERDICTS]

    true_fp = sum(1 for r in true_rows if r["verdict"] in WARNING_VERDICTS)  # 真→警告=误报
    true_abstain = sum(1 for r in true_rows if r["verdict"] == ABSTAIN_VERDICT)
    true_recall = sum(1 for r in true_rows if r["verdict"] == TRUE_VERDICT)  # 真→属实
    rumor_caught = sum(1 for r in rumor_rows if r["verdict"] in WARNING_VERDICTS)
    warning_actually_rumor = sum(1 for r in warning_rows if r["gold_label"] == 1)

    def safe(a: int, b: int) -> float:
        return a / b if b else 0.0

    true_recall_rate = safe(true_recall, n_true)
    rumor_recall_rate = safe(rumor_caught, n_rumor)

    # 混淆：(gold, action)，action ∈ {接受, 警告, 弃权}
    def action(v: str) -> str:
        if v == TRUE_VERDICT:
            return "接受"
        if v == ABSTAIN_VERDICT:
            return "弃权"
        return "警告"

    confusion = Counter(
        (("真" if r["gold_label"] == 0 else "谣"), action(r["verdict"])) for r in rows
    )

    return {
        "n": n,
        "n_true": n_true,
        "n_rumor": n_rumor,
        "accuracy": safe(correct, n),
        "coverage": safe(n_covered, n),
        "covered_accuracy": safe(correct_covered, n_covered),
        "true_false_positive_rate": safe(true_fp, n_true),
        "true_abstain_rate": safe(true_abstain, n_true),
        "rumor_recall": rumor_recall_rate,
        "warning_precision": safe(warning_actually_rumor, len(warning_rows)),
        "balanced_accuracy": (true_recall_rate + rumor_recall_rate) / 2,
        "confusion": {f"{g}→{a}": c for (g, a), c in sorted(confusion.items())},
    }


def _load_split(split: str) -> dict:
    path = f"{ROOT}/data/eval/{split}_ids.json"
    return json.load(open(path, encoding="utf-8"))


def _load_inputs_by_id(wanted_rumor: set[int], wanted_truth: set[int]) -> dict[int, dict]:
    """从全量 inputs 取出 wanted 的原文 + label。utf-8-sig，label 来自所在清单。"""
    out: dict[int, dict] = {}
    for pat, want, label in (
        ("data/eval/typology/inputs/chunk_*.jsonl", wanted_rumor, 1),
        ("data/eval/truth/inputs/chunk_*.jsonl", wanted_truth, 0),
    ):
        for f in sorted(glob.glob(f"{ROOT}/{pat}")):
            for line in open(f, encoding="utf-8-sig"):
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                rid = rec.get("id")
                if rid in want:
                    out[rid] = {
                        "id": rid,
                        "claim": rec.get("claim", ""),
                        "gold_label": label,
                        "domain": rec.get("domain", ""),
                    }
    return out


def _sample(split_data: dict, n: int, seed: int) -> list[dict]:
    """分层抽样：按 dev/lockbox 内的真假比例取 n 条。"""
    rng = random.Random(seed)
    rumor_ids = list(split_data["rumor_ids"])
    truth_ids = list(split_data["truth_ids"])
    total = len(rumor_ids) + len(truth_ids)
    n = min(n, total)
    n_rumor = round(n * len(rumor_ids) / total)
    n_truth = n - n_rumor
    rng.shuffle(rumor_ids)
    rng.shuffle(truth_ids)
    sel_rumor = set(rumor_ids[:n_rumor])
    sel_truth = set(truth_ids[:n_truth])
    by_id = _load_inputs_by_id(sel_rumor, sel_truth)
    missing = (sel_rumor | sel_truth) - set(by_id)
    if missing:
        print(f"⚠️  {len(missing)} 条 ID 在 inputs 里没找到，跳过")
    return [by_id[i] for i in (sel_rumor | sel_truth) if i in by_id]


def _save(out: str, split: str, seed: int, rows: list[dict], total: int) -> None:
    """每跑完一条就落盘（断点续跑的基础），并算当前已完成部分的指标。"""
    metrics = compute_metrics(rows)
    json.dump(
        {
            "split": split,
            "seed": seed,
            "done": len(rows),
            "target": total,
            "complete": len(rows) >= total,
            "metrics": metrics,
            "rows": rows,
        },
        open(out, "w", encoding="utf-8"),
        ensure_ascii=False,
        indent=1,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["dev", "lockbox"], required=True)
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--seed", type=int, default=20260529)
    ap.add_argument("--out", default="")
    ap.add_argument(
        "--mode",
        choices=["new", "baseline"],
        default="new",
        help="打分器模式（A/B 对照）：new=阶段1+2全改动 / baseline=原始打分器",
    )
    ap.add_argument(
        "--max-seconds",
        type=int,
        default=480,
        help="本次最多跑多少秒就保存退出（绕开命令行 10 分钟硬上限，分段续跑）",
    )
    ap.add_argument(
        "--i-am-running-final-lockbox-eval",
        action="store_true",
        help="lockbox 终评硬闸：不加这个标志拒绝跑 lockbox（预注册 §H）",
    )
    args = ap.parse_args()

    # ── lockbox 硬闸 ──
    if args.split == "lockbox" and not args.i_am_running_final_lockbox_eval:
        raise SystemExit(
            "❌ 拒绝：lockbox 是封箱终评、只跑一次。确认要跑请加 --i-am-running-final-lockbox-eval"
        )

    split_data = _load_split(args.split)
    samples = _sample(split_data, args.n, args.seed)
    total = len(samples)
    out = args.out or f"{ROOT}/data/eval/{args.split}_ab_{args.mode}_n{args.n}.json"

    # ── 断点续跑：读已完成，跳过 ──
    rows: list[dict] = []
    done_ids: set[int] = set()
    try:
        prev = json.load(open(out, encoding="utf-8"))
        rows = prev.get("rows", [])
        done_ids = {r["id"] for r in rows}
    except FileNotFoundError:
        pass
    todo = [s for s in samples if s["id"] not in done_ids]
    print(
        f"抽样 {total} 条（split={args.split}, n={args.n}, seed={args.seed}）"
        f" | 已完成 {len(done_ids)} | 本次待跑 {len(todo)} | 预算 {args.max_seconds}s"
    )
    if not todo:
        print("✅ 全部已完成。")
        _save(out, args.split, args.seed, rows, total)
        _print_metrics(rows)
        return

    # 懒加载 verify_message（避免测试时拉起整个 LLM 栈）
    import sys

    sys.path.insert(0, f"{ROOT}/src")
    from truthnote import dimensions as _dim
    from truthnote.pipeline import verify_message

    # A/B 打分器模式：函数运行时读 SCORER_MODE，程序化覆盖即可切换
    _dim.SCORER_MODE = args.mode
    print(f"打分器模式 SCORER_MODE = {_dim.SCORER_MODE}")

    t0 = time.time()
    for s in todo:
        elapsed = time.time() - t0
        if elapsed > args.max_seconds:
            print(f"⏸ 到预算 {args.max_seconds}s，保存退出。再次运行同命令续跑。")
            break
        try:
            resp = verify_message(s["claim"], use_memory=False)
            verdict = resp.overall_verdict.value
        except Exception as e:
            print(f"  id={s['id']} 异常：{e}")
            verdict = ABSTAIN_VERDICT  # 异常按弃权(算错)处理，不静默吞
        rows.append({**s, "verdict": verdict})
        _save(out, args.split, args.seed, rows, total)  # 每条都落盘，崩了不丢
        done = len(rows)
        el = time.time() - t0
        print(
            f"  [{done}/{total}] id={s['id']} label={s['gold_label']} → {verdict} "
            f"(本次 {el:.0f}s, ~{el / (done - len(done_ids)):.0f}s/条)"
        )

    complete = len(rows) >= total
    print(f"\n{'✅ 全部完成' if complete else '⏸ 未完成，再跑一次续上'}：{len(rows)}/{total}")
    _print_metrics(rows)
    print(f"落盘 {out}")


def _print_metrics(rows: list[dict]) -> None:
    metrics = compute_metrics(rows)
    print("=== 当前指标（已完成部分）===")
    for k, v in metrics.items():
        if k != "confusion":
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    print("  混淆:", metrics["confusion"])


if __name__ == "__main__":
    main()

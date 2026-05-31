"""阶段 1.1 — 给 design 数据（harvest 抽样 5000）打产品 MessageType 标。

为什么：先验表按 7 个 MessageType 键，但 harvest 标的是 info_type/domain，没 MessageType。
要算"每类真假计数"→ Δ 类型提升，必须先用产品分类器 ScenarioRouterAgent 标一遍。

成本：ScenarioRouter 只调 LLM（无搜索），单条 ~6s。默认子样本 800（400谣+400真），约 80min。
Jeffreys + 收缩能吃下小样本，不必标满 5000。

resumable：每条 checkpoint，可分段续跑（绕开命令行 10min 上限）。
只用 design 数据（sample_manifest 的 ID），绝不碰 dev/lockbox。

用法：
  python scripts/label_design_message_types.py --n 800 --max-seconds 480
"""

import argparse
import glob
import json
import random
import time
from collections import defaultdict

ROOT = r"d:/VIBE CODING/A-hackthon/02-wenke-song"
MANIFEST = ROOT + r"/data/eval/sample_manifest.json"
OUT = ROOT + r"/data/eval/design_message_types.json"


def _load_claims_by_id(rumor_ids: set[int], truth_ids: set[int]) -> dict[int, dict]:
    out: dict[int, dict] = {}
    for pat, want, label in (
        ("data/eval/typology/inputs/chunk_*.jsonl", rumor_ids, 1),
        ("data/eval/truth/inputs/chunk_*.jsonl", truth_ids, 0),
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
                if rec.get("id") in want:
                    out[rec["id"]] = {
                        "id": rec["id"],
                        "claim": rec.get("claim", ""),
                        "label": label,
                    }
    return out


def _sample(manifest: dict, n: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    rumor = list(manifest.get("rumor_ids", []))
    truth = list(manifest.get("truth_ids", []))
    rng.shuffle(rumor)
    rng.shuffle(truth)
    half = n // 2
    sel_rumor = set(rumor[:half])
    sel_truth = set(truth[: n - half])
    by_id = _load_claims_by_id(sel_rumor, sel_truth)
    return [by_id[i] for i in (sel_rumor | sel_truth) if i in by_id]


def _counts(rows: list[dict]) -> dict:
    """每个 MessageType 的 (真,谣) 计数，给 Δ 计算用。"""
    c: dict[str, dict[str, int]] = defaultdict(lambda: {"true": 0, "false": 0})
    for r in rows:
        mt = r["message_type"]
        c[mt]["false" if r["label"] == 1 else "true"] += 1
    return {k: v for k, v in c.items()}


def _save(rows: list[dict], total: int) -> None:
    json.dump(
        {
            "done": len(rows),
            "target": total,
            "complete": len(rows) >= total,
            "counts": _counts(rows),
            "rows": rows,
        },
        open(OUT, "w", encoding="utf-8"),
        ensure_ascii=False,
        indent=1,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=800)
    ap.add_argument("--seed", type=int, default=20260529)
    ap.add_argument("--max-seconds", type=int, default=480)
    args = ap.parse_args()

    manifest = json.load(open(MANIFEST, encoding="utf-8"))
    samples = _sample(manifest, args.n, args.seed)
    total = len(samples)

    rows: list[dict] = []
    done_ids: set[int] = set()
    try:
        prev = json.load(open(OUT, encoding="utf-8"))
        rows = prev.get("rows", [])
        done_ids = {r["id"] for r in rows}
    except FileNotFoundError:
        pass
    todo = [s for s in samples if s["id"] not in done_ids]
    print(
        f"目标 {total} | 已完成 {len(done_ids)} | 本次待跑 {len(todo)} | 预算 {args.max_seconds}s"
    )
    if not todo:
        print("✅ 全部已完成。计数：", _counts(rows))
        return

    import sys

    sys.path.insert(0, f"{ROOT}/src")
    from truthnote.agents import ScenarioRouterAgent

    router = ScenarioRouterAgent()
    t0 = time.time()
    for s in todo:
        if time.time() - t0 > args.max_seconds:
            print(f"⏸ 到预算，保存退出。再跑同命令续上。已完成 {len(rows)}/{total}")
            break
        try:
            frame = router.route_with_frame(s["claim"]).get("message_frame")
            mt = frame.message_type.value if frame else "其他"
        except Exception as e:
            print(f"  id={s['id']} 异常：{e}")
            mt = "其他"
        rows.append({**s, "message_type": mt})
        _save(rows, total)
        done = len(rows)
        el = time.time() - t0
        print(
            f"  [{done}/{total}] id={s['id']} label={s['label']} → {mt} (~{el / (done - len(done_ids)):.0f}s/条)"
        )

    print(f"\n{'✅ 全部完成' if len(rows) >= total else '⏸ 未完成，续跑'}：{len(rows)}/{total}")
    print("计数：", json.dumps(_counts(rows), ensure_ascii=False))


if __name__ == "__main__":
    main()

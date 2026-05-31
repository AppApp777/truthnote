"""从 2500 条真消息逐条分析里，聚合出 4 样可复用资产（镜像 build_harvest_assets.py 的谣言侧）：
  1) 证实信号词表     truth_signal_lexicon.json   —— 喂未来第 7 维 _dim_confirmation 的规则词表
  2) 证实路径表       confirmation_routing.json   —— 每个 info_type『该怎么找正面佐证』
  3) 误杀名单         false_positive_watchlist.json —— 产品该修的错题本（false_positive + over_abstention）
  4) 判别式库         discriminator_library.json  —— 真假边界规则（同题真消息 vs 谣言怎么分）
并生成人读汇总 docs/truth_signals_v1.md。

聚合主轴：info_type（与谣言侧同轴）+ domain_inferred（LLM 内联领域）。
base_rate 不在这里算——那是 oracle 待答的方法问题（HANDOFF §7-Q1），单独产出。
"""

import collections
import glob
import json
import os

ROOT = r"d:/VIBE CODING/A-hackthon/02-wenke-song"
OUT = ROOT + r"/data/eval/truth_sample/outputs"
ASSETS = ROOT + r"/data/eval/harvest_assets"
DOC = ROOT + r"/docs/truth_signals_v1.md"
WATCHDOC = ROOT + r"/docs/false_positive_watchlist.md"
os.makedirs(ASSETS, exist_ok=True)

# ── 加载全字段记录（utf-8-sig：subagent 写的文件带 BOM）──
recs = []
bad = 0
for f in sorted(glob.glob(os.path.join(OUT, "chunk_*.jsonl"))):
    for line in open(f, encoding="utf-8-sig"):
        line = line.strip()
        if not line:
            continue
        try:
            recs.append(json.loads(line))
        except Exception:
            bad += 1
N = len(recs)
if bad:
    print(f"⚠️  跳过 {bad} 条坏行")


def aslist(v):
    if isinstance(v, list):
        return [str(x).strip() for x in v if str(x).strip()]
    if v:
        return [str(v).strip()]
    return []


def dom(r):
    return str(r.get("domain_inferred") or "其他").strip()


def itype(r):
    return str(r.get("info_type") or "?").strip()


# ===== 1) 证实信号词表 =====
sig_global = collections.Counter()
sig_by_itype = collections.defaultdict(collections.Counter)
sig_examples = collections.defaultdict(list)  # signal_type -> 去重示例 evidence_text
sig_seen = collections.defaultdict(set)
for r in recs:
    it = itype(r)
    for s in r.get("truth_signals") or []:
        st = str(s.get("signal_type") or "?").strip()
        sig_global[st] += 1
        sig_by_itype[it][st] += 1
        ev = str(s.get("evidence_text") or "").strip()
        key = ev[:24]
        if ev and key not in sig_seen[st]:
            sig_seen[st].add(key)
            if len(sig_examples[st]) < 8:
                sig_examples[st].append(ev)
signal_lexicon = {
    "doc": "受控词证实信号分布，喂第 7 维 _dim_confirmation 规则词表",
    "total_signals": int(sum(sig_global.values())),
    "signal_type_dist": sig_global.most_common(),
    "by_info_type_top5": {it: c.most_common(5) for it, c in sig_by_itype.items()},
    "examples_by_signal": dict(sig_examples),
}
json.dump(
    signal_lexicon,
    open(ASSETS + "/truth_signal_lexicon.json", "w", encoding="utf-8"),
    ensure_ascii=False,
    indent=1,
)

# ===== 2) 证实路径表（按 info_type）=====
routing = {}
for it in sorted(set(itype(r) for r in recs)):
    rows = [r for r in recs if itype(r) == it]
    ev_types = collections.Counter()
    steps = []
    has_named = 0
    summaries, sseen = [], set()
    for r in rows:
        ce = r.get("confirmation_evidence") or {}
        for et in aslist(ce.get("evidence_types")):
            ev_types[et] += 1
        if isinstance(ce.get("confirmation_steps"), int | float):
            steps.append(ce["confirmation_steps"])
        src = str(ce.get("official_source_name") or "").strip().lower()
        if src and src not in ("unspecified", "none", "n/a", "无", ""):
            has_named += 1
        smy = str(ce.get("summary") or "").strip()
        k = smy[:20]
        if smy and k not in sseen and len(summaries) < 5:
            sseen.add(k)
            summaries.append(smy)
    routing[it] = {
        "n": len(rows),
        "named_official_source_rate": round(has_named / len(rows), 3) if rows else 0,
        "avg_confirmation_steps": round(sum(steps) / len(steps), 2) if steps else None,
        "evidence_types_dist": ev_types.most_common(),
        "sample_summaries": summaries,
    }
json.dump(
    routing,
    open(ASSETS + "/confirmation_routing.json", "w", encoding="utf-8"),
    ensure_ascii=False,
    indent=1,
)

# ===== 3) 误杀名单（错题本）=====
MISJUDGED = ("false_positive", "over_abstention")
watch = [r for r in recs if str(r.get("predicted_failure_mode")) in MISJUDGED]
by_mode = collections.Counter(str(r.get("predicted_failure_mode")) for r in watch)
by_itype = collections.Counter(itype(r) for r in watch)
by_dom = collections.Counter(dom(r) for r in watch)
by_verdict = collections.Counter(str(r.get("predicted_current_scorer_verdict")) for r in watch)
watch_records = [
    {
        "claim_id": r.get("claim_id"),
        "claim_preview": r.get("claim_preview"),
        "info_type": itype(r),
        "domain": dom(r),
        "failure_mode": r.get("predicted_failure_mode"),
        "predicted_verdict": r.get("predicted_current_scorer_verdict"),
        "prior_overpenalized": r.get("prior_overpenalized"),
        "surface_falseness_signals": aslist(r.get("surface_falseness_signals")),
        "why_misjudged": r.get("why_misjudged"),
        "gold_label_confidence": r.get("gold_label_confidence"),
    }
    for r in watch
]
watchlist = {
    "doc": "打分器会误杀的真消息错题本：false_positive=误判为假，over_abstention=过度沉默",
    "total_misjudged": len(watch),
    "misjudge_rate_in_sample": round(len(watch) / N, 3) if N else 0,
    "by_failure_mode": by_mode.most_common(),
    "by_info_type": by_itype.most_common(),
    "by_domain": by_dom.most_common(),
    "by_predicted_verdict": by_verdict.most_common(),
    "records": watch_records,
}
json.dump(
    watchlist,
    open(ASSETS + "/false_positive_watchlist.json", "w", encoding="utf-8"),
    ensure_ascii=False,
    indent=1,
)

# ===== 4) 判别式库（真假边界规则，按 info_type 去重）=====
disc_by_itype = collections.defaultdict(list)
disc_seen = collections.defaultdict(set)
disc_total = 0
for r in recs:
    d = str(r.get("discriminator_vs_rumor") or "").strip()
    if not d:
        continue
    disc_total += 1
    it = itype(r)
    key = d[:30]
    if key not in disc_seen[it]:
        disc_seen[it].add(key)
        disc_by_itype[it].append(d)
discriminators = {
    "doc": "同题真消息 vs 谣言的判别规则，喂 FactChecker / Skeptic 的真假边界",
    "total": disc_total,
    "by_info_type_samples": {it: ds[:10] for it, ds in disc_by_itype.items()},
}
json.dump(
    discriminators,
    open(ASSETS + "/discriminator_library.json", "w", encoding="utf-8"),
    ensure_ascii=False,
    indent=1,
)

# ===== 人读汇总 truth_signals_v1.md =====
L = [
    "# 真消息侧 harvest 资产 v1（从 2500 条真消息逐条分析聚合）\n",
    f"> 4 样机器可读资产在 `data/eval/harvest_assets/`（truth_signal_lexicon / confirmation_routing "
    f"/ false_positive_watchlist / discriminator_library），本文是人读摘要。共 {N} 条记录。\n",
    "## 0 · 头条数字（准确性 30% 的弹药）\n",
    f"- **{len(watch)} / {N} 条真消息（{len(watch) / N * 100:.1f}%）被当前打分器预测误判**："
    f"误判为假 {by_mode.get('false_positive', 0)} 条、过度沉默 {by_mode.get('over_abstention', 0)} 条。\n",
    "- 这是『证实维度缺位 + 偏假先验』的直接后果——产品改前 baseline，改后做盲测对比即得准确性硬数字。\n",
]

L.append("\n## 1 · 证实信号词表（受控词，喂第 7 维 _dim_confirmation）\n")
L.append("| 信号类型 | 出现次数 |\n|---|---:|")
for st, c in sig_global.most_common():
    L.append(f"| {st} | {c} |")

L.append("\n## 2 · 最易被误杀的 info_type（错题本 Top）\n")
L.append("| info_type | 误杀数 |\n|---|---:|")
for it, c in by_itype.most_common(12):
    L.append(f"| {it} | {c} |")
L.append("\n按失败模式：" + "、".join(f"{m} {c}" for m, c in by_mode.most_common()))
L.append("按预测裁决：" + "、".join(f"{v} {c}" for v, c in by_verdict.most_common()))

L.append("\n## 3 · 证实路径（每个 info_type『怎么找正面佐证』，前几类）\n")
L.append("| info_type | 样本数 | 具名官方源率 | 主导证据类型 |\n|---|---:|---:|---|")
for it, info in sorted(routing.items(), key=lambda x: -x[1]["n"])[:12]:
    top = "、".join(f"{k}×{v}" for k, v in info["evidence_types_dist"][:3])
    L.append(f"| {it} | {info['n']} | {info['named_official_source_rate'] * 100:.0f}% | {top} |")

L.append("\n## 4 · 判别式示例（真 vs 谣怎么分，节选健康/科学）\n")
for it in ("HEALTH_RISK_WARNING", "SCIENCE_FINDING_ACCURATE", "SAFETY_ADVISORY"):
    for d in disc_by_itype.get(it, [])[:2]:
        L.append(f"- **{it}**：{d}")

open(DOC, "w", encoding="utf-8").write("\n".join(L))

# ===== 误杀名单人读文档 false_positive_watchlist.md =====
W = [
    "# 误杀名单（false-positive watchlist）— 产品错题本\n",
    f"> 当前打分器会误判的真消息共 **{len(watch)} 条**（占抽样 {len(watch) / N * 100:.1f}%）。"
    "机器可读全量见 `data/eval/harvest_assets/false_positive_watchlist.json`。\n",
    "## 按 info_type × 失败模式\n",
    "| info_type | 误杀数 |\n|---|---:|",
]
for it, c in by_itype.most_common(20):
    W.append(f"| {it} | {c} |")
W.append("\n## 典型误杀案例（各失败模式取样）\n")
shown = collections.Counter()
for r in watch_records:
    m = r["failure_mode"]
    if shown[m] >= 5:
        continue
    shown[m] += 1
    W.append(
        f"- [{m}] **{r['info_type']}** id={r['claim_id']}：{str(r['claim_preview'])[:40]}\n"
        f"  - 预测裁决 {r['predicted_verdict']}；为何误判：{str(r['why_misjudged'])[:120]}"
    )
open(WATCHDOC, "w", encoding="utf-8").write("\n".join(W))

print(f"✅ 4 样真消息资产已落盘 {ASSETS}/：")
for fn in (
    "truth_signal_lexicon",
    "confirmation_routing",
    "false_positive_watchlist",
    "discriminator_library",
):
    print("   -", fn + ".json")
print("✅ 人读汇总:", DOC)
print("✅ 误杀名单:", WATCHDOC)
print(
    f"\n记录 {N} 条 | 误杀 {len(watch)} 条（{len(watch) / N * 100:.1f}%）"
    f" | 证实信号 {sum(sig_global.values())} 个（{len(sig_global)} 类）"
    f" | 判别式 {disc_total} 条 | info_type {len(routing)} 种"
)

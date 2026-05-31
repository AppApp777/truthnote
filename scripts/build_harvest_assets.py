"""从 2500 条谣言逐条分析里，聚合出 4 样可复用资产：
  1) 红旗词典         redflag_lexicon.json
  2) 核查路径路由表   verification_routing.json
  3) 质疑模板库       skeptic_question_bank.json
  4) 结构签名         structural_signatures.json
并生成人读汇总 docs/harvest_assets_v1.md。
domain 从 typology_clustered.json 按 claim_id 联接。
"""

import collections
import glob
import itertools
import json
import os

ROOT = r"d:/VIBE CODING/A-hackthon/02-wenke-song"
OUT = ROOT + r"/data/eval/typology_sample/outputs"
CLUSTERED = ROOT + r"/data/eval/typology_clustered.json"
ASSETS = ROOT + r"/data/eval/harvest_assets"
DOC = ROOT + r"/docs/harvest_assets_v1.md"
os.makedirs(ASSETS, exist_ok=True)

# domain 联接表
dom_by_id = {}
for r in json.load(open(CLUSTERED, encoding="utf-8")):
    dom_by_id[r.get("claim_id")] = r.get("domain", "其他")

# 加载全字段记录
recs = []
for f in sorted(glob.glob(os.path.join(OUT, "chunk_*.jsonl"))):
    for line in open(f, encoding="utf-8-sig"):
        line = line.strip()
        if line:
            try:
                d = json.loads(line)
                d["_domain"] = dom_by_id.get(d.get("claim_id"), "其他")
                recs.append(d)
            except Exception:
                pass
N = len(recs)


def aslist(v):
    if isinstance(v, list):
        return [str(x).strip() for x in v if str(x).strip()]
    if v:
        return [str(v).strip()]
    return []


# ===== 1) 红旗词典 =====
rf_global = collections.Counter()
rf_by_dom = collections.defaultdict(collections.Counter)
for r in recs:
    for p in aslist(r.get("red_flag_phrases")):
        rf_global[p] += 1
        rf_by_dom[r["_domain"]][p] += 1
redflag = {
    "total_distinct": len(rf_global),
    "top_global": rf_global.most_common(120),
    "by_domain_top20": {d: c.most_common(20) for d, c in rf_by_dom.items()},
}
json.dump(
    redflag,
    open(ASSETS + "/redflag_lexicon.json", "w", encoding="utf-8"),
    ensure_ascii=False,
    indent=1,
)

# ===== 2) 核查路径路由表 =====
routing = {}
for d in sorted(set(r["_domain"] for r in recs)):
    rows = [r for r in recs if r["_domain"] == d]
    est = collections.Counter(str(r.get("evidence_source_type", "?")) for r in rows)
    # 代表性 verification_path：优先 demo 适合的，去重，取 6 条
    paths, seen = [], set()
    pool = [r for r in rows if (r.get("demo_case_suitability") or {}).get("flag")] + rows
    for r in pool:
        vp = str(r.get("verification_path", "")).strip()
        key = vp[:20]
        if vp and key not in seen:
            seen.add(key)
            paths.append(vp)
        if len(paths) >= 6:
            break
    routing[d] = {
        "n": len(rows),
        "evidence_source_type_dist": est.most_common(),
        "sample_verification_paths": paths,
    }
json.dump(
    routing,
    open(ASSETS + "/verification_routing.json", "w", encoding="utf-8"),
    ensure_ascii=False,
    indent=1,
)

# ===== 3) 质疑模板库 =====
sk_by_dom = collections.defaultdict(list)
sk_seen = collections.defaultdict(set)
sk_total = 0
for r in recs:
    for q in aslist(r.get("skeptic_attack_questions")):
        sk_total += 1
        d = r["_domain"]
        key = q[:25]
        if key not in sk_seen[d]:
            sk_seen[d].add(key)
            sk_by_dom[d].append(q)
skeptic = {
    "total_questions": sk_total,
    "by_domain_samples": {d: qs[:12] for d, qs in sk_by_dom.items()},
}
json.dump(
    skeptic,
    open(ASSETS + "/skeptic_question_bank.json", "w", encoding="utf-8"),
    ensure_ascii=False,
    indent=1,
)

# ===== 4) 结构签名 =====
sp_global = collections.Counter()
sp_by_dom = collections.defaultdict(collections.Counter)
cooc = collections.Counter()
for r in recs:
    pats = aslist(r.get("structural_pattern"))
    for p in pats:
        sp_global[p] += 1
        sp_by_dom[r["_domain"]][p] += 1
    for a, b in itertools.combinations(sorted(set(pats)), 2):
        cooc[(a, b)] += 1
structural = {
    "global_dist": sp_global.most_common(),
    "by_domain_top3": {d: c.most_common(3) for d, c in sp_by_dom.items()},
    "top_cooccurrence": [[a, b, n] for (a, b), n in cooc.most_common(15)],
}
json.dump(
    structural,
    open(ASSETS + "/structural_signatures.json", "w", encoding="utf-8"),
    ensure_ascii=False,
    indent=1,
)

# ===== 人读汇总 =====
L = [
    "# Harvest 可复用资产 v1（从 2500 条谣言逐条分析聚合）\n",
    f"> 4 样机器可读资产在 `data/eval/harvest_assets/`，本文是人读摘要。共 {N} 条记录。\n",
]

L.append("## 1 · 红旗词典（谣言高频信号词，可做预筛/高亮）\n")
L.append(f"- 不同短语 {len(rf_global)} 个。高频前 30：\n")
L.append("| 短语 | 次数 |\n|---|---:|")
for p, c in rf_global.most_common(30):
    L.append(f"| {p} | {c} |")

L.append("\n## 2 · 核查路径路由表（每个领域『该查哪种源』）\n")
L.append("| 领域 | 主导证据源类型（前3） |\n|---|---|")
for d, info in sorted(routing.items(), key=lambda x: -x[1]["n"]):
    top3 = "、".join(f"{k}×{v}" for k, v in info["evidence_source_type_dist"][:3])
    L.append(f"| {d} | {top3} |")

L.append("\n## 3 · 质疑模板库（现成的『怎么质疑』，按领域）\n")
L.append(
    f"- 共 {sk_total} 条质疑问题。每领域抽样见 `skeptic_question_bank.json`。示例（健康医疗）：\n"
)
for q in sk_by_dom.get("健康医疗", [])[:4]:
    L.append(f"- {q}")

L.append("\n## 4 · 结构签名（谣言套路，可做规则层秒打标）\n")
L.append("| 套路 | 次数 |\n|---|---:|")
for p, c in sp_global.most_common(12):
    L.append(f"| {p} | {c} |")
L.append("\n高频共现套路对（常一起出现）：")
for a, b, n in structural["top_cooccurrence"][:8]:
    L.append(f"- {a} + {b}（{n} 次）")

open(DOC, "w", encoding="utf-8").write("\n".join(L))

print("✅ 4 样资产已落盘 data/eval/harvest_assets/：")
for fn in [
    "redflag_lexicon",
    "verification_routing",
    "skeptic_question_bank",
    "structural_signatures",
]:
    print("   -", fn + ".json")
print("✅ 人读汇总:", DOC)
print(
    f"\n红旗短语 {len(rf_global)} 种 | 质疑问题 {sk_total} 条 | 结构套路 {len(sp_global)} 种 | 领域 {len(routing)} 个"
)

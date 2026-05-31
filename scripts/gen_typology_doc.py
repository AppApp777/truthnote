"""读 typology_clustered.json，生成 docs/typology_v1.md 路演成品。"""

import collections
import json
import os

SRC = r"d:/VIBE CODING/A-hackthon/02-wenke-song/data/eval/typology_clustered.json"
DOC = r"d:/VIBE CODING/A-hackthon/02-wenke-song/docs/typology_v1.md"

recs = json.load(open(SRC, encoding="utf-8"))
N = len(recs)

DOMAIN_GAP = {
    "媒体影像错配": "看图/视频溯源（反向图搜）",
    "金融诈骗钓鱼": "诈骗话术/合规比对",
    "健康医疗": "因果强度 + 证据分级（临床指南）",
    "食品营养": "因果强度 + 剂量/口径核对",
    "政策法规": "结构化数据库查询（法规/批号原文）",
    "政治地缘": "多源交叉（默认流水线最适用）",
    "灾害天气": "官方通报缺位反证",
    "本地治安突发": "官方通报缺位反证（属地公安）",
    "科学统计": "因果强度 + 统计口径核对",
    "历史人物言论": "引语 / 史料溯源",
    "机构企业产品": "结构化数据库查询（资质核验）",
    "安全常识": "危害分级 + 正确替代",
    "事件通告类": "官方原文比对",
    "其他": "生活常识 / 长尾",
}
DISPLAY = {"其他": "生活常识/其他"}


def unworkable(rows):
    return sum(
        1 for r in rows if str(r.get("default_pipeline_works")).lower() in ("partial", "false")
    )


dom = collections.Counter(r["domain"] for r in recs)
mech = collections.Counter(r["mechanism"] for r in recs)
tot_uw = unworkable(recs)

# 6 大能力缺口 → 覆盖哪些领域
GAPS = [
    ("G1 看图/视频溯源", "反向图像/视频检索，定位原始拍摄时间地点，识破移花接木", ["媒体影像错配"]),
    (
        "G2 掰因果强度 / 概念辨析",
        "区分『相关 vs 因果』『传播力 vs 致病性』，对绝对化断言找反例",
        ["健康医疗", "科学统计", "食品营养"],
    ),
    (
        "G3 结构化数据库查询",
        "查法规全文/药监批号/卫健委资质/裁判文书，搜索引擎搜不到的库",
        ["政策法规", "机构企业产品"],
    ),
    (
        "G4 官方通报缺位反证",
        "本地突发/灾害类，靠『该有官方通报却没有』反向证伪，定向抓属地权威源",
        ["本地治安突发", "灾害天气", "事件通告类"],
    ),
    (
        "G5 危害分级",
        "对危险操作建议（捂汗退烧/地震躺平）按危害加权 + 给正确替代，而非单纯判真假",
        ["安全常识"],
    ),
    ("G6 诈骗话术比对", "识别钓鱼/诈骗话术模板与合规红线，而非核查事实真伪", ["金融诈骗钓鱼"]),
]


def pick_examples(domain, k=2):
    pool = [r for r in recs if r["domain"] == domain and r.get("demo_suitable")]
    if len(pool) < k:
        pool += [r for r in recs if r["domain"] == domain and not r.get("demo_suitable")]
    out = []
    seen = set()
    for r in pool:
        cp = str(r.get("claim_preview", "")).strip()
        if cp and cp not in seen:
            seen.add(cp)
            out.append(r)
        if len(out) >= k:
            break
    return out


L = []
L.append("# TruthNote 谣言类型学 v1（基于 2500 条真实谣言）\n")
L.append("> 数据来源：CANDY 中文谣言数据集分层抽样 2500 条（label=1 谣言），")
L.append("> 由 Opus 多智能体逐条分析，按【领域】+【造假手法】两轴确定性归类。\n")

L.append("## 核心数字（路演开场）\n")
L.append(f"- **{N} 条**真实谣言，逐条分析、可追溯。")
L.append(
    f"- **{tot_uw / N * 100:.0f}%** 的谣言，现有『搜索 + 多源交叉』默认流水线**无法独立搞定**（只有 {100 - tot_uw / N * 100:.0f}% 完全适用）。"
)
L.append(
    f"- 归纳出 **{len([d for d in dom if d != '其他'])} 个领域** × **{len([m for m in mech if m not in ('其他', '歪曲/误读其它')])} 种造假手法**，覆盖率 **{(N - dom.get('其他', 0)) / N * 100:.0f}%**。"
)
L.append("- → 据此定位出 **6 大产品能力缺口**（见下），这是 TruthNote 的差异化护城河。\n")

L.append("## 表 1 · 领域分布 × 默认搞不定率 × 缺的能力（主表）\n")
L.append("| 领域 | 谣言数 | 占比 | 默认搞不定率 | 主导手法 | 需补的能力 |")
L.append("|---|---:|---:|---:|---|---|")
for d, c in dom.most_common():
    rows = [r for r in recs if r["domain"] == d]
    uw = unworkable(rows) / len(rows) * 100
    topm = collections.Counter(r["mechanism"] for r in rows).most_common(1)[0][0]
    name = DISPLAY.get(d, d)
    L.append(
        f"| {name} | {c} | {c / N * 100:.0f}% | {uw:.0f}% | {topm} | {DOMAIN_GAP.get(d, '')} |"
    )

L.append("\n## 表 2 · 造假手法分布（『谣言是怎么撒谎的』— 洞察层）\n")
L.append("> 比『谣言讲什么』更深一层：同一套手法跨领域复现，是设计核查策略的真正抓手。\n")
L.append("| 造假手法 | 条数 | 占比 |")
L.append("|---|---:|---:|")
for m, c in mech.most_common():
    if m in ("其他", "歪曲/误读其它"):
        continue
    L.append(f"| {m} | {c} | {c / N * 100:.0f}% |")

L.append("\n## 6 大产品能力缺口（『我们分析完做了什么』）\n")
L.append(
    "> 默认流水线假设：事实可搜索 + 公开权威源可比对 + 证据是文本。每条假设的失效，对应一个要补的能力。\n"
)
for gid, desc, doms in GAPS:
    covered = sum(dom.get(d, 0) for d in doms)
    L.append(f"### {gid} — 覆盖约 {covered} 条（{covered / N * 100:.0f}%）")
    L.append(f"{desc}\n")
    L.append(f"*涉及领域：{'、'.join(DISPLAY.get(d, d) for d in doms)}*\n")
    exs = pick_examples(doms[0], 2)
    for r in exs:
        L.append(f"- 例：「{str(r.get('claim_preview', '')).strip()}」")
        vp = str(r.get("verification_path", "")).strip()
        if vp:
            L.append(f"  - 核查路径：{vp}")
    L.append("")

L.append("## 方法与可复现性（评委追问时的底气）\n")
L.append(
    "- **不靠人工拍脑袋分类**：2500 条产出 2094 个细标签（76% 唯一），按标签词形 +中文正文关键词**确定性**归入领域/手法，脚本可复现。"
)
L.append(
    f"- **覆盖率**：领域归类覆盖 {(N - dom.get('其他', 0)) / N * 100:.0f}%，未归类 {dom.get('其他', 0) / N * 100:.0f}%（生活常识长尾）。"
)
L.append(
    "- **脚本**：`scripts/cluster_typology.py`（归类）+ `scripts/gen_typology_doc.py`（生成本文）。"
)
L.append(
    "- **原始逐条分析**：`data/eval/typology_sample/outputs/`（125 chunk × 20 条，每条 20 字段）。"
)

os.makedirs(os.path.dirname(DOC), exist_ok=True)
open(DOC, "w", encoding="utf-8").write("\n".join(L))
print("已生成:", DOC)
print("总行数:", len(L))

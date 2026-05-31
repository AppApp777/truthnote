"""把 2500 条谣言的 2094 个碎标签，按【领域】+【造假手法】两轴确定性归类。
输出：data/eval/typology_clustered.json（每条带 domain/mechanism）+ 控制台路演数据。
方法：扫描整个 info_type 标签里的关键词，按优先级（最具体→最一般）归入领域/手法。
"""

import collections
import glob
import json
import os

OUT = r"d:/VIBE CODING/A-hackthon/02-wenke-song/data/eval/typology_sample/outputs"

# ===== 领域轴（优先级从上到下，命中即停）=====
DOMAINS = [
    (
        "媒体影像错配",
        [
            "IMAGE",
            "VIDEO",
            "PHOTO",
            "MEDIA",
            "FOOTAGE",
            "CLIP",
            "VISUAL",
            "DOCTORED",
            "MISCONTEXTUALIZ",
            "MISCAPTION",
            "RECYCLED",
            "STALE",
            "OLD_",
        ],
    ),
    (
        "金融诈骗钓鱼",
        [
            "SCAM",
            "PHISHING",
            "FRAUD",
            "FINANCIAL",
            "PONZI",
            "INVESTMENT",
            "RECRUITMENT",
            "LURE",
            "GIVEAWAY",
            "SOLICITATION",
            "MARKET",
            "MONEY",
            "CRYPTO",
            "INSURANCE",
            "LOAN",
            "ECONOMIC",
            "ECONOMY",
        ],
    ),
    ("食品营养", ["FOOD", "NUTRITION", "DIETARY", "DIET", "EDIBLE", "INGEST", "BEVERAGE", "DRINK"]),
    (
        "健康医疗",
        [
            "MEDICAL",
            "MEDICINE",
            "HEALTH",
            "DISEASE",
            "DRUG",
            "VACCINE",
            "MEDICATION",
            "SYMPTOM",
            "EPIDEMIC",
            "EPIDEMIOLOG",
            "PHYSIOLOG",
            "TCM",
            "QUACK",
            "SUPPLEMENT",
            "DOSE",
            "TOXICOLOG",
            "BIOLOGICAL",
            "TRANSMISSION",
            "CANCER",
            "MIRACLE",
            "THERAP",
            "REMEDY",
            "CLINICAL",
            "PATHOGEN",
            "VIRUS",
            "VIRAL",
            "INFECTION",
            "CURE",
            "DISORDER",
            "BODILY",
            "ANATOM",
            "NUTRIENT",
            "DETOX",
            "ACUPUNCTURE",
            "HERBAL",
        ],
    ),
    (
        "政策法规",
        [
            "POLICY",
            "LAW",
            "LEGAL",
            "REGULAT",
            "ADMINISTRATIVE",
            "ADMIN_",
            "PROCEDURAL",
            "TAX",
            "GOVERNMENT",
            "STATUTE",
            "JUDICIAL",
            "COURT",
            "PENSION",
            "WELFARE",
        ],
    ),
    (
        "政治地缘",
        ["POLITICAL", "GEOPOLITICAL", "MILITARY", "ELECTION", "FOREIGN", "DIPLOMA", "WAR"],
    ),
    (
        "灾害天气",
        [
            "DISASTER",
            "WEATHER",
            "EARTHQUAKE",
            "FLOOD",
            "TYPHOON",
            "NATURAL",
            "ENVIRONMENTAL",
            "CASUALTY",
            "CLIMATE",
            "STORM",
            "FIRE",
            "SEISMIC",
            "GEOSPATIAL",
            "GEOGRAPHIC",
            "TEMPERATURE",
            "MEASUREMENT",
        ],
    ),
    (
        "本地治安突发",
        [
            "LOCAL",
            "CRIME",
            "URBAN",
            "KIDNAP",
            "ABDUCT",
            "POLICE",
            "TRAFFICK",
            "INCIDENT",
            "ACCIDENT",
        ],
    ),
    (
        "科学统计",
        [
            "SCIENTIFIC",
            "STATISTIC",
            "PSEUDOSCIEN",
            "PSEUDO",
            "PHYSICS",
            "ENGINEERING",
            "DATA",
            "CORRELATION",
            "CAUSAL",
            "ACADEMIC",
            "SPECIES",
            "ARCHAEOLOG",
            "RESEARCH",
            "TECHNOLOG",
            "CHEMISTRY",
            "BIOLOGY",
        ],
    ),
    (
        "历史人物言论",
        [
            "HISTORICAL",
            "BIOGRAPHICAL",
            "CELEBRITY",
            "QUOTE",
            "EXPERT",
            "RELIGIOUS",
            "AUTHORITY",
            "PERSON",
            "ATTRIBUTION",
        ],
    ),
    (
        "机构企业产品",
        [
            "INSTITUTION",
            "CORPORATE",
            "ORG_",
            "PRODUCT",
            "TECH",
            "COSMETIC",
            "AI_",
            "PLATFORM",
            "SERVICE",
            "INFRASTRUCTURE",
            "MATERIAL",
            "BRAND",
            "COMPANY",
            "CREDENTIAL",
        ],
    ),
    ("安全常识", ["SAFETY", "PHYSICAL", "DANGEROUS", "DIY", "FIRST_AID", "SELF_"]),
    (
        "事件通告类",
        [
            "EVENT",
            "OFFICIAL",
            "ANNOUNCEMENT",
            "NOTICE",
            "PUBLIC",
            "REGULATION",
            "DOCUMENT",
            "CURRICULUM",
            "EDUCATION",
            "SPORTS",
            "CULTURAL",
            "ACADEMIC",
        ],
    ),
]

# ===== 手法轴（优先级从上到下，命中即停）=====
MECHANISMS = [
    (
        "凭空捏造",
        [
            "FABRICAT",
            "HOAX",
            "FALSIFICATION",
            "FORGERY",
            "FALSEHOOD",
            "FORGED",
            "NONEXISTENT",
            "INVENT",
            "FAKE",
        ],
    ),
    (
        "张冠李戴/错引",
        [
            "MISATTRIBUT",
            "MISMATCH",
            "MISLABEL",
            "MISCONTEXT",
            "MISIDENTIF",
            "MISCLASSIF",
            "RECONTEXT",
            "MISQUOTE",
            "RECYCLING",
            "MISCAPTION",
            "ATTRIBUTION",
        ],
    ),
    (
        "诈骗钓鱼话术",
        [
            "SCAM",
            "FRAUD",
            "PHISHING",
            "SOLICITATION",
            "LURE",
            "GIVEAWAY",
            "IMPERSONATION",
            "PRETEXT",
            "RECRUITMENT",
        ],
    ),
    (
        "恐慌煽动",
        [
            "SCARE",
            "PANIC",
            "FEARMONGER",
            "ALARM",
            "ALERT",
            "FUD",
            "SMEAR",
            "DEFAMATION",
            "ACCUSATION",
            "WARNING",
        ],
    ),
    (
        "功效夸大",
        [
            "OVERCLAIM",
            "MISCLAIM",
            "OVERSTATEMENT",
            "EXAGGERATION",
            "INFLATION",
            "HYPE",
            "PROMISE",
            "BENEFIT",
            "PROMO",
            "OVERCLAIM",
            "ABSOLUT",
        ],
    ),
    (
        "过度外推/以偏概全",
        [
            "OVERREACH",
            "OVERGENERALIZ",
            "OVERSIMPLIF",
            "GENERALIZATION",
            "EXTRAPOLAT",
            "FALLACY",
            "CONFLATION",
            "REVERSAL",
            "INVERSION",
            "CAUSATION",
            "CORRELATION",
            "OVERSTAT",
        ],
    ),
    (
        "歪曲解读",
        [
            "DISTORTION",
            "MISINTERPRET",
            "MISREAD",
            "MISREPRESENT",
            "MISCHARACTER",
            "MISFRAM",
            "FRAMING",
            "NARROWING",
            "REVISION",
            "MISSTATEMENT",
            "MISREPORT",
            "MISUNDERSTAND",
            "MISAPPLICATION",
        ],
    ),
    (
        "民间误解/伪科学",
        [
            "MYTH",
            "MISCONCEPTION",
            "PSEUDOSCIENCE",
            "LEGEND",
            "BELIEF",
            "THEORY",
            "SUPERSTITION",
            "FOLK",
        ],
    ),
    ("否认/淡化", ["DENIAL", "NEGATION", "DOWNPLAY", "UNDERSTATEMENT"]),
    (
        "危险建议/误导",
        [
            "MISADVICE",
            "MISGUIDANCE",
            "MISADVI",
            "ADVICE",
            "DIRECTIVE",
            "REMEDY",
            "PROCEDURE",
            "MISDIRECTION",
        ],
    ),
    ("预测臆测", ["PREDICTION", "SPECULATION", "PREMATURE", "UNCONFIRMED", "UNVERIFIABLE"]),
    ("统计操纵", ["MANIPULATION", "STATISTIC"]),
    ("功效声明", ["CLAIM", "ASSERTION", "PROMO", "TESTIMONIAL"]),
    (
        "歪曲/误读其它",
        [
            "ERROR",
            "CONFUSION",
            "MISINFO",
            "DISINFO",
            "MISINFORMATION",
            "NARRATIVE",
            "MISLEADING",
            "MISMATCH",
            "MISSTAT",
            "MISGUID",
            "OVERSIMPLIFICATION",
            "GENERALIZATION",
            "MISREAD",
            "DETAIL",
            "ABSOLUTE",
            "DENIAL",
            "NEGATION",
            "FALLACY",
            "MISCLASSIFICATION",
            "MISCONTEXTUALIZATION",
        ],
    ),
]


def classify(label, table):
    L = label.upper()
    for name, kws in table:
        for kw in kws:
            if kw in L:
                return name
    return "其他"


# 第二遍：标签认不出领域时，扫中文正文关键词（优先级同上）
DOMAIN_CLAIM_KW = [
    ("媒体影像错配", ["视频", "照片", "图片", "画面", "截图", "影片"]),
    (
        "金融诈骗钓鱼",
        [
            "贷",
            "投资",
            "理财",
            "诈骗",
            "转账",
            "银行",
            "保险",
            "基金",
            "信用卡",
            "扫码",
            "返利",
            "中奖",
        ],
    ),
    (
        "食品营养",
        [
            "食品",
            "食用",
            "西瓜",
            "鸡蛋",
            "牛奶",
            "水果",
            "蔬菜",
            "猪肉",
            "食用油",
            "白糖",
            "食盐",
            "饮食",
            "营养",
            "隔夜",
            "吃",
            "喝",
            "餐",
            "零食",
            "维生素",
        ],
    ),
    (
        "健康医疗",
        [
            "疫苗",
            "病毒",
            "药",
            "医",
            "癌",
            "血栓",
            "疾病",
            "医保",
            "症状",
            "治疗",
            "患者",
            "中毒",
            "细胞",
            "免疫",
            "感染",
            "发烧",
            "高血压",
            "糖尿病",
            "心脏",
            "新冠",
        ],
    ),
    (
        "政策法规",
        [
            "政策",
            "法律",
            "法规",
            "规定",
            "条例",
            "通知",
            "补贴",
            "社保",
            "养老",
            "公积金",
            "户籍",
            "退休",
            "税",
        ],
    ),
    (
        "政治地缘",
        ["美国", "中美", "政客", "选举", "外交", "脱钩", "总统", "拜登", "特朗普", "乌克兰", "俄"],
    ),
    (
        "灾害天气",
        [
            "地震",
            "台风",
            "洪水",
            "暴雨",
            "高温",
            "气温",
            "降温",
            "泥石流",
            "海啸",
            "雪灾",
            "干旱",
            "灾",
            "震级",
            "规模",
        ],
    ),
    (
        "本地治安突发",
        ["人贩子", "拐", "抢", "偷", "警", "治安", "失踪", "绑架", "命案", "持刀", "走散"],
    ),
    (
        "科学统计",
        [
            "物理",
            "化学",
            "科学",
            "研究",
            "实验",
            "理论",
            "数据",
            "统计",
            "概率",
            "宇宙",
            "量子",
            "基因",
        ],
    ),
    (
        "历史人物言论",
        ["说过", "名言", "历史", "古代", "将军", "教授", "专家", "院士", "曾说", "题词"],
    ),
    (
        "机构企业产品",
        ["公司", "机构", "品牌", "产品", "资质", "认证", "医院", "学校", "企业", "店"],
    ),
    ("安全常识", ["急救", "逃生", "触电", "防灾", "灭火", "溺水", "用电"]),
]


def classify_domain(r):
    d = classify(str(r.get("info_type_proposed", "?")), DOMAINS)
    if d != "其他":
        return d
    text = str(r.get("claim_preview", "")) + str(r.get("info_type_proposed", ""))
    for name, kws in DOMAIN_CLAIM_KW:
        for kw in kws:
            if kw in text:
                return name
    return "其他"


# 领域→缺的能力（路演用）
DOMAIN_GAP = {
    "媒体影像错配": "看图/视频溯源（反向图搜）",
    "金融诈骗钓鱼": "诈骗话术/合规比对",
    "健康医疗": "因果强度+证据分级（临床指南）",
    "食品营养": "因果强度+剂量/口径核对",
    "政策法规": "结构化数据库查询（法规/批号原文）",
    "政治地缘": "多源交叉（默认流水线最适用）",
    "灾害天气": "官方通报缺位反证",
    "本地治安突发": "官方通报缺位反证（属地公安）",
    "科学统计": "因果强度+统计口径核对",
    "历史人物言论": "引语/史料溯源",
    "机构企业产品": "结构化数据库查询（资质核验）",
    "安全常识": "危害分级+正确替代",
    "事件通告类": "官方原文比对",
    "其他": "—",
}

# ---- 加载 ----
records = []
for f in sorted(glob.glob(os.path.join(OUT, "chunk_*.jsonl"))):
    for line in open(f, encoding="utf-8-sig"):
        line = line.strip()
        if line:
            try:
                records.append(json.loads(line))
            except Exception:
                pass
N = len(records)

# ---- 归类 ----
for r in records:
    lab = str(r.get("info_type_proposed", "?"))
    r["_domain"] = classify_domain(r)
    r["_mechanism"] = classify(lab, MECHANISMS)

dom = collections.Counter(r["_domain"] for r in records)
mech = collections.Counter(r["_mechanism"] for r in records)


def unworkable_pct(rows):
    bad = sum(
        1 for r in rows if str(r.get("default_pipeline_works")).lower() in ("partial", "false")
    )
    return bad / len(rows) * 100 if rows else 0


print("=" * 72)
print(f"总记录 {N} 条 | 领域 {len(dom)} 类 | 手法 {len(mech)} 类")
print(
    f"领域未归类(其他): {dom.get('其他', 0)} ({dom.get('其他', 0) / N * 100:.1f}%)  手法未归类(其他): {mech.get('其他', 0)} ({mech.get('其他', 0) / N * 100:.1f}%)"
)
print("=" * 72)

print("\n【领域分布 × 默认搞不定率 × 主导手法 × 缺的能力】（路演主表）")
print(f"{'领域':<8}{'条数':>5}{'占比':>7}{'搞不定率':>9}  {'主导手法':<14}{'缺的能力'}")
for d, c in dom.most_common():
    rows = [r for r in records if r["_domain"] == d]
    uw = unworkable_pct(rows)
    topm = collections.Counter(r["_mechanism"] for r in rows).most_common(1)[0][0]
    print(f"{d:<8}{c:>5}{c / N * 100:>6.0f}%{uw:>8.0f}%  {topm:<14}{DOMAIN_GAP.get(d, '')}")

print("\n【造假手法分布】（'谣言是怎么撒谎的' — 路演洞察层）")
for m, c in mech.most_common():
    print(f"  {c:>4} ({c / N * 100:>4.0f}%)  {m}")

# ---- 落盘 ----
clustered = r"d:/VIBE CODING/A-hackthon/02-wenke-song/data/eval/typology_clustered.json"
with open(clustered, "w", encoding="utf-8") as fh:
    json.dump(
        [
            {
                "claim_id": r.get("claim_id"),
                "claim_preview": r.get("claim_preview"),
                "info_type_proposed": r.get("info_type_proposed"),
                "domain": r["_domain"],
                "mechanism": r["_mechanism"],
                "default_pipeline_works": r.get("default_pipeline_works"),
                "verification_path": r.get("verification_path"),
                "demo_suitable": (r.get("demo_case_suitability") or {}).get("flag"),
            }
            for r in records
        ],
        fh,
        ensure_ascii=False,
        indent=1,
    )
print(f"\n归类结果已落盘: data/eval/typology_clustered.json（{N} 条）")

# ---- 抽查"其他"看是否需要补关键词 ----
others = [r for r in records if r["_domain"] == "其他"][:15]
if others:
    print("\n[领域=其他 抽样，用于补关键词]")
    for r in others:
        print("  ", r.get("info_type_proposed"), "|", str(r.get("claim_preview"))[:30])

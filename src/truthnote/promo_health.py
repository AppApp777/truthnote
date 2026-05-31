"""PromoHealthVerifier · 健康产品推销专用核验子流水线（纯规则）。

来源：2026-05-28 oracle a8 复盘的 Snopes SOP + FTC Gut Check
+ 中文权威库（NMPA / SAMR / NHC）模板。

设计原则：
- 零 LLM——所有判断都基于规则（FTC 阈值、burden of proof 锚点检测）
- 输入 MessageFrame + 原文 → 输出 PromoHealthVerification dict
- 不替换 FactChecker，作为额外的 verdict_lean 信号喂给 DimensionAssessment

CONTRACTS：见 C0 INV-1 / INV-2。
"""

from __future__ import annotations

import re

from .schemas import MessageFrame, MessageType

# ── FTC Gut Check 阈值（来源：FTC "Gut Check: A Reference Guide for Media"） ──
# https://www.ftc.gov/business-guidance/resources/gut-check-reference-guide-media-spotting-false-weight-loss-claims

# FTC 定义"substantial weight loss" = >1 lb/week for >4 weeks OR >15 lb 任何时间段
FTC_LB_PER_WEEK_THRESHOLD = 2.0  # testimonial 触发"typical results disclosure 必须"
FTC_TOTAL_LB_THRESHOLD = 15.0
JIN_TO_KG = 0.5
KG_TO_LB = 2.20462

# ── 推销 / no-rebound / safe rapid / 永久 / 包治 短语 ──
_NO_REBOUND_PHRASES = ["不反弹", "不反复", "永不复胖", "终身不胖", "彻底告别"]
_SAFE_RAPID_PHRASES = ["安全快速", "健康减肥", "健康减重", "科学减肥", "健康安全"]
_PERMANENT_PHRASES = ["永久减重", "终身减肥", "一劳永逸", "一次见效"]
_DISCLOSURE_PHRASES = [
    "效果因人而异",
    "结果不代表所有",
    "请遵医嘱",
    "需配合饮食运动",
    "个体差异",
]

# ── 锚点（合法推销应包含的元素） ──
_REGISTRATION_PATTERNS = [
    r"国药准字",
    r"批准文号",
    r"备案号",
    r"国食[健药]字",
    r"卫食健字",
    r"消字号",
    r"妆字号",
    r"\bSFDA\b",
    r"\bNMPA\b",
]
_MANUFACTURER_PATTERNS = [r"生产厂家", r"生产商", r"\b[A-Z][a-z]+ Pharma\b", r"制药公司", r"集团"]
_CLINICAL_PATTERNS = [
    r"临床试验",
    r"双盲",
    r"对照试验",
    r"\d+\s*(?:人|名|例)\s*(?:参与|入组)",
    r"\bRCT\b",
    r"国家药监局",
]


_CN_NUMS = {
    "零": 0,
    "〇": 0,
    "一": 1,
    "二": 2,
    "两": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
    "十": 10,
}


def _cn_to_arabic(text: str) -> str:
    """简易中文数字转阿拉伯数字（仅处理 0-99 范围，覆盖常见时长）。

    例：「四个月」→「4个月」，「十八周」→「18周」，「二十一天」→「21天」
    """

    def replace_units(s: str) -> str:
        # "十" 单字 = 10
        s = re.sub(r"(?<![一二三四五六七八九])十(?![一二三四五六七八九])", "10", s)

        # "X十Y" = X*10+Y, "X十" = X*10
        def cn_tens(m):
            ten = _CN_NUMS.get(m.group(1), 0) * 10
            ones = _CN_NUMS.get(m.group(2), 0) if m.group(2) else 0
            return str(ten + ones)

        s = re.sub(r"([一二三四五六七八九])十([一二三四五六七八九])?", cn_tens, s)
        # "十Y" = 10+Y
        s = re.sub(r"十([一二三四五六七八九])", lambda m: str(10 + _CN_NUMS[m.group(1)]), s)
        # 单字
        for cn, ar in _CN_NUMS.items():
            if cn in s and cn not in ("十", "零", "〇"):
                s = s.replace(cn, str(ar))
        return s

    return replace_units(text)


def _normalize(text: str) -> str:
    return re.sub(r"\s+", "", (text or "").strip())


def _extract_weight_loss_rate(text: str) -> dict | None:
    """从文本里提取 "X 斤/kg 在 Y 个月/周" 模式 → kg/week。

    例：「四个月共计减重38斤」→ 19 kg / 17 周 ≈ 1.12 kg/week
    返回 dict(total_kg, weeks, kg_per_week, lb_per_week, total_lb, raw_match) 或 None。
    """
    raw = text or ""
    norm = _normalize(_cn_to_arabic(raw))

    # 数字 + 斤/公斤/kg + 几个月/几周
    pat = re.compile(
        r"(\d+(?:\.\d+)?)\s*(斤|公斤|kg|KG|千克|磅|lb|LB)"
        r"[^0-9]{0,30}?"
        r"(\d+(?:\.\d+)?)\s*(个月|月|周|天|天内|周内)"
    )
    m = pat.search(norm)
    if not m:
        # 反序：先时间后重量
        pat2 = re.compile(
            r"(\d+(?:\.\d+)?)\s*(个月|月|周|天)"
            r"[^0-9]{0,30}?"
            r"(\d+(?:\.\d+)?)\s*(斤|公斤|kg|KG|千克|磅|lb|LB)"
        )
        m = pat2.search(norm)
        if not m:
            return None
        time_n, time_u, w_n, w_u = m.groups()
    else:
        w_n, w_u, time_n, time_u = m.groups()

    try:
        w_val = float(w_n)
        t_val = float(time_n)
    except ValueError:
        return None

    # 重量统一到 kg
    if w_u in ("斤",):
        kg = w_val * JIN_TO_KG
    elif w_u in ("公斤", "kg", "KG", "千克"):
        kg = w_val
    elif w_u in ("磅", "lb", "LB"):
        kg = w_val / KG_TO_LB
    else:
        return None

    # 时间统一到周
    if time_u in ("个月", "月"):
        weeks = t_val * 4.345
    elif time_u in ("周",):
        weeks = t_val
    elif time_u in ("天", "天内", "周内"):
        weeks = t_val / 7
    else:
        return None

    if weeks <= 0:
        return None

    kg_per_week = kg / weeks
    lb_per_week = kg_per_week * KG_TO_LB
    total_lb = kg * KG_TO_LB
    return {
        "total_kg": round(kg, 2),
        "weeks": round(weeks, 1),
        "kg_per_week": round(kg_per_week, 3),
        "lb_per_week": round(lb_per_week, 3),
        "total_lb": round(total_lb, 2),
        "raw_match": m.group(0),
    }


def check_ftc_thresholds(text: str) -> dict:
    """FTC Gut Check 阈值检查。

    返回 dict:
      - rate_extracted: dict | None (kg/week + lb/week)
      - testimonial_over_15_lb: bool (>15 lb 总减重)
      - testimonial_over_2_lb_per_week_for_month: bool
      - typical_results_disclosure_present: bool
      - no_rebound_claim: bool
      - safe_rapid_loss_claim: bool
      - permanent_loss_claim: bool
      - flags_triggered: list[str]
    """
    norm = _normalize(text)
    rate = _extract_weight_loss_rate(text)

    flags: list[str] = []
    over_15_lb = bool(rate and rate["total_lb"] > FTC_TOTAL_LB_THRESHOLD)
    over_2_lb_week = bool(
        rate and rate["lb_per_week"] > FTC_LB_PER_WEEK_THRESHOLD and rate["weeks"] >= 4
    )
    if over_15_lb:
        flags.append("FTC-testimonial_over_15_lb")
    if over_2_lb_week:
        flags.append("FTC-testimonial_over_2_lb_per_week_for_month")

    no_rebound = any(p in norm for p in _NO_REBOUND_PHRASES)
    safe_rapid = any(p in norm for p in _SAFE_RAPID_PHRASES)
    permanent = any(p in norm for p in _PERMANENT_PHRASES)
    disclosure = any(p in norm for p in _DISCLOSURE_PHRASES)

    if no_rebound:
        flags.append("FTC-no_rebound_claim")
    if safe_rapid:
        flags.append("FTC-safe_rapid_loss_claim")
    if permanent:
        flags.append("FTC-permanent_loss_claim")

    return {
        "rate_extracted": rate,
        "testimonial_over_15_lb": over_15_lb,
        "testimonial_over_2_lb_per_week_for_month": over_2_lb_week,
        "typical_results_disclosure_present": disclosure,
        "no_rebound_claim": no_rebound,
        "safe_rapid_loss_claim": safe_rapid,
        "permanent_loss_claim": permanent,
        "flags_triggered": flags,
    }


def check_burden_of_proof(text: str, frame: MessageFrame) -> dict:
    """BurdenOfProof Gate：检测消息是否含合法推销应有的锚点。

    Snopes 风格：promoter 必须 substantiate registration / manufacturer / clinical evidence /
    typical-results disclosure。缺失这些是 substantiation failure，不是中性。
    """
    norm = text or ""
    has_registration = any(re.search(p, norm) for p in _REGISTRATION_PATTERNS)
    has_manufacturer = any(re.search(p, norm) for p in _MANUFACTURER_PATTERNS)
    has_clinical = any(re.search(p, norm) for p in _CLINICAL_PATTERNS)
    has_disclosure = any(p in norm for p in _DISCLOSURE_PHRASES)

    missing: list[str] = []
    if not has_registration:
        missing.append("产品注册或备案号（如国药准字、国食健字、消字号）")
    if not has_manufacturer:
        missing.append("生产厂家完整名称")
    if not has_clinical:
        missing.append("临床证据（试验/RCT/权威推荐）")
    if not has_disclosure and frame.message_type == MessageType.HEALTH_PRODUCT_PROMO:
        missing.append("效果因人而异等典型结果披露")

    return {
        "registration_anchor_present": has_registration,
        "manufacturer_anchor_present": has_manufacturer,
        "clinical_evidence_anchor_present": has_clinical,
        "disclosure_present": has_disclosure,
        "missing_anchors": missing,
        "burden_holder": "promoter",
    }


def suggest_regulatory_queries(frame: MessageFrame) -> list[str]:
    """为 health_product_promo 生成 NMPA/SAMR/NHC 搜索建议（oracle a8 SOP）。"""
    if frame.message_type != MessageType.HEALTH_PRODUCT_PROMO:
        return []
    entity = (frame.promoted_entity or "").strip()
    if not entity:
        return []
    return [
        f"{entity} 药品批准文号",
        f"{entity} 保健食品 注册 备案",
        f"{entity} 市场监管 虚假宣传 处罚",
        f"{entity} 国家药监局 NMPA",
        f"{entity} 消费投诉",
    ]


def verify_promo(text: str, frame: MessageFrame) -> dict:
    """主入口：对 health_product_promo 类型消息做完整 promo 核验。

    返回 dict:
      - ftc: FTC 阈值检查结果
      - burden: 锚点检查结果
      - suggested_queries: 推荐的额外搜索关键词
      - risk_level: low / medium / high
      - verdict_lean: 谣言 / 大部分不实 / 误导性信息 / 无法核实 / 属实（建议倾向）
      - reasoning: 一句话理由
      - applied: bool（是否被启用——非 health_product_promo 时 False）
    """
    if frame.message_type != MessageType.HEALTH_PRODUCT_PROMO:
        return {
            "applied": False,
            "ftc": {},
            "burden": {},
            "suggested_queries": [],
            "risk_level": "low",
            "verdict_lean": None,
            "reasoning": "非 health_product_promo 类型，跳过",
        }

    ftc = check_ftc_thresholds(text)
    burden = check_burden_of_proof(text, frame)
    queries = suggest_regulatory_queries(frame)

    # 风险分级（纯规则）
    ftc_count = len(ftc["flags_triggered"])
    missing_count = len(burden["missing_anchors"])

    if ftc_count >= 3 and missing_count >= 3:
        risk = "high"
        verdict_lean = "大部分不实"
        reason = (
            f"FTC 红旗 {ftc_count} 条 + 缺 {missing_count} 类合规锚点"
            f"（无注册号/厂家/临床证据/披露），符合典型 misleading health-product 模式"
        )
    elif ftc_count >= 2 and missing_count >= 2:
        risk = "medium"
        verdict_lean = "误导性信息"
        reason = f"FTC 红旗 {ftc_count} 条 + 缺 {missing_count} 类合规锚点"
    elif ftc_count >= 1 or missing_count >= 2:
        risk = "medium"
        verdict_lean = "无法核实"
        reason = "promo 信号或锚点缺失，需进一步证据"
    else:
        risk = "low"
        verdict_lean = None
        reason = "promo 信号有限"

    return {
        "applied": True,
        "ftc": ftc,
        "burden": burden,
        "suggested_queries": queries,
        "risk_level": risk,
        "verdict_lean": verdict_lean,
        "reasoning": reason,
    }

"""DimensionAssessment scorer · 6 维度独立评估 + VerdictDistribution。

复刻 Claude 单 LLM 在 case_213 上的推理链：
  prior + anchor + physiological + linguistic + counterfactual + error_cost
→ 让评委看到「78% 谣言、15% 大部分不实」的概率分布，而不是黑箱单标签。

来源：oracle a7 中 Claude 6 层独立检验链的分析。

设计：
- 纯规则（零 LLM），输入 promo_health + frame + verifications → 输出 6 维 + 分布
- 每维 score ∈ [0,1]：0 = 完全偏真，1 = 完全偏假
- 每维 verdict_lean ∈ {属实, 部分属实, 无法核实, 误导性信息, 大部分不实, 谣言}
- 加权聚合得到 VerdictDistribution
"""

from __future__ import annotations

import math
import os

from .schemas import (
    ClaimVerification,
    DimensionAssessment,
    MessageFrame,
    MessageType,
    Verdict,
    VerdictDistribution,
)

# 严重度序（按 verdict 距离"真"的远近排）
_VERDICT_ORDER = [
    Verdict.TRUE,
    Verdict.PARTLY_TRUE,
    Verdict.UNVERIFIABLE,
    Verdict.MISLEADING,
    Verdict.MOSTLY_FALSE,
    Verdict.FALSE,
]


def _score_to_verdict(score: float) -> Verdict:
    """score 0-1 → Verdict。"""
    if score >= 0.85:
        return Verdict.FALSE
    if score >= 0.70:
        return Verdict.MOSTLY_FALSE
    if score >= 0.55:
        return Verdict.MISLEADING
    if score >= 0.40:
        return Verdict.UNVERIFIABLE
    if score >= 0.20:
        return Verdict.PARTLY_TRUE
    return Verdict.TRUE


# ── 数据驱动两旋钮先验（oracle Q1）──
# 旧版手写假率（health_promo 0.85 / financial_scam 0.90 …）是凭感觉拍的，且把
# "数据集长什么样"当成了"真实世界长什么样"。改成两个解耦旋钮：
#   先验 = sigmoid(logit(部署基础率) + 强度·Δ类型)，封顶 PRIOR_CAP。
# - DEPLOYMENT_FALSE_BASE_RATE：全局截距，"部署环境里多少消息是假的"，dev 上可调。
# - _TYPE_LOG_ODDS_LIFT（Δ）：每类相对赔率提升，从 design 数据用对数似然比 + 收缩学出
#   （scripts/compute_type_priors.py，data/eval/type_log_odds_lift.json）。
#   等量 50:50 抽样下"假率"无效但对数似然比仍有效（case-control 校正）。
# 同一套 Δ，只换基础率即可在"CANDY 跑分(≈0.48)"和"真实产品(≈0.10)"间切换。
DEPLOYMENT_FALSE_BASE_RATE = 0.30  # dev 可调旋钮；0.30=温和中间档
TYPE_PRIOR_STRENGTH = 0.5  # λ：让数据驱动的类型信号影响多少
PRIOR_CAP = 0.60  # 先验封顶："先验只能升怀疑，定罪靠证据"（治 58.8% 误杀）

# Δ 类型对数赔率提升（scripts/compute_type_priors.py 从 800 条 design 标注算出）
_TYPE_LOG_ODDS_LIFT = {
    MessageType.HEALTH_PRODUCT_PROMO: 0.52,
    MessageType.FINANCIAL_SCAM: -0.22,  # 数据说 CANDY 里没那么谣（38%），不是手写的 0.90
    MessageType.POLITICAL_RUMOR: 0.0,  # design 无样本，运行时也不触发
    MessageType.HEALTH_ADVICE: 0.13,
    MessageType.FACT_ASSERTION: 0.06,
    MessageType.PERSONAL_EXPERIENCE: 0.0,  # design 无样本，运行时也不触发
    MessageType.OTHER: -0.37,
}


def _sigmoid(x: float) -> float:
    # 极端值保护，防 math.exp 溢出（审查 MEDIUM）
    if x >= 500:
        return 1.0
    if x <= -500:
        return 0.0
    return 1.0 / (1.0 + math.exp(-x))


def _logit(p: float) -> float:
    p = min(max(p, 1e-6), 1.0 - 1e-6)
    return math.log(p / (1.0 - p))


def prior_false_for_type(delta: float, base_rate: float, strength: float) -> float:
    """两旋钮先验：sigmoid(logit(基础率) + 强度·Δ)，封顶 PRIOR_CAP。"""
    return min(_sigmoid(_logit(base_rate) + strength * delta), PRIOR_CAP)


# 运行时先验表（由两旋钮 + Δ 计算得出，不再手写）
_TYPE_PRIOR_FALSE = {
    mt: prior_false_for_type(
        _TYPE_LOG_ODDS_LIFT.get(mt, 0.0), DEPLOYMENT_FALSE_BASE_RATE, TYPE_PRIOR_STRENGTH
    )
    for mt in MessageType
}

# ── A/B 开关（封箱终评用：同一批证据上跑 baseline vs new，隔离搜索变量，干净归因）──
# "baseline"=阶段1/2 改动前的原始打分器（旧手写先验 + 无证实维度）；"new"=全部改动生效。
# 运行时读 env TRUTHNOTE_SCORER_MODE，也可在 eval 里程序化覆盖 dimensions.SCORER_MODE。
SCORER_MODE = os.getenv("TRUTHNOTE_SCORER_MODE", "new")

# 阶段1 前的手写假率表，仅 baseline A/B 对照保留（勿用于生产）
_OLD_TYPE_PRIOR_FALSE = {
    MessageType.HEALTH_PRODUCT_PROMO: 0.85,
    MessageType.FINANCIAL_SCAM: 0.90,
    MessageType.POLITICAL_RUMOR: 0.60,
    MessageType.HEALTH_ADVICE: 0.35,
    MessageType.FACT_ASSERTION: 0.30,
    MessageType.PERSONAL_EXPERIENCE: 0.10,
    MessageType.OTHER: 0.30,
}


# ── 第7维证实：后处理折扣（设计卡 docs/dimension7_confirmation_spec.md，oracle Q3）──
# 不当第7个平均维度（会被分母稀释开后门），而是 F = max(H, B − gate·capped·discount)。
# 证实分只来自工具/harness 已验证的证据（不信文本"看起来像真"的话）。
_CONFIRM_WEIGHTS = {
    "NAMED_VERIFIABLE_SOURCE": 0.42,  # 易声称难验证：验证后高价值
    "FALSIFIABLE_SPECIFIC": 0.14,
    "CITES_CHECKABLE_DATE": 0.08,
    "QUANTIFIED_CONDITION": 0.08,
    "HEDGED_SCOPE": 0.10,
    "DISCLOSES_LIMITATION": 0.10,
    "CONSISTENT_WITH_COMMON_SENSE": 0.05,
    "NO_CALL_TO_ACTION": 0.03,  # 最易伪造（删个"转发"即可）
}
# 外部可核查信号（能显著移动分数）vs 内部风格信号（只能小幅、且需有外部支撑才生效）
_CONFIRM_EXTERNAL = {
    "NAMED_VERIFIABLE_SOURCE",
    "FALSIFIABLE_SPECIFIC",
    "CITES_CHECKABLE_DATE",
    "QUANTIFIED_CONDITION",
}
_CONFIRM_MIN = 0.15  # C_raw 低于此 → 归零（纯内部风格救不动任何东西，抗后门核心）
_CONFIRM_CAP = 0.30  # 普通折扣封顶
_CONFIRM_CAP_PRIMARY = 0.40  # 仅一手权威源直接支持且无强证伪时的折扣封顶
_CONFIRM_C_TO_DISCOUNT = (
    0.40  # C→折扣的转换系数（独立于封顶；审查 CRITICAL：避免与 CAP_PRIMARY 数值耦合）
)

# ⚠️ 信任边界契约（审查 HIGH，2.2 接线必须遵守）：
# 下面函数把传入的 q 值和 has_exact_primary_support 当作"已被工具/harness 验证过"。
# 调用方（assess_dimensions 接线）必须只在 Evidence.source_type ∈ 高权威集
# 且 verified_status=verified_support 且独立于原文时才给 q>0 / primary=True。
# 绝不能把"文本里声称的"当验证过的传进来——否则整个抗后门失效。


def gate(h: float) -> float:
    """证伪门控：强证伪(H≥0.75)时证实归零，中等时减弱，无证伪时全效。"""
    h = min(max(h, 0.0), 1.0)
    if h >= 0.75:
        return 0.0
    if h <= 0.45:
        return 1.0
    return (0.75 - h) / 0.30


def compute_confirmation_credit(q: dict[str, float]) -> float:
    """已验证证实信用 C ∈ [0,1]。q：信号类型→验证度(0/0.5/1)。

    外部可核查信号为主；内部风格信号封顶且需外部支撑；C_raw<0.15 归零（抗后门）。
    q 值 clamp 到 [0,1]，未知信号 key 安全忽略（审查 HIGH）。
    """

    def _w(k: str, v: float) -> float:
        return _CONFIRM_WEIGHTS.get(k, 0.0) * min(max(v, 0.0), 1.0)  # clamp + 未知 key→0

    c_ext = sum(_w(k, v) for k, v in q.items() if k in _CONFIRM_EXTERNAL)
    c_int = sum(
        _w(k, v) for k, v in q.items() if k in _CONFIRM_WEIGHTS and k not in _CONFIRM_EXTERNAL
    )
    i_cap = 0.08 + 0.12 * min(1.0, c_ext / 0.20)  # 内部信号上限随外部支撑放宽
    c_raw = c_ext + min(c_int, i_cap)
    return c_raw if c_raw >= _CONFIRM_MIN else 0.0


def confirmation_cap(has_exact_primary_support: bool, h: float) -> float:
    """折扣封顶：仅一手权威源直接支持且无强证伪(H≤0.25)时给 0.40，否则 0.30。"""
    if has_exact_primary_support and h <= 0.25:
        return _CONFIRM_CAP_PRIMARY
    return _CONFIRM_CAP


def apply_confirmation(
    base: float, hard_debunk: float, credit: float, has_exact_primary_support: bool
) -> float:
    """F = max(H, B − gate(H)·min(X, k·C))。证实只能在证伪地板之上有限拉回。

    参数↔公式：base=B（6维加权均值），hard_debunk=H（已验证硬证伪强度），
    credit=C（已验证证实信用），X=折扣封顶，k=_CONFIRM_C_TO_DISCOUNT。
    """
    x = confirmation_cap(has_exact_primary_support, hard_debunk)
    delta = gate(hard_debunk) * min(x, _CONFIRM_C_TO_DISCOUNT * credit)
    return max(hard_debunk, base - delta)


# 信任边界强制点（审查 HIGH）：证实/证伪只认"工具真检索到的高权威源"。
# authority_score ≥0.70 才参与（gov/监管/辟谣/医院/学术≈0.80+，权威媒体0.75）；
# 内容农场/社媒/未知(≤0.4)一律不参与——这是把"文本声称"挡在门外的硬闸。
_CONFIRM_SUPPORT_MIN_AUTH = 0.70  # 参与证实的最低权威
_CONFIRM_PRIMARY_AUTH = 0.85  # 一手权威源门槛（gov/监管/辟谣/医院）


_REL_SUPPORT = "直接支持"  # StructuredFC evidence_relations 标签
_REL_DEBUNK = "直接辟谣"


def confirmation_and_debunk_from_verifications(
    verifications: list[ClaimVerification] | None,
) -> tuple[float, float, bool]:
    """从核查结果的证据链推导 (H 硬证伪, C 证实信用, 是否一手权威源支持)。

    方向判断优先用 StructuredFC 的 evidence_relations 标签（真实流水线总会产出），
    回退到 Evidence.supports_claim。只认 authority_score≥0.70 的源——
    低权威源(内容农场/社媒/未知)被挡在门外（防文本声称蒙混）。
    高权威源「直接支持」→ 证实 q；「直接辟谣」→ 硬证伪 H。
    """
    h = 0.0
    best_support_auth = 0.0
    primary = False
    for v in verifications or []:
        chain = getattr(v, "evidence_chain", None) or []
        rel_by_idx: dict[int, str] = {}
        for la in getattr(v, "evidence_relations", None) or []:
            idx = la.get("index")
            if isinstance(idx, int):
                rel_by_idx[idx] = la.get("relation")
        for i, e in enumerate(chain):
            auth = getattr(e, "authority_score", 0.0) or 0.0
            if auth < _CONFIRM_SUPPORT_MIN_AUTH:
                continue  # 低权威源不参与证实/证伪
            rel = rel_by_idx.get(i)
            sup = getattr(e, "supports_claim", None)
            # 方向：优先标签，回退 supports_claim（标签缺失时）
            is_support = rel == _REL_SUPPORT or (rel is None and sup is True)
            is_debunk = rel == _REL_DEBUNK or (rel is None and sup is False)
            if is_support:
                best_support_auth = max(best_support_auth, auth)
                if auth >= _CONFIRM_PRIMARY_AUTH:
                    primary = True
            elif is_debunk:
                h = max(h, 0.90 if auth >= _CONFIRM_PRIMARY_AUTH else 0.78)
    q: dict[str, float] = {}
    if best_support_auth >= _CONFIRM_PRIMARY_AUTH:
        q["NAMED_VERIFIABLE_SOURCE"] = 1.0
    elif best_support_auth >= _CONFIRM_SUPPORT_MIN_AUTH:
        q["NAMED_VERIFIABLE_SOURCE"] = 0.5
    return h, compute_confirmation_credit(q), primary


def _dim_prior(frame: MessageFrame | None) -> DimensionAssessment:
    """先验维度：消息类型决定基础"偏假"概率。"""
    if frame is None:
        score = 0.5
        evidence_for = []
        reasoning = "无 MessageFrame，用中性先验 0.5"
    else:
        if SCORER_MODE == "baseline":
            # A/B 对照：用阶段1前的旧手写表
            score = _OLD_TYPE_PRIOR_FALSE.get(frame.message_type, 0.3)
        else:
            # 未知类型兜底由公式派生（Δ=0→基础率），不硬编码避免旋钮调整后静默漂移（审查 HIGH）
            _default_prior = prior_false_for_type(
                0.0, DEPLOYMENT_FALSE_BASE_RATE, TYPE_PRIOR_STRENGTH
            )
            score = _TYPE_PRIOR_FALSE.get(frame.message_type, _default_prior)
        evidence_for = [f"消息类型 {frame.message_type.value}"]
        if frame.red_flags:
            evidence_for += [f"红旗：{', '.join(frame.red_flags[:3])}"]
        reasoning = f"{frame.message_type.value} 类型基础先验 = {score:.2f}" + (
            f"，{len(frame.red_flags)} 条红旗" if frame.red_flags else ""
        )
    return DimensionAssessment(
        name="prior",
        label="先验概率",
        score=score,
        weight=0.15,
        verdict_lean=_score_to_verdict(score).value,
        evidence_for=evidence_for,
        evidence_against=[],
        reasoning=reasoning,
    )


def _dim_anchor(promo: dict | None) -> DimensionAssessment:
    """锚点维度：产品注册/厂家/临床证据是否可查。

    缺失锚点 → 偏假；齐全 → 偏真。
    """
    if not promo or not promo.get("applied"):
        return DimensionAssessment(
            name="anchor",
            label="监管锚点",
            score=0.5,
            weight=0.10,
            verdict_lean=Verdict.UNVERIFIABLE.value,
            evidence_for=[],
            evidence_against=[],
            reasoning="非推销类，锚点检查不适用",
        )
    burden = promo.get("burden", {})
    missing = burden.get("missing_anchors", [])
    total_anchors = 4  # registration + manufacturer + clinical + disclosure
    missing_ratio = len(missing) / total_anchors
    score = 0.20 + 0.70 * missing_ratio  # 0 missing → 0.2, 4 missing → 0.9
    return DimensionAssessment(
        name="anchor",
        label="监管锚点",
        score=score,
        weight=0.20,
        verdict_lean=_score_to_verdict(score).value,
        evidence_for=[f"缺失：{m}" for m in missing[:3]],
        evidence_against=[
            f"已有：{k}"
            for k, v in burden.items()
            if isinstance(v, bool) and v and k.endswith("_present")
        ][:3],
        reasoning=f"缺 {len(missing)}/{total_anchors} 类合规锚点",
    )


def _dim_physiological(promo: dict | None) -> DimensionAssessment:
    """生理/物理可能性维度：测算速率 vs 安全阈值。"""
    if not promo or not promo.get("applied"):
        return DimensionAssessment(
            name="physiological",
            label="生理可能性",
            score=0.5,
            weight=0.10,
            verdict_lean=Verdict.UNVERIFIABLE.value,
            evidence_for=[],
            evidence_against=[],
            reasoning="无速率/物理数据可分析",
        )
    ftc = promo.get("ftc", {})
    rate = ftc.get("rate_extracted")
    if not rate:
        return DimensionAssessment(
            name="physiological",
            label="生理可能性",
            score=0.45,
            weight=0.10,
            verdict_lean=Verdict.UNVERIFIABLE.value,
            evidence_for=[],
            evidence_against=[],
            reasoning="无具体速率，物理可能性不可计算",
        )
    kg_per_week = rate.get("kg_per_week", 0)
    # NHC/CDC 安全标准：≤1 kg/week；FTC：>2 lb/week = 0.91 kg/week 触发
    if kg_per_week > 1.5:
        score = 0.85
        lean_reason = "速率严重超过安全阈值"
    elif kg_per_week > 1.0:
        score = 0.70
        lean_reason = "速率超过 NHC 安全建议（0.5-1 kg/周）"
    elif kg_per_week > 0.5:
        score = 0.30
        lean_reason = "速率在合理范围"
    else:
        score = 0.15
        lean_reason = "速率保守安全"
    return DimensionAssessment(
        name="physiological",
        label="生理可能性",
        score=score,
        weight=0.15,
        verdict_lean=_score_to_verdict(score).value,
        evidence_for=[
            f"{rate['kg_per_week']:.2f} kg/周（{rate['total_kg']:.1f} kg / {rate['weeks']:.1f} 周）"
        ],
        evidence_against=[],
        reasoning=lean_reason,
    )


def _dim_linguistic(promo: dict | None, frame: MessageFrame | None) -> DimensionAssessment:
    """语言指纹维度：FTC 红旗（话术）数量。

    包含购买命令 + 个人见证 + 快速效果 + 安全承诺 + 竞品贬损 + 伪科学。
    """
    flag_count = 0
    flags: list[str] = []
    if promo and promo.get("applied"):
        ftc_flags = promo.get("ftc", {}).get("flags_triggered", [])
        flag_count += len(ftc_flags)
        flags += ftc_flags
    if frame and frame.red_flags:
        flag_count += len(frame.red_flags)
        flags += frame.red_flags

    if flag_count >= 6:
        score = 0.85
    elif flag_count >= 4:
        score = 0.72
    elif flag_count >= 2:
        score = 0.55
    elif flag_count >= 1:
        score = 0.40
    else:
        score = 0.20

    return DimensionAssessment(
        name="linguistic",
        label="语言指纹",
        score=score,
        weight=0.15,
        verdict_lean=_score_to_verdict(score).value,
        evidence_for=flags[:5],
        evidence_against=[],
        reasoning=f"{flag_count} 条话术/红旗指纹",
    )


def _dim_counterfactual(promo: dict | None, frame: MessageFrame | None) -> DimensionAssessment:
    """反事实维度：合法发送者会这么写吗？

    评分规则：缺失典型合法元素（披露、医生引用、批准号）→ 偏假。
    """
    if not promo or not promo.get("applied"):
        return DimensionAssessment(
            name="counterfactual",
            label="反事实测试",
            score=0.5,
            weight=0.10,
            verdict_lean=Verdict.UNVERIFIABLE.value,
            evidence_for=[],
            evidence_against=[],
            reasoning="非推销类，反事实测试不适用",
        )
    burden = promo.get("burden", {})
    omitted: list[str] = []
    if not burden.get("disclosure_present"):
        omitted.append("效果因人而异等典型披露")
    if not burden.get("registration_anchor_present"):
        omitted.append("批准文号/备案号")
    if not burden.get("clinical_evidence_anchor_present"):
        omitted.append("临床证据/医生引用")
    if not burden.get("manufacturer_anchor_present"):
        omitted.append("生产厂家完整信息")

    score = 0.20 + 0.18 * len(omitted)  # 0 omitted → 0.2, 4 → 0.92
    score = min(score, 0.95)

    return DimensionAssessment(
        name="counterfactual",
        label="反事实测试",
        score=score,
        weight=0.15,
        verdict_lean=_score_to_verdict(score).value,
        evidence_for=[f"合法卖家通常写：{x}" for x in omitted[:3]],
        evidence_against=[],
        reasoning=(
            "合法卖家会写而本消息没写的关键元素：" + "、".join(omitted[:3])
            if omitted
            else "合法元素齐全"
        ),
    )


def _dim_error_cost(promo: dict | None, frame: MessageFrame | None) -> DimensionAssessment:
    """错误代价维度：非对称损失。

    高风险话题（健康产品/金融）下，false negative（漏判）代价远高于 false positive。
    → 该维度倾向"保守偏假"（即使证据不齐也提示风险）。
    """
    if frame and frame.message_type in (
        MessageType.HEALTH_PRODUCT_PROMO,
        MessageType.FINANCIAL_SCAM,
    ):
        if promo and promo.get("risk_level") == "high":
            score = 0.80
            reasoning = "高风险健康/金融话题 + PromoHealth high risk，错误代价不对称偏假"
        elif promo and promo.get("risk_level") == "medium":
            score = 0.60
            reasoning = "高风险话题 + medium PromoHealth，错误代价偏保守"
        else:
            score = 0.50
            reasoning = "高风险话题但 promo 信号弱，中性"
        return DimensionAssessment(
            name="error_cost",
            label="错误代价",
            score=score,
            weight=0.15,
            verdict_lean=_score_to_verdict(score).value,
            evidence_for=[f"{frame.message_type.value} 类型 false-negative 代价高"],
            evidence_against=[],
            reasoning=reasoning,
        )
    return DimensionAssessment(
        name="error_cost",
        label="错误代价",
        score=0.40,
        weight=0.05,
        verdict_lean=Verdict.UNVERIFIABLE.value,
        evidence_for=[],
        evidence_against=[],
        reasoning="非高风险话题，错误代价对称",
    )


def assess_dimensions(
    *,
    frame: MessageFrame | None,
    promo: dict | None,
    verifications: list[ClaimVerification] | None = None,
) -> list[DimensionAssessment]:
    """生成 6 维度独立评估。"""
    return [
        _dim_prior(frame),
        _dim_anchor(promo),
        _dim_physiological(promo),
        _dim_linguistic(promo, frame),
        _dim_counterfactual(promo, frame),
        _dim_error_cost(promo, frame),
    ]


def aggregate_distribution(
    dimensions: list[DimensionAssessment],
    *,
    pipeline_verdict: Verdict | None = None,
    pipeline_confidence: float = 0.0,
    verifications: list[ClaimVerification] | None = None,
) -> VerdictDistribution:
    """把 6 维度加权聚合成 VerdictDistribution（路演杀器）。

    算法：
    1. 每维 score → 对应 verdict 索引（0=TRUE...5=FALSE）
    2. 加权平均得到加权 score
    3. 把加权 score 转换为 6 类 verdict 的概率分布
       （用一个简单的 mass-spread：主标签得最大权重，相邻类别按距离衰减）
    4. pipeline_verdict 作为额外信号叠加（如果存在）
    """
    if not dimensions:
        return VerdictDistribution(UNVERIFIABLE=1.0)

    # 加权 score（B）
    total_w = sum(d.weight for d in dimensions) or 1.0
    weighted = sum(d.score * d.weight for d in dimensions) / total_w
    weighted = max(0.0, min(1.0, weighted))

    # 阶段2 证实后处理：F = max(H, B − gate·min(X,k·C))，仅 new 模式 + 有核查证据时
    if SCORER_MODE != "baseline" and verifications:
        h, c, primary = confirmation_and_debunk_from_verifications(verifications)
        weighted = apply_confirmation(weighted, h, c, primary)
        weighted = max(0.0, min(1.0, weighted))

    # 把 weighted ∈ [0,1] 映射到 0..5 索引（连续）
    pos = weighted * (len(_VERDICT_ORDER) - 1)  # 0..5

    # mass-spread：每个 verdict 槽位的权重 = exp(-(i-pos)^2 / sigma^2)
    import math

    sigma = 0.95
    masses = [math.exp(-((i - pos) ** 2) / (sigma**2)) for i in range(len(_VERDICT_ORDER))]

    # 如果有 pipeline_verdict，叠加 0.5 * confidence 到对应位置
    if pipeline_verdict and pipeline_verdict in _VERDICT_ORDER:
        idx = _VERDICT_ORDER.index(pipeline_verdict)
        masses[idx] += 0.5 * max(0.0, min(1.0, pipeline_confidence))

    # 归一化
    total = sum(masses) or 1.0
    probs = [m / total for m in masses]

    return VerdictDistribution(
        TRUE=probs[0],
        PARTLY_TRUE=probs[1],
        UNVERIFIABLE=probs[2],
        MISLEADING=probs[3],
        MOSTLY_FALSE=probs[4],
        FALSE=probs[5],
    )


def compose_verdict_explanation(
    *,
    overall_verdict: Verdict,
    distribution: VerdictDistribution | None,
    dimensions: list[DimensionAssessment],
    frame: MessageFrame | None,
    promo: dict | None,
) -> str:
    """基于 6 维度 + PromoHealth + MessageFrame 数据生成人话解释。

    路演用：让评委/用户看到 "为什么是这个判定" 的结构化推理，而不是黑盒标签。
    返回 markdown 字符串。
    """
    parts: list[str] = []

    # 1. 头：判定 + 概率
    verdict_name = overall_verdict.value if overall_verdict else "未知"
    if distribution:
        top_prob = max(
            distribution.FALSE,
            distribution.MOSTLY_FALSE,
            distribution.MISLEADING,
            distribution.UNVERIFIABLE,
            distribution.PARTLY_TRUE,
            distribution.TRUE,
        )
        parts.append(f"## 综合判定：**{verdict_name}**（概率约 {top_prob:.0%}）")
    else:
        parts.append(f"## 综合判定：**{verdict_name}**")

    # 2. 消息类型 / MessageFrame
    if frame and frame.message_type.value not in ("other", "fact_assertion"):
        type_label = {
            "health_product_promo": "健康产品推销",
            "financial_scam": "金融诈骗",
            "political_rumor": "政治谣言",
            "health_advice": "健康建议",
            "personal_experience": "个人经历",
        }.get(frame.message_type.value, frame.message_type.value)
        parts.append("### 1. 这是什么类型的消息")
        parts.append(f"- 类型：**{type_label}**（不是普通事实声明）")
        if frame.central_action_claim:
            parts.append(f"- 它想让你做的事：{frame.central_action_claim}")
        if frame.promoted_entity:
            parts.append(f"- 被推销的对象：{frame.promoted_entity}")
        if frame.red_flags:
            parts.append(
                "- 检测到的话术红旗：" + "、".join(f"**{f}**" for f in frame.red_flags[:6])
            )

    # 3. PromoHealth 缺失锚点
    if promo and promo.get("applied"):
        burden = promo.get("burden", {})
        missing = burden.get("missing_anchors", [])
        ftc_flags = promo.get("ftc", {}).get("flags_triggered", [])
        rate = promo.get("ftc", {}).get("rate_extracted")

        if missing or ftc_flags:
            parts.append("### 2. 它缺了什么、违反了什么")
        if missing:
            parts.append("**合法销售应有但消息中没有：**")
            for m in missing[:6]:
                parts.append(f"- ❌ {m}")
        if ftc_flags:
            flag_translate = {
                "FTC-testimonial_over_15_lb": "总减重 > 15 lb（触发 FTC 虚假减肥广告阈值）",
                "FTC-testimonial_over_2_lb_per_week_for_month": (
                    "速率 > 2 lb/周持续 1 个月（FTC 阈值）"
                ),
                "FTC-no_rebound_claim": "用了「不反弹/不复胖」承诺（FTC 列为可疑话术）",
                "FTC-safe_rapid_loss_claim": "用了「健康安全快速」承诺（FTC 列为可疑话术）",
                "FTC-permanent_loss_claim": "用了「永久减重/终身有效」承诺",
            }
            parts.append("**触发的虚假减肥广告红旗：**")
            for f in ftc_flags[:6]:
                parts.append(f"- 🚩 {flag_translate.get(f, f)}")
        if rate:
            parts.append(
                f"**实际速率：{rate['kg_per_week']:.2f} kg/周**"
                f"（消息称 {rate['total_kg']:.1f} kg / {rate['weeks']:.1f} 周）"
                f"，对照国家卫健委安全建议 0.5-1 kg/周"
            )

    # 4. 6 维度独立投票
    if dimensions:
        parts.append("### 3. 6 个独立维度怎么投的")
        verdict_count: dict[str, int] = {}
        for d in dimensions:
            verdict_count[d.verdict_lean] = verdict_count.get(d.verdict_lean, 0) + 1
        sorted_v = sorted(verdict_count.items(), key=lambda x: -x[1])
        vote_summary = "、".join(f"{n}/6 维度投「{v}」" for v, n in sorted_v[:3])
        parts.append(vote_summary)
        # 详细列出每维倾向偏假的（score >= 0.55）
        biased_against = [d for d in dimensions if d.score >= 0.55]
        if biased_against:
            parts.append("**偏向「假」的维度（按权重）：**")
            for d in sorted(biased_against, key=lambda x: -x.score * x.weight):
                parts.append(f"- {d.label}（{d.score:.0%}）→ {d.reasoning}")

    # 5. 结论
    parts.append("### 4. 建议")
    if overall_verdict and overall_verdict.value in ("谣言", "大部分不实"):
        parts.append(
            "⚠️ 不建议购买、转发或采纳。如果消息提到了具体产品，建议要求卖家提供："
            "批准文号、生产厂家、临床证据、典型结果披露后再考虑。"
        )
    elif overall_verdict and overall_verdict.value == "误导性信息":
        parts.append("⚠️ 消息可能在用真话暗示假结论，请谨慎对待。")
    elif overall_verdict and overall_verdict.value == "无法核实":
        parts.append("❓ 证据不足以下结论，建议观望，不要急于转发或行动。")
    else:
        parts.append("✅ 当前证据下大体可信。")

    return "\n\n".join(parts)

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, Field


class SourceType(StrEnum):
    OFFICIAL_GOVERNMENT = "official_government"
    REGULATOR = "regulator"
    HOSPITAL_MEDICAL = "hospital_medical"
    ACADEMIC = "academic"
    FACT_CHECK_ORG = "fact_check_org"
    ESTABLISHED_MEDIA = "established_media"
    ENCYCLOPEDIA = "encyclopedia"
    BLOG_FORUM = "blog_forum"
    SOCIAL_MEDIA = "social_media"
    UNKNOWN = "unknown"


class RumorCategory(StrEnum):
    POLICY = "政策法规"
    HEALTH = "健康养生"
    SCAM = "诈骗套路"
    FAKE_SCREENSHOT = "伪造截图"
    OLD_NEWS = "旧闻翻炒"
    DISASTER = "灾难恐慌"
    FINANCE = "金融财经"
    AI_QUOTE = "AI名人语录"
    FOOD_SAFETY = "食品安全"
    OTHER = "其他"


class Verdict(StrEnum):
    FALSE = "谣言"
    MOSTLY_FALSE = "大部分不实"
    MISLEADING = "误导性信息"
    PARTLY_TRUE = "部分属实"
    TRUE = "属实"
    UNVERIFIABLE = "无法核实"


class DisplayBucket(StrEnum):
    """对外展示三档分组（评委看的是这个 + 二元徽章）。内部六值 Verdict 不变。"""

    REAL = "真实"
    MIXED = "真假混杂"
    RUMOR = "谣言"


class BinaryBadge(StrEnum):
    """对外二元徽章（投影层）。MIXED 与 RUMOR 都投影为 RUMOR，但带子类型说明。"""

    REAL = "真实"
    RUMOR = "谣言"


class BlockerType(StrEnum):
    """非裁定道主障碍类型（决策树按此顺序判定）。取代旧 developing/insufficient。"""

    NO_CHECKABLE_FACT = "非事实命题"
    MISSING_KEY_CONTEXT = "缺关键语境"
    SOURCE_ARTIFACT_AUTH = "来源不可溯源"
    NON_PUBLIC_EVIDENCE = "私域或超本地"
    NOT_YET_SETTLED = "尚未落定"
    CONFLICTING_AUTH = "权威证据冲突"
    NO_PUBLIC_EVIDENCE_FOUND = "可查但证据不足"


_VERDICT_TO_BUCKET: dict[Verdict, DisplayBucket] = {
    Verdict.TRUE: DisplayBucket.REAL,
    Verdict.PARTLY_TRUE: DisplayBucket.MIXED,
    Verdict.MISLEADING: DisplayBucket.MIXED,
    Verdict.MOSTLY_FALSE: DisplayBucket.RUMOR,
    Verdict.FALSE: DisplayBucket.RUMOR,
    # UNVERIFIABLE 不在表内 —— 不是裁定，走非裁定道，绝不给徽章
}
_BUCKET_TO_BADGE: dict[DisplayBucket, BinaryBadge] = {
    DisplayBucket.REAL: BinaryBadge.REAL,
    DisplayBucket.MIXED: BinaryBadge.RUMOR,
    DisplayBucket.RUMOR: BinaryBadge.RUMOR,
}
_VERDICT_SUBTYPE_LABEL: dict[Verdict, str] = {
    Verdict.TRUE: "属实",
    Verdict.PARTLY_TRUE: "部分属实——真假混杂，部分关键结论不成立",
    Verdict.MISLEADING: "误导性信息——事实大体为真，但框架/语境制造了错误印象",
    Verdict.MOSTLY_FALSE: "大部分不实——核心结论或数字是假的",
    Verdict.FALSE: "谣言——无可信的真实核心",
}


def project_to_binary(
    verdict: Verdict,
) -> tuple[BinaryBadge, DisplayBucket, str] | None:
    """裁定道投影：内部六值 → (二元徽章, 三档分组, 子类型文案)。

    返回 None 表示非裁定结论（UNVERIFIABLE/未知）——调用方改走非裁定道，
    绝不给徽章（INV-U1）。
    """
    bucket = _VERDICT_TO_BUCKET.get(verdict)
    if bucket is None:
        return None
    return (
        _BUCKET_TO_BADGE[bucket],
        bucket,
        _VERDICT_SUBTYPE_LABEL.get(verdict, verdict.value),
    )


class NonAdjudicatedAction(BaseModel):
    """非裁定输出（判定层之外）。UNVERIFIABLE 内部状态对外的呈现形态。

    INV-U1 不带徽章；INV-U2 blocker 必填+detail 非空；INV-U3 detail 不断真伪。
    """

    action_kind: str = Field(default="needs_first_hand_confirmation")
    primary_blocker: BlockerType
    secondary_flags: list[str] = Field(default_factory=list)
    claim_specific_detail: str = Field(description="这一条专属、不断真伪的具体障碍说明")
    verify_where: str = Field(default="", description="该去哪做一手确认")


class MessageType(StrEnum):
    """消息类型（用于 MessageFrame 路由）。

    见 CONTRACTS.md C0 INV-1：MessageFrame 必须绑定下游 Agent。
    case_213 失败的根因：ScenarioRouter 已知"健康养生"但 ClaimExtractor 没有
    type-aware 行为，把推销类当普通事实声明处理。
    """

    HEALTH_PRODUCT_PROMO = "health_product_promo"  # 健康产品推销（case_213 类型）
    FINANCIAL_SCAM = "financial_scam"  # 金融诈骗
    POLITICAL_RUMOR = "political_rumor"  # 政治谣言
    HEALTH_ADVICE = "health_advice"  # 健康建议（非推销）
    FACT_ASSERTION = "fact_assertion"  # 事实断言
    PERSONAL_EXPERIENCE = "personal_experience"  # 个人经历
    OTHER = "other"


class SpeechAct(BaseModel):
    """speech act 字段（D3 决策：合并进 MessageFrame，不单独建 Agent）。"""

    span: str = Field(description="原文片段")
    act: str = Field(description="assertive / directive / commissive / expressive / testimonial")
    intended_action: str = Field(default="", description="directive 的意图（如 purchase）")
    verification: str = Field(default="", description="该 act 的验证方式")


class MessageFrame(BaseModel):
    """消息框架（INV-1 强制对象）。

    所有下游 Agent 必须 consume 这个对象，而不是只看 Claim[]。
    """

    message_type: MessageType = MessageType.OTHER
    central_action_claim: str = Field(
        default="", description="消息的中心行动主张（推销类必须非空）"
    )
    central_public_meaning: str = Field(default="", description="消息的公共意义（不同于私人体验）")
    promoted_entity: str = Field(default="", description="被推销/讨论的实体名")
    target_audience: list[str] = Field(default_factory=list, description="消息瞄准的人群")
    speech_acts: list[SpeechAct] = Field(default_factory=list, description="speech act 分解")
    verification_burden: list[str] = Field(
        default_factory=list,
        description="必须核验的清单（注册/批准/疗效/安全等）",
    )
    red_flags: list[str] = Field(default_factory=list, description="结构化红旗清单")
    confidence: float = Field(default=0.0, ge=0, le=1)
    raw_router_hint: str = Field(default="", description="来自 ScenarioRouter 的 strategy_hint")


class DimensionAssessment(BaseModel):
    """6 维度独立评估（演示输出契约 · 透明推理链）。

    Claude 单 LLM 判 case_213 的方式：6 层独立检验链，每层独立投票。
    复刻这种推理结构，把"黑箱单标签"变成"可量化推理链"。
    """

    name: str = Field(description="prior/anchor/physiological/linguistic/counterfactual/error_cost")
    label: str = Field(description="中文展示名（先验/锚点/生理/语言/反事实/错误代价）")
    score: float = Field(ge=0, le=1, description="0=完全偏向真，1=完全偏向假")
    weight: float = Field(default=1.0, ge=0, le=1, description="该维度对总分的权重")
    verdict_lean: str = Field(description="该维度倾向的 verdict（如「谣言」「无法核实」）")
    evidence_for: list[str] = Field(default_factory=list, description="支持该方向的证据点")
    evidence_against: list[str] = Field(default_factory=list, description="反向证据点")
    reasoning: str = Field(default="", description="一句话理由")


class VerdictDistribution(BaseModel):
    """6 个 verdict 类别的概率分布（不是单一硬标签）。

    演示杀器：让评委看到「78% 谣言、15% 大部分不实、5% 无法核实」
    而不是只看到「谣言 80%」。
    """

    FALSE: float = Field(default=0.0, ge=0, le=1)
    MOSTLY_FALSE: float = Field(default=0.0, ge=0, le=1)
    MISLEADING: float = Field(default=0.0, ge=0, le=1)
    UNVERIFIABLE: float = Field(default=0.0, ge=0, le=1)
    PARTLY_TRUE: float = Field(default=0.0, ge=0, le=1)
    TRUE: float = Field(default=0.0, ge=0, le=1)

    def normalize(self) -> VerdictDistribution:
        total = (
            self.FALSE
            + self.MOSTLY_FALSE
            + self.MISLEADING
            + self.UNVERIFIABLE
            + self.PARTLY_TRUE
            + self.TRUE
        )
        if total <= 0:
            return self
        return VerdictDistribution(
            FALSE=self.FALSE / total,
            MOSTLY_FALSE=self.MOSTLY_FALSE / total,
            MISLEADING=self.MISLEADING / total,
            UNVERIFIABLE=self.UNVERIFIABLE / total,
            PARTLY_TRUE=self.PARTLY_TRUE / total,
            TRUE=self.TRUE / total,
        )

    def argmax_verdict(self) -> str:
        items = {
            "谣言": self.FALSE,
            "大部分不实": self.MOSTLY_FALSE,
            "误导性信息": self.MISLEADING,
            "无法核实": self.UNVERIFIABLE,
            "部分属实": self.PARTLY_TRUE,
            "属实": self.TRUE,
        }
        return max(items, key=items.get)


class Claim(BaseModel):
    text: str = Field(description="提取出的具体事实声明")
    category: RumorCategory = Field(default=RumorCategory.OTHER)
    original_context: str = Field(default="", description="声明在原文中的上下文")
    is_central_action: bool = Field(
        default=False, description="是否是消息的中心行动主张（INV-1 要求）"
    )


class Evidence(BaseModel):
    source: str = Field(description="来源名称/网站")
    url: str = Field(default="")
    title: str = Field(default="")
    snippet: str = Field(description="相关内容摘要")
    credibility: str = Field(default="未评估", description="来源可信度")
    supports_claim: bool | None = Field(default=None, description="是否支持该声明")
    source_tag: str = Field(default="", description="来源标签（如 360搜索）")
    source_type: SourceType = Field(default=SourceType.UNKNOWN)
    authority_score: float = Field(default=0.4, ge=0, le=1)
    published_date: str = Field(default="", description="发布日期 YYYY-MM-DD")
    is_original_source: bool = Field(default=False)
    # 可视化采信理由（仅官方辟谣库采信证据填）：结构化同命题核对标签 + 分数，
    # 让前端渲染干净的「为什么采信」chip，而不是从 snippet 散文里抠。默认空=非辟谣库证据。
    match_label: str = Field(
        default="", description="同命题核对标签（same_claim 等），仅辟谣库采信证据填"
    )
    match_score: float | None = Field(
        default=None, description="同命题核对分数（0-1），供可视化展示相关度"
    )

    def model_post_init(self, __context: object) -> None:
        """根据 source/url 自动标注 360 搜索来源。"""
        if not self.source_tag:
            s = (self.source or "").lower()
            u = (self.url or "").lower()
            if "360" in s or "so.com" in u or "360.cn" in u or "so.com" in s:
                self.source_tag = "360搜索"


class ClaimVerification(BaseModel):
    claim: Claim
    verdict: Verdict
    confidence: float = Field(ge=0, le=1, description="判定置信度")
    evidence_chain: list[Evidence] = Field(default_factory=list)
    reasoning: str = Field(default="", description="推理过程")
    evidence_relations: list[dict] = Field(
        default_factory=list,
        description="StructuredFC 的证据关系标签：[{index, relation}]",
    )
    dimensions: list[DimensionAssessment] = Field(
        default_factory=list,
        description="6 维度独立评估（演示输出契约，可选）",
    )
    verdict_distribution: VerdictDistribution | None = Field(
        default=None, description="6 类 verdict 概率分布（不是单一硬标签）"
    )
    # 单条声明无法核实时的细化归因（统一契约，序列化进 SSE done 事件的 claim 对象）。
    # 结构：{code, code_label, detail, blocked_condition, verify_where}
    #   （snake_case，前端转 camelCase）。仅 verdict==UNVERIFIABLE 的声明非空，其余 None。
    #   绝不携带任何能旁路裁决产 verdict 的字段（守 INV-4）。
    unverifiable_reason: dict | None = Field(
        default=None, description="无法核实细化归因（非裁定道·不断真伪）"
    )


class QueryPlan(BaseModel):
    queries: list[str] = Field(default_factory=list, description="搜索查询列表")
    strategy: str = Field(default="", description="搜索策略说明")
    official_sites: list[str] = Field(default_factory=list, description="指定查询的权威站点")


class EvidenceRanking(BaseModel):
    ranked_evidence: list[Evidence] = Field(default_factory=list)
    sufficiency: str = Field(
        default="insufficient", description="sufficient/insufficient/conflicting"
    )
    reasoning: str = Field(default="")


class SkepticChallenge(BaseModel):
    challenges: list[str] = Field(default_factory=list, description="质疑点列表")
    passed: bool = Field(default=False, description="是否通过质疑检验")
    revised_verdict: Verdict | None = Field(default=None)
    # P1 #5：INV-3 守护拦截标记。独立字段避免把 "[INV-3 拦截] ..." 内部诊断
    # 字符串塞进 challenges 数组，污染 humanize._h_skeptic 给用户看的质疑预览。
    inv3_blocked: bool = Field(default=False, description="是否被 INV-3 拦截了通用怀疑降级")


class VerificationState(BaseModel):
    """流水线完整状态对象。

    从消息进入到最终输出，每一步的中间结果都存在这里。
    Day 4 的 ScenarioRouter / QueryPlanner / EvidenceRanker / Skeptic
    都读写这个对象。
    """

    original_message: str = ""
    context: str = ""
    routed_scenario: RumorCategory = RumorCategory.OTHER
    claims: list[Claim] = Field(default_factory=list)
    query_plan: QueryPlan = Field(default_factory=QueryPlan)
    raw_evidence: list[Evidence] = Field(default_factory=list)
    evidence_ranking: EvidenceRanking = Field(default_factory=EvidenceRanking)
    verifications: list[ClaimVerification] = Field(default_factory=list)
    skeptic: SkepticChallenge = Field(default_factory=SkepticChallenge)
    overall_verdict: Verdict = Verdict.UNVERIFIABLE
    friendly_reply: str = ""
    summary: str = ""
    memory_hit: bool = Field(default=False, description="是否命中记忆库")
    memory_case_id: str | None = Field(default=None)


class VerifyRequest(BaseModel):
    message: str = Field(description="需要核查的群聊消息")
    context: str = Field(default="", description="可选的附加上下文")
    default_model: str = Field(default="", description="覆盖默认模型（简单 Agent）")
    strong_model: str = Field(
        default="", description="覆盖关键模型（ClaimExtractor/StructuredFC/Skeptic）"
    )
    request_id: str = Field(default="", description="客户端追踪 id，用于全链路日志关联")


class DecisionPoint(BaseModel):
    """单个决策节点：记录 pipeline 中每个判断点的输入、输出和理由。"""

    stage: str = Field(
        description=(
            "决策阶段：rule_engine / fact_checker / skeptic / d2_gate / "
            "true_rescue_gate / evidence_prescore / atom_aggregation"
        )
    )
    verdict_before: str = Field(default="", description="进入此阶段前的判定（如有）")
    verdict_after: str = Field(default="", description="此阶段输出的判定")
    fired: bool = Field(default=False, description="此决策点是否实际触发/改变了结果")
    detail: str = Field(default="", description="决策理由")
    signals: dict = Field(default_factory=dict, description="决策依据的关键信号")


class DiagnosticTrace(BaseModel):
    """全链路诊断追踪：记录从消息进入到最终判定的完整决策链。"""

    claim_text: str = Field(default="")
    decisions: list[DecisionPoint] = Field(default_factory=list)
    evidence_summary: dict = Field(
        default_factory=dict, description="证据概要：debunk/support/neutral 数量"
    )
    final_verdict: str = Field(default="")
    final_confidence: float = Field(default=0.0)

    def add(self, point: DecisionPoint) -> None:
        self.decisions.append(point)


class TraceStep(BaseModel):
    agent: str = Field(description="执行 Agent 名称")
    action: str = Field(description="执行动作")
    duration_ms: int = Field(description="耗时毫秒")
    input_summary: str = Field(default="")
    output_summary: str = Field(default="")
    output_data: dict | None = Field(default=None, description="结构化输出（前端 KV 渲染用）")
    human_narrative: str = Field(
        default="", description="评委友好的人话（render-engine 优先读这个）"
    )
    display: dict | None = Field(
        default=None,
        description="渲染模板 payload：{template, data}，对应 extension_event_contract.md",
    )


class InvestigationTrace(BaseModel):
    steps: list[TraceStep] = Field(default_factory=list)
    total_duration_ms: int = 0
    total_llm_calls: int = 0
    total_tokens_used: int = 0
    tokens_by_agent: dict[str, int] = Field(default_factory=dict)
    claims_extracted: int = 0
    claims_checkworthy: int = 0
    scenario: str = Field(default="")
    scenario_confidence: float = Field(default=0.0)
    strategy_hint: str = Field(default="")
    diagnostics: list[DiagnosticTrace] = Field(
        default_factory=list, description="每条 claim 的全链路决策诊断"
    )
    message_frame: MessageFrame | None = Field(
        default=None, description="MessageFrameBuilder 输出（INV-1）"
    )
    coverage_audit: dict | None = Field(
        default=None,
        description="CoverageAuditor 输出（INV-2）：{covered, missing, satisfied}",
    )
    promo_health: dict | None = Field(
        default=None, description="PromoHealthVerifier 输出（FTC + BurdenOfProof）"
    )
    dimensions: list[DimensionAssessment] = Field(
        default_factory=list, description="6 维度独立评估（演示输出契约）"
    )
    verdict_distribution: VerdictDistribution | None = Field(
        default=None, description="6 类 verdict 概率分布"
    )
    verdict_explanation: str = Field(
        default="",
        description="综合人话解释（compose_verdict_explanation 输出，markdown）",
    )


class RiskLevel(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class LifecycleState(StrEnum):
    DETECTED = "detected"
    EXTRACTED = "extracted"
    VERIFIED = "verified"
    JUDGED = "judged"
    INTERVENED = "intervened"
    TRACKING = "tracking"
    ESCALATED = "escalated"
    MEMORIZED = "memorized"
    REPEAT_BLOCKED = "repeat_blocked"


class LifecycleTransition(BaseModel):
    from_state: LifecycleState
    to_state: LifecycleState
    timestamp: datetime = Field(default_factory=datetime.now)
    detail: str = ""


class CaseLifecycle(BaseModel):
    case_id: int = 0
    current_state: LifecycleState = LifecycleState.DETECTED
    risk_level: RiskLevel = RiskLevel.MEDIUM
    intervention_type: str = ""
    transitions: list[LifecycleTransition] = Field(default_factory=list)
    self_correction_sent: bool = False
    self_correction_script: str = Field(default="", description="预写的发信人自纠话术")
    escalation_message: str = Field(default="", description="超时后的群发补充消息")
    escalation_deadline: datetime | None = Field(default=None, description="升级截止时间")
    escalation_timeout_sec: int = 300
    repeat_blocked: bool = Field(default=False, description="是否为重复拦截")
    created_at: datetime = Field(default_factory=datetime.now)

    def advance(self, new_state: LifecycleState, detail: str = "") -> None:
        self.transitions.append(
            LifecycleTransition(
                from_state=self.current_state,
                to_state=new_state,
                detail=detail,
            )
        )
        self.current_state = new_state


class DispositionStatus(BaseModel):
    case_id: int = 0
    copied: bool = False
    reported: bool = False
    tracked: bool = False
    clarification_generated: bool = False


class VerifyResponse(BaseModel):
    original_message: str
    claims: list[ClaimVerification]
    overall_verdict: Verdict
    summary: str = Field(description="核查结论摘要")
    friendly_reply: str = Field(description="发给爸妈版的温和回复")
    evidence_sources: list[str] = Field(default_factory=list)
    search_engines_used: list[str] = Field(
        default_factory=list, description="使用的搜索引擎（如 360搜索、Tavily）"
    )
    timestamp: datetime = Field(default_factory=datetime.now)
    trace: InvestigationTrace = Field(default_factory=InvestigationTrace)
    disposition: DispositionStatus = Field(default_factory=DispositionStatus)
    lifecycle: CaseLifecycle = Field(default_factory=CaseLifecycle)
    actions: list[dict] = Field(default_factory=list, description="闭环动作列表")
    claimreviews: list[dict] = Field(default_factory=list, description="ClaimReview JSON-LD")
    binary_badge: BinaryBadge | None = Field(
        default=None, description="对外二元徽章；非裁定时 None（INV-U1）"
    )
    display_bucket: DisplayBucket | None = Field(default=None, description="三档展示分组")
    display_subtype: str = Field(default="", description="子类型说明文案")
    non_adjudicated: NonAdjudicatedAction | None = Field(
        default=None, description="非裁定输出（判定层之外）"
    )

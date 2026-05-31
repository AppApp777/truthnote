from src.truthnote.schemas import (
    Claim,
    ClaimVerification,
    Evidence,
    EvidenceRanking,
    QueryPlan,
    RumorCategory,
    SkepticChallenge,
    Verdict,
    VerificationState,
    VerifyRequest,
    VerifyResponse,
)


def test_claim_defaults():
    c = Claim(text="测试声明")
    assert c.category == RumorCategory.OTHER
    assert c.original_context == ""


def test_evidence_model():
    e = Evidence(source="人民网", snippet="这是一段摘要")
    assert e.credibility == "未评估"
    assert e.supports_claim is None


def test_claim_verification():
    cv = ClaimVerification(
        claim=Claim(text="测试"),
        verdict=Verdict.FALSE,
        confidence=0.9,
    )
    assert cv.verdict == Verdict.FALSE
    assert 0 <= cv.confidence <= 1


def test_verify_request():
    req = VerifyRequest(message="测试消息")
    assert req.context == ""


def test_verify_response():
    resp = VerifyResponse(
        original_message="测试",
        claims=[],
        overall_verdict=Verdict.UNVERIFIABLE,
        summary="无声明",
        friendly_reply="没什么问题",
    )
    assert resp.evidence_sources == []
    assert resp.timestamp is not None


def test_new_categories():
    for cat in [RumorCategory.DISASTER, RumorCategory.FINANCE, RumorCategory.AI_QUOTE]:
        c = Claim(text="测试", category=cat)
        assert c.category == cat


def test_verification_state_defaults():
    state = VerificationState()
    assert state.routed_scenario == RumorCategory.OTHER
    assert state.claims == []
    assert state.query_plan.queries == []
    assert state.evidence_ranking.sufficiency == "insufficient"
    assert state.skeptic.passed is False
    assert state.overall_verdict == Verdict.UNVERIFIABLE
    assert state.memory_hit is False


def test_verification_state_full():
    claim = Claim(text="存款超5万要交税", category=RumorCategory.POLICY)
    evidence = Evidence(source="人民网", snippet="此为谣言")
    state = VerificationState(
        original_message="存款超5万要交税！",
        routed_scenario=RumorCategory.POLICY,
        claims=[claim],
        query_plan=QueryPlan(
            queries=["存款 交税 辟谣"],
            strategy="查权威来源",
            official_sites=["gov.cn"],
        ),
        raw_evidence=[evidence],
        evidence_ranking=EvidenceRanking(
            ranked_evidence=[evidence],
            sufficiency="sufficient",
            reasoning="权威来源明确辟谣",
        ),
        skeptic=SkepticChallenge(
            challenges=["是否有部分地区试点？"],
            passed=True,
        ),
        overall_verdict=Verdict.FALSE,
        friendly_reply="爸，这个消息不太准确哦",
        summary="经核查为谣言",
        memory_hit=False,
    )
    assert state.routed_scenario == RumorCategory.POLICY
    assert len(state.claims) == 1
    assert state.evidence_ranking.sufficiency == "sufficient"
    assert state.skeptic.passed is True

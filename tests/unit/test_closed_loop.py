"""闭环动作层 + ClaimReview JSON-LD 测试。"""

from src.truthnote.claimreview import claim_to_claimreview, response_to_claimreviews
from src.truthnote.closed_loop import (
    ActionStatus,
    ActionType,
    DispositionReceipt,
    RiskType,
    _build_subscription,
    _generate_correction_card,
    _generate_subscribe_card,
    _infer_risk_type,
    _recommend_action,
    build_disposition_receipt,
    generate_actions,
)
from src.truthnote.schemas import (
    Claim,
    ClaimVerification,
    Evidence,
    Verdict,
    VerifyResponse,
)


def _make_cv(text: str, verdict: Verdict, reasoning: str = "") -> ClaimVerification:
    return ClaimVerification(
        claim=Claim(text=text),
        verdict=verdict,
        confidence=0.85,
        evidence_chain=[
            Evidence(
                source="piyao.org.cn",
                url="https://www.piyao.org.cn/article/123",
                snippet="辟谣内容",
            )
        ],
        reasoning=reasoning,
    )


class TestInferRiskType:
    def test_scam(self):
        cv = _make_cv("test", Verdict.FALSE, "[规则判定·通用诈骗] ETC诈骗")
        assert _infer_risk_type(cv) == RiskType.SCAM

    def test_health(self):
        cv = _make_cv("test", Verdict.FALSE, "[规则判定·伪医疗] 养生偏方")
        assert _infer_risk_type(cv) == RiskType.HEALTH_MISINFORMATION

    def test_fake_policy(self):
        cv = _make_cv("test", Verdict.FALSE, "[规则判定·伪造官方公告] 政策不存在")
        assert _infer_risk_type(cv) == RiskType.FAKE_POLICY

    def test_general(self):
        cv = _make_cv("test", Verdict.UNVERIFIABLE, "无法判定")
        assert _infer_risk_type(cv) == RiskType.GENERAL


class TestRecommendAction:
    def test_true_no_action(self):
        assert _recommend_action(Verdict.TRUE, RiskType.GENERAL) == ActionType.NO_ACTION

    def test_scam_report(self):
        assert _recommend_action(Verdict.FALSE, RiskType.SCAM) == ActionType.REPORT_SCAM

    def test_unverifiable_subscribe_backfill(self):
        # 命题人校准：UNVERIFIABLE（还查不到定论）不再进"人工复核空表"，
        # 而是给"权威结论出来通知我"的订阅回填动作，把唯一真边界变成体面动作。
        assert (
            _recommend_action(Verdict.UNVERIFIABLE, RiskType.GENERAL)
            == ActionType.SUBSCRIBE_BACKFILL
        )

    def test_false_share_correction(self):
        assert _recommend_action(Verdict.FALSE, RiskType.FAKE_POLICY) == ActionType.SHARE_CORRECTION


class TestSubscribeBackfill:
    def test_subscription_built_for_unverifiable(self):
        cv = _make_cv("某地刚刚发生地铁事故", Verdict.UNVERIFIABLE, "时序边界：刚发生暂无权威定论")
        sub = _build_subscription(cv, RiskType.PANIC_CHAIN)
        assert sub["topic"]  # 订阅主题非空
        assert isinstance(sub["watch_sources"], list) and sub["watch_sources"]
        assert "channel" in sub

    def test_subscribe_card_text(self):
        cv = _make_cv("某地刚刚发生地铁事故", Verdict.UNVERIFIABLE, "时序边界：暂无权威定论")
        card = _generate_subscribe_card("某地刚刚发生地铁事故", cv, RiskType.PANIC_CHAIN)
        assert "TruthNote" in card
        assert "通知" in card  # 体面动作：权威结论出来后通知

    def test_generate_actions_attaches_subscription(self):
        response = VerifyResponse(
            original_message="某地刚刚发生地铁事故",
            claims=[_make_cv("某地刚发生地铁事故", Verdict.UNVERIFIABLE, "时序边界")],
            overall_verdict=Verdict.UNVERIFIABLE,
            summary="暂无定论",
            friendly_reply="先别转发",
        )
        actions = generate_actions(response)
        assert actions[0].recommended_action == ActionType.SUBSCRIBE_BACKFILL
        assert actions[0].subscription.get("topic")


class TestDispositionReceipt:
    def test_receipt_has_id_and_status(self):
        r = build_disposition_receipt(
            "act_deadbeef",
            ActionStatus.RESOLVED,
            claim_text="存款超5万要交税",
            recommended_action=ActionType.SHARE_CORRECTION,
        )
        assert isinstance(r, DispositionReceipt)
        assert r.receipt_id.startswith("TN-RC-")
        assert r.action_id == "act_deadbeef"
        assert r.status == ActionStatus.RESOLVED
        assert r.message  # 人话回执非空
        assert r.created_at  # 时间戳非空

    def test_receipt_deterministic_id(self):
        # 同一 (action_id, status) 回执号稳定，demo 可复现
        a = build_disposition_receipt("act_1", ActionStatus.SENT)
        b = build_disposition_receipt("act_1", ActionStatus.SENT)
        assert a.receipt_id == b.receipt_id

    def test_receipt_report_mentions_channel(self):
        r = build_disposition_receipt(
            "act_x",
            ActionStatus.RESOLVED,
            recommended_action=ActionType.REPORT_SCAM,
        )
        assert "举报" in r.message or "受理" in r.message

    def test_receipt_dismissed(self):
        r = build_disposition_receipt("act_y", ActionStatus.DISMISSED)
        assert r.status == ActionStatus.DISMISSED
        assert r.message


class TestCorrectionCard:
    def test_card_has_label(self):
        cv = _make_cv("存款超5万要交税", Verdict.FALSE, "[规则判定] 虚假")
        card = _generate_correction_card("紧急通知！存款超5万要交税！", cv, RiskType.FAKE_POLICY)
        assert "TruthNote" in card
        assert "经核查不实" in card
        assert "政府官网" in card

    def test_card_has_sources(self):
        cv = _make_cv("test", Verdict.FALSE, "test")
        card = _generate_correction_card("原消息", cv, RiskType.GENERAL)
        assert "piyao.org.cn" in card


class TestGenerateActions:
    def test_generates_from_response(self):
        response = VerifyResponse(
            original_message="紧急通知",
            claims=[_make_cv("存款要交税", Verdict.FALSE, "[诈骗] 假政策")],
            overall_verdict=Verdict.FALSE,
            summary="虚假",
            friendly_reply="别信",
        )
        actions = generate_actions(response)
        assert len(actions) == 1
        assert actions[0].verdict == Verdict.FALSE
        assert actions[0].action_id.startswith("act_")


class TestClaimReview:
    def test_single_claim(self):
        cv = _make_cv("存款超5万要交税", Verdict.FALSE, "虚假政策")
        cr = claim_to_claimreview(cv, "紧急通知")
        assert cr["@context"] == "https://schema.org"
        assert cr["@type"] == "ClaimReview"
        assert cr["claimReviewed"] == "存款超5万要交税"
        assert cr["reviewRating"]["ratingValue"] == 1
        assert cr["author"]["name"] == "TruthNote"

    def test_true_rating(self):
        cv = _make_cv("事实声明", Verdict.TRUE, "确认属实")
        cr = claim_to_claimreview(cv)
        assert cr["reviewRating"]["ratingValue"] == 5

    def test_response_to_list(self):
        response = VerifyResponse(
            original_message="test",
            claims=[
                _make_cv("声明1", Verdict.FALSE, "假"),
                _make_cv("声明2", Verdict.TRUE, "真"),
            ],
            overall_verdict=Verdict.FALSE,
            summary="mixed",
            friendly_reply="reply",
        )
        reviews = response_to_claimreviews(response)
        assert len(reviews) == 2
        assert reviews[0]["reviewRating"]["ratingValue"] == 1
        assert reviews[1]["reviewRating"]["ratingValue"] == 5

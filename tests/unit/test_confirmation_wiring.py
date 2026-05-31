"""阶段 2.2 — 证实维度接线测试：证据链 → H/C 推导 + A/B 开关。

信任边界核心验证：只认 authority≥0.70 的源，低权威（内容农场/社媒）被挡在门外。
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from truthnote import dimensions as D  # noqa: E402
from truthnote.schemas import (  # noqa: E402
    Claim,
    ClaimVerification,
    Evidence,
    MessageFrame,
    MessageType,
    SourceType,
    Verdict,
)


def _ev(auth, supports, st=SourceType.OFFICIAL_GOVERNMENT):
    return Evidence(
        source="src",
        snippet="...",
        url="http://x/a",
        supports_claim=supports,
        source_type=st,
        authority_score=auth,
    )


def _verif(evidences):
    return [
        ClaimVerification(
            claim=Claim(text="t"),
            verdict=Verdict.UNVERIFIABLE,
            confidence=0.5,
            evidence_chain=evidences,
        )
    ]


# ── 信任边界：低权威源被挡在门外 ──
def test_low_authority_support_gives_no_credit():
    """内容农场/问答站(authority 0.4)支持 → 不参与证实 → C=0（防文本声称/低质源）。"""
    h, c, primary = D.confirmation_and_debunk_from_verifications(_verif([_ev(0.40, True)]))
    assert c == 0.0
    assert primary is False
    assert h == 0.0


def test_high_authority_support_gives_credit():
    """gov/监管(authority 0.95)支持 → 证实 + 一手权威源。"""
    h, c, primary = D.confirmation_and_debunk_from_verifications(_verif([_ev(0.95, True)]))
    assert c >= D._CONFIRM_MIN
    assert primary is True
    assert h == 0.0


def test_medium_authority_support_half_credit():
    """权威媒体(0.75)支持 → q_N=0.5，非一手源。"""
    h, c, primary = D.confirmation_and_debunk_from_verifications(
        _verif([_ev(0.75, True, SourceType.ESTABLISHED_MEDIA)])
    )
    assert primary is False
    # q_N=0.5 → C_ext=0.21 ≥ 0.15 → C>0
    assert c > 0.0


def test_high_authority_refutation_sets_hard_debunk():
    """高权威源反对 → 硬证伪 H 高（直接辟谣地板）。"""
    h, c, primary = D.confirmation_and_debunk_from_verifications(_verif([_ev(0.95, False)]))
    assert h >= 0.85
    assert c == 0.0


def test_no_verifications_safe():
    assert D.confirmation_and_debunk_from_verifications(None) == (0.0, 0.0, False)
    assert D.confirmation_and_debunk_from_verifications([]) == (0.0, 0.0, False)


def test_real_path_uses_evidence_relations_not_just_supports_claim():
    """回归防线（审查 CRITICAL）：真实流水线 Evidence.supports_claim=None，
    方向靠 StructuredFC 的 evidence_relations 标签。证实必须照样生效。"""
    # 模拟真实证据：高权威但 supports_claim 未设（=None，真实搜索源的默认）
    e = Evidence(
        source="卫健委",
        snippet="...",
        url="http://nhc.gov.cn/a",
        source_type=SourceType.OFFICIAL_GOVERNMENT,
        authority_score=0.95,
    )
    assert e.supports_claim is None  # 真实路径默认
    cv = ClaimVerification(
        claim=Claim(text="t"),
        verdict=Verdict.UNVERIFIABLE,
        confidence=0.5,
        evidence_chain=[e],
        evidence_relations=[{"index": 0, "relation": "直接支持"}],  # StructuredFC 标签
    )
    h, c, primary = D.confirmation_and_debunk_from_verifications([cv])
    assert c >= D._CONFIRM_MIN  # 靠标签生效，不靠 supports_claim
    assert primary is True


def test_real_path_direct_debunk_via_relations():
    """高权威源标签『直接辟谣』→ 硬证伪 H（supports_claim 未设也生效）。"""
    e = Evidence(
        source="辟谣平台",
        snippet="...",
        url="http://piyao.org.cn/a",
        source_type=SourceType.FACT_CHECK_ORG,
        authority_score=0.90,
    )
    cv = ClaimVerification(
        claim=Claim(text="t"),
        verdict=Verdict.FALSE,
        confidence=0.8,
        evidence_chain=[e],
        evidence_relations=[{"index": 0, "relation": "直接辟谣"}],
    )
    h, c, primary = D.confirmation_and_debunk_from_verifications([cv])
    assert h >= 0.85


# ── A/B 开关 ──
def test_ab_toggle_prior_table():
    """baseline 模式用旧手写表（financial_scam=0.90），new 用重标表（更低）。"""
    frame = MessageFrame(message_type=MessageType.FINANCIAL_SCAM)
    old_mode = D.SCORER_MODE
    try:
        D.SCORER_MODE = "baseline"
        assert D._dim_prior(frame).score == 0.90  # 旧手写
        D.SCORER_MODE = "new"
        assert D._dim_prior(frame).score < 0.50  # 重标后明显更低
    finally:
        D.SCORER_MODE = old_mode


def test_ab_toggle_confirmation_applied_only_in_new():
    """new 模式下高权威支持把分往真拉；baseline 模式不应用证实。"""
    frame = MessageFrame(message_type=MessageType.FACT_ASSERTION)
    dims = D.assess_dimensions(frame=frame, promo=None, verifications=None)
    verifs = _verif([_ev(0.95, True)])
    old_mode = D.SCORER_MODE
    try:
        D.SCORER_MODE = "new"
        dist_new = D.aggregate_distribution(dims, verifications=verifs)
        D.SCORER_MODE = "baseline"
        dist_base = D.aggregate_distribution(dims, verifications=verifs)
    finally:
        D.SCORER_MODE = old_mode
    # new 模式证实生效 → 更偏"真"端（TRUE+PARTLY_TRUE 概率更高）
    new_true = dist_new.TRUE + dist_new.PARTLY_TRUE
    base_true = dist_base.TRUE + dist_base.PARTLY_TRUE
    assert new_true > base_true

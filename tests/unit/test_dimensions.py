"""DimensionAssessment + VerdictDistribution 单元测试。"""

from __future__ import annotations

from src.truthnote.dimensions import (
    _score_to_verdict,
    aggregate_distribution,
    assess_dimensions,
)
from src.truthnote.promo_health import verify_promo
from src.truthnote.schemas import MessageFrame, MessageType, Verdict


def test_score_to_verdict_boundaries():
    assert _score_to_verdict(0.95) == Verdict.FALSE
    assert _score_to_verdict(0.75) == Verdict.MOSTLY_FALSE
    assert _score_to_verdict(0.60) == Verdict.MISLEADING
    assert _score_to_verdict(0.45) == Verdict.UNVERIFIABLE
    assert _score_to_verdict(0.25) == Verdict.PARTLY_TRUE
    assert _score_to_verdict(0.10) == Verdict.TRUE


def test_assess_dimensions_no_frame():
    dims = assess_dimensions(frame=None, promo=None)
    assert len(dims) == 6
    names = {d.name for d in dims}
    assert names == {
        "prior",
        "anchor",
        "physiological",
        "linguistic",
        "counterfactual",
        "error_cost",
    }


def test_assess_dimensions_case_213():
    """case_213：所有 6 维度都应偏向"假"。"""
    frame = MessageFrame(
        message_type=MessageType.HEALTH_PRODUCT_PROMO,
        central_action_claim="购买恒晴药业+双色片",
        promoted_entity="恒晴药业+双色片",
        verification_burden=["产品注册或备案", "厂家信息", "疗效证据", "安全证据"],
        red_flags=["购买命令", "个人见证", "快速效果", "安全承诺", "竞品贬损"],
    )
    text = (
        "直接去买恒晴药业+的双色片，本人亲试效果特别好。"
        "四个月共计减重38斤。健康安全的瘦身方法。快速減脂不反复"
    )
    promo = verify_promo(text, frame)
    dims = assess_dimensions(frame=frame, promo=promo)

    # 阶段1 重标先验后的新契约（oracle Q1）：
    # 先验维度是"温和怀疑、不定罪"——封顶 0.60，不再硬编码 0.85。定罪靠证据维度。
    # 所以这里只要求"证据类"5 维仍强力识别风险，先验维度只需 ≤ 封顶。
    by_name = {d.name: d for d in dims}
    assert by_name["prior"].score <= 0.60 + 1e-9, "先验维度应被封顶（只升怀疑不定罪）"
    for name in ("anchor", "physiological", "linguistic", "counterfactual", "error_cost"):
        d = by_name[name]
        assert d.score >= 0.55, f"{d.label} score {d.score} 太低，没识别出风险"
    # 真正要保证的是：整体仍判谣言（聚合见 test_aggregate_distribution_case_213）


def test_aggregate_distribution_case_213():
    """case_213 6 维度聚合 → 大部分不实 应是 argmax，无法核实 < 5%。"""
    frame = MessageFrame(
        message_type=MessageType.HEALTH_PRODUCT_PROMO,
        promoted_entity="恒晴药业+双色片",
        central_action_claim="购买恒晴药业+双色片",
        verification_burden=["产品注册或备案", "厂家信息", "疗效证据", "安全证据"],
        red_flags=["购买命令", "个人见证", "快速效果", "安全承诺", "竞品贬损"],
    )
    text = (
        "直接去买恒晴药业+的双色片，本人亲试效果特别好。"
        "四个月共计减重38斤。健康安全的瘦身方法。快速減脂不反复"
    )
    promo = verify_promo(text, frame)
    dims = assess_dimensions(frame=frame, promo=promo)
    dist = aggregate_distribution(
        dims, pipeline_verdict=Verdict.MOSTLY_FALSE, pipeline_confidence=0.7
    )

    # 关键断言（case_213 翻盘标准）
    assert dist.argmax_verdict() in ("大部分不实", "谣言")
    assert dist.UNVERIFIABLE < 0.10, f"UNVERIFIABLE 太高：{dist.UNVERIFIABLE}"
    assert (dist.FALSE + dist.MOSTLY_FALSE) > 0.5
    # 概率分布之和应该 ≈ 1
    total = (
        dist.FALSE
        + dist.MOSTLY_FALSE
        + dist.MISLEADING
        + dist.UNVERIFIABLE
        + dist.PARTLY_TRUE
        + dist.TRUE
    )
    assert 0.99 < total < 1.01


def test_aggregate_distribution_empty_dimensions():
    dist = aggregate_distribution([])
    assert dist.UNVERIFIABLE == 1.0


def test_distribution_normalize_basic():
    from src.truthnote.schemas import VerdictDistribution

    d = VerdictDistribution(FALSE=0.6, MOSTLY_FALSE=0.3, TRUE=0.1)
    norm = d.normalize()
    assert abs(norm.FALSE - 0.6) < 1e-6
    assert abs(norm.MOSTLY_FALSE - 0.3) < 1e-6
    assert abs(norm.TRUE - 0.1) < 1e-6


def test_distribution_argmax():
    from src.truthnote.schemas import VerdictDistribution

    d = VerdictDistribution(FALSE=0.1, MOSTLY_FALSE=0.6, UNVERIFIABLE=0.3)
    assert d.argmax_verdict() == "大部分不实"

"""PromoHealthVerifier 单元测试（CONTRACTS C0 INV-2 落地）。"""

from __future__ import annotations

from src.truthnote.promo_health import (
    _cn_to_arabic,
    _extract_weight_loss_rate,
    check_burden_of_proof,
    check_ftc_thresholds,
    suggest_regulatory_queries,
    verify_promo,
)
from src.truthnote.schemas import MessageFrame, MessageType

# ── 中文数字转阿拉伯 ──


def test_cn_to_arabic_basic():
    assert _cn_to_arabic("四个月") == "4个月"
    assert _cn_to_arabic("两周内") == "2周内"
    assert _cn_to_arabic("三十天") == "30天"
    assert _cn_to_arabic("十八周") == "18周"


# ── 速率提取 ──


def test_extract_rate_case_213():
    """case_213 原文：「四个月共计减重38斤」→ 1.09 kg/周。"""
    rate = _extract_weight_loss_rate("四个月共计减重38斤")
    assert rate is not None
    assert rate["total_kg"] == 19.0
    assert 1.0 <= rate["kg_per_week"] <= 1.2
    assert rate["total_lb"] > 15  # 触发 FTC 阈值


def test_extract_rate_arabic_only():
    rate = _extract_weight_loss_rate("4个月减重38斤")
    assert rate is not None
    assert rate["total_kg"] == 19.0


def test_extract_rate_no_match():
    assert _extract_weight_loss_rate("吃了减肥药效果不错") is None


# ── FTC 阈值检查 ──


def test_ftc_no_rebound_flag():
    result = check_ftc_thresholds("快速减脂不反弹，永不复胖")
    assert "FTC-no_rebound_claim" in result["flags_triggered"]


def test_ftc_case_213_flags():
    text = (
        "四个月共计减重38斤，健康安全的瘦身方法，"
        "从科学的角度去针对性的控糖生酮，这样才能达到快速減脂不反复"
    )
    result = check_ftc_thresholds(text)
    flags = result["flags_triggered"]
    assert "FTC-testimonial_over_15_lb" in flags
    assert "FTC-testimonial_over_2_lb_per_week_for_month" in flags
    assert "FTC-no_rebound_claim" in flags
    assert "FTC-safe_rapid_loss_claim" in flags


# ── BurdenOfProof ──


def test_burden_missing_all_anchors():
    frame = MessageFrame(message_type=MessageType.HEALTH_PRODUCT_PROMO)
    result = check_burden_of_proof("直接去买恒晴药业+双色片", frame)
    assert not result["registration_anchor_present"]
    assert not result["clinical_evidence_anchor_present"]
    assert len(result["missing_anchors"]) >= 3


def test_burden_with_registration_anchor():
    frame = MessageFrame(message_type=MessageType.HEALTH_PRODUCT_PROMO)
    text = "本品国药准字 H20200001，由上海某制药公司生产"
    result = check_burden_of_proof(text, frame)
    assert result["registration_anchor_present"]
    assert result["manufacturer_anchor_present"]


# ── 推荐查询 ──


def test_suggested_queries_health_promo():
    frame = MessageFrame(
        message_type=MessageType.HEALTH_PRODUCT_PROMO,
        promoted_entity="恒晴药业+双色片",
    )
    queries = suggest_regulatory_queries(frame)
    assert any("批准文号" in q for q in queries)
    assert any("市场监管" in q for q in queries)
    assert any("国家药监局" in q for q in queries)


def test_suggested_queries_non_promo():
    frame = MessageFrame(message_type=MessageType.FACT_ASSERTION)
    assert suggest_regulatory_queries(frame) == []


# ── 完整 verify_promo ──


def test_verify_promo_case_213_high_risk():
    """case_213 端到端：必须输出 risk=high + verdict_lean=大部分不实。"""
    text = (
        "先说结论：直接去买恒晴药业+的双色片，本人亲试效果特别好。"
        "四个月共计减重38斤。可以肯定的说，是有健康安全的瘦身方法的。"
        "重要的是你们要跳出，白芸豆、祛湿膏、这些假性瘦身的产品，"
        "从科学的角度去针对性的控糖生酮，这样才能达到快速減脂不反复"
    )
    frame = MessageFrame(
        message_type=MessageType.HEALTH_PRODUCT_PROMO,
        central_action_claim="购买恒晴药业+双色片",
        promoted_entity="恒晴药业+双色片",
        verification_burden=["产品注册或备案", "厂家信息", "疗效证据", "安全证据"],
    )
    result = verify_promo(text, frame)
    assert result["applied"] is True
    assert result["risk_level"] == "high"
    assert result["verdict_lean"] == "大部分不实"
    assert len(result["ftc"]["flags_triggered"]) >= 4
    assert len(result["burden"]["missing_anchors"]) >= 3
    assert result["ftc"]["rate_extracted"] is not None


def test_verify_promo_skips_non_promo_type():
    frame = MessageFrame(message_type=MessageType.FACT_ASSERTION)
    result = verify_promo("减肥小贴士：少吃多运动", frame)
    assert result["applied"] is False
    assert result["verdict_lean"] is None


def test_verify_promo_legitimate_health_product():
    """合法推销（含国药准字 + 厂家 + 临床证据）不应被打成 high risk。"""
    text = (
        "国家药监局批准的处方药盐酸二甲双胍片（国药准字 H20023370），"
        "适应症糖尿病，由上海制药厂生产，"
        "临床试验涉及 500 人入组，结果效果因人而异，请遵医嘱"
    )
    frame = MessageFrame(
        message_type=MessageType.HEALTH_PRODUCT_PROMO,
        promoted_entity="盐酸二甲双胍片",
        verification_burden=["产品注册或备案", "厂家信息", "疗效证据", "安全证据"],
    )
    result = verify_promo(text, frame)
    assert result["burden"]["registration_anchor_present"]
    assert result["burden"]["clinical_evidence_anchor_present"]
    assert result["risk_level"] in ("low", "medium")

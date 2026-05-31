"""阶段 1.3 失败测试先行 — 两旋钮数据驱动先验（oracle Q1）。

验收：先验 = sigmoid(logit(部署基础率) + λ·Δ类型)，封顶 0.60。
先写本测试（在旧手写表 0.85/0.90 上会红）→ 再实现（绿）。
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from truthnote import dimensions as D  # noqa: E402
from truthnote.schemas import MessageType  # noqa: E402


def test_all_priors_capped_at_060():
    """先验封顶 0.60——'先验只能升怀疑，定罪靠证据'。旧表 0.85/0.90 会失败。"""
    for mt in MessageType:
        p = D._TYPE_PRIOR_FALSE[mt]
        assert p <= 0.60 + 1e-9, f"{mt.value} 先验 {p} 超过封顶 0.60"


def test_financial_scam_no_longer_extreme():
    """金融诈骗旧先验 0.90，数据说没那么谣，新先验应明显更低（收紧阈值，审查 LOW）。"""
    assert D._TYPE_PRIOR_FALSE[MessageType.FINANCIAL_SCAM] < 0.45


def test_lift_table_matches_computed_json():
    """契约：dimensions.py 内嵌的 Δ 表必须与 compute_type_priors.py 产出的 JSON 一致，
    防硬编码 Δ 和数据脚本静默漂移（审查 HIGH）。"""
    import json

    p = Path(__file__).resolve().parents[2] / "data/eval/type_log_odds_lift.json"
    if not p.exists():
        import pytest

        pytest.skip("type_log_odds_lift.json 未生成，跑 scripts/compute_type_priors.py")
    with open(p, encoding="utf-8") as fh:
        j = json.load(fh)["log_odds_lift"]
    for mt, delta in D._TYPE_LOG_ODDS_LIFT.items():
        if mt.value in j:
            assert abs(j[mt.value] - delta) < 0.005, (
                f"{mt.value} Δ 漂移：json={j[mt.value]} 代码={delta}"
            )


def test_two_knob_formula_delta_zero_equals_base_rate():
    """Δ=0 时先验 = 部署基础率（两旋钮解耦的核心）。"""
    p = D.prior_false_for_type(0.0, base_rate=0.10, strength=0.5)
    assert abs(p - 0.10) < 1e-6
    p2 = D.prior_false_for_type(0.0, base_rate=0.48, strength=0.5)
    assert abs(p2 - 0.48) < 1e-6


def test_base_rate_knob_monotonic():
    """同一 Δ，基础率越高先验越高（旋钮生效）。"""
    delta = 0.5
    low = D.prior_false_for_type(delta, base_rate=0.10, strength=0.5)
    high = D.prior_false_for_type(delta, base_rate=0.48, strength=0.5)
    assert high > low


def test_positive_delta_raises_above_base_rate():
    """正 Δ（偏谣类型）先验高于基础率，负 Δ 低于基础率。"""
    base = 0.30
    hi = D.prior_false_for_type(1.0, base_rate=base, strength=0.5)
    lo = D.prior_false_for_type(-1.0, base_rate=base, strength=0.5)
    assert hi > base > lo


def test_lift_table_financial_scam_negative():
    """实测 financial_scam 在 CANDY 里没那么谣 → Δ 应为负（低于平均）。"""
    assert D._TYPE_LOG_ODDS_LIFT[MessageType.FINANCIAL_SCAM] < 0

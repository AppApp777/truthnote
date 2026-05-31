"""阶段 2 抗后门对抗测试（失败测试先行）——证实维度的核心安全不变量。

设计卡 docs/dimension7_confirmation_spec.md §7。假设有人想往谣言里塞假证实信号骗高分。
这些是纯数学函数，瞬间跑、不调 LLM、不耗博查配额。
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from truthnote import dimensions as D  # noqa: E402

# 证实信号 q 值（0=未验证/缺失，0.5=部分核对，1=完全核对）
N = "NAMED_VERIFIABLE_SOURCE"
F = "FALSIFIABLE_SPECIFIC"
DT = "CITES_CHECKABLE_DATE"
Q = "QUANTIFIED_CONDITION"
H = "HEDGED_SCOPE"
L = "DISCLOSES_LIMITATION"
S = "CONSISTENT_WITH_COMMON_SENSE"
A = "NO_CALL_TO_ACTION"


# ── gate(H)：强证伪时证实失效 ──
def test_gate_disables_confirmation_when_strong_debunk():
    assert D.gate(0.05) == 1.0  # 无证伪，证实全效
    assert D.gate(0.45) == 1.0  # 边界
    assert abs(D.gate(0.60) - 0.5) < 1e-9  # 中等证伪，减半
    assert D.gate(0.75) == 0.0  # 强证伪，证实归零
    assert D.gate(0.95) == 0.0


# ── 抗后门核心：纯内部风格信号救不动任何东西 ──
def test_internal_only_signals_give_zero_credit():
    """谣言堆 hedge+limitation+常识+无CTA（全内部），C 必须 < 0.15 → 归零。"""
    q = {H: 1.0, L: 1.0, S: 1.0, A: 1.0}  # 全部内部信号拉满，外部为 0
    c = D.compute_confirmation_credit(q)
    assert c < 0.15
    assert c == 0.0  # 阈值以下直接归零


def test_fake_named_source_string_gives_zero():
    """文本写'据新华社'但未验证(q_N=0) → 无外部证实 → C=0。"""
    q = {N: 0.0, DT: 0.0, Q: 0.0}  # 都只是文本声称，未核对
    assert D.compute_confirmation_credit(q) == 0.0


def test_verified_named_source_gives_real_credit():
    """真验证到独立权威源支持(q_N=1) → 显著证实。断言绑到实际权重，不写死 0.40。"""
    q = {N: 1.0}
    c = D.compute_confirmation_credit(q)
    assert c >= D._CONFIRM_WEIGHTS[N] - 1e-9  # 至少 = q_N 权重


def test_partial_external_unlocks_internal_cap_is_intended():
    """记录预期行为（审查 MEDIUM）：q_N=0.5 部分外部信号把内部封顶从 0.08 放宽到 0.20。

    C ≈ 0.41，是 oracle 公式的有意设计——但前提是 q_N=0.5 来自验证器真找到的部分支持源，
    不是文本声称。验证器把关（2.2 接线职责）才是抗后门的最后防线，不是这个数学函数。
    """
    q = {N: 0.5, H: 1.0, L: 1.0, S: 1.0, A: 1.0}
    c = D.compute_confirmation_credit(q)
    assert 0.38 < c < 0.44  # c_ext=0.21 + min(0.28, i_cap=0.20)=0.20 → 0.41
    # 但折扣仍被封顶：H=0.20、非一手源 → 折扣 ≤ 0.30
    f = D.apply_confirmation(0.62, 0.20, c, has_exact_primary_support=False)
    assert 0.62 - f <= 0.30 + 1e-9


def test_unknown_signal_keys_safely_ignored():
    """未知信号 key（拼写错/伪造类型）安全忽略，不崩、不计分（审查 HIGH）。"""
    assert D.compute_confirmation_credit({"FAKE_SIGNAL": 1.0, "xyz": 9.9}) == 0.0


def test_out_of_range_q_clamped():
    """q 越界（负值/>1）被 clamp，不产生超额信用（审查 HIGH）。"""
    over = D.compute_confirmation_credit({N: 5.0})  # clamp 到 1.0
    assert over == D.compute_confirmation_credit({N: 1.0})
    neg = D.compute_confirmation_credit({N: -3.0})  # clamp 到 0
    assert neg == 0.0


# ── F = max(H, B − gated·capped·discount) ──
def test_fake_confirmation_no_discount_on_rumor():
    """谣言注入假信号但 C=0 → 无折扣 → F == max(B,H)（抗后门不变量）。"""
    B, Hd, C = 0.62, 0.20, 0.0
    f = D.apply_confirmation(B, Hd, C, has_exact_primary_support=False)
    assert abs(f - max(B, Hd)) < 1e-9


def test_direct_debunk_blocks_confirmation():
    """直接辟谣 H≥0.75 → gate=0 → 再多证实也 F==max(B,H)。"""
    B, Hd, C = 0.58, 0.95, 0.74  # 即使 C 很高
    f = D.apply_confirmation(B, Hd, C, has_exact_primary_support=True)
    assert abs(f - 0.95) < 1e-9


def test_verified_confirmation_rescues_true_message():
    """真消息 + 真验证证实 → F < B，且降幅 ≤ 0.40（封顶）。"""
    B, Hd, C = 0.62, 0.05, 0.74
    f = D.apply_confirmation(B, Hd, C, has_exact_primary_support=True)
    assert f < B
    assert B - f <= 0.40 + 1e-9


def test_confirmation_discount_capped_without_primary():
    """无一手权威源时降幅封顶 0.30。"""
    B, Hd, C = 0.90, 0.10, 1.0  # C 拉满
    f = D.apply_confirmation(B, Hd, C, has_exact_primary_support=False)
    assert B - f <= 0.30 + 1e-9


def test_confirmation_never_below_floor():
    """证实绝不能把分压到证伪地板 H 以下。"""
    B, Hd, C = 0.50, 0.45, 1.0
    f = D.apply_confirmation(B, Hd, C, has_exact_primary_support=True)
    assert f >= Hd - 1e-9

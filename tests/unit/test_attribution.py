"""非裁定道 + 二元投影 单元测试（判定重构 v2）。

主线落地：把本文件 cp 到 tests/unit/test_attribution.py。
先跑红（attribution / schemas 新代码未落时全红）→ 实现 → 跑绿。

假 LLM：注入 str->str 的 lambda 即可（真实接口 llm.chat_text(prompt)->str）。
"""

import pytest

from src.truthnote.attribution import (
    build_non_adjudicated,
    classify_blocker,
    extract_claim_features,
    fallback_detail,
    is_checkable_announcement,
    validate_attribution_detail,
)
from src.truthnote.schemas import (
    BinaryBadge,
    BlockerType,
    DisplayBucket,
    Verdict,
    project_to_binary,
)


# ── 5.1 决策树各类型 ──
@pytest.mark.parametrize(
    "text,expected",
    [
        ("同事说公司下月裁员30%", BlockerType.NON_PUBLIC_EVIDENCE),
        ("某银行下周要倒闭赶紧取钱", BlockerType.MISSING_KEY_CONTEXT),  # 缺主体优先于未来
        ("明星私聊截图承认吸毒", BlockerType.SOURCE_ARTIFACT_AUTH),
        ("这个楼盘风水旺，住进去财运会变好", BlockerType.NO_CHECKABLE_FACT),
        ("点这个二维码领取医保补贴，今天最后一天", BlockerType.MISSING_KEY_CONTEXT),  # 缺链接目标
    ],
)
def test_blocker_per_type(text, expected):
    blocker, _ = classify_blocker(text, [])
    assert blocker == expected


# ── 5.2 杀手测试（Oracle 给的回潮陷阱）──
def test_killer_hyperlocal_elevator():
    """电梯困人：已发生/有主体/有时间/非截图/无冲突/公开搜不到 → 绝不能判谣言，归私域超本地。"""
    text = "昨晚9点广州天河华景新城A栋电梯困人20分钟，消防解救，物业说不让外传"
    blocker, _ = classify_blocker(text, [])
    assert blocker == BlockerType.NON_PUBLIC_EVIDENCE  # 不是 RUMOR


def test_killer_airport_live_status():
    """机场 live 运营态：五类全不沾，兜底归私域超本地，不判谣言。"""
    text = "现在浦东机场T2安检排队超过4小时，东航柜台全停了"
    blocker, _ = classify_blocker(text, [])
    assert blocker in (BlockerType.NON_PUBLIC_EVIDENCE, BlockerType.NOT_YET_SETTLED)


def test_killer_official_notice_future_is_checkable():
    """教委通知/停水通知：声称官方通知+点名 → 可查，不该停在非裁定。"""
    assert is_checkable_announcement("明天朝阳区教委通知全区幼儿园停课，号文〔2026〕18号") is True
    assert is_checkable_announcement("明天海淀部分小区停水，供水公司已发布通知") is True


def test_pure_future_prediction_not_checkable():
    assert is_checkable_announcement("明年房价一定翻倍") is False


def test_checkable_official_notice_not_self_contradictory_blocker():
    """HIGH#1：可查官方通知类（点名机构+号文）停在非裁定时，blocker 必须是
    NO_PUBLIC_EVIDENCE_FOUND（可查但证据不足），绝不能是 NON_PUBLIC_EVIDENCE
    （其文案『本就不会出现在公开渠道』对官方公告自相矛盾，INV-U4 判定层失效）。"""
    blocker, _ = classify_blocker("明天朝阳区教委通知全区幼儿园停课，号文〔2026〕18号", [])
    assert blocker == BlockerType.NO_PUBLIC_EVIDENCE_FOUND
    assert blocker != BlockerType.NON_PUBLIC_EVIDENCE


# ── 5.3 MECE：多命中 tie-break 确定 + 零命中兜底 ──
def test_multi_hit_deterministic_tiebreak():
    """银行+截图+未来+缺主体全中 → 决策树确定取『缺关键语境』(优先级2)，且 secondary 记录其余。"""
    text = "朋友转的微信群截图：某银行下周要暴雷，赶紧取钱"
    blocker, flags = classify_blocker(text, [])
    assert blocker == BlockerType.MISSING_KEY_CONTEXT
    assert "unsourced_screenshot" in flags
    assert "private_channel" in flags
    assert "future_time_reference" in flags


def test_zero_signal_falls_back_to_non_public_not_rumor():
    text = "昨天下午华景新城B区水管爆了维修两小时"
    blocker, _ = classify_blocker(text, [])
    assert blocker == BlockerType.NON_PUBLIC_EVIDENCE  # 兜底绝不判谣言


# ── 5.4 二元投影 ──
@pytest.mark.parametrize(
    "verdict,badge,bucket",
    [
        (Verdict.TRUE, BinaryBadge.REAL, DisplayBucket.REAL),
        (Verdict.FALSE, BinaryBadge.RUMOR, DisplayBucket.RUMOR),
        (Verdict.MOSTLY_FALSE, BinaryBadge.RUMOR, DisplayBucket.RUMOR),
        (Verdict.MISLEADING, BinaryBadge.RUMOR, DisplayBucket.MIXED),
        (Verdict.PARTLY_TRUE, BinaryBadge.RUMOR, DisplayBucket.MIXED),
    ],
)
def test_projection_adjudicated(verdict, badge, bucket):
    proj = project_to_binary(verdict)
    assert proj is not None
    assert proj[0] == badge
    assert proj[1] == bucket
    assert proj[2]  # subtype 非空


def test_projection_unverifiable_is_none():
    """INV-U1：UNVERIFIABLE 不投影为徽章。"""
    assert project_to_binary(Verdict.UNVERIFIABLE) is None


# ── 5.5 校验层（INV-U3 防回潮 + 防编理由）──
@pytest.mark.parametrize(
    "bad",
    [
        "网上查不到这条消息",  # 缺席话术
        "没有官方公告支持",  # 缺席话术
        "经核查该说法属实",  # 真伪断言
    ],
)
def test_validate_rejects_absence_and_verdict(bad):
    feats = extract_claim_features("某消息", [])
    assert validate_attribution_detail(bad, BlockerType.NON_PUBLIC_EVIDENCE, feats) is False


def test_validate_rejects_false_missing_claim():
    """谎称缺时间但其实有时间 → 拒绝（防编理由）。"""
    feats = extract_claim_features("昨晚9点电梯困人", [])
    assert feats["has_date_time"] is True
    assert (
        validate_attribution_detail(
            "这条消息没有时间无法核查", BlockerType.MISSING_KEY_CONTEXT, feats
        )
        is False
    )


def test_validate_accepts_good_detail():
    feats = extract_claim_features("同事说下月裁员", [])
    good = "这是公司内部人事消息，按其性质本就不会出现在公开渠道"
    assert validate_attribution_detail(good, BlockerType.NON_PUBLIC_EVIDENCE, feats) is True


# ── 5.6 INV-U2：LLM 失败也不出空壳 ──
def test_fallback_never_bare_unverifiable():
    feats = extract_claim_features("某银行要倒闭", [])
    d = fallback_detail(BlockerType.MISSING_KEY_CONTEXT, feats)
    assert d
    assert "无法核实" not in d
    assert "具体主体" in d


def test_build_non_adjudicated_llm_fail_uses_fallback():
    """LLM 返回缺席话术 → 校验拒 → 回退确定性兜底，仍带具体 blocker + 非空 detail。"""
    na = build_non_adjudicated("同事说下月裁员30%", [], llm_fn=lambda p: "网上查不到所以可疑")
    assert na.primary_blocker == BlockerType.NON_PUBLIC_EVIDENCE
    assert na.claim_specific_detail
    assert "无法核实" not in na.claim_specific_detail
    assert "查不到" not in na.claim_specific_detail  # 缺席话术被挡


def test_output_has_no_bare_unverifiable_label():
    """整条非裁定输出不出现『无法核实』四字（防回潮硬断言）。"""
    na = build_non_adjudicated("明天小区停水三天", [], llm_fn=lambda p: "")
    blob = f"{na.primary_blocker.value}{na.claim_specific_detail}{na.verify_where}"
    assert "无法核实" not in blob

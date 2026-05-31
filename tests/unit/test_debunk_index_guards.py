# tests/unit/test_debunk_index_guards.py
"""同命题守卫单元测试 —— 不碰真实爬虫库，直接喂合成 DebunkCandidate 测守卫逻辑。

这些是「先写失败测试」的桩：先确认 verify_same_claim 能拒掉 4 类陷阱
（异地同模板 / 状态翻转 / 否定翻转 / 过期日期 / 数字不同），再让 debunk_index.py 实现通过。
"""

from truthnote.debunk_index import (
    DebunkCandidate,
    confirmed_candidate_to_evidence,
    verify_same_claim,
)
from truthnote.schemas import SourceType


def _cand(text: str, *, score: float = 0.93) -> DebunkCandidate:
    return DebunkCandidate(
        item_id="test_001",
        claim_text=text,
        verdict="谣言",
        source="中国互联网联合辟谣平台",
        url="https://piyao.example/test",
        title=text[:80],
        snippet=f"官方辟谣库收录：{text}",
        category="综合",
        published_date="2026-01-01",
        lexical_score=score,
        bm25_score=score,
        ngram_score=score,
        token_score=score,
        entity_score=1.0,
    )


def test_same_claim_positive_allows_evidence_conversion():
    result = verify_same_claim(
        "北京发放1万元补贴",
        _cand("北京发放1万元补贴"),
    )

    assert result.label == "same_claim"
    assert result.score >= 0.72

    ev = confirmed_candidate_to_evidence(_cand("北京发放1万元补贴"), result)

    assert ev.source == "中国互联网联合辟谣平台"
    assert ev.url == "https://piyao.example/test"
    assert ev.source_type == SourceType.FACT_CHECK_ORG
    assert ev.authority_score == 0.90
    assert ev.credibility == "S-辟谣库命中"
    assert ev.supports_claim is False


def test_rejects_same_template_different_city_beijing_vs_shanghai():
    result = verify_same_claim(
        "北京发放1万元补贴",
        _cand("上海发放1万元补贴"),
    )

    assert result.label == "same_topic_different_claim"
    assert "main_entity_conflict" in result.failed_guards
    assert any("地域实体冲突" in r for r in result.reasons)


def test_rejects_implemented_vs_consultation_status_flip():
    result = verify_same_claim(
        "北京垃圾分类新规已经实施",
        _cand("北京垃圾分类新规正在征求意见"),
    )

    assert result.label == "same_topic_different_claim"
    assert "status_or_modality_conflict" in result.failed_guards
    assert any("状态/时态冲突" in r for r in result.reasons)


def test_rejects_negation_flip_will_not_cause_vs_will_cause():
    result = verify_same_claim(
        "疫苗不会导致不孕",
        _cand("疫苗会导致不孕"),
    )

    assert result.label == "opposite_claim"
    assert "negation_polarity_conflict" in result.failed_guards
    assert any("否定极性冲突" in r for r in result.reasons)


def test_rejects_stale_date_debunk_2020_vs_2026():
    result = verify_same_claim(
        "2026年某地新增房产税试点",
        _cand("2020年某地新增房产税试点"),
    )

    assert result.label == "same_topic_different_claim"
    assert "number_or_date_conflict" in result.failed_guards
    assert any("数字/日期实体冲突" in r for r in result.reasons)


def test_rejects_same_template_different_number():
    result = verify_same_claim(
        "北京发放1万元补贴",
        _cand("北京发放2万元补贴"),
    )

    assert result.label == "same_topic_different_claim"
    assert "number_or_date_conflict" in result.failed_guards
    assert any("数字/日期实体冲突" in r for r in result.reasons)


def test_rejects_modal_negation_flip_keyi():
    # 真实库实测发现：模态谓词「可以」的否定翻转曾漏过守卫（误把相反命题判 same_claim）。
    result = verify_same_claim(
        "吃苦瓜不可以降血糖",
        _cand("吃苦瓜可以降血糖"),
    )

    assert result.label == "opposite_claim"
    assert "negation_polarity_conflict" in result.failed_guards


def test_rejects_modal_negation_flip_neng():
    result = verify_same_claim(
        "茶水醋与盐水不能杀灭新冠病毒",
        _cand("茶水醋与盐水能杀灭新冠病毒"),
    )

    assert result.label == "opposite_claim"
    assert "negation_polarity_conflict" in result.failed_guards


def test_debunk_verdict_framed_as_debunk_signal():
    """「谣言」家族裁决 → 辟谣证据：S-辟谣库命中 + 文本触发下游辟谣信号。"""
    from truthnote.orchestrator import _has_debunk_signal

    cand = _cand("某地发放万元补贴")  # _cand 默认 verdict="谣言"
    ev = confirmed_candidate_to_evidence(cand)
    assert ev.credibility == "S-辟谣库命中"
    assert ev.supports_claim is False
    assert _has_debunk_signal(f"{ev.title} {ev.snippet}") is True


def test_unverifiable_verdict_not_framed_as_debunk_signal():
    """adversarial-review HIGH 修：「无法核实」非证伪定性，被同命题命中也不得伪装成辟谣，
    否则会把「官方都查不到」的消息错误推向「假」。中性入链、不触发下游辟谣信号。"""
    from truthnote.orchestrator import _has_debunk_signal

    cand = DebunkCandidate(
        item_id="u1",
        claim_text="吃黄桃罐头能缓解新冠症状",
        verdict="无法核实",
        source="中国互联网联合辟谣平台",
        url="https://piyao.example/u1",
        title="吃黄桃罐头能缓解新冠症状",
        snippet="官方核查库收录",
        category="健康",
        published_date="2026-01-01",
        lexical_score=0.9,
        ngram_score=0.9,
        entity_score=1.0,
    )
    ev = confirmed_candidate_to_evidence(cand)
    assert ev.credibility == "S-官方核查库", "非证伪定性不得标成辟谣库命中"
    assert _has_debunk_signal(f"{ev.title} {ev.snippet}") is False, (
        "无法核实证据不得触发下游辟谣信号"
    )


def test_opposite_or_related_candidates_are_not_convertible_to_evidence():
    cand = _cand("疫苗会导致不孕")
    result = verify_same_claim("疫苗不会导致不孕", cand)

    assert result.label == "opposite_claim"

    try:
        confirmed_candidate_to_evidence(cand, result)
    except ValueError as exc:
        assert "requires SameClaimResult.label == 'same_claim'" in str(exc)
    else:
        raise AssertionError("opposite_claim must not be convertible to Evidence")


# ── §2 已知限制（辟谣结论式标题）+ b2 反向错防护 ────────────────────────────────
# 背景：副线施工卡 §2 担心辟谣结论式标题（"X并没有Y"）会被极性守卫误杀。实测：
# 爬虫库 11466 条里 6.2% 是否定式，其中辟谣结论式（真会误杀的）约占全库 1%。
# 决策（2026-05-30，用户拍板）：**不修**。理由——放行"官方源 opposite_claim 当辟谣"
# 会撞 INV-4（守卫偏精确、漏判=安全），且制造下面 b2 那种反向错（把真话推向假）。
# 这两条测试把"安全方向"钉死：宁可漏（落回正常流程）也不误收。


def test_conclusion_form_official_title_dropped_is_known_safe_limitation():
    """§2 已知限制（实测特征化）：部分辟谣结论式标题会被漏掉，落回正常流程。

    实测真相比施工卡假设更细：极性守卫识别「是/不是、能/不能」等谓词翻转，但**对
    「并没有/并非」这类否定不敏感**。后果分两种，都不致命：
      - 「吃苦瓜能降血糖」vs「吃苦瓜并没有降血糖的作用」→ 未达同命题阈值 → 漏掉
        （损召回，但 INV-4「漏判=安全」——落回正常取证，绝不反向错）。本测试锁这条。
      - 「无菌蛋完全没菌」vs「无菌蛋并非完全没菌」→ 反而 same_claim 被采信，且方向正确
        （用户claim本身就是谣言，被正确辟掉）——守卫对「并非」不敏感这里恰好帮了召回。
    结论：守卫这点「不敏感」利大于弊，**不修**（修激进了反而把上面第二种正确采信也丢掉）。
    """
    # 用户给肯定式谣言，库里存的是「并没有」式辟谣结论 → 被漏掉（非 same_claim）
    result = verify_same_claim(
        "吃苦瓜能降血糖",
        _cand("吃苦瓜并没有降血糖的作用"),
    )

    assert result.label != "same_claim", "该辟谣结论式标题当前被漏掉（已知召回限制）"
    assert result.label == "same_topic_different_claim", "实测落在同主题档（未达同命题阈值）"


def test_b2_truth_aligned_claim_not_polluted_by_negative_rumor():
    """b2 反向错防护（这就是不放行 opposite_claim 的根本原因）：

    库里存在「否定式谣言」如「疫苗不能预防重症」(verdict=谣言)。若用户说的是与之
    相反的**真话**「疫苗能预防重症」，极性相反 → opposite_claim → 必须丢弃。
    若误收，会把「疫苗不能预防重症是谣言」当 supports_claim=False 的辟谣证据喂下游，
    等于用一条谣言的辟谣去**反向打击用户的真话**，把真推向假——事实核查工具最致命的错。
    """
    negative_rumor = _cand("疫苗不能预防重症")  # 否定式谣言，verdict=谣言
    result = verify_same_claim("疫苗能预防重症", negative_rumor)

    assert result.label == "opposite_claim"
    try:
        confirmed_candidate_to_evidence(negative_rumor, result)
    except ValueError:
        pass
    else:
        raise AssertionError("否定式谣言绝不能被误收成对真话的辟谣证据（b2 反向错）")


# ── 可视化采信理由：结构化 verify 标签（施工卡 §5）──────────────────────────────


def test_confirmed_evidence_carries_structured_match_label():
    """采信证据必须带结构化的同命题核对标签，供可视化渲染「为什么采信」的干净 chip，
    而不是让前端从 snippet 散文里抠。施工卡 §5「新增 verify 标签字段给可视化」。"""
    result = verify_same_claim("北京发放1万元补贴", _cand("北京发放1万元补贴"))
    assert result.label == "same_claim"

    ev = confirmed_candidate_to_evidence(_cand("北京发放1万元补贴"), result)
    assert ev.match_label == "same_claim", "采信证据应携带结构化同命题标签"
    assert ev.match_score >= 0.72, "采信证据应携带同命题核对分数（可视化展示相关度）"


def test_non_debunk_evidence_has_empty_match_label():
    """普通证据（非官方辟谣库命中）match_label 默认空——该字段是辟谣库采信专属元数据，
    不污染其它来源证据。"""
    from truthnote.schemas import Evidence

    ev = Evidence(source="人民网", snippet="一般证据")
    assert ev.match_label == ""
    assert ev.match_score is None

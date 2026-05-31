from src.truthnote.orchestrator import (
    Orchestrator,
    _apply_skeptic_invariants,
    _evidence_matches_claim_strict,
    _extract_claim_key_terms,
    _pick_overall_verdict,
    classify_unverifiable_type,
    detect_financial_scam,
    detect_obsolete_policy,
    detect_personal_content,
    detect_stale_evidence,
    detect_unverified_official_claim,
    prescore_evidence,
    rule_based_verdict,
)
from src.truthnote.schemas import (
    Claim,
    ClaimVerification,
    Evidence,
    SkepticChallenge,
    Verdict,
)


def test_pick_overall_verdict_worst_wins():
    vs = [
        ClaimVerification(claim=Claim(text="a"), verdict=Verdict.TRUE, confidence=0.9),
        ClaimVerification(claim=Claim(text="b"), verdict=Verdict.FALSE, confidence=0.9),
    ]
    assert _pick_overall_verdict(vs) == Verdict.FALSE


def test_pick_overall_verdict_empty():
    assert _pick_overall_verdict([]) == Verdict.UNVERIFIABLE


def test_pick_overall_verdict_unverifiable_mixed_with_true():
    vs = [
        ClaimVerification(claim=Claim(text="a"), verdict=Verdict.TRUE, confidence=0.9),
        ClaimVerification(claim=Claim(text="b"), verdict=Verdict.UNVERIFIABLE, confidence=0.3),
    ]
    assert _pick_overall_verdict(vs) == Verdict.PARTLY_TRUE


def test_pick_overall_verdict_all_true():
    vs = [
        ClaimVerification(claim=Claim(text="a"), verdict=Verdict.TRUE, confidence=0.9),
        ClaimVerification(claim=Claim(text="b"), verdict=Verdict.TRUE, confidence=0.8),
    ]
    assert _pick_overall_verdict(vs) == Verdict.TRUE


def test_pick_overall_verdict_single_unverifiable():
    vs = [
        ClaimVerification(claim=Claim(text="a"), verdict=Verdict.UNVERIFIABLE, confidence=0.3),
    ]
    assert _pick_overall_verdict(vs) == Verdict.UNVERIFIABLE


def test_obsolete_policy_split_claims():
    """声明提取器拆分关键词后，用完整文本仍能检测过时政策。"""
    claim_only = "朝阳区即日起实施全域静态管理"
    full_text = (
        "官方通报：北京市今天新增3例本土新冠确诊病例，"
        "朝阳区即日起实施全域静态管理。 朝阳区即日起实施全域静态管理"
    )
    assert detect_obsolete_policy(claim_only) is None
    result = detect_obsolete_policy(full_text)
    assert result is not None
    assert result["name"] == "新冠封控措施"


def test_unverified_official_claim_stripped_entity():
    """声明提取器去掉了政府机构名后，用完整文本仍能检测伪造公告。"""
    claim_only = "2026年7月1日起，房贷利率统一降至2%"
    full_text = (
        "重大利好！住建部确认：2026年7月1日起，房贷利率统一降至2%！"
        " 2026年7月1日起，房贷利率统一降至2%"
    )
    assert detect_unverified_official_claim(claim_only, []) is None
    result = detect_unverified_official_claim(full_text, [])
    assert result is not None
    assert result["entity"] == "住建部"


def test_unverified_official_both_number_and_date_false():
    """含精确数字+精确日期但无辟谣证据 → UNVERIFIABLE（搜索缺失≠证伪）。
    有辟谣证据+精确数字+精确日期 → FALSE。"""
    from src.truthnote.orchestrator import rule_based_verdict

    # 无辟谣证据时：搜索缺失不等于证伪 → UNVERIFIABLE
    score_no_debunk = {
        "signal": "neutral",
        "debunk_count": 0,
        "authority_debunk_count": 0,
        "authority_count": 0,
        "debunk_snippets": [],
        "screenshot_claim": None,
        "obsolete_policy": None,
        "unverified_official_claim": {
            "entity": "住建部",
            "action": "确认",
            "reason": "声明称「住建部」确认了相关政策，但搜索结果中无任何 gov.cn 官方来源佐证",
        },
        "_claim_text": "重大利好 住建部确认 2026年7月1日起 房贷利率统一降至2%",
        "financial_scam": None,
        "miracle_cure": None,
        "fake_subsidy_scam": None,
        "general_scam": None,
        "local_panic": None,
        "food_incompatibility": None,
        "ai_celebrity_quote": None,
        "stale_evidence": {"signal": "neutral"},
        "suppress_stale_rule": True,
    }
    result = rule_based_verdict(score_no_debunk)
    assert result is not None
    assert result["verdict"] == Verdict.UNVERIFIABLE

    # 有辟谣证据时 → FALSE
    score_with_debunk = dict(score_no_debunk)
    score_with_debunk["debunk_count"] = 1
    score_with_debunk["signal"] = "weak_debunk"
    result2 = rule_based_verdict(score_with_debunk)
    assert result2 is not None
    assert result2["verdict"] == Verdict.FALSE


def test_stale_evidence_single_old_year():
    """即使只有1条旧年份证据，也应触发旧闻信号。"""
    evidence = [
        Evidence(
            url="https://news.example.com/article",
            title="2019年某化工厂爆炸事故回顾",
            snippet="2019年3月发生的化工厂爆炸事故造成重大伤亡",
            source="example.com",
        )
    ]
    result = detect_stale_evidence("刚刚！某地化工厂发生大爆炸！", evidence)
    assert result["signal"] == "stale_evidence"
    assert result["has_immediacy"] is True
    assert 2019 in result["stale_years"]


def test_stale_evidence_keyword_pattern():
    """证据中含"N年前"等旧闻关键词也应触发。"""
    evidence = [
        Evidence(
            url="https://news.example.com/1",
            title="某超市关店事件回顾",
            snippet="早在多年前该超市就已开始关闭门店，此前报道显示经营困难",
            source="example.com",
        )
    ]
    result = detect_stale_evidence("最新消息：大型超市宣布全面关闭中国门店！", evidence)
    assert result["signal"] == "stale_evidence"


def test_stale_evidence_no_immediacy():
    """没有即时性词汇时不应触发旧闻信号。"""
    evidence = [
        Evidence(
            url="https://news.example.com/1",
            title="2019年某事件",
            snippet="2019年发生的事件",
            source="example.com",
        )
    ]
    result = detect_stale_evidence("某地化工厂曾经发生过爆炸", evidence)
    assert result["signal"] == "neutral"


def test_financial_scam_detection():
    """金融内幕/荐股诈骗应被规则检测到。"""
    result = detect_financial_scam(
        "重大利好！某知名基金经理透露：下周一A股将暴涨500点！赶紧满仓干！"
    )
    assert result is not None
    assert "暴涨" in result["keyword"] or "满仓" in result["keyword"]


def test_financial_scam_no_match():
    """正常金融新闻不应触发。"""
    result = detect_financial_scam("央行今日公布最新利率决议")
    assert result is None


def test_miracle_cure_detection():
    """万能养生/伪医疗声明应被检测到。"""
    from src.truthnote.orchestrator import detect_miracle_cure

    result = detect_miracle_cure(
        "协和退休医生提醒：高血压不是病，每天喝芹菜籽粉加陈醋，21天见效，不用吃药"
    )
    assert result is not None
    assert "伪医疗" in result["reason"] or "不用吃药" in result["keyword"]


def test_miracle_cure_negation_skip():
    """否定语境中的养生关键词不应触发。"""
    from src.truthnote.orchestrator import detect_miracle_cure

    result = detect_miracle_cure("医生提醒：不要相信那些声称能根治癌症的保健品")
    assert result is None


def test_fake_subsidy_detection():
    """假补贴诈骗应被检测到。"""
    from src.truthnote.orchestrator import detect_fake_subsidy_scam

    result = detect_fake_subsidy_scam(
        "点链接领2026年度综合补贴，登录官方页面填写身份证、银行卡、验证码"
    )
    assert result is not None
    assert "补贴" in result["keyword"]


def test_local_panic_detection():
    """本地恐慌链应被检测到。"""
    from src.truthnote.orchestrator import detect_local_panic

    result = detect_local_panic("紧急扩散！白色面包车在小区门口用乙醚迷晕小孩，已经丢了6个")
    assert result is not None


def test_earthquake_prediction_false():
    """精确地震预测应标记为伪地震预测。"""
    from src.truthnote.orchestrator import detect_local_panic

    result = detect_local_panic("今晚20点北京会发生7级大地震，地震局内部通知")
    assert result is not None
    assert result["is_earthquake_prediction"] is True


def test_food_incompatibility_detection():
    """食物相克谣言应被检测到。"""
    from src.truthnote.orchestrator import detect_food_incompatibility

    result = detect_food_incompatibility("桃子和西瓜同食会产生砒霜一样的中毒")
    assert result is not None


def test_food_drug_interaction_skip():
    """药物-食物相互作用不应触发食物相克规则。"""
    from src.truthnote.orchestrator import detect_food_incompatibility

    result = detect_food_incompatibility("服用降压药期间不能同吃西柚，会导致中毒")
    assert result is None


def test_ai_celebrity_quote_detection():
    """AI 名人语录应被检测到。"""
    from src.truthnote.orchestrator import detect_ai_celebrity_quote

    result = detect_ai_celebrity_quote(
        "钟南山院士直播推荐某保健品，说吃了就不用去医院",
        [],
    )
    assert result is not None
    assert "钟南山" in result["celebrity"]


def test_strong_rule_overrides_unverified_official_weak_branch():
    """回归测试：unverified_official 弱分支不应挡住后续强规则。
    场景：消息同时触发 unverified_official（弱 → UNVERIFIABLE）和
    local_panic 地震预测（强 → FALSE），应判 FALSE。
    """
    score = {
        "signal": "none",
        "debunk_count": 0,
        "authority_debunk_count": 0,
        "authority_count": 0,
        "debunk_snippets": [],
        "_claim_text": "国家地震局发布紧急通知今晚8点北京7级地震",
        "unverified_official_claim": {
            "entity": "国家地震局",
            "action": "发布",
            "reason": "声称官方发布但无原文",
        },
        "local_panic": {
            "reason": "精确地震预测无科学依据",
            "is_earthquake_prediction": True,
        },
    }
    result = rule_based_verdict(score)
    assert result is not None
    assert result["verdict"] == Verdict.FALSE
    assert "地震" in result["reasoning"]


def test_strong_rule_miracle_cure_overrides_unverified_official():
    """unverified_official 弱分支不应挡住 miracle_cure 强规则。"""
    score = {
        "signal": "none",
        "debunk_count": 0,
        "authority_debunk_count": 0,
        "authority_count": 0,
        "debunk_snippets": [],
        "_claim_text": "卫健委发布通告称某草药可治愈癌症",
        "unverified_official_claim": {
            "entity": "卫健委",
            "action": "发布",
            "reason": "声称官方发布但无原文",
        },
        "miracle_cure": {
            "reason": "声称草药治愈癌症，典型伪医疗",
        },
    }
    result = rule_based_verdict(score)
    assert result is not None
    assert result["verdict"] == Verdict.FALSE
    assert "伪医疗" in result["reasoning"]


def test_structured_verdict_true_message_protection():
    """真消息不应被判为谣言：有支持证据+无辟谣 → 属实。"""
    from src.truthnote.agents import _structured_verdict

    labels = [
        {"index": 0, "relation": "直接支持"},
        {"index": 1, "relation": "话题相关"},
    ]
    key_facts = [
        {"fact": "150分钟", "status": "有原文"},
        {"fact": "中等强度", "status": "无原文"},
    ]
    prescore = {"signal": "neutral"}
    verdict, conf, reasoning = _structured_verdict(labels, key_facts, prescore)
    assert verdict == Verdict.TRUE


def test_structured_verdict_support_outweighs_debunk():
    """支持多于辟谣时不应判 FALSE。"""
    from src.truthnote.agents import _structured_verdict

    labels = [
        {"index": 0, "relation": "直接支持"},
        {"index": 1, "relation": "直接支持"},
        {"index": 2, "relation": "直接辟谣"},
    ]
    key_facts = [{"fact": "某事实", "status": "有原文"}]
    prescore = {"signal": "neutral"}
    verdict, conf, reasoning = _structured_verdict(labels, key_facts, prescore)
    assert verdict != Verdict.FALSE


def test_structured_verdict_no_debunk_unverified_not_false():
    """无辟谣 + 全部无原文 + 话题相关 → 不应判 FALSE。"""
    from src.truthnote.agents import _structured_verdict

    labels = [
        {"index": 0, "relation": "话题相关"},
        {"index": 1, "relation": "话题相关"},
    ]
    key_facts = [{"fact": "150分钟", "status": "无原文"}]
    prescore = {"signal": "neutral"}
    verdict, conf, reasoning = _structured_verdict(labels, key_facts, prescore)
    assert verdict != Verdict.FALSE


def test_indirect_contradict_single_mostly_false():
    """单条间接矛盾 + 无支持 → MOSTLY_FALSE。"""
    from src.truthnote.agents import _structured_verdict

    labels = [
        {"index": 0, "relation": "间接矛盾"},
        {"index": 1, "relation": "话题相关"},
    ]
    key_facts = [{"fact": "今晚8点地震", "status": "无原文"}]
    prescore = {"signal": "neutral"}
    verdict, conf, reasoning = _structured_verdict(labels, key_facts, prescore)
    assert verdict == Verdict.MOSTLY_FALSE
    assert "矛盾" in reasoning


def test_indirect_contradict_multiple_false():
    """2条间接矛盾 + 无支持 → FALSE。"""
    from src.truthnote.agents import _structured_verdict

    labels = [
        {"index": 0, "relation": "间接矛盾"},
        {"index": 1, "relation": "间接矛盾"},
    ]
    key_facts = [{"fact": "精确预测地震", "status": "无原文"}]
    prescore = {"signal": "neutral"}
    verdict, conf, reasoning = _structured_verdict(labels, key_facts, prescore)
    assert verdict == Verdict.FALSE
    assert "矛盾" in reasoning


def test_indirect_contradict_with_support_conflict():
    """间接矛盾 + 直接支持 → MOSTLY_FALSE（矛盾优先）或 PARTLY_TRUE（支持多）。"""
    from src.truthnote.agents import _structured_verdict

    labels = [
        {"index": 0, "relation": "间接矛盾"},
        {"index": 1, "relation": "直接支持"},
    ]
    key_facts = [{"fact": "WHO建议", "status": "有原文"}]
    prescore = {"signal": "neutral"}
    verdict, conf, reasoning = _structured_verdict(labels, key_facts, prescore)
    assert verdict == Verdict.MOSTLY_FALSE


def test_indirect_contradict_support_majority_partly_true():
    """支持多于间接矛盾 → PARTLY_TRUE。"""
    from src.truthnote.agents import _structured_verdict

    labels = [
        {"index": 0, "relation": "间接矛盾"},
        {"index": 1, "relation": "直接支持"},
        {"index": 2, "relation": "直接支持"},
    ]
    key_facts = [{"fact": "运动量", "status": "有原文"}]
    prescore = {"signal": "neutral"}
    verdict, conf, reasoning = _structured_verdict(labels, key_facts, prescore)
    assert verdict == Verdict.PARTLY_TRUE


def test_orchestrator_full_run(mock_llm_pipeline):
    orch = Orchestrator()
    result = orch.run("紧急通知！存款超5万交税！")

    assert result.overall_verdict == Verdict.FALSE
    assert len(result.claims) == 1
    # 强辟谣规则命中时跳过 FactChecker+Skeptic，步骤更少
    assert len(orch.trace.steps) >= 9
    assert orch.trace.total_llm_calls >= 6


# ── Feature 1: 个人内容预过滤测试 ──


def test_personal_content_pure_rant():
    """纯吐槽/情绪宣泄应被识别为个人内容。"""
    result = detect_personal_content(
        "好自私，别人为什么要谅解你？公共教室没课的时候任何人都可以来自习，"
        "你在那里学习不代表那个位置就是你的"
    )
    assert result is not None
    assert result["type"] == "personal_experience"


def test_personal_content_experience_sharing():
    """个人经历分享应被识别。"""
    result = detect_personal_content("我今天遇到一个特别无语的事情，超市排队有人插队，真的服了")
    assert result is not None
    assert result["type"] == "personal_experience"


def test_personal_content_daily_sharing():
    """日常碎碎念应被识别。"""
    result = detect_personal_content("分享一下今天的日常，早上起来跑了五公里，感觉不错")
    assert result is not None
    assert result["type"] == "personal_experience"


def test_personal_content_opinion_with_verifiable_fact():
    """有情绪标记但也含可核查事实（政策类）→ 不应触发。"""
    result = detect_personal_content("太气人了，存款超5万居然要交税！")
    assert result is None


def test_personal_content_opinion_with_health_claim():
    """有情绪标记但含健康声明 → 不应触发。"""
    result = detect_personal_content("真的服了，我妈又在转那些说芹菜汁能根治高血压的文章")
    assert result is None


def test_personal_content_opinion_with_government_entity():
    """有情绪标记但提及政府机构 → 不应触发。"""
    result = detect_personal_content("太过分了，教育部居然要取消学区房了")
    assert result is None


def test_personal_content_opinion_with_scam_keywords():
    """有情绪标记但含诈骗关键词 → 不应触发。"""
    result = detect_personal_content("太气人了，有人加我微信说可以刷单赚钱")
    assert result is None


def test_personal_content_no_markers():
    """不含任何个人/情绪标记 → 不应触发。"""
    result = detect_personal_content("明天天气不错适合出门")
    assert result is None


def test_personal_content_in_rule_based_verdict():
    """rule_based_verdict 应优先检查个人内容规则。"""
    score = {
        "signal": "neutral",
        "debunk_count": 0,
        "authority_debunk_count": 0,
        "authority_count": 0,
        "debunk_snippets": [],
        "_claim_text": "好自私别人为什么要谅解你",
        "personal_content": {
            "reason": "个人经历/观点分享，不包含可公开核查的事实声明",
            "type": "personal_experience",
        },
        "screenshot_claim": None,
        "obsolete_policy": None,
        "unverified_official_claim": None,
        "financial_scam": None,
        "miracle_cure": None,
        "fake_subsidy_scam": None,
        "general_scam": None,
        "local_panic": None,
        "food_incompatibility": None,
        "ai_celebrity_quote": None,
        "stale_evidence": {"signal": "neutral"},
        "suppress_stale_rule": False,
    }
    result = rule_based_verdict(score)
    assert result is not None
    assert result["verdict"] == Verdict.UNVERIFIABLE
    assert "个人内容" in result["reasoning"]


# ── Feature 2: UNVERIFIABLE 细分测试 ──


def test_classify_developing_specific_institution_and_action():
    """具体机构 + 具体行动 → developing。"""
    result = classify_unverifiable_type("北大学硕取消了", [])
    assert result == "developing"


def test_classify_developing_government_entity():
    """政府机构 + 政策行动 → developing。"""
    result = classify_unverifiable_type("教育部要调整高考政策", [])
    assert result == "developing"


def test_classify_insufficient_vague():
    """模糊引用（听说、某）→ insufficient。"""
    result = classify_unverifiable_type("听说某大学要取消一个专业", [])
    assert result == "insufficient"


def test_classify_insufficient_no_specifics():
    """无任何具体指标 → insufficient。"""
    result = classify_unverifiable_type("据说有个新规定出来了", [])
    assert result == "insufficient"


def test_classify_developing_with_discussion():
    """具体机构 + 多方讨论 → developing。"""
    result = classify_unverifiable_type("清华大学要合并某学院，有人说是真的，也有人说还在讨论", [])
    assert result == "developing"


def test_classify_developing_company():
    """具体企业 + 行动 → developing。"""
    result = classify_unverifiable_type("华为要停止某项业务了", [])
    assert result == "developing"


# ── Feature 3: 搜索相关性过滤测试 ──


def test_extract_claim_key_terms_anchor_and_generic():
    """锚点词（4+ 字）和通用词（2-3 字）正确分类。"""
    anchor, generic = _extract_claim_key_terms("传媒大学失火了")
    assert "传媒大学" in anchor
    # "失火" 是 2 字通用词，不应在锚点中
    assert "失火" not in anchor
    assert any("失火" in g for g in generic)


def test_extract_claim_key_terms_proper_nouns():
    """骨质疏松、维生素 等专有名词应被提取为锚点。"""
    anchor, generic = _extract_claim_key_terms("骨质疏松患者补充维生素D")
    assert "骨质疏松" in anchor
    assert "维生素" in anchor or any("维生素" in a for a in anchor)


def test_strict_match_requires_anchor():
    """有锚点词时，必须匹配至少一个锚点才算相关。"""
    # "传媒大学失火" → anchor=["传媒大学"], generic=["失火"]
    # 不相关辟谣文章只提到"大学"和"失火"但不含"传媒大学"
    assert (
        _evidence_matches_claim_strict(
            "某大学宿舍失火系谣言 官方辟谣",
            anchor_terms=["传媒大学"],
            generic_terms=["失火"],
        )
        is False
    )

    # 相关辟谣文章提到"传媒大学"
    assert (
        _evidence_matches_claim_strict(
            "传媒大学失火系谣言 官方辟谣",
            anchor_terms=["传媒大学"],
            generic_terms=["失火"],
        )
        is True
    )


def test_strict_match_no_anchor_falls_back():
    """无锚点词时退回通用词匹配。"""
    assert (
        _evidence_matches_claim_strict(
            "失火 起火 消防队",
            anchor_terms=[],
            generic_terms=["失火", "起火"],
        )
        is True
    )


def test_prescore_filters_unrelated_debunk():
    """prescore_evidence 应过滤掉不相关的辟谣文章。"""
    # 声明：传媒大学失火
    # 证据：关于"某小学失火"的辟谣 → 不应被计为辟谣信号
    unrelated_evidence = Evidence(
        source="piyao.org.cn",
        url="https://www.piyao.org.cn/example",
        title="某小学失火消息不实 官方辟谣",
        snippet="经核实，网传某小学失火消息与事实不符。",
    )
    result = prescore_evidence(
        [unrelated_evidence],
        claim_text="传媒大学失火了",
    )
    assert result["debunk_count"] == 0
    assert result["signal"] == "neutral"


def test_prescore_keeps_related_debunk():
    """prescore_evidence 应保留相关的辟谣文章。"""
    related_evidence = Evidence(
        source="piyao.org.cn",
        url="https://www.piyao.org.cn/example",
        title="传媒大学失火消息不实 官方辟谣",
        snippet="经核实，网传传媒大学失火消息纯属编造。",
    )
    result = prescore_evidence(
        [related_evidence],
        claim_text="传媒大学失火了",
    )
    assert result["debunk_count"] >= 1


# ── Feature 4: 官方声明规则修复测试 ──


def test_unverified_official_claim_with_trusted_media_support():
    """有权威媒体提及同一机构 → 不应触发假冒官方规则。"""
    xinhua_evidence = Evidence(
        source="新华网",
        url="https://www.xinhuanet.com/policy/2025-01/01/c_123.htm",
        title="国家医保局发布罕见病用药保障方案",
        snippet="国家医保局近日发文，将部分罕见病用药纳入医保目录。",
    )
    result = detect_unverified_official_claim(
        "国家医保局正式发文，2025年起将部分罕见病用药纳入医保",
        [xinhua_evidence],
    )
    # 新华网提到了国家医保局 → 不应被标为假冒
    assert result is None


def test_unverified_official_claim_no_support():
    """无任何支持来源 → 应触发假冒官方规则。"""
    random_evidence = Evidence(
        source="某博客",
        url="https://blog.example.com/post123",
        title="今日财经新闻",
        snippet="市场波动加大。",
    )
    result = detect_unverified_official_claim(
        "教育部发文取消学区房政策，2025年起全面实施",
        [random_evidence],
    )
    assert result is not None
    assert "教育部" in result["entity"]


def test_unverified_official_claim_debunked_by_trusted():
    """权威媒体有辟谣 → 应仍然触发（辟谣 = 说这是假的）。"""
    debunk_evidence = Evidence(
        source="新华网",
        url="https://www.xinhuanet.com/example",
        title="教育部辟谣：从未发布取消学区房文件",
        snippet="针对网传教育部取消学区房的不实消息，教育部表示从未发布相关文件。",
    )
    result = detect_unverified_official_claim(
        "教育部发文取消学区房政策",
        [debunk_evidence],
    )
    # 辟谣证据不算 trusted support（_has_debunk_signal 检测到辟谣关键词）
    # 但规则本身可能返回 None 或非 None 取决于逻辑
    # 关键：不应因为有辟谣文章就认为是真的官方声明
    # 实际上：xinhua + "教育部" in combined + 无辟谣 → 才算 trusted
    # 这里有辟谣信号 → has_trusted_support=False → 规则仍触发
    assert result is not None


# -- INV-3 . Skeptic downgrade guard (C0 / case_213 baseline) --


def _make_false_verification_with_contradictions(
    *, verdict: Verdict = Verdict.FALSE
) -> ClaimVerification:
    return ClaimVerification(
        claim=Claim(text="四个月共计减重38斤"),
        verdict=verdict,
        confidence=0.8,
        evidence_chain=[
            Evidence(source="www.wnxrmyy.com", snippet="快速减重背后往往隐藏着健康风险"),
            Evidence(source="thread.com", snippet="安全减重 0.5-1 kg/周"),
        ],
        reasoning="[结构化质询] 2条证据的事实均与声明矛盾",
        evidence_relations=[
            {"index": 0, "relation": "间接矛盾"},
            {"index": 1, "relation": "间接矛盾"},
        ],
    )


def test_inv3_blocks_generic_doubt_downgrade_case_213():
    verification = _make_false_verification_with_contradictions()
    skeptic = SkepticChallenge(
        challenges=[
            "是否存在部分地区/时间段的例外？",
            "声明是否可能被过度简化？原始说法可能更复杂？",
        ],
        passed=False,
        revised_verdict=Verdict.UNVERIFIABLE,
    )
    fixed, notice = _apply_skeptic_invariants(verification, skeptic)
    assert notice, "应触发 INV-3 拦截"
    assert fixed.revised_verdict is None
    assert fixed.passed is True
    # P1 #5: INV-3 标记走 inv3_blocked 独立字段，不污染 challenges
    assert fixed.inv3_blocked is True
    assert not any("INV-3" in c for c in fixed.challenges)


def test_inv3_blocks_mostly_false_downgrade():
    verification = _make_false_verification_with_contradictions(verdict=Verdict.MOSTLY_FALSE)
    skeptic = SkepticChallenge(
        challenges=["也许有例外", "可能被过度简化"],
        passed=False,
        revised_verdict=Verdict.UNVERIFIABLE,
    )
    fixed, notice = _apply_skeptic_invariants(verification, skeptic)
    assert notice
    assert fixed.revised_verdict is None


def test_inv3_blocks_partly_true_downgrade_illusion_case():
    """Bug A 回归（HANDOFF 2026-05-28）：StructuredFC 判 PARTLY_TRUE 70%，
    Skeptic 用「也许有例外/证据可能过时/可能被过度简化」三连通用怀疑
    降到 UNVERIFIABLE 49%——必须拦住。"""
    verification = _make_false_verification_with_contradictions(verdict=Verdict.PARTLY_TRUE)
    skeptic = SkepticChallenge(
        challenges=[
            "也许有例外情况",
            "可能是过时信息",
            "可能被过度简化",
        ],
        passed=False,
        revised_verdict=Verdict.UNVERIFIABLE,
    )
    fixed, notice = _apply_skeptic_invariants(verification, skeptic)
    assert notice, "PARTLY_TRUE + 通用怀疑应触发 INV-3 拦截"
    assert fixed.revised_verdict is None
    # P1 #5: INV-3 标记走 inv3_blocked 独立字段，不污染 challenges
    assert fixed.inv3_blocked is True
    assert not any("INV-3" in c for c in fixed.challenges)


def test_inv3_partly_true_single_indirect_contradiction_can_downgrade():
    """P1 #4 v2 (adversarial HIGH 修): PARTLY_TRUE + 仅 1 条「间接矛盾」（弱矛盾）
    时允许 Skeptic 降级。「部分属实」判定本身就脆弱，仅弱矛盾下应能合理降级。

    替代 v1 的 confidence < 0.6 判定——避免把 LLM ±0.1 confidence 抖动直接传到
    verdict 破坏 determinism。改用 relation 离散类型判定更稳。
    """
    verification = ClaimVerification(
        claim=Claim(text="某产品成分A可以辅助减肥"),
        verdict=Verdict.PARTLY_TRUE,
        confidence=0.7,  # 高 confidence 也允许，因为靠 relation 判定不靠 confidence
        evidence_relations=[{"index": 0, "relation": "间接矛盾"}],  # 仅 1 条弱矛盾
    )
    skeptic = SkepticChallenge(
        challenges=["也许有例外", "可能被过度简化"],
        passed=False,
        revised_verdict=Verdict.UNVERIFIABLE,
    )
    fixed, notice = _apply_skeptic_invariants(verification, skeptic)
    assert not notice, "薄弱 PARTLY_TRUE（仅 1 条间接矛盾）应允许 Skeptic 降级"
    assert fixed.revised_verdict == Verdict.UNVERIFIABLE


def test_inv3_partly_true_single_direct_refutation_still_blocks():
    """P1 #4 v2: PARTLY_TRUE + 即使仅 1 条「直接辟谣」（强矛盾）也锁死。
    「直接辟谣」是更强的信号，1 条已足够，不应该允许通用怀疑降级。"""
    verification = ClaimVerification(
        claim=Claim(text="某产品成分A可以辅助减肥"),
        verdict=Verdict.PARTLY_TRUE,
        confidence=0.4,  # 低 confidence 也锁死——「直接辟谣」是强信号
        evidence_relations=[{"index": 0, "relation": "直接辟谣"}],
    )
    skeptic = SkepticChallenge(
        challenges=["也许有例外", "可能被过度简化"],
        passed=False,
        revised_verdict=Verdict.UNVERIFIABLE,
    )
    fixed, notice = _apply_skeptic_invariants(verification, skeptic)
    assert notice, "PARTLY_TRUE + 1 条直接辟谣应锁死，不允许通用怀疑降级"


def test_inv3_blocked_field_independent_of_challenges():
    """P1 #5: INV-3 拦截走 inv3_blocked 独立字段，不再追加 notice 到 challenges。

    防回退：HANDOFF 描述「INV-3 notice 字符串拼进 challenges 数组会污染下游
    reasoning 文本」。具体污染路径是 humanize._h_skeptic 把 challenges 前 2 条
    各取 18 字给用户看——notice 字符串会被截成「[INV-3 拦截] 当前已有」
    塞进用户可见的质疑预览。本测试锁死 challenges 不含 INV-3 字样。
    """
    verification = _make_false_verification_with_contradictions(verdict=Verdict.PARTLY_TRUE)
    orig_challenges = ["也许有例外", "可能被过度简化"]
    skeptic = SkepticChallenge(
        challenges=orig_challenges,
        passed=False,
        revised_verdict=Verdict.UNVERIFIABLE,
    )
    fixed, notice = _apply_skeptic_invariants(verification, skeptic)
    # notice 字符串通过第二个返回值传播，供 orchestrator 走 diag 通道
    assert notice and "INV-3" in notice
    # inv3_blocked 独立字段为 True
    assert fixed.inv3_blocked is True
    # challenges 保持原样，没有 notice 字符串
    assert fixed.challenges == orig_challenges
    assert not any("INV-3" in c for c in fixed.challenges)


def test_skeptic_challenge_inv3_blocked_default_false():
    """P1 #5: SkepticChallenge.inv3_blocked 默认 False，旧调用代码不影响。"""
    c = SkepticChallenge()
    assert c.inv3_blocked is False
    c2 = SkepticChallenge(challenges=["x"], passed=True, revised_verdict=None)
    assert c2.inv3_blocked is False


def test_humanize_skeptic_appends_inv3_suffix():
    """P1 #5: humanize._h_skeptic 检测 inv3_blocked 时拼独立提示给用户，
    不再混进 challenges 预览。"""
    from src.truthnote.humanize import _h_skeptic

    data = {
        "challenges": ["也许有例外", "可能被过度简化"],
        "passed": True,
        "revised_verdict": None,
        "inv3_blocked": True,
    }
    line, _ = _h_skeptic(action=None, result=None, output_data=data, output_summary=None)
    assert "INV-3 守护" in line, "inv3_blocked=True 时应拼 INV-3 守护提示"
    # 原 challenges 文本仍在预览里（用户能看到正常质疑）
    assert "也许有例外" in line


def test_inv3_writeback_to_trace_step_exists_in_orchestrator():
    """P1 #5 CRITICAL 修（adversarial subagent 抓出）: orchestrator 在
    _apply_skeptic_invariants 拦截后必须回写最近一个 Skeptic trace step 的
    output_data + 重新 humanize。

    根因：_timed 在 _apply_skeptic_invariants 之前快照 output_data（拿到 raw
    SkepticChallenge inv3_blocked=False），导致 humanize._h_skeptic 永远拿到
    False，P1 #5 的用户可见效果在真实流程中是死代码。

    单元测试用 inspect.getsource 锁死回写逻辑存在——避免未来重构时把这段
    代码删掉而单测仍过（mock 测试无法覆盖此路径）。
    """
    import inspect

    from src.truthnote import orchestrator as orch_mod

    # 在 ClaimOrchestrator 类源码里找 verify_claim / _verify_claim 等关键方法
    src = inspect.getsource(orch_mod)
    # 回写必须做：拦截后找最近 Skeptic step + 改 output_data + 重新 humanize
    assert "inv3_blocked" in src and "trace.steps" in src, (
        "P1 #5 CRITICAL: orchestrator 缺少 INV-3 拦截后的 trace 回写逻辑"
    )
    # 必须用 reversed 找最近 Skeptic step（顺序从最新往前）
    assert "reversed(self.trace.steps)" in src or "self.trace.steps[::-1]" in src, (
        "回写必须用 reversed 顺序找最近 Skeptic step"
    )
    # adversarial v2 HIGH 修：多 claim 并发走 ThreadPoolExecutor 共享 self.trace.steps，
    # 单纯找 agent=="Skeptic" 会串扰到其他 claim 的 step。必须用 action 字符串
    # （含 claim.text[:15]）精确匹配锁定当前 claim 的 step。
    assert "target_action" in src and "claim.text[:15]" in src, (
        "回写必须用 action 精确匹配防多 claim 并发串扰"
    )
    assert "_step.action == target_action" in src, "必须断言 action 一致才回写"


def test_humanize_skeptic_no_inv3_suffix_when_not_blocked():
    """P1 #5: 未拦截时不应混入 INV-3 提示，避免误导用户。"""
    from src.truthnote.humanize import _h_skeptic

    data = {
        "challenges": ["也许有例外"],
        "passed": True,
        "revised_verdict": None,
        "inv3_blocked": False,
    }
    line, _ = _h_skeptic(action=None, result=None, output_data=data, output_summary=None)
    assert "INV-3" not in line, "未拦截时不应混入 INV-3 提示"


def test_inv3_partly_true_two_contradictions_blocks_regardless():
    """P1 #4: PARTLY_TRUE + ≥2 条强矛盾时锁死，不看 confidence。
    多条独立强矛盾本身就是高确定性证据，不需要 confidence 兜底。"""
    verification = ClaimVerification(
        claim=Claim(text="某产品成分A可以辅助减肥"),
        verdict=Verdict.PARTLY_TRUE,
        confidence=0.4,  # 即使 confidence 很低
        evidence_relations=[
            {"index": 0, "relation": "间接矛盾"},
            {"index": 1, "relation": "直接辟谣"},
        ],
    )
    skeptic = SkepticChallenge(
        challenges=["也许有例外", "可能被过度简化"],
        passed=False,
        revised_verdict=Verdict.UNVERIFIABLE,
    )
    fixed, notice = _apply_skeptic_invariants(verification, skeptic)
    assert notice, "PARTLY_TRUE ≥2 强矛盾应锁死不看 confidence"


def test_inv3_allows_specific_evidence_gap_downgrade():
    verification = _make_false_verification_with_contradictions()
    skeptic = SkepticChallenge(
        challenges=[
            "证据来自台湾医师的微博，没有大陆官方卫健委的直接表态，对大陆产品的同主张证据存在缺口"
        ],
        passed=False,
        revised_verdict=Verdict.UNVERIFIABLE,
    )
    fixed, notice = _apply_skeptic_invariants(verification, skeptic)
    assert not notice
    assert fixed.revised_verdict == Verdict.UNVERIFIABLE


def test_inv3_skip_when_no_strong_contradiction():
    verification = ClaimVerification(
        claim=Claim(text="x"),
        verdict=Verdict.FALSE,
        confidence=0.6,
        evidence_relations=[{"index": 0, "relation": "话题相关"}],
    )
    skeptic = SkepticChallenge(
        challenges=["也许有例外"],
        passed=False,
        revised_verdict=Verdict.UNVERIFIABLE,
    )
    fixed, notice = _apply_skeptic_invariants(verification, skeptic)
    assert not notice
    assert fixed.revised_verdict == Verdict.UNVERIFIABLE


def test_inv3_skip_when_verdict_already_unverifiable():
    verification = ClaimVerification(
        claim=Claim(text="x"),
        verdict=Verdict.UNVERIFIABLE,
        confidence=0.5,
        evidence_relations=[{"index": 0, "relation": "间接矛盾"}],
    )
    skeptic = SkepticChallenge(
        challenges=["也许有例外"],
        passed=False,
        revised_verdict=Verdict.UNVERIFIABLE,
    )
    fixed, notice = _apply_skeptic_invariants(verification, skeptic)
    assert not notice


def test_inv3_allows_upgrade_to_true():
    verification = _make_false_verification_with_contradictions()
    skeptic = SkepticChallenge(
        challenges=["有 WHO 权威支持"],
        passed=False,
        revised_verdict=Verdict.TRUE,
    )
    fixed, notice = _apply_skeptic_invariants(verification, skeptic)
    assert not notice
    assert fixed.revised_verdict == Verdict.TRUE


def test_inv3_blocks_empty_challenges():
    verification = _make_false_verification_with_contradictions()
    skeptic = SkepticChallenge(
        challenges=[],
        passed=False,
        revised_verdict=Verdict.UNVERIFIABLE,
    )
    fixed, notice = _apply_skeptic_invariants(verification, skeptic)
    assert notice
    assert fixed.revised_verdict is None


# -- INV-1/INV-2 . MessageFrame + CoverageAuditor --


def _make_health_promo_frame(*, central="购买恒晴药业+双色片可安全减肥", entity="恒晴药业+双色片"):
    from src.truthnote.schemas import MessageFrame, MessageType

    return MessageFrame(
        message_type=MessageType.HEALTH_PRODUCT_PROMO,
        central_action_claim=central,
        promoted_entity=entity,
        verification_burden=["产品注册或备案", "厂家信息", "疗效证据", "安全证据"],
        red_flags=["购买命令", "个人见证"],
    )


def test_audit_coverage_central_action_missing():
    from src.truthnote.orchestrator import _audit_coverage

    frame = _make_health_promo_frame()
    claims = [Claim(text="四个月共计减重38斤")]  # 漏抽 central action
    audit = _audit_coverage(frame, claims, [])
    assert audit["central_action_present"] is False
    assert audit["downgrade_blocked"] is True


def test_audit_coverage_central_action_present():
    from src.truthnote.orchestrator import _audit_coverage

    frame = _make_health_promo_frame()
    claim = Claim(text="购买恒晴药业+双色片可安全减肥")
    claim.is_central_action = True
    audit = _audit_coverage(frame, [claim], [])
    assert audit["central_action_present"] is True


def test_audit_coverage_burden_tracking():
    from src.truthnote.orchestrator import _audit_coverage

    frame = _make_health_promo_frame()
    claim = Claim(text="恒晴药业+双色片是否有产品注册或备案")
    claim.is_central_action = True
    verifications = [
        ClaimVerification(
            claim=claim,
            verdict=Verdict.UNVERIFIABLE,
            confidence=0.5,
            reasoning="未在 NMPA 查到该产品注册或备案，厂家信息缺失",
        ),
    ]
    audit = _audit_coverage(frame, [claim], verifications)
    assert "产品注册或备案" in audit["covered"]


def test_scenario_router_build_frame_health_promo():
    from src.truthnote.agents import ScenarioRouterAgent
    from src.truthnote.schemas import MessageType

    frame = ScenarioRouterAgent._build_frame(
        scenario="健康养生",
        strategy_hint="核实产品认证",
        confidence=0.95,
        frame_raw={
            "central_action_claim": "购买恒晴药业",
            "promoted_entity": "恒晴药业",
            "red_flags": ["购买命令"],
            "verification_burden": [],
        },
        key_entities=["恒晴药业"],
    )
    assert frame.message_type == MessageType.HEALTH_PRODUCT_PROMO
    # 强制注入最小 burden 清单
    assert "产品注册或备案" in frame.verification_burden
    assert "疗效证据" in frame.verification_burden


def test_scenario_router_build_frame_health_advice_degrade():
    """健康养生 + 无产品名/无中心行动 → 降级 HEALTH_ADVICE。"""
    from src.truthnote.agents import ScenarioRouterAgent
    from src.truthnote.schemas import MessageType

    frame = ScenarioRouterAgent._build_frame(
        scenario="健康养生",
        strategy_hint="",
        confidence=0.8,
        frame_raw={"central_action_claim": "", "promoted_entity": ""},
        key_entities=[],
    )
    assert frame.message_type == MessageType.HEALTH_ADVICE


def test_inv3_does_not_block_legitimate_specific_doubt():
    """HIGH-2 回归：代含“也可能是”但包含具体内容的质疑不应被误拦。"""
    verification = _make_false_verification_with_contradictions()
    skeptic = SkepticChallenge(
        challenges=[
            "证据来自台湾医师的微博，也可能是大陆不同各医院的不同实践。同主张的大陆卫健委表态并未在证据中出现",
        ],
        passed=False,
        revised_verdict=Verdict.UNVERIFIABLE,
    )
    fixed, notice = _apply_skeptic_invariants(verification, skeptic)
    assert not notice, "含具体内容的质疑不应被 INV-3 拦截"
    assert fixed.revised_verdict == Verdict.UNVERIFIABLE


def test_inv1_exemption_central_action_survives_checkworthy_filter():
    """HIGH-1 回归：is_central_action=True 的 claim 被 CheckWorthy 过滤吗？
    这里模拟 CheckWorthy 返回空列表（全部过滤掉），验证豁免逻辑能否裥回 central_action_claim。"""
    central = Claim(text="购买某品牌减肥药", is_central_action=True)
    normal = Claim(text="别的报道")
    claims = [central, normal]

    central_claims = [c for c in claims if c.is_central_action]
    filtered: list = []  # CheckWorthy LLM 返回空列表（过滤掉全部）
    for cc in central_claims:
        if not any(cc.text == c.text or cc.text in c.text or c.text in cc.text for c in filtered):
            filtered = [cc] + filtered
    assert len(filtered) == 1
    assert filtered[0].is_central_action
    assert filtered[0].text == "购买某品牌减肥药"

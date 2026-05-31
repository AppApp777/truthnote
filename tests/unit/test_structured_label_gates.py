"""D2 / TRUE 救援门用 StructuredFC 标签而非关键词计数的测试。

Oracle Q2 指出的最大代码风险：D2/TRUE 门当前用 `debunk_count`（关键词正则计数）
判断要不要降级/升级，但 StructuredFC 已经用 LLM 给每条证据打了关系标签
（直接辟谣 / 间接矛盾 / 直接支持 / 话题相关 / 不相关）。

风险：LLM 标出 1 条「直接辟谣」但证据原文不含 _DEBUNK_KEYWORDS 列表中的
关键词时，关键词计数 = 0，D2 门会把 LLM 的 FALSE 判定误降级为 UNVERIFIABLE。

修复：D2/TRUE 门优先用标签，关键词计数仅作兜底。
"""

from src.truthnote.schemas import Claim, ClaimVerification, Evidence, SourceType, Verdict

# ── 基础：ClaimVerification 必须能携带证据关系标签 ──


def test_claim_verification_accepts_evidence_relations():
    """ClaimVerification 应允许 evidence_relations 字段持久化标签。"""
    cv = ClaimVerification(
        claim=Claim(text="测试"),
        verdict=Verdict.FALSE,
        confidence=0.85,
        evidence_relations=[
            {"index": 0, "relation": "直接辟谣"},
            {"index": 1, "relation": "话题相关"},
        ],
    )
    assert cv.evidence_relations[0]["relation"] == "直接辟谣"


def test_claim_verification_default_empty_relations():
    """没传 evidence_relations 时应该默认空列表，保持向后兼容。"""
    cv = ClaimVerification(
        claim=Claim(text="测试"),
        verdict=Verdict.FALSE,
        confidence=0.85,
    )
    assert cv.evidence_relations == []


# ── StructuredFC 必须把标签暴露给上层 ──


def test_structured_fact_checker_persists_labels(monkeypatch):
    """StructuredFC.check() 必须把 labels 持久化到 ClaimVerification.evidence_relations。"""
    from src.truthnote.agents import StructuredFactCheckerAgent

    agent = StructuredFactCheckerAgent()

    expected_labels = [
        {"index": 0, "relation": "直接辟谣"},
        {"index": 1, "relation": "话题相关"},
    ]
    step2_payload = {
        "key_facts": [{"fact": "5万元", "status": "无原文"}],
        "all_verified": False,
    }

    call_counter = {"n": 0}

    def fake_call_json(self, prompt, system=None):
        call_counter["n"] += 1
        if call_counter["n"] == 1:
            return {"labels": expected_labels}
        return step2_payload

    monkeypatch.setattr(StructuredFactCheckerAgent, "_call_json", fake_call_json)

    claim = Claim(text="存款超 5 万要交 20% 税")
    evidence = [
        Evidence(
            source="人民网",
            url="https://people.com.cn/x",
            title="官方否认",
            snippet="经核实，无此政策",
        ),
        Evidence(source="某网", url="https://other.com/y", title="相关讨论", snippet="不少人在问"),
    ]
    result = agent.check(claim, evidence, prescore={"signal": "neutral"})
    assert result.verdict == Verdict.FALSE
    assert result.evidence_relations == expected_labels


# ── D2 保守门：标签里有「直接辟谣」时不能降级 ──


def _build_evidence(
    url: str,
    support: bool | None,
    authority: float = 0.5,
    title: str = "t",
    snippet: str = "s",
):
    return Evidence(
        source="测试",
        url=url,
        title=title,
        snippet=snippet,
        supports_claim=support,
        source_type=SourceType.ESTABLISHED_MEDIA,
        authority_score=authority,
    )


def _debunk_evidence(url: str = "https://x.com/a") -> Evidence:
    """构造一条"直接辟谣"标签能通过交叉验证的证据（snippet 含辟谣关键词）。"""
    return Evidence(
        source="人民网",
        url=url,
        title="官方辟谣",
        snippet="经核实该说法不实，纯属编造",
        supports_claim=False,
        source_type=SourceType.OFFICIAL_GOVERNMENT,
        authority_score=0.9,
    )


def test_d2_gate_skips_when_labels_show_direct_debunk(monkeypatch):
    """证据原文含辟谣关键词 + StructuredFC 标签「直接辟谣」时，D2 门不应降级。

    注：旧版规则层只看 _DEBUNK_KEYWORDS 关键词 + 权威域名两个信号同时存在才算
    keyword_debunk；这里 source="人民网" 但 url 不在 _AUTHORITY_DOMAINS 集合中，
    所以 prescore_evidence 给的是 debunk_count=0、weak_debunk 信号缺失。
    本测试验证：即使关键词信号弱，LLM 标签也能保护 FALSE 判定不被降级。
    """
    from src.truthnote import orchestrator as orch_mod

    orch = orch_mod.Orchestrator.__new__(orch_mod.Orchestrator)
    orch.use_gates = True
    orch._current_diagnostics = []

    claim = Claim(text="网传新规取消高考")
    evidence = [_debunk_evidence()]
    verification = ClaimVerification(
        claim=claim,
        verdict=Verdict.FALSE,
        confidence=0.85,
        evidence_chain=evidence,
        reasoning="LLM 判 FALSE",
        evidence_relations=[{"index": 0, "relation": "直接辟谣"}],
    )
    # 关键词计数路径未触发 strong/weak debunk 时，标签救场
    score = {"debunk_count": 0, "signal": "neutral"}

    new_verification = orch._apply_conservative_gates(
        verification, evidence=evidence, score=score, diag=orch_mod.DiagnosticTrace()
    )
    assert new_verification.verdict == Verdict.FALSE  # 不应降级


def test_d2_gate_skips_when_labels_show_indirect_contradict(monkeypatch):
    """标签有「间接矛盾」时同样保护 FALSE 不被降级。"""
    from src.truthnote import orchestrator as orch_mod

    orch = orch_mod.Orchestrator.__new__(orch_mod.Orchestrator)
    orch.use_gates = True
    orch._current_diagnostics = []

    claim = Claim(text="今晚北京 8 级地震")
    evidence = [_build_evidence("https://x.com/a", support=False)]
    verification = ClaimVerification(
        claim=claim,
        verdict=Verdict.FALSE,
        confidence=0.85,
        evidence_chain=evidence,
        reasoning="间接矛盾",
        evidence_relations=[{"index": 0, "relation": "间接矛盾"}],
    )
    score = {"debunk_count": 0, "signal": "neutral"}
    new_verification = orch._apply_conservative_gates(
        verification, evidence=evidence, score=score, diag=orch_mod.DiagnosticTrace()
    )
    assert new_verification.verdict == Verdict.FALSE


def test_d2_gate_downgrades_when_no_labels_and_no_keywords():
    """无标签 + 无关键词 → 保持原行为：降级为 UNVERIFIABLE。"""
    from src.truthnote import orchestrator as orch_mod

    orch = orch_mod.Orchestrator.__new__(orch_mod.Orchestrator)
    orch.use_gates = True
    orch._current_diagnostics = []

    claim = Claim(text="某未知谣言")
    evidence = [_build_evidence("https://x.com/a", support=None)]
    verification = ClaimVerification(
        claim=claim,
        verdict=Verdict.FALSE,
        confidence=0.85,
        evidence_chain=evidence,
        reasoning="LLM 判 FALSE 但无任何辟谣支撑",
        evidence_relations=[
            {"index": 0, "relation": "话题相关"},
        ],
    )
    score = {"debunk_count": 0, "signal": "neutral"}
    new_verification = orch._apply_conservative_gates(
        verification, evidence=evidence, score=score, diag=orch_mod.DiagnosticTrace()
    )
    assert new_verification.verdict == Verdict.UNVERIFIABLE


# ── TRUE 救援门：标签里有「直接支持」可触发救援 ──


def test_true_rescue_fires_on_direct_support_label():
    """LLM 标签有「直接支持」 + 无标签辟谣/矛盾 + 无关键词辟谣 → 救援为 PARTLY_TRUE。"""
    from src.truthnote import orchestrator as orch_mod

    orch = orch_mod.Orchestrator.__new__(orch_mod.Orchestrator)
    orch.use_gates = True
    orch._current_diagnostics = []

    claim = Claim(text="WHO 推荐成人每周 150 分钟中等强度运动")
    # 即使 authority_score 不到 0.70，标签判定也应救援
    evidence = [_build_evidence("https://x.com/a", support=True, authority=0.4)]
    verification = ClaimVerification(
        claim=claim,
        verdict=Verdict.FALSE,
        confidence=0.85,
        evidence_chain=evidence,
        reasoning="误判为 FALSE",
        evidence_relations=[{"index": 0, "relation": "直接支持"}],
    )
    score = {"debunk_count": 0, "signal": "neutral"}
    new_verification = orch._apply_conservative_gates(
        verification, evidence=evidence, score=score, diag=orch_mod.DiagnosticTrace()
    )
    assert new_verification.verdict == Verdict.PARTLY_TRUE


# ── HIGH 1：Skeptic 翻转后 TRUE 救援不应反弹 ──


def test_true_rescue_skips_when_skeptic_revised():
    """Skeptic 把 TRUE→FALSE 后：
    - TRUE 救援门跳过（不撤销 Skeptic 修正）
    - D2 仍然生效：无辟谣证据时降级为 UNVERIFIABLE（安全兜底）
    """
    from src.truthnote import orchestrator as orch_mod

    orch = orch_mod.Orchestrator.__new__(orch_mod.Orchestrator)
    orch.use_gates = True
    orch._current_diagnostics = []

    claim = Claim(text="某数字伪造但权威来源覆盖话题")
    evidence = [_build_evidence("https://x.com/a", support=True, authority=0.85)]
    verification = ClaimVerification(
        claim=claim,
        verdict=Verdict.FALSE,
        confidence=0.6,
        evidence_chain=evidence,
        reasoning="Skeptic 检出数字造假，翻转为 FALSE",
        evidence_relations=[{"index": 0, "relation": "直接支持"}],
    )
    score = {"debunk_count": 0, "signal": "neutral"}
    new_verification = orch._apply_conservative_gates(
        verification,
        evidence=evidence,
        score=score,
        diag=orch_mod.DiagnosticTrace(),
        skeptic_revised=True,
    )
    # D2 在 Skeptic 修正后仍然适用：无辟谣证据 → UNVERIFIABLE
    assert new_verification.verdict == Verdict.UNVERIFIABLE


# ── HIGH 2：标签交叉验证 ──


def test_direct_debunk_label_discarded_when_evidence_lacks_keywords():
    """LLM 标「直接辟谣」但证据原文无任何辟谣关键词 → 标签被丢弃，D2 门正常触发降级。"""
    from src.truthnote import orchestrator as orch_mod

    orch = orch_mod.Orchestrator.__new__(orch_mod.Orchestrator)
    orch.use_gates = True
    orch._current_diagnostics = []

    claim = Claim(text="某未知传言")
    # 证据 snippet 完全不含辟谣字眼，"直接辟谣"标签不可信
    evidence = [
        Evidence(
            source="某网",
            url="https://x.com/a",
            title="相关主题报道",
            snippet="本文讨论了相关主题的多个方面",
        )
    ]
    verification = ClaimVerification(
        claim=claim,
        verdict=Verdict.FALSE,
        confidence=0.85,
        evidence_chain=evidence,
        reasoning="LLM 误标",
        evidence_relations=[{"index": 0, "relation": "直接辟谣"}],
    )
    score = {"debunk_count": 0, "signal": "neutral"}
    new_verification = orch._apply_conservative_gates(
        verification, evidence=evidence, score=score, diag=orch_mod.DiagnosticTrace()
    )
    assert new_verification.verdict == Verdict.UNVERIFIABLE  # 标签被验伪，D2 触发


def test_direct_support_label_discarded_when_evidence_lacks_auth_or_supports():
    """LLM 标「直接支持」但证据 supports_claim=False 且 authority 不足 → 标签被丢弃，不救援。"""
    from src.truthnote import orchestrator as orch_mod

    orch = orch_mod.Orchestrator.__new__(orch_mod.Orchestrator)
    orch.use_gates = True
    orch._current_diagnostics = []

    claim = Claim(text="某主张")
    evidence = [
        Evidence(
            source="某博客",
            url="https://blog.x.com/a",
            title="t",
            snippet="s",
            supports_claim=False,
            source_type=SourceType.BLOG_FORUM,
            authority_score=0.3,
        )
    ]
    verification = ClaimVerification(
        claim=claim,
        verdict=Verdict.FALSE,
        confidence=0.85,
        evidence_chain=evidence,
        reasoning="LLM 误标",
        evidence_relations=[{"index": 0, "relation": "直接支持"}],
    )
    score = {"debunk_count": 0, "signal": "neutral"}
    new_verification = orch._apply_conservative_gates(
        verification, evidence=evidence, score=score, diag=orch_mod.DiagnosticTrace()
    )
    # 标签被验伪，TRUE 救援不触发；D2 也无 label_debunk → 降级
    assert new_verification.verdict == Verdict.UNVERIFIABLE


# ── MEDIUM：D2 门覆盖 MOSTLY_FALSE ──


def test_d2_gate_downgrades_mostly_false_when_no_debunk():
    """MOSTLY_FALSE 也无辟谣支撑时，D2 门同样降级为 UNVERIFIABLE。"""
    from src.truthnote import orchestrator as orch_mod

    orch = orch_mod.Orchestrator.__new__(orch_mod.Orchestrator)
    orch.use_gates = True
    orch._current_diagnostics = []

    claim = Claim(text="某可疑主张")
    evidence = [_build_evidence("https://x.com/a", support=None)]
    verification = ClaimVerification(
        claim=claim,
        verdict=Verdict.MOSTLY_FALSE,
        confidence=0.75,
        evidence_chain=evidence,
        reasoning="LLM 判 MOSTLY_FALSE",
        evidence_relations=[{"index": 0, "relation": "话题相关"}],
    )
    score = {"debunk_count": 0, "signal": "neutral"}
    new_verification = orch._apply_conservative_gates(
        verification, evidence=evidence, score=score, diag=orch_mod.DiagnosticTrace()
    )
    assert new_verification.verdict == Verdict.UNVERIFIABLE


# ── MEDIUM：_count_relations 校验 index/relation ──


def test_count_relations_filters_bad_index():
    """非法 index（越界、负数、非整数）应被忽略。"""
    from src.truthnote.orchestrator import Orchestrator

    evidence = [_build_evidence("https://x.com/a", support=False)]
    verification = ClaimVerification(
        claim=Claim(text="测试"),
        verdict=Verdict.FALSE,
        confidence=0.8,
        evidence_chain=evidence,
        evidence_relations=[
            {"index": 99, "relation": "直接辟谣"},  # 越界
            {"index": -1, "relation": "直接辟谣"},  # 负数
            {"index": "0", "relation": "直接辟谣"},  # 非整数
        ],
    )
    rel = Orchestrator._count_relations(verification, evidence)
    assert rel["direct_debunk"] == 0


def test_count_relations_filters_unknown_relation():
    """未知 relation 类别应被忽略。"""
    from src.truthnote.orchestrator import Orchestrator

    evidence = [_build_evidence("https://x.com/a", support=False)]
    verification = ClaimVerification(
        claim=Claim(text="测试"),
        verdict=Verdict.FALSE,
        confidence=0.8,
        evidence_chain=evidence,
        evidence_relations=[
            {"index": 0, "relation": "directly_debunked"},  # 英文
            {"index": 0, "relation": "未知类别"},
        ],
    )
    rel = Orchestrator._count_relations(verification, evidence)
    assert rel["direct_debunk"] == 0


# ── MEDIUM：StructuredFC step1 失败时 evidence_relations 应该为空（降级标记）──


def test_structured_fc_step1_failure_persists_empty_relations(monkeypatch):
    """LLM step1 异常时，evidence_relations 应为空列表，让 gates 退回关键词路径。"""
    from src.truthnote.agents import StructuredFactCheckerAgent

    agent = StructuredFactCheckerAgent()

    call_counter = {"n": 0}

    def fake_call_json(self, prompt, system=None):
        call_counter["n"] += 1
        if call_counter["n"] == 1:
            raise RuntimeError("LLM 调用失败")
        return {"key_facts": [], "all_verified": False}

    monkeypatch.setattr(StructuredFactCheckerAgent, "_call_json", fake_call_json)

    claim = Claim(text="某主张")
    evidence = [Evidence(source="某网", url="https://x.com/a", title="t", snippet="s")]
    result = agent.check(claim, evidence, prescore={"signal": "neutral"})
    # 失败时不应把 [{话题相关}*N] 持久化到 evidence_relations，避免误导 gates
    assert result.evidence_relations == []


def test_true_rescue_does_not_fire_with_label_debunk():
    """有「直接支持」但同时也有「直接辟谣」标签 → 不救援。"""
    from src.truthnote import orchestrator as orch_mod

    orch = orch_mod.Orchestrator.__new__(orch_mod.Orchestrator)
    orch.use_gates = True
    orch._current_diagnostics = []

    claim = Claim(text="某争议主张")
    evidence = [
        _build_evidence("https://x.com/a", support=True, authority=0.85),
        _debunk_evidence("https://x.com/b"),
    ]
    verification = ClaimVerification(
        claim=claim,
        verdict=Verdict.FALSE,
        confidence=0.85,
        evidence_chain=evidence,
        reasoning="冲突证据",
        evidence_relations=[
            {"index": 0, "relation": "直接支持"},
            {"index": 1, "relation": "直接辟谣"},
        ],
    )
    score = {"debunk_count": 0, "signal": "neutral"}
    new_verification = orch._apply_conservative_gates(
        verification, evidence=evidence, score=score, diag=orch_mod.DiagnosticTrace()
    )
    assert new_verification.verdict == Verdict.FALSE  # 不救援

"""契约测试 · 采信漏斗（官方辟谣库证据检索的「只把采信的几条上桌」边界）。

来源：副线施工卡《采信筛选与可视化》。核心产品契约——
召回会捞回跑题/反命题候选；**只有 same_claim 才允许进证据链（上桌），
opposite_claim / same_topic_different_claim / no_match 只进调试 trace（留后厨），
绝不进入用户可见的 evidence_chain。一条都不采信时不许伪造证据。**

与既有契约的分工：
  - tests/contracts/test_debunk_index_no_shortcircuit.py 守 INV-4「命中只进证据不短路 verdict」。
  - 本文件守「采信筛选」的另一半：哪些候选**配进证据链**、哪些只配进 trace。
这两者一起把「11000 条爬虫库 → 检索 → 同命题核对 → 证据」这条链的采信边界钉死，
是「爬虫价值链冻结」的回归护栏。

注：这些多为**特征化/锁定测试**——锁住当前已正确的行为，防回退（这正是"冻结"的含义）。
"""

from src.truthnote.orchestrator import Orchestrator


def _fake_cand(claim_text: str = "银行存款超过5万元要交20%的税"):
    from src.truthnote import debunk_index as di

    return di.DebunkCandidate(
        item_id="funnel_fake",
        claim_text=claim_text,
        verdict="谣言",
        source="中国互联网联合辟谣平台",
        url="https://piyao.example/funnel",
        title=claim_text[:80],
        snippet=f"官方辟谣库收录：{claim_text}",
        category="政策法规",
        published_date="2026-01-01",
        lexical_score=0.95,
        ngram_score=0.95,
        entity_score=1.0,
    )


def _fake_result(label: str, score: float):
    from src.truthnote import debunk_index as di

    # 让失败守卫与 label 自洽（避免造一个真实 verify_same_claim 永不产出的状态）：
    failed_by_label = {
        "same_claim": [],
        "opposite_claim": ["negation_polarity_conflict"],
        "same_topic_different_claim": ["same_claim_threshold"],
        "no_match": ["lexical_gate"],
    }
    return di.SameClaimResult(
        label=label,
        score=score,
        candidate_id="funnel_fake",
        candidate_url="https://piyao.example/funnel",
        matched_claim="银行存款超过5万元要交20%的税",
        failed_guards=failed_by_label.get(label, []),
        passed_guards=["lexical_gate"],
    )


def _debunk_index_evidence(result):
    """从 run 结果里取出由本地官方辟谣库贡献的证据（source_tag 唯一标识）。"""
    return [
        ev
        for c in result.claims
        for ev in c.evidence_chain
        if getattr(ev, "source_tag", "") == "official_debunk_index"
    ]


# ── 采信漏斗：只有 same_claim 上桌，其余留后厨 ──────────────────────────────


def test_same_claim_candidate_is_adopted_into_evidence_chain(mock_llm_pipeline, monkeypatch):
    """正例（上桌）：same_claim 候选 → 作为官方辟谣证据进入 evidence_chain。"""
    from src.truthnote import debunk_index as di

    monkeypatch.setattr(di, "retrieve_debunk_candidates", lambda text, top_k=3: [_fake_cand()])
    monkeypatch.setattr(
        di, "verify_same_claim", lambda text, cand: _fake_result("same_claim", 0.95)
    )

    orch = Orchestrator(use_debunk_index=True)
    result = orch.run("紧急通知！存款超5万交税！")

    adopted = _debunk_index_evidence(result)
    assert adopted, "same_claim 候选应被采信进入 evidence_chain"
    assert adopted[0].url == "https://piyao.example/funnel"


def test_opposite_claim_candidate_is_kept_backstage(mock_llm_pipeline, monkeypatch):
    """后厨（不上桌）：opposite_claim → 绝不进 evidence_chain，只在 trace 里留痕。

    这同时锁死了 §2 的安全方向：辟谣结论式标题被极性守卫判 opposite 时，
    宁可漏掉（落回正常流程）也不误收——符合 INV-4「漏判=安全」。
    """
    from src.truthnote import debunk_index as di

    monkeypatch.setattr(di, "retrieve_debunk_candidates", lambda text, top_k=3: [_fake_cand()])
    monkeypatch.setattr(
        di, "verify_same_claim", lambda text, cand: _fake_result("opposite_claim", 0.80)
    )

    orch = Orchestrator(use_debunk_index=True)
    result = orch.run("紧急通知！存款超5万交税！")

    assert not _debunk_index_evidence(result), "opposite_claim 候选绝不允许进 evidence_chain"
    # 仍应在 trace 里留痕（透明：后厨可查，只是不上桌）
    actions = [s.action for s in orch.trace.steps]
    assert any("同命题核对：未通过" in a for a in actions), (
        f"opposite 候选应在 trace 留痕：{actions}"
    )


def test_same_topic_different_claim_candidate_is_kept_backstage(mock_llm_pipeline, monkeypatch):
    """后厨：same_topic_different_claim（同话题不同说法）→ 不进 evidence_chain。"""
    from src.truthnote import debunk_index as di

    monkeypatch.setattr(di, "retrieve_debunk_candidates", lambda text, top_k=3: [_fake_cand()])
    monkeypatch.setattr(
        di,
        "verify_same_claim",
        lambda text, cand: _fake_result("same_topic_different_claim", 0.60),
    )

    orch = Orchestrator(use_debunk_index=True)
    result = orch.run("紧急通知！存款超5万交税！")

    assert not _debunk_index_evidence(result), (
        "same_topic_different_claim 候选只配进 trace，不配进 evidence_chain"
    )


def test_all_miss_fabricates_no_debunk_evidence(mock_llm_pipeline, monkeypatch):
    """一条都不采信时绝不伪造：召回到候选但全 no_match → 无任何官方辟谣库证据进链。

    呼应施工卡 §3「一条不剩 → 标'未找到官方辟谣' → 走无法核实/科学归因，绝不硬凑」。
    """
    from src.truthnote import debunk_index as di

    monkeypatch.setattr(
        di, "retrieve_debunk_candidates", lambda text, top_k=3: [_fake_cand(), _fake_cand()]
    )
    monkeypatch.setattr(di, "verify_same_claim", lambda text, cand: _fake_result("no_match", 0.10))

    orch = Orchestrator(use_debunk_index=True)
    result = orch.run("紧急通知！存款超5万交税！")

    assert not _debunk_index_evidence(result), "全 miss 时不得伪造官方辟谣库证据"

"""契约测试 · INV-4：官方辟谣库命中只进证据、绝不短路最终 verdict。

红线（CONTRACTS.md INV-4）：debunk_index 检索到的官方辟谣命中，只能作为一条
高权威证据进入 prior_evidence → evidence 链，由下游 6 维度引擎 + 规则裁决出
最终 verdict。**任何**"命中即直接判假/直接 return verdict"的短路都违反本不变量。

本文件两道防线：
  1. 结构守护（inspect.getsource）——锁死 orchestrator 里 debunk 块不含 verdict
     短路。防止以后有人为了"更快"偷偷加 `if 命中: return FALSE`。
  2. 行为集成（mock 全流水线）——强制 debunk 命中后跑完整 orchestrator，断言
     命中只是被当作证据，下游取证/裁决管线照常执行（没有被短路）。
"""

import inspect

from src.truthnote.orchestrator import Orchestrator
from src.truthnote.schemas import Verdict

# ── 1) 结构守护：debunk 块只产证据，不产 verdict ──────────────────────────────


def _method_source() -> str:
    return inspect.getsource(Orchestrator._verify_single_claim)


_PART1_START = "# 1) 离线/本地官方辟谣库证据检索"
_PART1_END = "# 2) 现有的 live piyao 定向预搜索"


def _part1_debunk_slice() -> str:
    """切出 Part 1（本地官方辟谣库检索）那一段源码。

    边界靠两处注释锚点；若有人改注释文案导致找不到，给出明确报错而非裸 ValueError
    （code-review MEDIUM：避免重构时 6 个结构测试集体崩且不指向真因）。
    """
    src = _method_source()
    if _PART1_START not in src or _PART1_END not in src:
        raise AssertionError(
            "INV-4 结构测试找不到 debunk Part1 锚点注释——若你重构了 ClaimMatcher 块，"
            f"请同步更新锚点：{_PART1_START!r} / {_PART1_END!r}"
        )
    start = src.index(_PART1_START)
    end = src.index(_PART1_END)
    return src[start:end]


def test_debunk_block_exists_and_marked_no_shortcircuit():
    src = _method_source()
    assert "官方辟谣库" in src, "ClaimMatcher debunk 块应存在于 _verify_single_claim"
    # 红线注释必须在场（INV-4 标记），既是文档也是防回退锚点
    assert "INV-4" in src, "debunk 块必须带 INV-4 红线注释"
    assert "不短路" in src or "绝不短路" in src


def test_debunk_confirmed_hit_only_becomes_evidence():
    """确认命中只能 → confirmed_candidate_to_evidence → prior_evidence.append。"""
    part1 = _part1_debunk_slice()
    assert "retrieve_debunk_candidates" in part1
    assert "verify_same_claim" in part1
    assert "confirmed_candidate_to_evidence" in part1
    assert "prior_evidence.append(ev)" in part1


def test_debunk_block_has_no_verdict_shortcircuit():
    """Part 1 块内绝不出现 verdict 短路：无 Verdict 赋值、无 early return。

    这是 INV-4 的核心防线——若有人加 `if 命中: return ClaimVerification(FALSE)`
    或 `verdict = Verdict.FALSE`，本断言立即失败。
    """
    part1 = _part1_debunk_slice()
    assert "Verdict" not in part1, "debunk 块不得引用 Verdict（命中不出判定）"
    assert "return" not in part1, "debunk 块不得 early return（命中不短路）"
    # 不得直接给 ClaimVerification 赋判定
    assert "ClaimVerification(" not in part1
    assert "_pick_overall_verdict" not in part1


def test_debunk_runs_before_evidence_hunter():
    """debunk 块必须在 EvidenceHunter / 最终裁决之前——它喂的是 prior_evidence，
    再由后续正常管线合并、打分。位置上证明它不可能"替代"判定。"""
    src = _method_source()
    debunk_pos = src.index("官方辟谣库")
    hunter_pos = src.index("hunter.hunt")
    assert debunk_pos < hunter_pos, "debunk 检索应在 EvidenceHunter 之前（只做证据前置）"


# ── 2) 行为集成：强制命中后，完整管线照常跑，命中只当证据 ────────────────────


def test_debunk_hit_does_not_shortcircuit_full_run(mock_llm_pipeline, monkeypatch):
    """强制 debunk 命中，跑完整 orchestrator.run，断言：
    - ClaimMatcher 检索 step 出现在 trace（集成确实接上）
    - EvidenceHunter step 仍出现（管线未被 debunk 短路）
    - debunk 命中证据进入了证据链（命中=证据）
    - 最终 verdict 由引擎/规则产出（此 mock 下为 FALSE）
    """
    from src.truthnote import debunk_index as di

    fake_cand = di.DebunkCandidate(
        item_id="inv4_fake",
        claim_text="银行存款超过5万元要交20%的税",
        verdict="谣言",
        source="中国互联网联合辟谣平台",
        url="https://piyao.example/inv4",
        title="存款超5万交税系谣言",
        snippet="官方辟谣库收录：银行存款超过5万元要交20%的税",
        category="政策法规",
        published_date="2026-01-01",
        lexical_score=0.95,
        ngram_score=0.95,
        entity_score=1.0,
    )
    fake_result = di.SameClaimResult(
        label="same_claim",
        score=0.95,
        candidate_id="inv4_fake",
        candidate_url="https://piyao.example/inv4",
        matched_claim=fake_cand.claim_text,
        passed_guards=["exact_fingerprint"],
    )

    # 懒导入在方法内 `from .debunk_index import ...`，运行时从源模块取属性 → patch 源模块即可生效
    monkeypatch.setattr(di, "retrieve_debunk_candidates", lambda text, top_k=3: [fake_cand])
    monkeypatch.setattr(di, "verify_same_claim", lambda text, cand: fake_result)

    orch = Orchestrator(use_debunk_index=True)
    result = orch.run("紧急通知！存款超5万交税！")

    actions = [s.action for s in orch.trace.steps]

    # 集成接上：debunk 检索 step 出现
    assert any("官方辟谣库检索" in a for a in actions), f"未见 debunk 检索 step：{actions}"
    # 同命题核对 step 出现且通过
    assert any("同命题核对：通过" in a for a in actions), f"未见同命题核对通过 step：{actions}"
    # 管线未短路：debunk 在 hunter 之前，EvidenceHunter 仍跑了
    assert any("搜索" in a for a in actions), f"EvidenceHunter 未运行（疑似被短路）：{actions}"

    # 命中=证据：debunk 命中以官方辟谣源进入证据链
    all_ev = [ev for c in result.claims for ev in c.evidence_chain]
    assert any(
        getattr(ev, "source_tag", "") == "official_debunk_index"
        or ev.url == "https://piyao.example/inv4"
        for ev in all_ev
    ), "debunk 命中未作为证据进入证据链"

    # 最终 verdict 由引擎/规则产出（mock 给「直接辟谣」标签 → FALSE）
    assert result.overall_verdict == Verdict.FALSE


def test_debunk_disabled_flag_skips_local_retrieval(mock_llm_pipeline, monkeypatch):
    """use_debunk_index=False（demo 一键关）时，本地辟谣库检索整段不跑。"""
    from src.truthnote import debunk_index as di

    called = {"n": 0}

    def _boom(text, top_k=3):
        called["n"] += 1
        return []

    monkeypatch.setattr(di, "retrieve_debunk_candidates", _boom)

    orch = Orchestrator(use_debunk_index=False)
    orch.run("紧急通知！存款超5万交税！")

    assert called["n"] == 0, "关闭开关后不应调用本地辟谣库检索"
    actions = [s.action for s in orch.trace.steps]
    assert not any("官方辟谣库检索" in a for a in actions), "关闭后不应出现 debunk 检索 step"

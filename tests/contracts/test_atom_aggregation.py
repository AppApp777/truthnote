"""契约测试：原子化路径必须把 evidence_relations 传播到聚合结果。

回归 BUG（2026-05-29）：_verify_single_claim 拆原子聚合时，
1. 中间记录 dict 漏 "relations" key；
2. 更根本的是 _aggregate_atom_verdicts 根本不收集 / 不重映射 / 不设置 evidence_relations。
两层叠加 → 聚合后方向标签恒空 → 下游证实维度 C 恒 0 → 真消息即便有高权威支持源
也无法救回（id=13140 / 18351 误判谣言）。

依赖契约：CONTRACTS.md C4.4 原子化聚合传播 relations / C5.2 TRUE 救援门 / C5.5 标签交叉验证。
"""

from src.truthnote.dimensions import confirmation_and_debunk_from_verifications
from src.truthnote.orchestrator import (
    _aggregate_atom_verdicts,
    _atom_verification_record,
)
from src.truthnote.schemas import Claim, ClaimVerification, Evidence, Verdict


def _ev(source: str, auth: float, support: bool | None = None) -> Evidence:
    return Evidence(
        source=source,
        url=f"http://{source}",
        snippet="x",
        authority_score=auth,
        supports_claim=support,
    )


def _atom(atom_id, text, verdict, evidence, relations, is_core=True):
    return {
        "atom_id": atom_id,
        "text": text,
        "is_core": is_core,
        "verdict": verdict,
        "confidence": 0.8,
        "evidence": evidence,
        "relations": relations,
        "reasoning": "r",
    }


class TestAtomVerificationRecord:
    """helper 产出的 dict 的 key 必须和 _aggregate_atom_verdicts 消费的 key 对齐。"""

    def test_record_carries_relations_and_evidence(self):
        ev = [_ev("gov", 0.95)]
        rels = [{"index": 0, "relation": "直接支持"}]
        cv = ClaimVerification(
            claim=Claim(text="原子1"),
            verdict=Verdict.TRUE,
            confidence=0.8,
            evidence_chain=ev,
            evidence_relations=rels,
            reasoning="r",
        )
        rec = _atom_verification_record({"id": "A1", "text": "原子1"}, cv)
        assert rec["relations"] == rels
        assert rec["evidence"] == ev
        assert rec["verdict"] == Verdict.TRUE
        assert rec["atom_id"] == "A1"
        assert rec["is_core"] is True

    def test_record_relations_empty_when_cv_has_none(self):
        cv = ClaimVerification(
            claim=Claim(text="x"),
            verdict=Verdict.UNVERIFIABLE,
            confidence=0.2,
            evidence_chain=[],
            reasoning="",
        )
        rec = _atom_verification_record({"id": "A1", "text": "x"}, cv)
        assert rec["relations"] == []


class TestAggregatePropagatesRelations:
    def test_relations_propagated_with_offset(self):
        # 两个原子各 1 条证据 + 各 1 条 index=0 的标签
        # 聚合 evidence_chain = [gov1, gov2]，relation index 必须重映射为 0 和 1
        a1 = _atom(
            "A1", "事实1", Verdict.TRUE, [_ev("gov1", 0.95)], [{"index": 0, "relation": "直接支持"}]
        )
        a2 = _atom(
            "A2", "事实2", Verdict.TRUE, [_ev("gov2", 0.95)], [{"index": 0, "relation": "直接辟谣"}]
        )
        result = _aggregate_atom_verdicts([a1, a2], Claim(text="复合"))

        assert len(result.evidence_chain) == 2
        by_idx = {r["index"]: r["relation"] for r in result.evidence_relations}
        assert set(by_idx) == {0, 1}
        # 重映射后 index 必须指向正确证据
        assert result.evidence_chain[0].source == "gov1"
        assert result.evidence_chain[1].source == "gov2"
        assert by_idx[0] == "直接支持"
        assert by_idx[1] == "直接辟谣"

    def test_out_of_range_atom_local_index_dropped(self):
        # 原子内 index 越界（StructuredFC 抖动产出脏标签）→ 丢弃，不重映射到错误证据
        a1 = _atom(
            "A1", "事实1", Verdict.TRUE, [_ev("gov1", 0.95)], [{"index": 5, "relation": "直接支持"}]
        )
        result = _aggregate_atom_verdicts([a1], Claim(text="复合"))
        assert result.evidence_relations == []

    def test_relations_beyond_truncation_dropped(self):
        # evidence_chain 截断到 10；超出截断的 relation 必须丢弃（否则 index 指向不存在的证据）
        atoms = [
            _atom(
                f"A{i}",
                f"f{i}",
                Verdict.TRUE,
                [_ev(f"gov{i}", 0.95)],
                [{"index": 0, "relation": "直接支持"}],
            )
            for i in range(12)
        ]
        result = _aggregate_atom_verdicts(atoms, Claim(text="复合"))
        assert len(result.evidence_chain) == 10
        assert all(r["index"] < 10 for r in result.evidence_relations)
        assert len(result.evidence_relations) == 10

    def test_missing_relations_key_tolerated(self):
        # 旧记录无 relations key（向后兼容）→ 不崩，relations 空
        old = {
            "atom_id": "A1",
            "text": "x",
            "is_core": True,
            "verdict": Verdict.TRUE,
            "confidence": 0.8,
            "evidence": [_ev("gov", 0.95)],
            "reasoning": "r",
        }
        result = _aggregate_atom_verdicts([old], Claim(text="复合"))
        assert result.evidence_relations == []


class TestEndToEndConfirmationRevived:
    """聚合 CV 喂给 confirmation 维度，证实分 C 必须复活（真消息救回的核心证明）。"""

    def test_high_authority_support_revives_confirmation(self):
        a1 = _atom(
            "A1",
            "事实1",
            Verdict.TRUE,
            [_ev("piyao.org.cn", 0.95, support=True)],
            [{"index": 0, "relation": "直接支持"}],
        )
        agg = _aggregate_atom_verdicts([a1], Claim(text="复合"))
        h, c, primary = confirmation_and_debunk_from_verifications([agg])
        assert c > 0.0  # 修复前因 relations 丢失恒为 0
        assert primary is True

    def test_relations_lost_keeps_confirmation_zero(self):
        # 对照组：relations 丢失且证据无 supports_claim → 无方向 → C 保守地等于 0
        a1 = _atom(
            "A1",
            "事实1",
            Verdict.TRUE,
            [_ev("piyao.org.cn", 0.95, support=None)],
            [],
        )
        agg = _aggregate_atom_verdicts([a1], Claim(text="复合"))
        h, c, primary = confirmation_and_debunk_from_verifications([agg])
        assert c == 0.0

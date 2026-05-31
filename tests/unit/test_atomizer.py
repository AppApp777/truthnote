"""AtomicFactExtractor + 聚合规则测试。"""

from src.truthnote.agents import AtomicFactExtractorAgent
from src.truthnote.orchestrator import _aggregate_atom_verdicts
from src.truthnote.schemas import Claim, Evidence, Verdict


class TestShouldTryAtomize:
    def test_single_fact_skip(self):
        assert not AtomicFactExtractorAgent._should_try_atomize("存款超5万要交税")

    def test_two_numbers_trigger(self):
        assert AtomicFactExtractorAgent._should_try_atomize("存款超5万元要交20%的税")

    def test_connector_trigger(self):
        assert AtomicFactExtractorAgent._should_try_atomize("中国人口14亿并且GDP超过美国")

    def test_short_single_fact(self):
        assert not AtomicFactExtractorAgent._should_try_atomize("今天会下雨")

    def test_multiple_clauses_trigger(self):
        text = "教育部发布通知，取消中考体育，同时推迟高考时间，并且增加艺术科目"
        assert AtomicFactExtractorAgent._should_try_atomize(text)

    def test_quoted_entities_trigger(self):
        text = "《新闻联播》报道「张三」和「李四」同时被任命，取代了「王五」和「赵六」"
        assert AtomicFactExtractorAgent._should_try_atomize(text)


class TestAggregateAtomVerdicts:
    def _make_atom(self, atom_id, text, verdict, is_core=True):
        return {
            "atom_id": atom_id,
            "text": text,
            "is_core": is_core,
            "verdict": verdict,
            "confidence": 0.8,
            "evidence": [Evidence(source="test", url="http://test.com", snippet="test")],
            "reasoning": "test",
        }

    def test_all_core_false(self):
        atoms = [
            self._make_atom("A1", "数字1错误", Verdict.FALSE),
            self._make_atom("A2", "数字2错误", Verdict.FALSE),
        ]
        result = _aggregate_atom_verdicts(atoms, Claim(text="复合声明"))
        assert result.verdict == Verdict.FALSE
        assert result.confidence >= 0.85

    def test_mixed_core_false_and_true(self):
        atoms = [
            self._make_atom("A1", "数字1正确", Verdict.TRUE),
            self._make_atom("A2", "数字2错误", Verdict.FALSE),
        ]
        result = _aggregate_atom_verdicts(atoms, Claim(text="复合声明"))
        assert result.verdict == Verdict.MOSTLY_FALSE

    def test_all_core_true(self):
        atoms = [
            self._make_atom("A1", "事实1", Verdict.TRUE),
            self._make_atom("A2", "事实2", Verdict.TRUE),
        ]
        result = _aggregate_atom_verdicts(atoms, Claim(text="复合声明"))
        assert result.verdict == Verdict.TRUE

    def test_core_true_and_unverifiable(self):
        atoms = [
            self._make_atom("A1", "事实1", Verdict.TRUE),
            self._make_atom("A2", "事实2", Verdict.UNVERIFIABLE),
        ]
        result = _aggregate_atom_verdicts(atoms, Claim(text="复合声明"))
        assert result.verdict == Verdict.PARTLY_TRUE

    def test_all_unverifiable(self):
        atoms = [
            self._make_atom("A1", "事实1", Verdict.UNVERIFIABLE),
            self._make_atom("A2", "事实2", Verdict.UNVERIFIABLE),
        ]
        result = _aggregate_atom_verdicts(atoms, Claim(text="复合声明"))
        assert result.verdict == Verdict.UNVERIFIABLE

    def test_empty_atoms(self):
        result = _aggregate_atom_verdicts([], Claim(text="空"))
        assert result.verdict == Verdict.UNVERIFIABLE

    def test_non_core_doesnt_override(self):
        atoms = [
            self._make_atom("A1", "核心事实", Verdict.TRUE, is_core=True),
            self._make_atom("A2", "背景信息", Verdict.FALSE, is_core=False),
        ]
        result = _aggregate_atom_verdicts(atoms, Claim(text="复合声明"))
        assert result.verdict == Verdict.TRUE

    def test_misleading_atom(self):
        atoms = [
            self._make_atom("A1", "事实1", Verdict.TRUE),
            self._make_atom("A2", "旧闻", Verdict.MISLEADING),
        ]
        result = _aggregate_atom_verdicts(atoms, Claim(text="复合声明"))
        assert result.verdict == Verdict.MISLEADING

    def test_reasoning_contains_atom_ids(self):
        atoms = [
            self._make_atom("A1", "事实1", Verdict.TRUE),
            self._make_atom("A2", "事实2", Verdict.FALSE),
        ]
        result = _aggregate_atom_verdicts(atoms, Claim(text="复合声明"))
        assert "[A1]" in result.reasoning
        assert "[A2]" in result.reasoning
        assert "[原子化聚合]" in result.reasoning

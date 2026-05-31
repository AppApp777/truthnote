"""CommonsenseCheckerAgent 单元测试。

覆盖：
- 科学共识类常识命中 → 快速路径返回 FALSE
- 事实性常识命中 → 快速路径返回 MOSTLY_FALSE
- 非常识声明 → 不走快速路径
- confidence 低于阈值 → 不走快速路径
- LLM 返回不允许的 verdict → 安全降级
- LLM 调用失败 → 安全降级，走完整流水线
- 有争议的科学问题 → 不走快速路径
- is_commonsense=true 但无 verdict → 安全降级
- confidence 阈值边界测试
- 输出格式兜底（commonsense_type 无效值）
"""

import json
from unittest.mock import patch

from src.truthnote.agents import VERDICT_MAP, CommonsenseCheckerAgent
from src.truthnote.schemas import Claim, RumorCategory, Verdict


def _mock_chat_result(content):
    return {
        "content": json.dumps(content, ensure_ascii=False)
        if isinstance(content, dict)
        else content,
        "tool_calls": [],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 0, "output_tokens": 0},
    }


class TestCommonsenseCheckerAgent:
    """CommonsenseCheckerAgent 的核心行为测试。"""

    def test_scientific_consensus_false(self):
        """科学共识：出汗越多越燃脂 → 常识级谣言。"""
        response = _mock_chat_result(
            {
                "is_commonsense": True,
                "commonsense_type": "scientific_consensus",
                "llm_verdict": "谣言",
                "confidence": 0.95,
                "reasoning": "出汗量与脂肪燃烧无直接关系，出汗主要是体温调节机制。",
            }
        )
        with patch("src.truthnote.llm.chat", return_value=response):
            agent = CommonsenseCheckerAgent()
            claim = Claim(text="运动出汗越多越燃脂", category=RumorCategory.HEALTH)
            result = agent.check(claim)

            assert result["is_commonsense"] is True
            assert result["commonsense_type"] == "scientific_consensus"
            assert result["llm_verdict"] == "谣言"
            assert result["confidence"] >= 0.85
            assert "出汗" in result["reasoning"] or "燃烧" in result["reasoning"]

    def test_scientific_consensus_mostly_false(self):
        """科学共识：酸碱体质 → 大部分不实。"""
        response = _mock_chat_result(
            {
                "is_commonsense": True,
                "commonsense_type": "scientific_consensus",
                "llm_verdict": "大部分不实",
                "confidence": 0.90,
                "reasoning": "人体有精密的酸碱缓冲系统，'酸性体质'概念缺乏科学依据。",
            }
        )
        with patch("src.truthnote.llm.chat", return_value=response):
            agent = CommonsenseCheckerAgent()
            claim = Claim(text="酸性体质容易得癌症", category=RumorCategory.HEALTH)
            result = agent.check(claim)

            assert result["is_commonsense"] is True
            assert result["llm_verdict"] == "大部分不实"
            assert result["confidence"] >= 0.85

    def test_factual_history_commonsense(self):
        """事实性常识：历史事实错误 → 常识级。"""
        response = _mock_chat_result(
            {
                "is_commonsense": True,
                "commonsense_type": "factual_history",
                "llm_verdict": "谣言",
                "confidence": 0.98,
                "reasoning": "长城历经多个朝代修建，非秦始皇一人独建。",
            }
        )
        with patch("src.truthnote.llm.chat", return_value=response):
            agent = CommonsenseCheckerAgent()
            claim = Claim(text="长城是秦始皇一个人修的", category=RumorCategory.OTHER)
            result = agent.check(claim)

            assert result["is_commonsense"] is True
            assert result["commonsense_type"] == "factual_history"
            assert result["llm_verdict"] == "谣言"

    def test_not_commonsense_policy(self):
        """需要查政策的声明 → 非常识，走完整流水线。"""
        response = _mock_chat_result(
            {
                "is_commonsense": False,
                "commonsense_type": "n/a",
                "llm_verdict": None,
                "confidence": 0.3,
                "reasoning": "需要查最新税收政策，不是常识级问题。",
            }
        )
        with patch("src.truthnote.llm.chat", return_value=response):
            agent = CommonsenseCheckerAgent()
            claim = Claim(text="存款超5万要交税", category=RumorCategory.POLICY)
            result = agent.check(claim)

            assert result["is_commonsense"] is False
            assert result["llm_verdict"] is None

    def test_low_confidence_no_fast_path(self):
        """confidence 低于 0.85 → 虽然标记常识但不走快速路径。"""
        response = _mock_chat_result(
            {
                "is_commonsense": True,
                "commonsense_type": "scientific_consensus",
                "llm_verdict": "谣言",
                "confidence": 0.70,
                "reasoning": "可能是伪科学，但不完全确定。",
            }
        )
        with patch("src.truthnote.llm.chat", return_value=response):
            agent = CommonsenseCheckerAgent()
            claim = Claim(text="某种草药可以降血压", category=RumorCategory.HEALTH)
            result = agent.check(claim)

            # Agent 本身返回这些值，由 orchestrator 判断是否走快速路径
            assert result["is_commonsense"] is True
            assert result["confidence"] == 0.70
            # 在 orchestrator 中 0.70 < 0.85 不会触发快速路径

    def test_disallowed_verdict_sanitized(self):
        """LLM 返回不允许的 verdict（如"无法确定"）→ 安全降级为 None。"""
        response = _mock_chat_result(
            {
                "is_commonsense": True,
                "commonsense_type": "scientific_consensus",
                "llm_verdict": "无法确定",
                "confidence": 0.95,
                "reasoning": "这个说法不太确定。",
            }
        )
        with patch("src.truthnote.llm.chat", return_value=response):
            agent = CommonsenseCheckerAgent()
            claim = Claim(text="维生素C可以预防感冒", category=RumorCategory.HEALTH)
            result = agent.check(claim)

            # "无法确定" 不在允许列表，应被清空
            assert result["llm_verdict"] is None
            # verdict 被清空后，is_commonsense 也应降级为 False
            assert result["is_commonsense"] is False

    def test_llm_failure_safe_fallback(self):
        """LLM 调用完全失败 → 返回安全默认值。"""
        with patch("src.truthnote.llm.chat", side_effect=Exception("网络超时")):
            agent = CommonsenseCheckerAgent()
            claim = Claim(text="吃味精会致癌", category=RumorCategory.HEALTH)
            result = agent.check(claim)

            assert result["is_commonsense"] is False
            assert result["llm_verdict"] is None
            assert result["confidence"] == 0.0

    def test_controversial_science_not_commonsense(self):
        """有争议的科学问题（中医/传统养生）→ 非常识。"""
        response = _mock_chat_result(
            {
                "is_commonsense": False,
                "commonsense_type": "n/a",
                "llm_verdict": None,
                "confidence": 0.4,
                "reasoning": "生姜祛湿在中医理论中有依据，但现代医学有不同看法，存在学术分歧。",
            }
        )
        with patch("src.truthnote.llm.chat", return_value=response):
            agent = CommonsenseCheckerAgent()
            claim = Claim(text="喝生姜水可以祛湿气", category=RumorCategory.HEALTH)
            result = agent.check(claim)

            assert result["is_commonsense"] is False

    def test_commonsense_true_but_no_verdict_degrades(self):
        """is_commonsense=true 但 llm_verdict=null → 安全降级为非常识。"""
        response = _mock_chat_result(
            {
                "is_commonsense": True,
                "commonsense_type": "scientific_consensus",
                "llm_verdict": None,
                "confidence": 0.90,
                "reasoning": "是常识但无法给出判定。",
            }
        )
        with patch("src.truthnote.llm.chat", return_value=response):
            agent = CommonsenseCheckerAgent()
            claim = Claim(text="地球是平的", category=RumorCategory.OTHER)
            result = agent.check(claim)

            # 有 is_commonsense 但无 verdict → 降级
            assert result["is_commonsense"] is False

    def test_confidence_boundary_exactly_085(self):
        """confidence 刚好 0.85 → 在 orchestrator 中应满足阈值。"""
        response = _mock_chat_result(
            {
                "is_commonsense": True,
                "commonsense_type": "scientific_consensus",
                "llm_verdict": "谣言",
                "confidence": 0.85,
                "reasoning": "味精（谷氨酸钠）在正常用量下不致癌，WHO和FDA均认可其安全性。",
            }
        )
        with patch("src.truthnote.llm.chat", return_value=response):
            agent = CommonsenseCheckerAgent()
            claim = Claim(text="味精吃多了会致癌", category=RumorCategory.HEALTH)
            result = agent.check(claim)

            assert result["is_commonsense"] is True
            assert result["confidence"] == 0.85
            assert result["llm_verdict"] == "谣言"
            # 0.85 >= 0.85 → orchestrator 中会走快速路径

    def test_invalid_commonsense_type_sanitized(self):
        """commonsense_type 为无效值 → 归一化为 'n/a'。"""
        response = _mock_chat_result(
            {
                "is_commonsense": True,
                "commonsense_type": "random_garbage",
                "llm_verdict": "谣言",
                "confidence": 0.92,
                "reasoning": "科学共识。",
            }
        )
        with patch("src.truthnote.llm.chat", return_value=response):
            agent = CommonsenseCheckerAgent()
            claim = Claim(text="手机辐射致癌", category=RumorCategory.HEALTH)
            result = agent.check(claim)

            assert result["commonsense_type"] == "n/a"

    def test_confidence_as_percentage_normalized(self):
        """LLM 返回百分制 confidence（如 95）→ 归一化为 0.95。"""
        response = _mock_chat_result(
            {
                "is_commonsense": True,
                "commonsense_type": "scientific_consensus",
                "llm_verdict": "谣言",
                "confidence": 95,
                "reasoning": "血型决定性格没有科学依据。",
            }
        )
        with patch("src.truthnote.llm.chat", return_value=response):
            agent = CommonsenseCheckerAgent()
            claim = Claim(text="A型血的人性格比较内向", category=RumorCategory.HEALTH)
            result = agent.check(claim)

            assert result["confidence"] == 0.95  # _safe_confidence 处理百分制


class TestCommonsenseCheckerVerdictMapIntegration:
    """测试 CommonsenseChecker 输出与 VERDICT_MAP 的兼容性。"""

    def test_verdict_map_has_allowed_values(self):
        """确认 VERDICT_MAP 包含常识路径允许的 verdict 字符串。"""
        assert "谣言" in VERDICT_MAP
        assert VERDICT_MAP["谣言"] == Verdict.FALSE
        assert "大部分不实" in VERDICT_MAP
        assert VERDICT_MAP["大部分不实"] == Verdict.MOSTLY_FALSE

    def test_allowed_verdicts_map_to_correct_enums(self):
        """允许的 verdict 字符串 → 正确的枚举值。"""
        assert VERDICT_MAP["谣言"] in (Verdict.FALSE, Verdict.MOSTLY_FALSE)
        assert VERDICT_MAP["大部分不实"] in (Verdict.FALSE, Verdict.MOSTLY_FALSE)

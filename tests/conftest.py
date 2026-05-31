import json
from unittest.mock import AsyncMock, patch

import pytest

from src.truthnote.schemas import Evidence


@pytest.fixture
def mock_llm_pipeline():
    """模拟完整流水线的 LLM 调用序列。

    Agent 调用顺序（StructuredFactChecker 版）：
    0. ScenarioRouter
    1. ClaimExtractor
    2. QueryPlanner (via EvidenceHunter)
    3. EvidenceRanker
    4. StructuredFactChecker 步骤1：证据标注
    5. StructuredFactChecker 步骤2：关键事实核验
    6. Skeptic
    7. ResponseComposer
    """
    responses = [
        # 0. ScenarioRouter: 场景路由
        {
            "content": json.dumps(
                {
                    "scenario": "政策法规",
                    "confidence": 0.95,
                    "strategy_hint": "查政府官网",
                    "key_entities": ["存款", "交税"],
                },
                ensure_ascii=False,
            ),
            "tool_calls": [],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 0, "output_tokens": 0},
        },
        # 1. ClaimExtractor: 提取声明
        {
            "content": json.dumps(
                {
                    "claims": [
                        {
                            "text": "银行存款超过5万元要交20%的税",
                            "category": "政策法规",
                            "original_context": "紧急通知！",
                        }
                    ]
                },
                ensure_ascii=False,
            ),
            "tool_calls": [],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 0, "output_tokens": 0},
        },
        # 2. CheckWorthy: 核查价值过滤
        {
            "content": json.dumps(
                {
                    "results": [
                        {
                            "claim": "银行存款超过5万元要交20%的税",
                            "checkworthy": True,
                            "reason": "涉及具体政策和数字，可验证",
                        }
                    ]
                },
                ensure_ascii=False,
            ),
            "tool_calls": [],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 0, "output_tokens": 0},
        },
        # 2.5. AtomicFactExtractor: 原子化判断
        {
            "content": json.dumps(
                {
                    "should_atomize": False,
                    "atoms": [],
                    "atomization_risk": "low",
                    "reason": "单一政策声明，无需拆分",
                },
                ensure_ascii=False,
            ),
            "tool_calls": [],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 0, "output_tokens": 0},
        },
        # 3. EvidenceHunter: 搜索查询
        {
            "content": json.dumps(
                {
                    "queries": ["存款超过5万交税 辟谣", "银行存款税收新规 核查"],
                    "analysis": "该声明疑似政策类谣言",
                },
                ensure_ascii=False,
            ),
            "tool_calls": [],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 0, "output_tokens": 0},
        },
        # 3. EvidenceRanker: 排序+充分性
        {
            "content": json.dumps(
                {
                    "ranked_indices": [0, 1],
                    "sufficiency": "sufficient",
                    "reasoning": "两条权威来源一致辟谣",
                },
                ensure_ascii=False,
            ),
            "tool_calls": [],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 0, "output_tokens": 0},
        },
        # 4. StructuredFactChecker 步骤1：证据标注
        {
            "content": json.dumps(
                {
                    "labels": [
                        {"index": 0, "relation": "直接辟谣"},
                        {"index": 1, "relation": "直接辟谣"},
                    ]
                },
                ensure_ascii=False,
            ),
            "tool_calls": [],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 0, "output_tokens": 0},
        },
        # 5. StructuredFactChecker 步骤2：关键事实核验
        {
            "content": json.dumps(
                {
                    "key_facts": [
                        {"fact": "5万元", "status": "无原文"},
                        {"fact": "20%税", "status": "无原文"},
                    ],
                    "all_verified": False,
                },
                ensure_ascii=False,
            ),
            "tool_calls": [],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 0, "output_tokens": 0},
        },
        # 6. Skeptic: 质疑检验
        {
            "content": json.dumps(
                {
                    "challenges": ["是否有地区试点？"],
                    "passed": True,
                    "revised_verdict": None,
                },
                ensure_ascii=False,
            ),
            "tool_calls": [],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 0, "output_tokens": 0},
        },
        # 7. ResponseComposer: 生成回复
        {
            "content": json.dumps(
                {
                    "friendly_reply": (
                        "爸妈，这个消息我查了一下，目前没有这样的政策哦，咱们的存款是安全的～"
                    ),
                    "summary": (
                        "网传「存款超5万要交20%税」消息不实。"
                        "经核查，目前没有任何官方文件支持此说法。"
                    ),
                },
                ensure_ascii=False,
            ),
            "tool_calls": [],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 0, "output_tokens": 0},
        },
    ]

    call_count = {"n": 0}

    def fake_chat(messages, model=None, system=None, temperature=0, max_tokens=4096, provider=None):
        idx = call_count["n"]
        call_count["n"] += 1
        if idx < len(responses):
            return responses[idx]
        return responses[-1]

    mock_evidence = [
        Evidence(
            source="人民网",
            url="https://www.people.com.cn/example",
            title="官方辟谣：存款超5万交税系谣言",
            snippet="经核实，网传'银行存款超过5万元要交20%的税'不实。中国目前无任何对个人银行存款征收额外税款的政策。",
            credibility="权威媒体",
        ),
        Evidence(
            source="新华社",
            url="https://www.xinhuanet.com/example",
            title="辟谣：网传存款超5万交税不实",
            snippet="经核实，网传存款超过5万元要交税为虚假消息。个人储蓄存款利息税自2008年起已暂免征收。",
            credibility="国家通讯社",
        ),
    ]
    mock_searcher = AsyncMock()
    mock_searcher.search = AsyncMock(return_value=mock_evidence)

    with (
        patch("src.truthnote.llm.chat", side_effect=fake_chat),
        patch("src.truthnote.orchestrator.get_search_provider", return_value=mock_searcher),
        patch("src.truthnote.pipeline.get_search_provider", return_value=mock_searcher),
    ):
        yield call_count


@pytest.fixture
def mock_llm_empty():
    """模拟无声明的情况。"""
    response = {
        "content": json.dumps({"claims": []}, ensure_ascii=False),
        "tool_calls": [],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 0, "output_tokens": 0},
    }

    with patch("src.truthnote.llm.chat", return_value=response):
        yield

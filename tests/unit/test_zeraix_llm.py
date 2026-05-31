"""zeraix 助理接口适配器单测：聚焦 SSE 解析纯函数 _parse_zeraix_stream。"""

from __future__ import annotations

import json

import pytest

from src.truthnote import llm


def _data(event: dict) -> str:
    return "data: " + json.dumps(event, ensure_ascii=False)


@pytest.fixture(autouse=True)
def _silence_stream(monkeypatch):
    # 关掉 stdout 流式打印，保持测试输出干净
    monkeypatch.setattr(llm, "_STREAM_LLM", False)


def test_assembles_text_deltas_in_order():
    lines = [
        _data({"type": "start"}),
        _data({"type": "text-start", "payload": {"id": "txt-0"}}),
        _data({"type": "text-delta", "payload": {"id": "txt-0", "text": "你好"}}),
        _data({"type": "text-delta", "payload": {"id": "txt-0", "text": "，世界"}}),
        _data({"type": "text-end", "payload": {"id": "txt-0"}}),
        _data({"type": "finish"}),
    ]
    content, finish = llm._parse_zeraix_stream(iter(lines))
    assert content == "你好，世界"
    assert finish == "stop"


def test_ignores_non_data_lines_and_bad_json():
    lines = [
        "",
        ": keep-alive comment",
        "event: message",
        "data: not-json-at-all",
        _data({"type": "text-delta", "payload": {"text": "X"}}),
        "data: [DONE]",
        _data({"type": "text-delta", "payload": {"text": "SHOULD-NOT-APPEAR"}}),
    ]
    content, finish = llm._parse_zeraix_stream(iter(lines))
    assert content == "X"  # [DONE] 后的内容不计入


def test_empty_text_delta_skipped():
    lines = [
        _data({"type": "text-delta", "payload": {"id": "txt-0", "text": ""}}),
        _data({"type": "text-delta", "payload": {"id": "txt-0", "text": "ok"}}),
    ]
    content, _ = llm._parse_zeraix_stream(iter(lines))
    assert content == "ok"


def test_error_event_sets_finish_error():
    lines = [
        _data({"type": "text-delta", "payload": {"text": "部分"}}),
        _data({"type": "error", "message": "rate limited"}),
    ]
    content, finish = llm._parse_zeraix_stream(iter(lines))
    assert content == "部分"
    assert finish == "error"


def test_model_alias_resolution(monkeypatch):
    # 验证 zeraix/<model> 与裸 zeraix 的 provider 推断
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    assert llm._infer_provider("zeraix/qwen3.5-flash") == "zeraix"
    assert llm._infer_provider("zeraix") == "zeraix"
    # 非 zeraix 不受影响
    assert llm._infer_provider("qwen-max") == "openai"

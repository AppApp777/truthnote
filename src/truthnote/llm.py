"""LLM 抽象层。支持：Claude CLI（默认）、Anthropic API、OpenAI 兼容。

参考 agent-eval 的多提供商架构，统一接口返回 {content, tool_calls, stop_reason, usage}。
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv(Path(__file__).parents[2] / ".env")

logger = logging.getLogger(__name__)


def _get_provider() -> str:
    return os.getenv("LLM_PROVIDER", "claude_cli")


def _get_default_model() -> str | None:
    env_model = os.getenv("DEFAULT_MODEL")
    if env_model:
        return env_model
    from .config import settings

    return settings.default_model or None


def _call_claude_cli(
    messages: list[dict],
    system: str | None = None,
    temperature: float = 0,
    max_tokens: int = 4096,
    model: str | None = None,
) -> dict:
    import shutil

    parts = []
    for msg in messages:
        role = msg["role"]
        content = msg.get("content", "")
        if isinstance(content, list):
            texts = [
                b.get("text", b.get("content", str(b))) if isinstance(b, dict) else str(b)
                for b in content
            ]
            content = "\n".join(texts)
        if role == "user":
            parts.append(content)
        elif role == "assistant":
            parts.append(f"[Assistant]\n{content}")

    full_prompt = "\n\n".join(parts)

    claude_bin = shutil.which("claude")
    if not claude_bin:
        raise FileNotFoundError("claude CLI not found in PATH")

    cmd_args = [claude_bin, "-p", "--output-format", "text"]
    if model:
        cmd_args.extend(["--model", model])

    sys_path = None
    if system:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        ) as sf:
            sf.write(system)
            sys_path = sf.name
        cmd_args.extend(["--system-prompt-file", sys_path])

    isolated_cwd = tempfile.gettempdir()

    try:
        result = subprocess.run(
            cmd_args,
            input=full_prompt,
            capture_output=True,
            text=True,
            timeout=180,
            encoding="utf-8",
            cwd=isolated_cwd,
        )
        response_text = result.stdout.strip()
        if result.returncode != 0 and not response_text:
            response_text = f"[CLI Error] {result.stderr.strip()}"
    except subprocess.TimeoutExpired:
        response_text = "[CLI Error] Claude CLI timed out after 180s"
    except Exception as e:
        response_text = f"[CLI Error] {e}"
    finally:
        if sys_path:
            with contextlib.suppress(OSError):
                os.unlink(sys_path)

    return {
        "content": response_text,
        "tool_calls": [],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 0, "output_tokens": 0},
    }


def _call_anthropic(
    messages: list[dict],
    model: str,
    system: str | None = None,
    temperature: float = 0,
    max_tokens: int = 4096,
) -> dict:
    import anthropic

    # P1 #2 MEDIUM 修（adversarial 抓出）：anthropic.Anthropic() 默认 timeout ~600s，
    # ResponseComposer worker daemon thread 在 anthropic 路径仍会游离。统一套
    # httpx.Timeout(read=20)，与 OpenAI 兼容路径行为一致。
    client = anthropic.Anthropic(timeout=_build_http_timeout())
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if system:
        kwargs["system"] = system

    response = client.messages.create(**kwargs)

    result: dict[str, Any] = {"content": "", "tool_calls": []}
    for block in response.content:
        if block.type == "text":
            result["content"] += block.text
    result["stop_reason"] = response.stop_reason
    result["usage"] = {
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
    }
    return result


_MODEL_ENDPOINTS: dict[str, tuple[str, str]] = {
    "qwen": ("DASHSCOPE_API_KEY", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
    "glm": ("GLM_API_KEY", "https://open.bigmodel.cn/api/paas/v4"),
    "deepseek": ("DEEPSEEK_API_KEY", "https://api.deepseek.com"),
    "mimo": (
        "MIMO_API_KEY",
        os.getenv("MIMO_BASE_URL", "https://token-plan-cn.xiaomimimo.com/v1"),
    ),
    "360gpt": ("QIHOO_API_KEY", "https://api.360.cn/v1"),
    "longcat": ("LONGCAT_API_KEY", "https://api.longcat.chat/openai"),
    "anyrouter": ("ANYROUTER_API_KEY", "https://anyrouter.top/v1"),
    "ark": ("ARK_API_KEY", "https://ark.cn-beijing.volces.com/api/coding/v3"),
}


def _build_http_timeout():
    """构造 httpx.Timeout：stream 模式下 read 是「两个 chunk 之间最大空闲」语义。

    P1 #2 根治 ResponseComposer worker daemon thread 游离：
    - 旧 timeout=30.0 在 stream 模式只控「请求总时长 = 收到末 chunk 之前的总耗时」。
      服务端逐 chunk 慢吐字累计 30s 也不触发——主线兜底返回后 worker 继续跑。
    - 新 httpx.Timeout(read=20) 是 chunk-to-chunk idle 上限，超时即抛 ReadTimeout，
      worker 必定在 ~20s 内退出，不再游离。
    """
    import httpx

    return httpx.Timeout(connect=10.0, read=20.0, write=10.0, pool=10.0)


def _resolve_openai_client(model: str):
    from openai import OpenAI

    timeout = _build_http_timeout()

    # "anyrouter/claude-sonnet-4-5" → prefix="anyrouter", actual_model="claude-sonnet-4-5"
    prefix = model.split("/", 1)[0].lower() if "/" in model else None

    model_lower = model.lower()
    for ep_prefix, (key_env, base_url) in _MODEL_ENDPOINTS.items():
        if (prefix and prefix == ep_prefix) or (not prefix and model_lower.startswith(ep_prefix)):
            api_key = os.getenv(key_env)
            if api_key:
                return OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)
    return OpenAI(
        base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"), timeout=timeout
    )


def _call_vertex(
    messages: list[dict],
    model: str,
    system: str | None = None,
    temperature: float = 0,
    max_tokens: int = 4096,
) -> dict:
    from google import genai

    project = os.getenv("GCP_PROJECT_ID", "")
    location = os.getenv("GCP_LOCATION", "us-central1")
    if not project:
        raise ValueError("GCP_PROJECT_ID 未设置，无法使用 Vertex AI")

    client = genai.Client(vertexai=True, project=project, location=location)

    contents = []
    for m in messages:
        role = "user" if m["role"] == "user" else "model"
        text = m.get("content", "")
        if isinstance(text, list):
            text = "\n".join(b.get("text", str(b)) if isinstance(b, dict) else str(b) for b in text)
        contents.append({"role": role, "parts": [{"text": text or "(无内容)"}]})

    config = {
        "temperature": temperature,
        "max_output_tokens": max_tokens,
    }
    if system:
        config["system_instruction"] = system

    response = client.models.generate_content(
        model=model,
        contents=contents,
        config=config,
    )

    text_out = response.text or ""
    usage = response.usage_metadata
    return {
        "content": text_out,
        "tool_calls": [],
        "stop_reason": "end_turn",
        "usage": {
            "input_tokens": getattr(usage, "prompt_token_count", 0) or 0,
            "output_tokens": getattr(usage, "candidates_token_count", 0) or 0,
        },
    }


def _call_gemini_direct(
    messages: list[dict],
    model: str,
    system: str | None = None,
    temperature: float = 0,
    max_tokens: int = 4096,
) -> dict:
    """Google AI Studio 直连（API Key），不走 Vertex，速度更快。"""
    from google import genai

    api_key = os.getenv("GOOGLE_API_KEY", "")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY 未设置，无法使用 Gemini 直连")

    client = genai.Client(api_key=api_key)

    contents = []
    for m in messages:
        role = "user" if m["role"] == "user" else "model"
        text = m.get("content", "")
        if isinstance(text, list):
            text = "\n".join(b.get("text", str(b)) if isinstance(b, dict) else str(b) for b in text)
        contents.append({"role": role, "parts": [{"text": text or "(无内容)"}]})

    config = {
        "temperature": temperature,
        "max_output_tokens": max_tokens,
    }
    if system:
        config["system_instruction"] = system

    response = client.models.generate_content(
        model=model,
        contents=contents,
        config=config,
    )

    text_out = response.text or ""
    usage = response.usage_metadata
    return {
        "content": text_out,
        "tool_calls": [],
        "stop_reason": "end_turn",
        "usage": {
            "input_tokens": getattr(usage, "prompt_token_count", 0) or 0,
            "output_tokens": getattr(usage, "candidates_token_count", 0) or 0,
        },
    }


_STREAM_LLM = os.getenv("STREAM_LLM", "1").lower() not in ("0", "false", "no")


def _call_openai(
    messages: list[dict],
    model: str,
    system: str | None = None,
    temperature: float = 0,
    max_tokens: int = 4096,
) -> dict:
    client = _resolve_openai_client(model)
    api_model = model.split("/", 1)[1] if "/" in model else model
    full_messages = []
    if system:
        full_messages.append({"role": "system", "content": system})
    for m in messages:
        full_messages.append(m if m.get("content") else {**m, "content": "(无内容)"})

    try:
        if _STREAM_LLM:
            return _call_openai_stream(client, api_model, full_messages, max_tokens, temperature)
        response = client.chat.completions.create(
            model=api_model,
            messages=full_messages,
            max_tokens=max_tokens,
            temperature=max(temperature, 0.01),
        )
    except Exception as e:
        error_str = str(e)
        if "1301" in error_str or "contentFilter" in error_str or "不安全" in error_str:
            logger.warning("内容过滤器拦截: %s", error_str[:200])
            return {
                "content": "",
                "tool_calls": [],
                "stop_reason": "content_filter",
                "usage": {"input_tokens": 0, "output_tokens": 0},
            }
        raise
    msg = response.choices[0].message
    return {
        "content": msg.content or "",
        "tool_calls": [],
        "stop_reason": response.choices[0].finish_reason,
        "usage": {
            "input_tokens": response.usage.prompt_tokens if response.usage else 0,
            "output_tokens": response.usage.completion_tokens if response.usage else 0,
        },
    }


def _call_openai_stream(client, model, messages, max_tokens, temperature) -> dict:
    import sys

    stream = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=max(temperature, 0.01),
        stream=True,
        stream_options={"include_usage": True},
    )
    chunks = []
    finish_reason = None
    usage_data = None

    _pipeline_progress = None
    with contextlib.suppress(Exception):
        from .agents import _pipeline_progress

    for chunk in stream:
        if chunk.usage:
            usage_data = chunk.usage
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        if chunk.choices[0].finish_reason:
            finish_reason = chunk.choices[0].finish_reason
        if delta and delta.content:
            sys.stdout.write(delta.content)
            sys.stdout.flush()
            chunks.append(delta.content)
            if _pipeline_progress is not None:
                _pipeline_progress._push("llm_token", {"text": delta.content})
    sys.stdout.write("\n")
    sys.stdout.flush()
    content = "".join(chunks)
    return {
        "content": content,
        "tool_calls": [],
        "stop_reason": finish_reason or "stop",
        "usage": {
            "input_tokens": usage_data.prompt_tokens if usage_data else 0,
            "output_tokens": usage_data.completion_tokens if usage_data else 0,
        },
    }


# ── zeraix 个人助理接口适配（SSE 流 + JWT 滚动续期）─────────────────────
# 形态说明：这是 zeraix「个人助理」产品接口（非裸模型 / 非 OpenAI 兼容），
# 单条 message + threadId 维护服务端会话。为评测隔离，本适配器「不带 threadId」，
# 每次调用都让服务端开新会话，用例互不污染。凭证 JWT 24h 过期 + 每请求滚动续期，
# 自动从响应头 x-renewed-token 刷新落盘。
_ZERAIX_URL = os.getenv("ZERAIX_URL", "https://api.zeraix.com/api/personal-assistant/chat")
_ZERAIX_TOKEN_FILE = Path(__file__).parents[2] / ".zeraix_token"
_ZERAIX_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/148.0.0.0 Safari/537.36"
)


def _zeraix_token() -> str:
    """取 token：优先读滚动续期落盘的 sidecar 文件，回退到 .env 的 ZERAIX_TOKEN。"""
    with contextlib.suppress(OSError):
        if _ZERAIX_TOKEN_FILE.exists():
            tok = _ZERAIX_TOKEN_FILE.read_text(encoding="utf-8").strip()
            if tok:
                return tok
    return os.getenv("ZERAIX_TOKEN", "").strip()


def _zeraix_save_token(tok: str) -> None:
    """把服务端返回的 x-renewed-token 落盘，下次调用自动用最新的（解决 24h 过期）。"""
    if not tok:
        return
    with contextlib.suppress(OSError):
        _ZERAIX_TOKEN_FILE.write_text(tok.strip(), encoding="utf-8")


def _parse_zeraix_stream(line_iter) -> tuple[str, str]:
    """解析 zeraix SSE 流。line_iter 产出已解码的 str 行（每行形如 'data: {...}'）。

    事件：text-delta（payload.text 是内容 token）/ finish / error。
    返回 (content, stop_reason)。抽成纯函数便于单测。
    """
    chunks: list[str] = []
    finish = "stop"

    _pipeline_progress = None
    with contextlib.suppress(Exception):
        from .agents import _pipeline_progress

    for line in line_iter:
        if not line or not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if not data:
            continue
        if data == "[DONE]":
            break
        try:
            evt = json.loads(data)
        except json.JSONDecodeError:
            continue
        etype = evt.get("type")
        if etype == "text-delta":
            text = (evt.get("payload") or {}).get("text", "")
            if text:
                chunks.append(text)
                if _STREAM_LLM:
                    import sys

                    sys.stdout.write(text)
                    sys.stdout.flush()
                if _pipeline_progress is not None:
                    _pipeline_progress._push("llm_token", {"text": text})
        elif etype == "error":
            finish = "error"
            msg = evt.get("message") or (evt.get("payload") or {}).get("message", "")
            if msg:
                logger.warning("zeraix 流内错误: %s", str(msg)[:200])
        elif etype == "finish":
            finish = "stop"

    if _STREAM_LLM and chunks:
        import sys

        sys.stdout.write("\n")
        sys.stdout.flush()
    return "".join(chunks), finish


def _call_zeraix(
    messages: list[dict],
    model: str | None = None,
    system: str | None = None,
    temperature: float = 0,
    max_tokens: int = 4096,
) -> dict:
    """zeraix 个人助理接口（SSE）。无状态调用：不带 threadId → 每次新会话。"""
    import httpx

    token = _zeraix_token()
    if not token:
        raise ValueError("ZERAIX_TOKEN 未设置（.env 填 ZERAIX_TOKEN=<浏览器抓的 JWT>）")

    # 模型名：zeraix/qwen3.5-flash → qwen3.5-flash；裸 "zeraix" → 用 ZERAIX_MODEL 默认
    zmodel = model.split("/", 1)[1] if model and "/" in model else (model or "")
    if not zmodel or zmodel.lower() == "zeraix":
        zmodel = os.getenv("ZERAIX_MODEL", "qwen3.5-flash")

    # 扁平化 messages + system 成单条 message（接口是单 message + threadId 维护上下文）
    parts: list[str] = []
    if system:
        parts.append(system)
    for m in messages:
        content = m.get("content", "")
        if isinstance(content, list):
            content = "\n".join(
                b.get("text", str(b)) if isinstance(b, dict) else str(b) for b in content
            )
        if m.get("role") == "assistant":
            parts.append(f"[助手]\n{content}")
        else:
            parts.append(content)
    message = "\n\n".join(p for p in parts if p) or "(无内容)"

    body = {
        "message": message,
        "attachments": [],
        "imageModel": "default",
        "model": zmodel,
        "reasoningEffort": os.getenv("ZERAIX_REASONING", "none"),
        # 故意不带 threadId → 服务端开新会话，测试用例互不污染
    }
    headers = {
        "authorization": f"Bearer {token}",
        "content-type": "application/json",
        "accept": "text/event-stream",
        "origin": "https://zeraix.com",
        "referer": "https://zeraix.com/",
        "user-agent": _ZERAIX_UA,
    }
    proxy = os.getenv("ZERAIX_PROXY", "").strip() or None
    timeout = httpx.Timeout(connect=10.0, read=60.0, write=10.0, pool=10.0)

    client_kwargs: dict[str, Any] = {"timeout": timeout, "trust_env": False}
    if proxy:
        client_kwargs["proxy"] = proxy

    with (
        httpx.Client(**client_kwargs) as client,
        client.stream("POST", _ZERAIX_URL, headers=headers, json=body) as r,
    ):
        _zeraix_save_token(r.headers.get("x-renewed-token", ""))
        if r.status_code != 200:
            raw = r.read().decode("utf-8", "replace")[:300]
            raise RuntimeError(f"zeraix HTTP {r.status_code}: {raw}")
        r.encoding = "utf-8"  # 强制 utf-8，避免中文乱码（流默认解码会错）
        content, finish = _parse_zeraix_stream(r.iter_lines())

    return {
        "content": content,
        "tool_calls": [],
        "stop_reason": finish,
        "usage": {"input_tokens": 0, "output_tokens": 0},  # 接口不返回 token 计数
    }


def _infer_provider(model: str | None) -> str | None:
    # LLM_PROVIDER 显式设置时，优先使用（覆盖自动推断）
    explicit = os.getenv("LLM_PROVIDER", "").strip()
    if explicit:
        return explicit

    if not model:
        return None
    ml = model.lower()
    if ml == "zeraix" or ml.startswith(("zeraix/", "zeraix-")):
        return "zeraix"
    if "/" in model:
        return "openai"
    if model.startswith("claude-"):
        if os.getenv("ANTHROPIC_API_KEY"):
            return "anthropic"
        return "claude_cli"
    if model.startswith(("gpt-", "o1", "o3")):
        return "openai"
    if model.lower().startswith("gemini"):
        if os.getenv("GOOGLE_API_KEY"):
            return "gemini_direct"
        if os.getenv("GCP_PROJECT_ID"):
            return "vertex"
        return "openai"
    if model.lower().startswith(("qwen", "glm", "deepseek", "mimo", "360gpt", "longcat")):
        return "openai"
    return None


_THINK_RE = re.compile(r"<think>[\s\S]*?</think>\s*", re.IGNORECASE)


def _strip_artifacts(text: str) -> str:
    return _THINK_RE.sub("", text).strip()


def chat(
    messages: list[dict],
    model: str | None = None,
    system: str | None = None,
    temperature: float = 0,
    max_tokens: int = 4096,
    provider: str | None = None,
) -> dict:
    """统一调用接口。返回 {content, tool_calls, stop_reason, usage}。"""
    model = model or _get_default_model()
    p = provider or _infer_provider(model) or _get_provider()

    if p == "claude_cli":
        result = _call_claude_cli(messages, system, temperature, max_tokens, model=model)
    elif p == "anthropic":
        m = model or "claude-sonnet-4-6"
        result = _call_anthropic(messages, m, system, temperature, max_tokens)
    elif p == "gemini_direct":
        m = model or "gemini-2.0-flash"
        result = _call_gemini_direct(messages, m, system, temperature, max_tokens)
    elif p == "vertex":
        m = model or "gemini-2.0-flash"
        result = _call_vertex(messages, m, system, temperature, max_tokens)
    elif p == "zeraix":
        result = _call_zeraix(messages, model, system, temperature, max_tokens)
    else:
        m = model or "gpt-4o"
        result = _call_openai(messages, m, system, temperature, max_tokens)

    if result.get("content"):
        result["content"] = _strip_artifacts(result["content"])
    return result


def chat_text(
    prompt: str,
    model: str | None = None,
    system: str | None = None,
    temperature: float = 0,
    max_tokens: int = 4096,
) -> str:
    """简单的 text-in text-out 辅助函数。"""
    result = chat(
        messages=[{"role": "user", "content": prompt}],
        model=model,
        system=system,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return result["content"]


def chat_json(
    prompt: str,
    model: str | None = None,
    system: str | None = None,
) -> dict:
    """调用 LLM 并解析 JSON 返回。"""
    text = chat_text(prompt, model=model, system=system)
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*\n(.*?)```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        text = text[start : end + 1]
    return json.loads(text)


def get_embedding(text: str, model: str | None = None) -> list[float]:
    """获取文本 embedding 向量。支持 DashScope / OpenAI 兼容接口。"""
    model = model or os.getenv("EMBEDDING_MODEL", "text-embedding-v3")
    api_key = os.getenv("DASHSCOPE_API_KEY") or os.getenv("OPENAI_API_KEY", "")
    base_url = os.getenv(
        "EMBEDDING_BASE_URL",
        "https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    if not api_key:
        return []

    import urllib.request

    url = f"{base_url.rstrip('/')}/embeddings"
    payload = json.dumps({"model": model, "input": text}).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data["data"][0]["embedding"]
    except Exception:
        return []

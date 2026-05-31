"""
TruthNote MCP Server —— 把已有的 /api/verify 核查引擎包成一个标准 MCP 工具。

【这是什么】
一层薄适配器:不改核心引擎,只通过 HTTP 转调它,对外暴露一个标准 MCP 工具
`verify_truth`。任何支持 MCP 的客户端(Claude Desktop / Cherry Studio /
ChatWise 等)挂上这个 server,就能在对话里核查任意中文内容真伪。

────────────────────────────────────────────────────────
依赖:
    pip install "mcp[cli]" requests

跑之前:先把后端引擎跑起来
    python run.py                 # 默认 http://localhost:8000

启动本 server(三选一,看客户端要哪种传输):
    python mcp_server.py          # stdio   —— Claude Desktop / Cherry Studio 默认
    python mcp_server.py --http   # streamable-http —— 填地址的 MCP 客户端,默认 127.0.0.1:8765/mcp
    python mcp_server.py --sse    # sse     —— 旧客户端兜底,默认 127.0.0.1:8765/sse

环境变量(都可不填,用默认):
    TRUTHNOTE_ENGINE   引擎地址,默认 http://localhost:8000
                       ⚠️ 若 config.app_port 不是 8000,改这个对齐
    TRUTHNOTE_TIMEOUT  单次核查超时秒数,默认 120
    MCP_HOST / MCP_PORT  http/sse 传输的监听地址,默认 127.0.0.1:8765
────────────────────────────────────────────────────────
"""

from __future__ import annotations

import os
import sys

import requests
from mcp.server.fastmcp import FastMCP

ENGINE = os.getenv("TRUTHNOTE_ENGINE", "http://localhost:8000").rstrip("/")
TIMEOUT = float(os.getenv("TRUTHNOTE_TIMEOUT", "120"))

mcp = FastMCP("truthnote")


@mcp.tool()
def verify_truth(message: str, context: str = "") -> dict:
    """核查一段中文内容/群聊消息的真伪。

    什么时候用:用户在问"这是真的吗 / 这条消息可信吗 / 这是不是谣言",
    或拿到一段可疑的紧急通知、保健品宣传、养老理财话术、政策传言时。

    参数:
        message: 需要核查的内容原文(必填)
        context: 可选,消息来源或背景(如 "家庭群转发" "某公众号文章")

    返回:
        verdict           —— 真 / 假 / 存疑 的倾向判定
        summary           —— 核查结论摘要
        friendly_reply    —— 可直接转发给长辈的温和辟谣话术
        evidence_sources  —— 证据来源链接列表
        claim_count       —— 拆出的独立声明条数
    """
    if not message or not message.strip():
        return {"error": "message 不能为空"}

    try:
        resp = requests.post(
            f"{ENGINE}/api/verify",
            json={"message": message, "context": context},
            timeout=TIMEOUT,
        )
    except requests.exceptions.ConnectionError:
        return {
            "error": f"连不上核查引擎 {ENGINE},请先启动后端(python run.py),或检查 TRUTHNOTE_ENGINE"
        }
    except requests.exceptions.Timeout:
        return {"error": f"核查超时(>{TIMEOUT}s),请稍后重试或缩短内容"}

    if resp.status_code != 200:
        return {"error": f"引擎返回 {resp.status_code}: {resp.text[:200]}"}

    data = resp.json()
    return {
        "verdict": data.get("overall_verdict"),
        "summary": data.get("summary", ""),
        "friendly_reply": data.get("friendly_reply", ""),
        "evidence_sources": data.get("evidence_sources", []),
        "claim_count": len(data.get("claims") or []),
    }


def _run_networked(transport: str) -> None:
    """http / sse 这类需要监听端口的传输,设好 host/port 再跑。"""
    host = os.getenv("MCP_HOST", "127.0.0.1")
    port = int(os.getenv("MCP_PORT", "8765"))
    # FastMCP 用 settings 持有 host/port(避开引擎的 8000 端口)
    mcp.settings.host = host
    mcp.settings.port = port
    path = "/mcp" if transport == "streamable-http" else "/sse"
    print(f"[truthnote-mcp] {transport} on http://{host}:{port}{path}", file=sys.stderr)
    mcp.run(transport=transport)


if __name__ == "__main__":
    if "--http" in sys.argv:
        _run_networked("streamable-http")  # 填地址的 MCP 客户端
    elif "--sse" in sys.argv:
        _run_networked("sse")  # 旧客户端兜底
    else:
        print("[truthnote-mcp] stdio transport (Claude Desktop / Cherry Studio)", file=sys.stderr)
        mcp.run(transport="stdio")

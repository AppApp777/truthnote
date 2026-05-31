"""图片多模态模块：提取图片中的文字/描述，送入文本核查流水线。

支持两种模式：
1. 通义千问 VL（DASHSCOPE_API_KEY）— 优先
2. anyrouter 路由到支持视觉的模型 — 兜底
"""

from __future__ import annotations

import base64
import logging
import os
from pathlib import Path

from . import llm

logger = logging.getLogger(__name__)

_VL_MODEL = os.getenv("VL_MODEL", "qwen-vl-max")

_EXTRACT_SYSTEM = (
    "你是图片内容提取专家。分析图片并提取所有可核查的文字信息。\n\n"
    "## 任务\n"
    "1. 提取图片中所有文字（OCR）\n"
    "2. 描述图片中的关键视觉元素（人物、场景、数据图表等）\n"
    "3. 识别图片类型：新闻截图/聊天记录/公告通知/数据图表/社交媒体帖子/其他\n\n"
    "## 输出格式\n"
    "直接输出提取的文字内容，用自然语言描述。不要用 JSON。\n"
    "如果图片包含聊天记录，按顺序还原对话。\n"
    "如果图片包含公告/通知，还原完整文字。\n"
    "如果图片是数据图表，描述关键数据点。"
)


def _encode_image_base64(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _get_mime_type(path: str) -> str:
    ext = Path(path).suffix.lower()
    return {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".gif": "image/gif",
        ".webp": "image/webp",
        ".bmp": "image/bmp",
    }.get(ext, "image/jpeg")


def extract_text_from_image(
    image_source: str,
    model: str | None = None,
) -> str:
    """从图片提取文字内容。

    Args:
        image_source: 图片路径（本地文件）或 URL
        model: VL 模型名，默认从 VL_MODEL 环境变量读取

    Returns:
        提取的文字内容描述
    """
    model = model or _VL_MODEL

    if image_source.startswith(("http://", "https://")):
        image_content = {"type": "image_url", "image_url": {"url": image_source}}
    elif os.path.isfile(image_source):
        b64 = _encode_image_base64(image_source)
        mime = _get_mime_type(image_source)
        image_content = {
            "type": "image_url",
            "image_url": {"url": f"data:{mime};base64,{b64}"},
        }
    else:
        logger.warning("[Image] 无效的图片源: %s", image_source[:60])
        return ""

    messages = [
        {
            "role": "user",
            "content": [
                image_content,
                {"type": "text", "text": "请提取并描述这张图片中的所有文字和关键信息。"},
            ],
        }
    ]

    try:
        result = llm.chat(
            messages=messages,
            model=model,
            system=_EXTRACT_SYSTEM,
            temperature=0,
        )
        text = result.get("content", "").strip()
        logger.info("[Image] 提取到 %d 字文本", len(text))
        return text
    except Exception as e:
        logger.warning("[Image] VL 模型调用失败: %s", e)
        return ""


def verify_image(
    image_source: str,
    context: str = "",
    model: str | None = None,
) -> dict:
    """图片核查入口：提取文字 → 送入文本核查流水线。

    Returns:
        dict with keys: extracted_text, verify_result
    """
    from .pipeline import verify_message

    extracted = extract_text_from_image(image_source, model=model)
    if not extracted:
        return {
            "extracted_text": "",
            "verify_result": None,
            "error": "无法从图片中提取文字内容",
        }

    combined_context = f"[图片内容提取] {context}" if context else "[图片内容提取]"
    result = verify_message(extracted, context=combined_context)

    return {
        "extracted_text": extracted,
        "verify_result": result,
    }

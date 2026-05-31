"""图片溯源（旧图新用 / 张冠李戴反查）。

设计要点（副线已想清楚，主力直接用）：
- **双后端**：国内走百度识图（免费、逆向、覆盖中文圈，已实测）；国外走 TinEye（官方 API、
  稳定、付费、需 key，待接桩）。provider="auto" 先百度、没找到再 TinEye。demo 用 baidu。
- **缓存优先**：命中缓存瞬间返回、不联网——这是 demo 台上零风险的关键。
- 实时路径（百度）走逆向接口，脆弱、无 SLA，失败**永不抛异常**，返回 found=False。
- 只做"找出这张图在全网哪些网页出现过"，**不判真假**（溯源 ≠ 判定，overall_verdict 保持中性）。
- **诚实红线**：返回的是"已知出现过的网页"，不等于"首发时间"——文案里写死这句话。
- **SSRF 防护**：image_url 来自网页（外部输入），下载前校验，拒私网/环回/非 http(s)。

依赖：pip install PicImageSearch（百度识图逆向流封装）。本模块顶层不 import 它，
缺库时模块仍可导入，调用时优雅失败。

放置位置：src/truthnote/reverse_image.py
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import ipaddress
import json
import logging
import os
import socket
import tempfile
from pathlib import Path
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

_CACHE_PATH = Path(os.getenv("REVERSE_IMAGE_CACHE", "data/reverse_image_cache.json"))
_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)
_DOWNLOAD_TIMEOUT = 20.0
_MAX_BYTES = 8 * 1024 * 1024
_MIN_BYTES = 256  # 小于这个多半是错误页 / 防盗链 JSON
_TOP_N = 8

# dev 开关：仅跳过"主机名 DNS 解析后是否为内网"的校验。
# 适用于 透明代理把所有域名解析到内网/保留段 的开发环境（否则会误杀真实公网图）。
# 即使开了，IP 字面量内网地址 / localhost 仍然永远拦死。生产环境务必保持关闭（默认）。
_ALLOW_INSECURE = os.getenv("REVERSE_IMAGE_ALLOW_INSECURE", "").lower() in ("1", "true", "yes")


# ---------------- 缓存（demo 台上零风险的命脉） ----------------
def _empty_cache() -> dict:
    return {"by_url": {}, "by_hash": {}}


def _load_cache() -> dict:
    try:
        if _CACHE_PATH.is_file():
            data = json.loads(_CACHE_PATH.read_text(encoding="utf-8"))
            data.setdefault("by_url", {})
            data.setdefault("by_hash", {})
            return data
    except Exception as e:  # noqa: BLE001 — 缓存坏了也不能拖垮核查
        logger.warning("[ReverseImage] 读缓存失败：%s", e)
    return _empty_cache()


def _save_cache(cache: dict) -> None:
    try:
        _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        _CACHE_PATH.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as e:  # noqa: BLE001
        logger.warning("[ReverseImage] 写缓存失败：%s", e)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# ---------------- SSRF 防护 + 下载 ----------------
def _is_public_ip(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    return not (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_reserved
        or ip.is_multicast
        or ip.is_unspecified
    )


def _is_safe_public_url(url: str) -> bool:
    """只允许 http(s)，且不指向内网/环回/链路本地，防 SSRF。

    分层校验（越靠前越硬，永不被开关绕过）：
    1. scheme 必须是 http/https；
    2. host 是 IP 字面量 → 直接判公网性（内网 IP 字面量永远拒，含 insecure 模式）；
    3. localhost / *.local / *.internal 等明文内网名 → 永远拒；
    4. 普通主机名 → DNS 解析后逐个 IP 校验；
       - dev 开关 _ALLOW_INSECURE 仅跳过第 4 步（应对透明代理把域名解析到保留段的环境）；
       - 解析失败 → 保守拒绝。

    注意：第 4 步在「下载前」解析，httpx 实际请求会再解析一次，存在 DNS rebinding 的
    理论 TOCTOU 窗口。黑客松可接受；生产应改用固定 resolver + 锁定 IP 直连。
    """
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        return False
    host = (parsed.hostname or "").strip()
    if not host:
        return False

    # IP 字面量：内网/环回永远拒（即使 insecure 模式）
    try:
        return _is_public_ip(ipaddress.ip_address(host))
    except ValueError:
        pass  # 不是 IP，继续按主机名处理

    low = host.lower()
    if low == "localhost" or low.endswith((".local", ".internal", ".localdomain")):
        return False

    if _ALLOW_INSECURE:
        return True

    try:
        infos = socket.getaddrinfo(host, None)
    except Exception:  # noqa: BLE001 — 解析不了，保守拒绝
        return False
    return all(_is_public_ip(ipaddress.ip_address(i[4][0])) for i in infos)


def _download_image(url: str) -> bytes | None:
    """下载图片字节。带 UA、大小上限、SSRF 校验。永不抛。"""
    if not _is_safe_public_url(url):
        logger.warning("[ReverseImage] URL 不安全或非公网，拒下载：%s", url[:80])
        return None
    try:
        import httpx

        # follow_redirects=False：防 SSRF 重定向绕过——恶意外站可 302 跳到内网/云元数据，
        # 而预校验只验了原始 URL。demo 走缓存不受影响；live 偶有 CDN 跳转失败可接受（安全优先）。
        with httpx.Client(
            timeout=_DOWNLOAD_TIMEOUT,
            follow_redirects=False,
            headers={"User-Agent": _UA},
        ) as client:
            resp = client.get(url)
            resp.raise_for_status()
            data = resp.content
        if len(data) > _MAX_BYTES:
            logger.warning("[ReverseImage] 图过大(%d B)，跳过：%s", len(data), url[:80])
            return None
        if len(data) < _MIN_BYTES:
            logger.warning(
                "[ReverseImage] 下载内容过小(%d B)，疑似错误页/防盗链：%s",
                len(data),
                url[:80],
            )
            return None
        return data
    except Exception as e:  # noqa: BLE001
        logger.warning("[ReverseImage] 下载失败：%s（%s）", e, url[:80])
        return None


# ---------------- 百度识图（逆向，经 PicImageSearch） ----------------
async def _baidu_search_async(image_path: str) -> tuple[list[dict], str]:
    from PicImageSearch import BaiDu, Network

    async with Network() as client:
        baidu = BaiDu(client=client)
        resp = await baidu.search(file=image_path)
    result_url = getattr(resp, "url", "") or ""
    pages: list[dict] = []
    for it in getattr(resp, "raw", []) or []:
        url = getattr(it, "url", "") or ""
        if not url:
            continue
        pages.append(
            {
                "url": url,
                # 注：百度改版后 title 常为空（PicImageSearch 已 deprecated），属正常
                "title": (getattr(it, "title", "") or "").strip(),
                "domain": _domain_of(url),
            }
        )
    return pages, result_url


def _domain_of(url: str) -> str:
    try:
        host = urlparse(url).hostname or ""
        return host[4:] if host.startswith("www.") else host
    except Exception:  # noqa: BLE001
        return ""


def _run_async_blocking(coro):
    """在同步上下文里跑协程，兼容"已有运行中事件循环"的环境（对齐 agents.py 的模式）。

    FastAPI 同步路由当前跑在线程池里（无运行中循环），asyncio.run 可直接用；
    但若日后改 async 路由或调度变化，裸 asyncio.run 会抛 RuntimeError。这里探测后再决定。
    """
    import concurrent.futures

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop and loop.is_running():
        with concurrent.futures.ThreadPoolExecutor() as pool:
            return pool.submit(asyncio.run, coro).result(timeout=_DOWNLOAD_TIMEOUT + 10)
    return asyncio.run(coro)


def _run_baidu(image_bytes: bytes) -> dict:
    """国内反查：实时调百度识图（逆向）。永不抛。"""
    tmp_path = ""
    try:
        with tempfile.NamedTemporaryFile(suffix=".img", delete=False) as f:
            f.write(image_bytes)
            tmp_path = f.name
        pages, result_url = _run_async_blocking(_baidu_search_async(tmp_path))
    except Exception as e:  # noqa: BLE001 — 逆向接口随时可能崩，不能拖垮请求
        logger.warning("[ReverseImage] 百度识图调用失败：%s", e)
        return {
            "found": False,
            "source_pages": [],
            "total": 0,
            "error": f"reverse_search_failed:{type(e).__name__}",
            "result_page": "",
        }
    finally:
        if tmp_path:
            with contextlib.suppress(OSError):
                os.unlink(tmp_path)

    # 裸 /upload 结果页 = 上传被拒（无效图），区别于"真搜了但没找到"（结果页是 /s?...）
    if not pages and result_url.rstrip("/").endswith("/upload"):
        return {
            "found": False,
            "source_pages": [],
            "total": 0,
            "error": "upload_rejected_invalid_image",
            "result_page": result_url,
        }

    return {
        "found": bool(pages),
        "source_pages": pages[:_TOP_N],
        "total": len(pages),
        "error": "",
        "result_page": result_url,
    }


# ---------------- TinEye（国外·官方 API，待接桩） ----------------
# 国外/全球流传的图走 TinEye 官方 API（稳定、付费、需 key），与国内百度形成双后端。
# 没配 TINEYE_API_KEY 时 fail-soft 返回 not_configured，auto 模式自动退回百度——
# 所以 demo（只用百度）完全不受影响。主力拿到 key 后按下面契约把实现核对/补全即可。
# 接入契约：pip install pytineye；官方 REST POST 图片字节；用 key 鉴权；
#   sort=crawl_date asc 取最早收录；每个 match.backlinks[].backlink 即来源网页 URL。
#   文档：https://services.tineye.com/TinEyeAPI
_TINEYE_API_KEY = os.getenv("TINEYE_API_KEY", "")


def _tineye_configured() -> bool:
    return bool(_TINEYE_API_KEY)


def _run_tineye(image_bytes: bytes) -> dict:
    """国外反查：TinEye 官方 API。⚠️未实测（无 key），主力接入时核对 pytineye 版本 API。永不抛。"""
    if not _tineye_configured():
        return {
            "found": False,
            "source_pages": [],
            "total": 0,
            "error": "tineye_not_configured",
            "result_page": "",
        }
    try:
        from pytineye import TinEyeAPIRequest  # type: ignore

        api = TinEyeAPIRequest(api_key=_TINEYE_API_KEY)
        resp = api.search_data(image_bytes, sort="crawl_date", order="asc")
        pages: list[dict] = []
        for m in getattr(resp, "matches", []) or []:
            for bl in getattr(m, "backlinks", []) or []:
                url = getattr(bl, "backlink", "") or getattr(bl, "url", "") or ""
                if url:
                    pages.append({"url": url, "title": "", "domain": _domain_of(url)})
        return {
            "found": bool(pages),
            "source_pages": pages[:_TOP_N],
            "total": len(pages),
            "error": "",
            "result_page": "tineye",
        }
    except Exception as e:  # noqa: BLE001 — 未实测/网络/版本差异都不能拖垮请求
        logger.warning("[ReverseImage] TinEye 调用失败：%s", e)
        return {
            "found": False,
            "source_pages": [],
            "total": 0,
            "error": f"tineye_failed:{type(e).__name__}",
            "result_page": "",
        }


def _run_live(image_bytes: bytes, provider: str = "baidu") -> dict:
    """按 provider 实时反查。永不抛。

    provider:
      - "baidu"（默认，国内，免费逆向）
      - "tineye"（国外，官方 API，需 key）
      - "auto"（先百度；百度没找到且配了 TinEye key → 再试 TinEye，覆盖国外流传图）
    """
    if provider == "tineye":
        return _run_tineye(image_bytes)
    result = _run_baidu(image_bytes)
    if provider == "auto" and not result.get("found") and _tineye_configured():
        ty = _run_tineye(image_bytes)
        if ty.get("found"):
            return ty
    return result


# ---------------- 对外主入口 ----------------
def reverse_search_image(
    image_url: str = "",
    image_bytes: bytes | None = None,
    *,
    provider: str = "baidu",
    use_cache: bool = True,
    allow_live: bool = True,
) -> dict:
    """图片溯源主入口。

    Args:
        image_url: 图片 URL（右键"复制图片地址"得到的 srcUrl）
        image_bytes: 或直接给图片字节（与 image_url 二选一）
        provider: 反查后端 —— "baidu"（国内，默认，免费）/ "tineye"（国外，官方API，需key）/
                  "auto"（先百度，没找到再试 TinEye）。demo 用 baidu。
        use_cache: 是否查/写缓存（demo 务必 True）
        allow_live: 缓存未命中时是否允许实时联网
                    （demo 台上想绝对保险，可设 False 强制只走缓存）

    Returns dict:
        found(bool) / source_pages(list[{url,domain,title}]) / total(int) /
        from_cache(bool) / error(str) / result_page(str) / query_key(str)

    注：缓存按 图URL/字节哈希 索引，不区分 provider（demo 缓存是百度结果）。
    """
    cache = _load_cache() if use_cache else _empty_cache()

    # 1) 按 URL 命中缓存（最快：右键 srcUrl 直接命中，无需下载、无需联网）
    if use_cache and image_url and image_url in cache["by_url"]:
        hit = dict(cache["by_url"][image_url])
        hit["from_cache"] = True
        hit["query_key"] = f"url:{image_url[:60]}"
        return hit

    # 2) 拿到字节（下载或直接给）
    data = image_bytes
    if data is None and image_url:
        # allow_live=False：连"下载图片"这一次联网都不做——demo 台上零联网的硬保证。
        # （by_url 缓存已在步骤 1 查过；没字节就算不出哈希，无法再查 by_hash，直接判 miss）
        if not allow_live:
            return {
                "found": False,
                "source_pages": [],
                "total": 0,
                "from_cache": False,
                "error": "cache_miss_live_disabled",
                "result_page": "",
                "query_key": "",
            }
        data = _download_image(image_url)
    if data is None:
        return {
            "found": False,
            "source_pages": [],
            "total": 0,
            "from_cache": False,
            "error": "no_image_bytes",
            "result_page": "",
            "query_key": "",
        }

    h = _sha256(data)

    # 3) 按字节哈希命中缓存（同图不同 URL 也能命中）
    if use_cache and h in cache["by_hash"]:
        hit = dict(cache["by_hash"][h])
        hit["from_cache"] = True
        hit["query_key"] = f"hash:{h[:12]}"
        return hit

    # 4) 缓存全 miss
    if not allow_live:
        return {
            "found": False,
            "source_pages": [],
            "total": 0,
            "from_cache": False,
            "error": "cache_miss_live_disabled",
            "result_page": "",
            "query_key": f"hash:{h[:12]}",
        }

    # 5) 实时反查（按 provider：百度/TinEye/auto）
    result = _run_live(data, provider)
    result["from_cache"] = False
    result["query_key"] = f"hash:{h[:12]}"

    # 6) 回写缓存（URL + 哈希 双索引）
    if use_cache:
        store = {k: result[k] for k in ("found", "source_pages", "total", "error", "result_page")}
        cache["by_hash"][h] = store
        if image_url:
            cache["by_url"][image_url] = store
        _save_cache(cache)

    return result


# ---------------- 塑成"推理卡片"可渲染的对象（content.js SHOW_RESULT 直接吃） ----------------
def build_source_card(result: dict, image_url: str = "") -> dict:
    """把溯源结果塑成 content.js 能渲染的对象。

    复用现有卡片：来源网页放进 claims[0].evidence[]，
    content.js 的 renderEvidence 直接渲染成"证据来源"链接区，零改渲染层。
    """
    pages = result.get("source_pages", [])
    total = result.get("total", 0)

    evidence = [
        {
            "source": p.get("domain") or p.get("url", ""),
            "url": p["url"],
            "source_url": p["url"],  # content.js 优先读 source_url
            "title": p.get("title") or p.get("domain") or p["url"],
            "snippet": "",
            "credibility": "未评估",
        }
        for p in pages
    ]

    if not result.get("found"):
        err = result.get("error", "")
        if err == "upload_rejected_invalid_image":
            summary = "无法解析这张图（可能不是有效图片或下载失败），未能溯源。"
        elif err.startswith(("reverse_search", "tineye")):
            summary = "图片溯源服务暂时不可用（溯源后端未返回结果），稍后重试。"
        elif err == "no_image_bytes":
            summary = "拿不到这张图（URL 不可达、被防盗链、或非公网地址），无法溯源。"
        else:
            summary = "未在全网检索到这张图的其他出处——可能是原创图，或检索引擎尚未收录。"
        return {
            "overall_verdict": "无法核实",
            "friendly_reply": "这张图没查到全网其他出处，无法据此判断是否为旧图新用。",
            "summary": summary,
            "claims": [],
            "original_message": f"[图片溯源] {image_url}",
            "image_source": result,
        }

    summary = (
        f"这张图已知在全网至少 {total} 个网页出现过。"
        "⚠️ 以图搜图给出的是『已知出现过的网页』，不等于首发时间——"
        "请据此人工核对：该图是否被挪用到了不相干的事件上（旧图新用 / 张冠李戴）。"
    )
    reply = (
        f"我在全网找到这张图的 {total} 处出处（见下方证据来源）。"
        "若这些出处指向的事件/时间与当前网页声称的不一致，就很可能是旧图新用。"
    )
    return {
        "overall_verdict": "无法核实",  # 溯源不直接判真假，保持中性
        "friendly_reply": reply,
        "summary": summary,
        "claims": [
            {
                "text": f"图片溯源：全网发现 {total} 处出处",
                "verdict": "—",
                "evidence": evidence,
            }
        ],
        "original_message": f"[图片溯源] {image_url}",
        "image_source": result,
    }

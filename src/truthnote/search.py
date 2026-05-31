from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
import sqlite3
import threading
import time
from abc import ABC, abstractmethod
from pathlib import Path

import httpx

from .config import settings
from .schemas import Evidence, SourceType

logger = logging.getLogger(__name__)

# ── source_type 域名映射 + authority_score 计算 ──

DOMAIN_SOURCE_TYPE: dict[str, SourceType] = {
    "gov.cn": SourceType.OFFICIAL_GOVERNMENT,
    "gov.hk": SourceType.OFFICIAL_GOVERNMENT,
    "gov.mo": SourceType.OFFICIAL_GOVERNMENT,
    "nhc.gov.cn": SourceType.OFFICIAL_GOVERNMENT,
    "who.int": SourceType.OFFICIAL_GOVERNMENT,
    "mof.gov.cn": SourceType.OFFICIAL_GOVERNMENT,
    "pbc.gov.cn": SourceType.REGULATOR,
    "csrc.gov.cn": SourceType.REGULATOR,
    "cbirc.gov.cn": SourceType.REGULATOR,
    "samr.gov.cn": SourceType.REGULATOR,
    "piyao.org.cn": SourceType.FACT_CHECK_ORG,
    "factcheck.org": SourceType.FACT_CHECK_ORG,
    "snopes.com": SourceType.FACT_CHECK_ORG,
    "reuters.com": SourceType.ESTABLISHED_MEDIA,
    "xinhuanet.com": SourceType.ESTABLISHED_MEDIA,
    "people.com.cn": SourceType.ESTABLISHED_MEDIA,
    "cctv.com": SourceType.ESTABLISHED_MEDIA,
    "chinanews.com": SourceType.ESTABLISHED_MEDIA,
    "thepaper.cn": SourceType.ESTABLISHED_MEDIA,
    "bjnews.com.cn": SourceType.ESTABLISHED_MEDIA,
    "caixin.com": SourceType.ESTABLISHED_MEDIA,
    "wikipedia.org": SourceType.ENCYCLOPEDIA,
    "baike.baidu.com": SourceType.ENCYCLOPEDIA,
    "weibo.com": SourceType.SOCIAL_MEDIA,
    "weixin.qq.com": SourceType.SOCIAL_MEDIA,
    "mp.weixin.qq.com": SourceType.SOCIAL_MEDIA,
    "zhihu.com": SourceType.BLOG_FORUM,
    "douban.com": SourceType.BLOG_FORUM,
    "tieba.baidu.com": SourceType.BLOG_FORUM,
    "toutiao.com": SourceType.SOCIAL_MEDIA,
}

SOURCE_TYPE_AUTHORITY: dict[SourceType, float] = {
    SourceType.OFFICIAL_GOVERNMENT: 0.95,
    SourceType.REGULATOR: 0.90,
    SourceType.FACT_CHECK_ORG: 0.90,
    SourceType.HOSPITAL_MEDICAL: 0.85,
    SourceType.ACADEMIC: 0.80,
    SourceType.ESTABLISHED_MEDIA: 0.75,
    SourceType.ENCYCLOPEDIA: 0.70,
    SourceType.BLOG_FORUM: 0.30,
    SourceType.SOCIAL_MEDIA: 0.20,
    SourceType.UNKNOWN: 0.40,
}


def classify_source(url: str) -> tuple[SourceType, float]:
    """根据 URL 域名返回 (source_type, authority_score)。"""
    if not url or "/" not in url:
        return SourceType.UNKNOWN, 0.40
    try:
        domain = url.split("/")[2].lower()
    except IndexError:
        return SourceType.UNKNOWN, 0.40
    # 精确匹配优先
    if domain in DOMAIN_SOURCE_TYPE:
        st = DOMAIN_SOURCE_TYPE[domain]
        return st, SOURCE_TYPE_AUTHORITY[st]
    # 后缀匹配（如 www.nhc.gov.cn → gov.cn）
    for suffix, st in DOMAIN_SOURCE_TYPE.items():
        if domain.endswith("." + suffix) or domain == suffix:
            return st, SOURCE_TYPE_AUTHORITY[st]
    return SourceType.UNKNOWN, 0.40


def enrich_evidence(ev: Evidence) -> Evidence:
    """给 Evidence 填充 source_type 和 authority_score（如果还是默认值）。"""
    if ev.source_type == SourceType.UNKNOWN and ev.url:
        st, score = classify_source(ev.url)
        ev.source_type = st
        ev.authority_score = score
    return ev


def _root_domain(url: str) -> str:
    """提取根域名（去 www 前缀，保留主域+顶级域）。"""
    if not url or "/" not in url:
        return ""
    try:
        domain = url.split("/")[2].lower()
    except IndexError:
        return ""
    if domain.startswith("www."):
        domain = domain[4:]
    parts = domain.split(".")
    if len(parts) >= 2:
        return ".".join(parts[-2:])
    return domain


def deduplicate_evidence(evidence: list[Evidence]) -> list[Evidence]:
    """按根域名去重：同一根域名只保留 authority_score 最高的一条。"""
    domain_best: dict[str, Evidence] = {}
    for e in evidence:
        rd = _root_domain(e.url) or e.url
        if rd not in domain_best or e.authority_score > domain_best[rd].authority_score:
            domain_best[rd] = e
    deduped = list(domain_best.values())
    deduped.sort(key=lambda e: e.authority_score, reverse=True)
    return deduped


OFFICIAL_SITE_TEMPLATES: dict[str, list[str]] = {
    "政策法规": ["site:gov.cn", "site:piyao.org.cn", "site:xinhuanet.com"],
    "健康养生": ["site:piyao.org.cn", "site:nhc.gov.cn", "site:who.int"],
    "诈骗套路": ["site:12321.cn", "site:police.gov.cn", "site:piyao.org.cn"],
    "灾难恐慌": ["site:gov.cn", "site:cma.gov.cn", "site:xinhuanet.com"],
    "金融财经": ["site:pbc.gov.cn", "site:csrc.gov.cn", "site:gov.cn"],
    "AI名人语录": ["site:reuters.com", "site:xinhuanet.com"],
    "伪造截图": ["site:piyao.org.cn", "site:xinhuanet.com"],
    "旧闻翻炒": ["site:piyao.org.cn", "site:xinhuanet.com"],
    "食品安全": ["site:samr.gov.cn", "site:piyao.org.cn", "site:nhc.gov.cn"],
}


def build_official_queries(claim_text: str, category: str) -> list[str]:
    """根据类别生成带官方站点过滤的搜索查询。"""
    sites = OFFICIAL_SITE_TEMPLATES.get(category, [])
    queries = []
    keywords = claim_text[:30]
    for site in sites[:2]:
        queries.append(f"{keywords} {site}")
    queries.append(f"{keywords} 辟谣 核查")
    return queries


class SearchProvider(ABC):
    @abstractmethod
    async def search(self, query: str, *, max_results: int = 5) -> list[Evidence]: ...


class MockSearchProvider(SearchProvider):
    """开发阶段用的 mock 搜索，返回预设结果。"""

    MOCK_DB: dict[str, list[dict]] = {
        "默认": [
            {
                "source": "人民网",
                "url": "https://www.people.com.cn/example",
                "title": "官方辟谣：该消息不实",
                "snippet": "经核实，网传消息与事实不符。相关部门已发布澄清声明。",
                "credibility": "权威媒体",
                "supports_claim": False,
            },
            {
                "source": "新华社",
                "url": "https://www.xinhuanet.com/example",
                "title": "权威发布：最新政策解读",
                "snippet": "根据最新发布的政策文件，实际情况如下……",
                "credibility": "国家通讯社",
                "supports_claim": False,
            },
        ]
    }

    async def search(self, query: str, *, max_results: int = 5) -> list[Evidence]:
        logger.info("[MockSearch] query=%s", query)
        results = self.MOCK_DB.get("默认", [])[:max_results]
        return [Evidence(**r) for r in results]


class Qihoo360SearchProvider(SearchProvider):
    """360 智搜 API 适配器。"""

    def __init__(self) -> None:
        self.api_key = settings.qihoo_api_key
        self.base_url = settings.qihoo_base_url

    async def search(self, query: str, *, max_results: int = 5) -> list[Evidence]:
        if not self.api_key:
            logger.warning("360 API key 未配置，返回空结果")
            return []

        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(
                f"{self.base_url}/v1/search",
                params={"q": query, "count": max_results},
                headers={"Authorization": f"Bearer {self.api_key}"},
            )
            resp.raise_for_status()
            data = resp.json()

        results = []
        for item in data.get("results", [])[:max_results]:
            url = item.get("url", "")
            st, auth = classify_source(url)
            results.append(
                Evidence(
                    source=item.get("source", "360搜索"),
                    url=url,
                    title=item.get("title", ""),
                    snippet=item.get("snippet", "")[:600],
                    credibility="搜索引擎结果",
                    source_type=st,
                    authority_score=auth,
                )
            )
        return results


class So360WebSearchProvider(SearchProvider):
    """360 搜索网页版（so.com）适配器，无需 API key。

    直接请求 https://www.so.com/s?q=xxx 并从 HTML 中提取搜索结果。
    作为 360 产品集成的核心证据来源。
    """

    SEARCH_URL = "https://www.so.com/s"
    _last_request_time = 0.0
    _captcha_until = 0.0
    _lock = threading.Lock()
    HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
        "Referer": "https://www.so.com/",
    }

    # 360 搜索结果页 HTML 结构：
    # <li class="res-list"> 每个搜索结果
    #   <h3 class="res-title"><a href="...">标题</a></h3>
    #   <p class="res-desc">摘要...</p> 或 <div class="res-rich">...</div>
    #   <cite>来源域名</cite>
    _RESULT_PATTERN = re.compile(
        r'<li[^>]*class="res-list"[^>]*>(.*?)</li>',
        re.DOTALL,
    )
    _TITLE_LINK_PATTERN = re.compile(
        r'<h3[^>]*>\s*<a[^>]*href="([^"]*)"[^>]*>(.*?)</a>',
        re.DOTALL,
    )
    _DESC_PATTERN = re.compile(
        r'<p[^>]*class="res-desc"[^>]*>(.*?)</p>',
        re.DOTALL,
    )
    _CITE_PATTERN = re.compile(
        r"<cite[^>]*>(.*?)</cite>",
        re.DOTALL,
    )

    @staticmethod
    def _strip_tags(html: str) -> str:
        """移除 HTML 标签，保留纯文本。"""
        from html import unescape

        return unescape(re.sub(r"<[^>]+>", "", html)).strip()

    async def search(self, query: str, *, max_results: int = 5) -> list[Evidence]:
        import asyncio

        logger.info("[So360Web] 搜索: %s", query[:60])
        with So360WebSearchProvider._lock:
            now = time.time()
            if now < So360WebSearchProvider._captcha_until:
                logger.info("[So360Web] CAPTCHA 冷却中，跳过请求")
                return []
            elapsed = now - So360WebSearchProvider._last_request_time
            wait = max(0, 1.5 - elapsed)
            So360WebSearchProvider._last_request_time = now + wait
        if wait > 0:
            await asyncio.sleep(wait)
        try:
            async with httpx.AsyncClient(timeout=15, follow_redirects=True) as client:
                resp = await client.get(
                    self.SEARCH_URL,
                    params={"q": query, "src": "srp", "fr": "none"},
                    headers=self.HEADERS,
                )
                resp.raise_for_status()
                html = resp.text
        except Exception as exc:
            logger.warning("[So360Web] 请求失败: %s，返回空结果", exc)
            return []

        if "验证码" in html or "/captcha" in html:
            So360WebSearchProvider._captcha_until = time.time() + 300
            logger.warning("[So360Web] 触发验证码，5 分钟内跳过请求")
            return []

        results: list[Evidence] = []
        blocks = self._RESULT_PATTERN.findall(html)

        if not blocks:
            blocks = re.findall(r"<li[^>]*data-res[^>]*>(.*?)</li>", html, re.DOTALL)

        for block in blocks[:max_results]:
            title_match = self._TITLE_LINK_PATTERN.search(block)
            if not title_match:
                continue

            url = title_match.group(1)
            from urllib.parse import urlparse

            parsed = urlparse(url)
            if parsed.scheme not in ("http", "https", ""):
                continue
            title = self._strip_tags(title_match.group(2))

            desc_match = self._DESC_PATTERN.search(block)
            snippet = self._strip_tags(desc_match.group(1)) if desc_match else ""

            cite_match = self._CITE_PATTERN.search(block)
            source_domain = self._strip_tags(cite_match.group(1)) if cite_match else "360搜索"

            st, auth = classify_source(url)
            results.append(
                Evidence(
                    source=f"360搜索 · {source_domain}"
                    if source_domain != "360搜索"
                    else "360搜索",
                    url=url,
                    title=title,
                    snippet=snippet[:600],
                    credibility="360搜索引擎结果",
                    source_type=st,
                    authority_score=auth,
                )
            )

        if not results:
            logger.warning("[So360Web] 未解析到结果，返回空")
            return []

        logger.info("[So360Web] 返回 %d 条结果", len(results))
        return results


class TavilySearchProvider(SearchProvider):
    """Tavily 搜索作为备选。"""

    _MAX_RETRIES = 2
    _RETRY_DELAYS = [2.0, 5.0]
    _MIN_INTERVAL = 1.0

    _global_lock = threading.Lock()
    _global_last_request = 0.0
    _key_pool: list[str] = []
    _key_index = 0

    @classmethod
    def _init_key_pool(cls):
        if cls._key_pool:
            return
        from dotenv import load_dotenv

        load_dotenv(Path(__file__).parents[2] / ".env")
        primary = settings.tavily_api_key or ""
        extras = os.getenv("TAVILY_API_KEYS", "").split(",")
        all_keys = [k.strip() for k in [primary] + extras if k.strip()]
        seen = set()
        for k in all_keys:
            if k not in seen:
                cls._key_pool.append(k)
                seen.add(k)
        if cls._key_pool:
            logger.info("[Tavily] 加载 %d 个 API key", len(cls._key_pool))

    @classmethod
    def _next_key(cls) -> str:
        if not cls._key_pool:
            return ""
        with cls._global_lock:
            key = cls._key_pool[cls._key_index % len(cls._key_pool)]
            cls._key_index += 1
            return key

    def __init__(self) -> None:
        self.api_key = settings.tavily_api_key
        TavilySearchProvider._init_key_pool()

    async def search(self, query: str, *, max_results: int = 5) -> list[Evidence]:
        use_key = self._next_key() if self._key_pool else self.api_key
        if not use_key:
            logger.warning("Tavily API key 未配置，返回空结果")
            return []

        import asyncio

        # 全局速率限制（多 key 时间隔更短，单 key 保守）
        interval = self._MIN_INTERVAL / max(len(self._key_pool), 1)
        with TavilySearchProvider._global_lock:
            now = time.time()
            wait = interval - (now - TavilySearchProvider._global_last_request)
            if wait > 0:
                TavilySearchProvider._global_last_request = now + wait
            else:
                TavilySearchProvider._global_last_request = now
                wait = 0
        if wait > 0:
            await asyncio.sleep(wait)

        for attempt in range(self._MAX_RETRIES + 1):
            try:
                async with httpx.AsyncClient(timeout=15) as client:
                    resp = await client.post(
                        "https://api.tavily.com/search",
                        json={
                            "api_key": use_key,
                            "query": query,
                            "max_results": max_results,
                            "search_depth": "advanced",
                        },
                    )
                    resp.raise_for_status()
                    data = resp.json()
                break
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code in (429, 432):
                    if len(self._key_pool) > 1 and use_key in self._key_pool:
                        with self._global_lock:
                            if use_key in self._key_pool:
                                self._key_pool.remove(use_key)
                                logger.warning(
                                    "[Tavily] Key %s...额度耗尽，移除（剩余 %d 个）",
                                    use_key[:15],
                                    len(self._key_pool),
                                )
                        use_key = self._next_key()
                        if use_key:
                            continue
                    if attempt < self._MAX_RETRIES:
                        delay = self._RETRY_DELAYS[attempt]
                        logger.warning(
                            "[Tavily] 限速 %d，%0.fs 后重试", exc.response.status_code, delay
                        )
                        await asyncio.sleep(delay)
                        continue
                logger.warning("[Tavily] 请求失败: %s", exc)
                return []
            except Exception as exc:
                logger.warning("[Tavily] 请求异常: %s", exc)
                return []

        results = []
        for item in data.get("results", [])[:max_results]:
            url = item.get("url", "")
            content = item.get("content", "")
            st, auth = classify_source(url)
            results.append(
                Evidence(
                    source=url.split("/")[2] if url else "Tavily",
                    url=url,
                    title=item.get("title", ""),
                    snippet=content[:800],
                    credibility="搜索引擎结果",
                    source_type=st,
                    authority_score=auth,
                )
            )
        return results


class BochaSearchProvider(SearchProvider):
    """博查（Bocha）Web Search API 适配器。中文权威源召回优于通用引擎，是证实维度命脉。

    API（Bocha Web Search）：POST https://api.bocha.cn/v1/web-search，Bearer 鉴权，
    body {"query","summary":true,"count","freshness":"noLimit"}；
    响应 data.webPages.value[]（name/url/snippet/summary/siteName/datePublished）。
    解析对响应结构做防御（字段缺失/结构变动不崩，降级取可得字段）。

    付费档限流（写死在 provider，保守留余量，绝不突破）：
    - 并发 ≤5：类级 asyncio.Semaphore(5)，请求全程持有。
    - ≤200 次/分钟：起始最小间隔 0.34s（≈176/min，留余量），短临界区只管节流不挡并发。
    - ≤1000 次/天：用户设定每日上限 1000，撞 950 安全线即停（留 50 缓冲），跨进程累计。
    """

    _ENDPOINT = "https://api.bocha.cn/v1/web-search"
    _MAX_RETRIES = 2
    _RETRY_DELAYS = [2.0, 5.0]
    _MIN_INTERVAL = 0.34  # 200 次/分钟 → 间隔 ≥0.3s，取 0.34（≈176/min，保守）
    _DAILY_CAP = 1000  # 用户设定每日上限 1000
    _DAILY_SAFE = 950  # 撞这条线就停，留 50 缓冲
    _USAGE_FILE = Path(__file__).parents[2] / "data" / ".bocha_usage.json"
    # ⚠️ 用 threading.Lock（跨事件循环安全）——orchestrator 每次搜索用独立 asyncio.run（新建并关闭
    # 循环），绝不能用 asyncio.Lock/Semaphore（会绑死在已关闭的循环上导致死锁，2026-05-29 踩坑）。
    # 搜索调用本就串行（每 query 一个 asyncio.run），付费档并发5 在此架构下用不上，串行远低于限速。
    _throttle = threading.Lock()
    _last_request = 0.0

    def __init__(self) -> None:
        self.api_key = settings.bocha_api_key

    @classmethod
    def _today_count(cls) -> int:
        """读今日已用次数（跨进程持久化，分段续跑也累计）。"""
        try:
            u = json.loads(cls._USAGE_FILE.read_text(encoding="utf-8"))
            return int(u.get("count", 0)) if u.get("date") == time.strftime("%Y-%m-%d") else 0
        except Exception:
            return 0

    @classmethod
    def _bump_count(cls) -> None:
        today = time.strftime("%Y-%m-%d")
        cur = cls._today_count()
        try:
            cls._USAGE_FILE.parent.mkdir(parents=True, exist_ok=True)
            cls._USAGE_FILE.write_text(
                json.dumps({"date": today, "count": cur + 1}), encoding="utf-8"
            )
        except Exception as exc:
            logger.warning("[Bocha] 日计数写入失败: %s", exc)

    async def search(self, query: str, *, max_results: int = 5) -> list[Evidence]:
        if not self.api_key:
            logger.warning("[Bocha] API key 未配置，返回空结果")
            return []

        # 短临界区（threading 锁，跨循环安全）：日计数守卫 + 起始节流时间戳预留
        with BochaSearchProvider._throttle:
            if self._today_count() >= self._DAILY_SAFE:
                logger.warning(
                    "[Bocha] 今日已用 ≥%d 次（安全线），停用至次日，返回空结果", self._DAILY_SAFE
                )
                return []
            now = time.time()
            wait = BochaSearchProvider._MIN_INTERVAL - (now - BochaSearchProvider._last_request)
            # 预留下一个槽位（即使并发也按序间隔，≤176/min）
            BochaSearchProvider._last_request = now + max(0.0, wait)
            self._bump_count()
        if wait > 0:
            await asyncio.sleep(wait)

        data = None
        for attempt in range(self._MAX_RETRIES + 1):
            try:
                async with httpx.AsyncClient(timeout=20) as client:
                    resp = await client.post(
                        self._ENDPOINT,
                        headers={
                            "Authorization": f"Bearer {self.api_key}",
                            "Content-Type": "application/json",
                        },
                        json={
                            "query": query,
                            "summary": True,
                            "count": max_results,
                            "freshness": "noLimit",
                        },
                    )
                    resp.raise_for_status()
                    data = resp.json()
                break
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code == 429 and attempt < self._MAX_RETRIES:
                    await asyncio.sleep(self._RETRY_DELAYS[attempt])
                    BochaSearchProvider._last_request = time.time()
                    continue
                logger.warning("[Bocha] 请求失败: %s", exc)
                return []
            except Exception as exc:
                logger.warning("[Bocha] 请求异常: %s", exc)
                return []

        if not isinstance(data, dict):
            return []
        # 防御性解析：data.webPages.value（标准结构），结构变动时降级
        payload = data.get("data") if isinstance(data.get("data"), dict) else data
        web_pages = (payload or {}).get("webPages") or {}
        items = web_pages.get("value") if isinstance(web_pages, dict) else None
        if not isinstance(items, list):
            logger.warning("[Bocha] 响应结构非预期，原始 keys=%s", list(data.keys()))
            return []

        results = []
        for item in items[:max_results]:
            if not isinstance(item, dict):
                continue
            url = item.get("url", "") or ""
            # summary 比 snippet 信息量大，优先
            content = item.get("summary") or item.get("snippet") or ""
            st, auth = classify_source(url)
            results.append(
                Evidence(
                    source=item.get("siteName") or (url.split("/")[2] if "/" in url else "Bocha"),
                    url=url,
                    title=item.get("name", ""),
                    snippet=str(content)[:800],
                    credibility="博查搜索结果",
                    source_type=st,
                    authority_score=auth,
                    published_date=str(item.get("datePublished") or "")[:10],
                )
            )
        return results


class WikipediaSearchProvider(SearchProvider):
    """中文维基百科 API 适配器。免费、无 key、无限调用。

    两步查询：
    1. list=search 搜出 top N 标题
    2. prop=extracts 批量取摘要（无 HTML 标签）

    要求：必须发送 User-Agent，否则维基返回 403。
    """

    BASE_URL = "https://zh.wikipedia.org/w/api.php"
    HEADERS = {
        "User-Agent": (
            "TruthNote/1.0 (https://github.com/AppApp777/wenke-song; noreply@github.com)"
        )
    }

    async def search(self, query: str, *, max_results: int = 5) -> list[Evidence]:
        logger.info("[WikiSearch] 查询: %s", query[:60])

        try:
            async with httpx.AsyncClient(timeout=15, headers=self.HEADERS) as client:
                resp = await client.get(
                    self.BASE_URL,
                    params={
                        "action": "query",
                        "list": "search",
                        "srsearch": query,
                        "srlimit": max_results,
                        "format": "json",
                    },
                )
                resp.raise_for_status()
                data = resp.json()
        except Exception as exc:
            logger.warning("[WikiSearch] 搜索失败: %s", exc)
            return []

        hits = data.get("query", {}).get("search", [])
        if not hits:
            logger.info("[WikiSearch] 无结果")
            return []

        titles = [h["title"] for h in hits[:max_results]]
        titles_joined = "|".join(titles)

        try:
            async with httpx.AsyncClient(timeout=15, headers=self.HEADERS) as client:
                resp = await client.get(
                    self.BASE_URL,
                    params={
                        "action": "query",
                        "prop": "extracts|info",
                        "exintro": True,
                        "explaintext": True,
                        "exchars": 500,
                        "exlimit": "max",
                        "inprop": "url",
                        "titles": titles_joined,
                        "format": "json",
                        "redirects": 1,
                    },
                )
                resp.raise_for_status()
                detail = resp.json()
        except Exception as exc:
            logger.warning("[WikiSearch] 取摘要失败: %s", exc)
            return []

        pages = detail.get("query", {}).get("pages", {})
        page_by_title = {p.get("title"): p for p in pages.values()}

        results: list[Evidence] = []
        for title in titles:
            page = page_by_title.get(title)
            if not page:
                continue
            extract = (page.get("extract") or "").strip()
            if not extract:
                continue
            url = page.get("fullurl") or f"https://zh.wikipedia.org/wiki/{title}"
            results.append(
                Evidence(
                    source="维基百科",
                    url=url,
                    title=title,
                    snippet=extract[:600],
                    credibility="维基百科条目",
                    source_type=SourceType.ENCYCLOPEDIA,
                    authority_score=0.70,
                )
            )

        logger.info("[WikiSearch] 返回 %d 条结果", len(results))
        return results


class FirecrawlSearchProvider(SearchProvider):
    """Firecrawl /search API：网页搜索 + 自动正文提取。"""

    _BASE_URL = "https://api.firecrawl.dev/v1"
    _MAX_RETRIES = 2
    _RETRY_DELAYS = [2.0, 5.0]

    def __init__(self) -> None:
        self.api_key = settings.firecrawl_api_key

    async def search(self, query: str, *, max_results: int = 5) -> list[Evidence]:
        if not self.api_key:
            logger.warning("[Firecrawl] API key 未配置")
            return []

        import asyncio

        data = None
        for attempt in range(self._MAX_RETRIES + 1):
            try:
                async with httpx.AsyncClient(timeout=20) as client:
                    resp = await client.post(
                        f"{self._BASE_URL}/search",
                        headers={"Authorization": f"Bearer {self.api_key}"},
                        json={
                            "query": query,
                            "limit": max_results,
                            "lang": "zh",
                            "scrapeOptions": {"formats": ["markdown"]},
                        },
                    )
                    resp.raise_for_status()
                    data = resp.json()
                break
            except httpx.HTTPStatusError as exc:
                status = exc.response.status_code
                if status in (429, 500, 502, 503) and attempt < self._MAX_RETRIES:
                    delay = self._RETRY_DELAYS[attempt]
                    logger.warning(
                        "[Firecrawl] %d，%.0fs 后重试 (%d/%d)",
                        status,
                        delay,
                        attempt + 1,
                        self._MAX_RETRIES,
                    )
                    await asyncio.sleep(delay)
                    continue
                logger.warning("[Firecrawl] 请求失败: %s", exc)
                return []
            except Exception as exc:
                logger.warning("[Firecrawl] 请求异常: %s", exc)
                return []

        if data is None:
            return []

        results = []
        for item in data.get("data", [])[:max_results]:
            url = item.get("url", "")
            source = url.split("/")[2] if url else "Firecrawl"
            meta = item.get("metadata", {})
            desc = (
                item.get("description", "")
                or meta.get("og:description", "")
                or meta.get("description", "")
            )
            md = item.get("markdown", "")
            snippet = (desc[:800] if desc else md[:800]) or ""
            st, auth = classify_source(url)
            results.append(
                Evidence(
                    source=source,
                    url=url,
                    title=item.get("title", meta.get("title", "")),
                    snippet=snippet,
                    credibility="搜索引擎结果",
                    source_type=st,
                    authority_score=auth,
                )
            )
        logger.info("[Firecrawl] 返回 %d 条结果", len(results))
        return results


class CachedSearchProvider(SearchProvider):
    """SQLite 缓存包装器，对任意 SearchProvider 添加缓存。"""

    def __init__(
        self, inner: SearchProvider, db_path: str | Path | None = None, ttl_hours: int = 24
    ):
        self.inner = inner
        self.provider_name = type(inner).__name__
        self.db_path = Path(db_path) if db_path else Path(settings.search_cache_db)
        self.ttl_seconds = ttl_hours * 3600
        self._init_db()

    def _open_db(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=30.0)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=30000")
        return conn

    def _init_db(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = self._open_db()
        conn.execute(
            """CREATE TABLE IF NOT EXISTS search_cache (
                query_hash TEXT PRIMARY KEY,
                query TEXT NOT NULL,
                results_json TEXT NOT NULL,
                created_at REAL NOT NULL
            )"""
        )
        conn.commit()
        conn.close()

    def _hash(self, query: str, max_results: int) -> str:
        return hashlib.sha256(f"{self.provider_name}||{query}||{max_results}".encode()).hexdigest()

    @staticmethod
    def _legacy_hash(query: str, max_results: int) -> str:
        return hashlib.sha256(f"{query}||{max_results}".encode()).hexdigest()

    def _get_cached(self, query: str, max_results: int) -> list[Evidence] | None:
        qh = self._hash(query, max_results)
        conn = self._open_db()
        row = conn.execute(
            "SELECT results_json, created_at FROM search_cache WHERE query_hash = ?", (qh,)
        ).fetchone()
        if row is None:
            legacy_qh = self._legacy_hash(query, max_results)
            row = conn.execute(
                "SELECT results_json, created_at FROM search_cache WHERE query_hash = ?",
                (legacy_qh,),
            ).fetchone()
        conn.close()
        if row is None:
            return None
        results_json, created_at = row
        if time.time() - created_at > self.ttl_seconds:
            return None
        data = json.loads(results_json)
        return [Evidence(**item) for item in data]

    def _set_cache(self, query: str, max_results: int, results: list[Evidence]) -> None:
        qh = self._hash(query, max_results)
        data = json.dumps([r.model_dump() for r in results], ensure_ascii=False)
        conn = self._open_db()
        conn.execute(
            "INSERT OR REPLACE INTO search_cache"
            " (query_hash, query, results_json, created_at)"
            " VALUES (?, ?, ?, ?)",
            (qh, query, data, time.time()),
        )
        conn.commit()
        conn.close()

    async def search(self, query: str, *, max_results: int = 5) -> list[Evidence]:
        cached = self._get_cached(query, max_results)
        if cached is not None:
            logger.info("[CachedSearch] 缓存命中: %s", query[:40])
            return cached
        results = await self.inner.search(query, max_results=max_results)
        if results:
            self._set_cache(query, max_results, results)
            logger.info("[CachedSearch] 缓存写入: %s (%d 条)", query[:40], len(results))
        else:
            logger.info("[CachedSearch] 搜索返回空，不写入缓存: %s", query[:40])
        return results


class FallbackSearchProvider(SearchProvider):
    """主引擎返回空时自动切到备用引擎。"""

    def __init__(self, primary: SearchProvider, fallback: SearchProvider) -> None:
        self.primary = primary
        self.fallback = fallback

    async def search(self, query: str, *, max_results: int = 5) -> list[Evidence]:
        results = await self.primary.search(query, max_results=max_results)
        if results:
            return results
        logger.info("[Fallback] 主引擎无结果，切换备用引擎")
        return await self.fallback.search(query, max_results=max_results)


def get_search_provider(*, use_cache: bool = True) -> SearchProvider:
    provider_name = settings.search_provider.lower()
    if provider_name == "qihoo360":
        if settings.qihoo_api_key:
            inner: SearchProvider = Qihoo360SearchProvider()
        else:
            logger.info("360 API key 未配置，使用 360 网页搜索（so.com）")
            inner = So360WebSearchProvider()
    elif provider_name == "tavily":
        inner = FallbackSearchProvider(TavilySearchProvider(), So360WebSearchProvider())
    elif provider_name == "so360":
        inner = So360WebSearchProvider()
    elif provider_name == "wikipedia":
        inner = WikipediaSearchProvider()
    elif provider_name == "wiki_so360":
        inner = FallbackSearchProvider(WikipediaSearchProvider(), So360WebSearchProvider())
    elif provider_name == "so360_wiki":
        inner = FallbackSearchProvider(So360WebSearchProvider(), WikipediaSearchProvider())
    elif provider_name == "firecrawl":
        inner = FirecrawlSearchProvider()
    elif provider_name == "firecrawl_wiki":
        inner = FallbackSearchProvider(FirecrawlSearchProvider(), WikipediaSearchProvider())
    elif provider_name == "bocha":
        inner = BochaSearchProvider()
    elif provider_name == "bocha_wiki":
        # 博查主用（中文权威源召回好），wiki 兜底（科普级反驳）
        inner = FallbackSearchProvider(BochaSearchProvider(), WikipediaSearchProvider())
    elif provider_name == "wiki_bocha":
        inner = FallbackSearchProvider(WikipediaSearchProvider(), BochaSearchProvider())
    elif provider_name == "wiki_tavily":
        # wiki 主用（科普级反驳快稳），tavily 兜底（事件级查证）
        inner = FallbackSearchProvider(WikipediaSearchProvider(), TavilySearchProvider())
    elif provider_name == "tavily_wiki":
        inner = FallbackSearchProvider(TavilySearchProvider(), WikipediaSearchProvider())
    else:
        inner = MockSearchProvider()

    if use_cache and provider_name != "mock":
        return CachedSearchProvider(inner)
    return inner

"""图片溯源（reverse_image）单元测试。

覆盖信任边界与不变量（不联网、不依赖 PicImageSearch）：
- SSRF 防护：内网/环回/非 http(s) 必拒，公网 IP 字面量放行
- 溯源不判真假：build_source_card 的 overall_verdict 恒中性"无法核实"
- fail-soft：拿不到图/缓存未命中+禁联网 → found=False，绝不抛异常
- 缓存：URL/哈希双索引命中、from_cache 标记
"""

from __future__ import annotations

import importlib
import sys

import pytest

from src.truthnote import reverse_image as ri


# ---------------- SSRF 防护 ----------------
@pytest.mark.parametrize(
    "url",
    [
        "http://localhost/x.jpg",
        "http://127.0.0.1/x.jpg",
        "http://10.0.0.1/x.jpg",
        "http://192.168.1.5/x.jpg",
        "http://169.254.169.254/latest/meta-data",  # 云元数据 SSRF 经典目标
        "http://[::1]/x.jpg",
        "http://foo.local/x.jpg",
        "http://bar.internal/x.jpg",
        "ftp://example.com/x.jpg",  # 非 http(s)
        "file:///etc/passwd",
        "  ",
        "",
    ],
)
def test_ssrf_rejects_unsafe(url):
    assert ri._is_safe_public_url(url) is False


def test_ssrf_allows_public_ip_literal():
    # 公网 IP 字面量无需 DNS，确定性放行
    assert ri._is_safe_public_url("http://8.8.8.8/x.jpg") is True
    assert ri._is_safe_public_url("https://1.1.1.1/img.png") is True


def test_internal_ip_literal_rejected_even_in_insecure_mode(monkeypatch):
    # insecure 开关只跳过"主机名解析"，IP 字面量内网永远拦死
    monkeypatch.setattr(ri, "_ALLOW_INSECURE", True)
    assert ri._is_safe_public_url("http://10.0.0.1/x.jpg") is False
    assert ri._is_safe_public_url("http://127.0.0.1/x.jpg") is False


def test_download_unsafe_url_returns_none_no_raise():
    # 不安全 URL 在下载前被 SSRF 拦截，返回 None 不抛
    assert ri._download_image("http://127.0.0.1/x.jpg") is None


# ---------------- 溯源不判真假（核心不变量） ----------------
def test_build_source_card_found_is_neutral():
    result = {
        "found": True,
        "total": 30,
        "source_pages": [
            {"url": "https://www.sohu.com/a/414131840", "domain": "sohu.com", "title": ""},
            {"url": "https://weibo.com/x", "domain": "weibo.com", "title": "现场"},
        ],
        "error": "",
    }
    card = ri.build_source_card(result, image_url="https://img/x.jpg")
    # 溯源绝不输出真/假，恒中性
    assert card["overall_verdict"] == "无法核实"
    # 来源网页塑成 claims[0].evidence[]，渲染层零改
    assert len(card["claims"]) == 1
    ev = card["claims"][0]["evidence"]
    assert len(ev) == 2
    assert ev[0]["url"] == "https://www.sohu.com/a/414131840"
    assert ev[0]["source_url"] == ev[0]["url"]  # content.js 优先读 source_url
    # title 为空时域名兜底
    assert ev[0]["title"] == "sohu.com"
    # 诚实红线：summary 必须说"已知出现过"而非"首发时间"
    assert "首发时间" in card["summary"]


def test_build_source_card_not_found_is_neutral():
    card = ri.build_source_card({"found": False, "error": "", "source_pages": [], "total": 0})
    assert card["overall_verdict"] == "无法核实"
    assert card["claims"] == []
    assert card["friendly_reply"]


def test_build_source_card_error_messages():
    for err, needle in [
        ("upload_rejected_invalid_image", "无法解析"),
        ("reverse_search_failed:RuntimeError", "暂时不可用"),
        ("no_image_bytes", "拿不到这张图"),
    ]:
        card = ri.build_source_card({"found": False, "error": err, "source_pages": [], "total": 0})
        assert needle in card["summary"]


# ---------------- fail-soft ----------------
def test_no_image_source_failsoft():
    out = ri.reverse_search_image(image_url="", image_bytes=None, use_cache=False)
    assert out["found"] is False
    assert out["error"] == "no_image_bytes"


def test_cache_miss_live_disabled(tmp_path, monkeypatch):
    monkeypatch.setattr(ri, "_CACHE_PATH", tmp_path / "rev_cache.json")
    out = ri.reverse_search_image(image_bytes=b"x" * 300, allow_live=False)
    assert out["found"] is False
    assert out["error"] == "cache_miss_live_disabled"
    assert out["from_cache"] is False


def test_allow_live_false_with_url_does_not_download(tmp_path, monkeypatch):
    # allow_live=False + 给 URL（无缓存）→ 绝不联网下载，直接 miss（demo 零联网硬保证）
    monkeypatch.setattr(ri, "_CACHE_PATH", tmp_path / "rev_cache.json")
    called = {"n": 0}

    def _boom_download(url):
        called["n"] += 1
        raise AssertionError("allow_live=False 不应触发下载")

    monkeypatch.setattr(ri, "_download_image", _boom_download)
    out = ri.reverse_search_image(image_url="https://img/x.jpg", allow_live=False)
    assert out["found"] is False
    assert out["error"] == "cache_miss_live_disabled"
    assert called["n"] == 0  # 一次下载都没发生


def test_run_baidu_failsoft_when_backend_raises(tmp_path, monkeypatch):
    # 真覆盖 fail-soft：让百度协程抛异常，_run_baidu 的 except 必须兜住 → found=False，不抛
    monkeypatch.setattr(ri, "_CACHE_PATH", tmp_path / "rev_cache.json")

    async def _boom(_path):
        raise RuntimeError("baidu down")

    monkeypatch.setattr(ri, "_baidu_search_async", _boom)
    out = ri.reverse_search_image(image_bytes=b"y" * 300, allow_live=True)
    assert out["found"] is False
    assert out["from_cache"] is False
    assert out["error"].startswith("reverse_search_failed")


# ---------------- 缓存双索引 ----------------
def test_cache_roundtrip_by_hash_and_url(tmp_path, monkeypatch):
    monkeypatch.setattr(ri, "_CACHE_PATH", tmp_path / "rev_cache.json")
    fake = {
        "found": True,
        "total": 2,
        "source_pages": [{"url": "https://a.com/1", "domain": "a.com", "title": ""}],
        "error": "",
        "result_page": "https://baidu/s?x",
    }
    monkeypatch.setattr(ri, "_run_live", lambda data, provider="baidu": dict(fake))

    # 第一次：live → 写缓存
    first = ri.reverse_search_image(image_url="https://img/x.jpg", image_bytes=b"z" * 300)
    assert first["found"] is True and first["from_cache"] is False

    # 第二次同 URL：命中 by_url 缓存，瞬回（哪怕不再给 bytes）
    second = ri.reverse_search_image(image_url="https://img/x.jpg")
    assert second["from_cache"] is True
    assert second["total"] == 2


def test_module_imports_without_picimagesearch(monkeypatch):
    # 真模拟缺库：从 sys.modules 删掉 PicImageSearch 再 reload，模块仍能导入（调用时才优雅失败）
    monkeypatch.delitem(sys.modules, "PicImageSearch", raising=False)
    mod = importlib.reload(ri)
    assert hasattr(mod, "reverse_search_image")
    assert hasattr(mod, "build_source_card")

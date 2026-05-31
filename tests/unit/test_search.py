import tempfile

import pytest

from src.truthnote.search import (
    CachedSearchProvider,
    MockSearchProvider,
    build_official_queries,
    get_search_provider,
)


@pytest.mark.asyncio
async def test_mock_search_returns_results():
    provider = MockSearchProvider()
    results = await provider.search("测试查询")
    assert len(results) == 2
    assert results[0].source == "人民网"
    assert results[1].source == "新华社"


@pytest.mark.asyncio
async def test_mock_search_max_results():
    provider = MockSearchProvider()
    results = await provider.search("测试查询", max_results=1)
    assert len(results) == 1


def test_get_search_provider_default(monkeypatch):
    from src.truthnote import config

    monkeypatch.setattr(config.settings, "search_provider", "mock")
    provider = get_search_provider()
    assert isinstance(provider, MockSearchProvider)


def test_build_official_queries_policy():
    queries = build_official_queries("存款超5万要交税", "政策法规")
    assert len(queries) == 3
    assert any("site:gov.cn" in q for q in queries)
    assert any("辟谣" in q for q in queries)


def test_build_official_queries_unknown_category():
    queries = build_official_queries("测试", "未知类别")
    assert len(queries) == 1
    assert "辟谣" in queries[0]


@pytest.mark.asyncio
async def test_cached_search_hit():
    with tempfile.TemporaryDirectory() as tmpdir:
        mock = MockSearchProvider()
        cached = CachedSearchProvider(mock, db_path=f"{tmpdir}/test_cache.db")
        r1 = await cached.search("缓存测试", max_results=2)
        r2 = await cached.search("缓存测试", max_results=2)
        assert len(r1) == len(r2)
        assert r1[0].source == r2[0].source


@pytest.mark.asyncio
async def test_cached_search_different_queries():
    with tempfile.TemporaryDirectory() as tmpdir:
        mock = MockSearchProvider()
        cached = CachedSearchProvider(mock, db_path=f"{tmpdir}/test_cache.db")
        r1 = await cached.search("查询A")
        r2 = await cached.search("查询B")
        assert len(r1) == len(r2)

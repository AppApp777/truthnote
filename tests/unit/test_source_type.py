"""source_type 域名映射 + authority_score 计算测试。"""

from src.truthnote.schemas import Evidence, SourceType
from src.truthnote.search import classify_source, enrich_evidence


class TestClassifySource:
    def test_gov_cn(self):
        st, score = classify_source("https://www.nhc.gov.cn/some/page")
        assert st == SourceType.OFFICIAL_GOVERNMENT
        assert score == 0.95

    def test_piyao(self):
        st, score = classify_source("https://www.piyao.org.cn/article/123")
        assert st == SourceType.FACT_CHECK_ORG
        assert score == 0.90

    def test_wikipedia(self):
        st, score = classify_source("https://zh.wikipedia.org/wiki/Test")
        assert st == SourceType.ENCYCLOPEDIA
        assert score == 0.70

    def test_xinhuanet(self):
        st, score = classify_source("https://www.xinhuanet.com/news")
        assert st == SourceType.ESTABLISHED_MEDIA
        assert score == 0.75

    def test_weibo(self):
        st, score = classify_source("https://weibo.com/user/123")
        assert st == SourceType.SOCIAL_MEDIA
        assert score == 0.20

    def test_zhihu(self):
        st, score = classify_source("https://www.zhihu.com/question/123")
        assert st == SourceType.BLOG_FORUM
        assert score == 0.30

    def test_unknown_domain(self):
        st, score = classify_source("https://random-blog.xyz/post")
        assert st == SourceType.UNKNOWN
        assert score == 0.40

    def test_empty_url(self):
        st, score = classify_source("")
        assert st == SourceType.UNKNOWN

    def test_no_slash(self):
        st, score = classify_source("not-a-url")
        assert st == SourceType.UNKNOWN


class TestEnrichEvidence:
    def test_fills_unknown(self):
        ev = Evidence(
            source="test",
            url="https://www.people.com.cn/article",
            snippet="test snippet",
        )
        assert ev.source_type == SourceType.UNKNOWN
        enrich_evidence(ev)
        assert ev.source_type == SourceType.ESTABLISHED_MEDIA
        assert ev.authority_score == 0.75

    def test_preserves_existing(self):
        ev = Evidence(
            source="test",
            url="https://weibo.com/user",
            snippet="test",
            source_type=SourceType.FACT_CHECK_ORG,
            authority_score=0.90,
        )
        enrich_evidence(ev)
        assert ev.source_type == SourceType.FACT_CHECK_ORG
        assert ev.authority_score == 0.90

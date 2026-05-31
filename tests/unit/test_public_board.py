"""社会层面闭环 · 公示墙后端测试。"""

from src.truthnote.closed_loop import RiskType
from src.truthnote.public_board import (
    BoardStatus,
    _parse_verdict_str,
    board_stats,
    get_public_board,
    load_crawler_rumors,
    risk_to_category,
    seed_board,
    verdict_to_status,
)
from src.truthnote.schemas import Verdict


class TestMappings:
    def test_risk_to_category(self):
        assert risk_to_category(RiskType.SCAM) == "诈骗"
        assert risk_to_category(RiskType.HEALTH_MISINFORMATION) == "健康养生"
        assert risk_to_category("不存在的") == "综合"

    def test_verdict_to_status(self):
        assert verdict_to_status(Verdict.FALSE) == BoardStatus.DEBUNKED
        assert verdict_to_status(Verdict.UNVERIFIABLE) == BoardStatus.AWAITING_AUTHORITY
        assert verdict_to_status(Verdict.TRUE) == BoardStatus.CONFIRMED_TRUE
        assert verdict_to_status(Verdict.PARTLY_TRUE) == BoardStatus.LABELED

    def test_reported_status(self):
        # 高危 + 已上报 → 已上报国家平台
        assert verdict_to_status(Verdict.FALSE, heat=2000, reported=True) == BoardStatus.REPORTED
        # 未上报仍是已辟谣
        assert verdict_to_status(Verdict.FALSE, reported=False) == BoardStatus.DEBUNKED


class TestSeedBoard:
    def test_seed_never_empty(self):
        items = seed_board()
        assert len(items) >= 10  # 永不空表

    def test_seed_covers_all_statuses(self):
        statuses = {x.status for x in seed_board()}
        # 至少覆盖：已辟谣 / 已上报 / 待权威定论 / 属实，证明处置状态有流转
        assert BoardStatus.DEBUNKED in statuses
        assert BoardStatus.REPORTED in statuses
        assert BoardStatus.AWAITING_AUTHORITY in statuses
        assert BoardStatus.CONFIRMED_TRUE in statuses

    def test_seed_no_user_data(self):
        # 脱敏：条目里只有谣言文本 + 判定，无用户字段
        item = seed_board()[0]
        assert item.claim_text
        assert not hasattr(item, "user_id")


class TestGetBoard:
    def test_store_rows_on_top(self):
        # 真实核查（新）应排在种子样本（旧）之前；不依赖爬虫库
        rows = [
            {
                "action_id": "act_real",
                "claim_text": "刚核查的一条真实谣言",
                "verdict": Verdict.FALSE.value,
                "risk_type": RiskType.SCAM.value,
                "evidence_urls": "[]",
                "created_at": "2099-01-01T00:00:00",  # 故意设很新
            }
        ]
        items = get_public_board(store_rows=rows, include_crawler=False)
        assert items[0].item_id == "act_real"

    def test_limit(self):
        items = get_public_board(limit=3, include_crawler=False)
        assert len(items) == 3


class TestCrawlerLoader:
    """官方辟谣库爬虫产物加载（fixture 隔离，不耦合真 5000 行文件）。"""

    def _write(self, tmp_path, name, text):
        d = tmp_path / "output"
        d.mkdir(exist_ok=True)
        (d / name).write_text(text, encoding="utf-8")
        return d

    def test_loads_and_maps(self, tmp_path):
        d = self._write(
            tmp_path,
            "a.jsonl",
            '{"id":"k1","message":"某谣言一","verdict":"谣言","source_site":"科普中国",'
            '"source_url":"https://x.cn/1","published_date":"2026-05-02"}\n'
            '{"id":"k2","title":"某谣言二","verdict":"谣言",'
            '"source_url":"https://x.cn/2","published_date":"2026-05-01"}\n',
        )
        items = load_crawler_rumors(dirs=[d])
        assert len(items) == 2
        assert items[0].verdict == Verdict.FALSE
        assert items[0].evidence_urls == ["https://x.cn/1"]  # source_url 进证据
        assert items[0].reported_to == "科普中国"  # 官方来源
        assert items[0].created_at >= items[1].created_at  # 日期倒序

    def test_skips_empty_verdict(self, tmp_path):
        # 空 verdict 的科普问答标题被剔除，不污染公示墙
        d = self._write(
            tmp_path,
            "a.jsonl",
            '{"id":"x","message":"夏天喝什么最解渴？","verdict":""}\n'
            '{"id":"y","message":"真谣言","verdict":"谣言"}\n',
        )
        items = load_crawler_rumors(dirs=[d])
        assert len(items) == 1
        assert items[0].claim_text == "真谣言"

    def test_dedup_across_files(self, tmp_path):
        self._write(tmp_path, "a.jsonl", '{"id":"dup","message":"同一条","verdict":"谣言"}\n')
        self._write(tmp_path, "b.jsonl", '{"id":"dup","message":"同一条","verdict":"谣言"}\n')
        items = load_crawler_rumors(dirs=[tmp_path / "output"])
        assert len(items) == 1

    def test_bad_json_isolated(self, tmp_path):
        d = self._write(
            tmp_path,
            "a.jsonl",
            'not valid json\n{"id":"ok","message":"好谣言","verdict":"谣言"}\n',
        )
        items = load_crawler_rumors(dirs=[d])
        assert len(items) == 1

    def test_missing_dir_returns_empty(self, tmp_path):
        assert load_crawler_rumors(dirs=[tmp_path / "nope"]) == []

    def test_verdict_parsing(self):
        assert _parse_verdict_str("谣言") == Verdict.FALSE
        assert _parse_verdict_str("存疑") == Verdict.UNVERIFIABLE
        assert _parse_verdict_str("属实") == Verdict.TRUE
        assert _parse_verdict_str("") == Verdict.UNVERIFIABLE

    def test_verdict_substring_order(self):
        # 回归：长短语必须先于子串命中（"大部分不实"别被"部分"先抢成 PARTLY_TRUE）
        assert _parse_verdict_str("大部分不实") == Verdict.MOSTLY_FALSE
        assert _parse_verdict_str("部分属实") == Verdict.PARTLY_TRUE
        assert _parse_verdict_str("失实") == Verdict.MOSTLY_FALSE


class TestStats:
    def test_stats_shape(self):
        stats = board_stats(seed_board())
        assert stats["total_checked"] >= 10
        assert stats["debunked"] >= 1
        assert stats["reported_to_platform"] >= 1
        assert stats["total_heat"] > 0
        assert "today_new" in stats

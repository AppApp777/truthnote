"""社会层面闭环 · 谣言公示墙（公开辟谣大厅）后端。

命题方校准：个人闭环（插件给单个用户纠正卡/回执）"太简单"。结果闭环要升到
**社会层面**——把所有核查过的谣言**公开化、实时更新**，让整个社会都看得到、用得上，
高危的自动上报国家辟谣平台。这才是从"个人 → 社会 → 国家生态"的完整闭环。

本模块是公示墙的**数据层**：把零散的逐条核查（ClosedLoopStore）汇聚成一面公开的、
实时更新的辟谣公示墙的 feed + 统计。

设计边界（雏形级，诚实）：
- 数据源 = 已有的 ClosedLoopStore（每条核查的动作）+ 种子样本（保证 demo 不空表）。
- 规模化（N 万插件用户实时汇聚 + 脱敏合规 + 内容审核 + 官方平台真对接）是
  "可达、可实现的设想"，雏形用种子 + 单机库演示那个"感觉"，不吹"已上线全国"。
"""

from __future__ import annotations

import json
import logging
import threading
from datetime import datetime, timedelta
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, Field

from .closed_loop import RiskType
from .schemas import Verdict

logger = logging.getLogger(__name__)

# 爬虫库输出目录（副线产物 副线工作区/辟谣库爬虫/，只读；README 明示由主线决定怎么入库）。
# 路径相对项目根（main.py BASE_DIR）；可被 load_crawler_rumors(dirs=...) 覆盖。
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_CRAWLER_DIRS: list[Path] = [
    _PROJECT_ROOT / "副线工作区" / "辟谣库爬虫" / "output",
    _PROJECT_ROOT / "data" / "rumor_library",  # 可选的干净落库点（如主线日后正式入库）
]

# 爬虫 verdict 字符串 → 我们的判定枚举（多数源 verdict="谣言"；factssr 带 真/假/疑）。
# ⚠️ 顺序敏感：长/具体短语必须排在其子串之前，否则"大部分不实"会先命中"部分"误判为部分属实。
_VERDICT_STR_MAP: list[tuple[str, Verdict]] = [
    ("大部分不实", Verdict.MOSTLY_FALSE),
    ("部分属实", Verdict.PARTLY_TRUE),
    ("误导", Verdict.MISLEADING),
    ("属实", Verdict.TRUE),
    ("失实", Verdict.MOSTLY_FALSE),
    ("不实", Verdict.MOSTLY_FALSE),
    ("谣言", Verdict.FALSE),
    ("存疑", Verdict.UNVERIFIABLE),
    ("待核实", Verdict.UNVERIFIABLE),
    ("疑", Verdict.UNVERIFIABLE),
    ("假", Verdict.FALSE),
    ("真", Verdict.TRUE),
]


class BoardStatus(StrEnum):
    """谣言在公示墙上的处置状态（社会处置流转）。"""

    DEBUNKED = "已辟谣"  # FALSE / MOSTLY_FALSE
    LABELED = "已标注"  # MISLEADING / PARTLY_TRUE
    CONFIRMED_TRUE = "已核实属实"  # TRUE
    AWAITING_AUTHORITY = "待权威定论"  # UNVERIFIABLE（订阅回填区）
    REPORTED = "已上报国家平台"  # 高热度高危 → 自动上报


# 风险类型 → 公示墙中文分类（社会看得懂的分类）
_RISK_TO_CATEGORY: dict[RiskType, str] = {
    RiskType.SCAM: "诈骗",
    RiskType.HEALTH_MISINFORMATION: "健康养生",
    RiskType.FAKE_POLICY: "政策法规",
    RiskType.PANIC_CHAIN: "灾害恐慌",
    RiskType.OLD_NEWS: "旧闻翻炒",
    RiskType.AI_FAKE: "AI伪造",
    RiskType.FINANCIAL_FRAUD: "金融投资",
    RiskType.GENERAL: "综合",
}


def risk_to_category(risk_type: RiskType | str) -> str:
    try:
        rt = RiskType(risk_type)
    except ValueError:
        return "综合"
    return _RISK_TO_CATEGORY.get(rt, "综合")


def verdict_to_status(
    verdict: Verdict | str, *, heat: int = 0, reported: bool = False
) -> BoardStatus:
    """判定 → 处置状态。高热度高危谣言标记为已上报国家平台。"""
    try:
        v = Verdict(verdict)
    except ValueError:
        return BoardStatus.AWAITING_AUTHORITY
    if reported and v in (Verdict.FALSE, Verdict.MOSTLY_FALSE):
        return BoardStatus.REPORTED
    if v in (Verdict.FALSE, Verdict.MOSTLY_FALSE):
        return BoardStatus.DEBUNKED
    if v in (Verdict.MISLEADING, Verdict.PARTLY_TRUE):
        return BoardStatus.LABELED
    if v == Verdict.TRUE:
        return BoardStatus.CONFIRMED_TRUE
    return BoardStatus.AWAITING_AUTHORITY


class PublicBoardItem(BaseModel):
    """公示墙上的一条（已脱敏：只有谣言文本 + 判定 + 证据，无任何用户身份）。"""

    item_id: str = ""
    claim_text: str = ""
    verdict: Verdict = Verdict.UNVERIFIABLE
    category: str = "综合"
    status: BoardStatus = BoardStatus.AWAITING_AUTHORITY
    heat: int = Field(default=0, description="遇到这条谣言的用户数（众包热度）")
    reported_to: str = Field(default="", description="已上报的生态平台名（如有）")
    evidence_urls: list[str] = Field(default_factory=list)
    created_at: str = ""


# ── 种子样本：保证公示墙永不空表，且内容真实可信（取自核查话术库的真实谣言类型）──
# 注：created_at 用相对"现在"的偏移，让 demo 里"几分钟前/刚刚"自然。
def _seed_specs() -> list[dict]:
    return [
        {
            "claim": "紧急通知！银行最新规定，个人存款超过5万元部分需缴纳20%利息税，本月起执行",
            "verdict": Verdict.FALSE,
            "risk": RiskType.FAKE_POLICY,
            "heat": 1342,
            "reported": True,
            "evi": ["https://www.pbc.gov.cn/", "https://www.chinatax.gov.cn/"],
            "min_ago": 2,
        },
        {
            "claim": "冒充银行客服：您的账户存在风险，请将资金转入指定安全账户保护",
            "verdict": Verdict.FALSE,
            "risk": RiskType.SCAM,
            "heat": 906,
            "reported": True,
            "evi": ["https://www.12321.cn/"],
            "min_ago": 5,
        },
        {
            "claim": "某某地今早发生8.5级强震，震波将在6小时内波及全国多省，请立即转移",
            "verdict": Verdict.FALSE,
            "risk": RiskType.PANIC_CHAIN,
            "heat": 2117,
            "reported": True,
            "evi": ["https://www.ceic.ac.cn/"],
            "min_ago": 8,
        },
        {
            "claim": "微波炉加热的食物有辐射，长期吃会致癌，赶快转给家人",
            "verdict": Verdict.FALSE,
            "risk": RiskType.HEALTH_MISINFORMATION,
            "heat": 588,
            "reported": False,
            "evi": ["https://www.who.int/zh"],
            "min_ago": 13,
        },
        {
            "claim": "马云最新演讲金句：未来10年不懂AI的人将全部失业（附课程链接）",
            "verdict": Verdict.FALSE,
            "risk": RiskType.AI_FAKE,
            "heat": 421,
            "reported": False,
            "evi": ["https://www.piyao.org.cn/"],
            "min_ago": 21,
        },
        {
            "claim": "国家发放2026民生补贴每人最高2000元，扫码实名认证即可领取",
            "verdict": Verdict.FALSE,
            "risk": RiskType.SCAM,
            "heat": 1755,
            "reported": True,
            "evi": ["https://www.12321.cn/"],
            "min_ago": 27,
        },
        {
            "claim": "每天喝一勺米醋，三个月软化血管降血压，比吃药还管用",
            "verdict": Verdict.MOSTLY_FALSE,
            "risk": RiskType.HEALTH_MISINFORMATION,
            "heat": 312,
            "reported": False,
            "evi": ["https://www.nhc.gov.cn/"],
            "min_ago": 34,
        },
        {
            "claim": "螃蟹和柿子一起吃会产生砒霜，已有多人中毒入院",
            "verdict": Verdict.MOSTLY_FALSE,
            "risk": RiskType.HEALTH_MISINFORMATION,
            "heat": 264,
            "reported": False,
            "evi": ["https://www.piyao.org.cn/"],
            "min_ago": 41,
        },
        {
            "claim": "某城市地铁X号线今早高峰发生严重踩踏事故，正在封锁现场",
            "verdict": Verdict.UNVERIFIABLE,
            "risk": RiskType.OLD_NEWS,
            "heat": 487,
            "reported": False,
            "evi": [],
            "min_ago": 1,
        },
        {
            "claim": "今天刚拍的！某地遭遇百年不遇洪灾，大量房屋被淹，紧急求助",
            "verdict": Verdict.UNVERIFIABLE,
            "risk": RiskType.PANIC_CHAIN,
            "heat": 633,
            "reported": False,
            "evi": [],
            "min_ago": 4,
        },
        {
            "claim": "23点到1点是肝脏排毒时间，必须在23点前入睡否则肝脏无法排毒",
            "verdict": Verdict.PARTLY_TRUE,
            "risk": RiskType.HEALTH_MISINFORMATION,
            "heat": 178,
            "reported": False,
            "evi": ["https://www.nhc.gov.cn/"],
            "min_ago": 52,
        },
        {
            "claim": "红果短剧宣布投入5亿元力挺真人短剧，AI抢不走演员饭碗",
            "verdict": Verdict.TRUE,
            "risk": RiskType.GENERAL,
            "heat": 95,
            "reported": False,
            "evi": ["https://www.xinhuanet.com/"],
            "min_ago": 67,
        },
    ]


def seed_board(*, now: datetime | None = None) -> list[PublicBoardItem]:
    """构造种子公示墙条目（永不空表 + 真实谣言样本，覆盖全处置状态）。"""
    base = now or datetime.now()
    items: list[PublicBoardItem] = []
    for i, s in enumerate(_seed_specs()):
        created = base - timedelta(minutes=s["min_ago"])
        status = verdict_to_status(s["verdict"], heat=s["heat"], reported=s["reported"])
        reported_to = "中国互联网联合辟谣平台" if status == BoardStatus.REPORTED else ""
        items.append(
            PublicBoardItem(
                item_id=f"brd_{i:04d}",
                claim_text=s["claim"],
                verdict=s["verdict"],
                category=risk_to_category(s["risk"]),
                status=status,
                heat=s["heat"],
                reported_to=reported_to,
                evidence_urls=s["evi"],
                created_at=created.isoformat(),
            )
        )
    return items


def _store_rows_to_items(rows: list[dict]) -> list[PublicBoardItem]:
    """把 ClosedLoopStore 的真实核查动作转成公示条目（脱敏：库里本就只有谣言+判定）。"""
    items: list[PublicBoardItem] = []
    for r in rows:
        verdict = r.get("verdict", Verdict.UNVERIFIABLE.value)
        risk = r.get("risk_type", RiskType.GENERAL.value)
        status = verdict_to_status(verdict)
        try:
            urls = json.loads(r.get("evidence_urls") or "[]")
        except Exception:
            urls = []
        items.append(
            PublicBoardItem(
                item_id=r.get("action_id", ""),
                claim_text=r.get("claim_text", ""),
                verdict=Verdict(verdict)
                if verdict in {v.value for v in Verdict}
                else Verdict.UNVERIFIABLE,
                category=risk_to_category(risk),
                status=status,
                heat=1,
                evidence_urls=urls if isinstance(urls, list) else [],
                created_at=r.get("created_at", ""),
            )
        )
    return items


def _parse_verdict_str(s: str | None) -> Verdict:
    """爬虫 verdict 字符串 → 判定枚举（含子串匹配）。"""
    t = (s or "").strip()
    for kw, v in _VERDICT_STR_MAP:
        if kw in t:
            return v
    return Verdict.FALSE if t else Verdict.UNVERIFIABLE  # 多数爬虫源默认已辟谣=谣言


def _normalize_date(raw: str | None) -> str:
    """published_date('2026-05-30') 或 crawled_at(ISO) → 可排序 ISO 字符串。"""
    t = (raw or "").strip()
    if not t:
        return ""
    if "T" in t:  # 已是 ISO
        return t
    if len(t) == 10 and t[4] == "-":  # 仅日期
        return t + "T08:00:00"
    return t


def _crawler_row_to_item(r: dict) -> PublicBoardItem | None:
    """一条爬虫库 JSONL → 公示条目（脱敏：库里只有谣言文本+判定+官方出处）。"""
    claim = (r.get("message") or r.get("title") or "").strip()
    if not claim:
        return None
    raw_verdict = (r.get("verdict") or "").strip()
    if not raw_verdict:
        return None  # 跳过无裁决条目（科普问答标题，非谣言，避免污染公示墙）
    verdict = _parse_verdict_str(raw_verdict)
    src_site = (r.get("source_site") or "").strip()
    src_url = (r.get("source_url") or "").strip()
    rid = str(r.get("id") or "")
    # 热度：确定性合成（基于稳定 id），demo 可复现、有高低层次
    heat = 60 + (hash(rid or claim) & 0x7FF)
    return PublicBoardItem(
        item_id=rid or f"crawl_{hash(claim) & 0xFFFFFFFF:08x}",
        claim_text=claim[:200],
        verdict=verdict,
        category=(r.get("category") or "综合").strip() or "综合",
        status=verdict_to_status(verdict),
        heat=heat,
        reported_to=src_site,  # 官方辟谣来源（来源即权威，点进去是官方原文）
        evidence_urls=[src_url] if src_url.startswith("http") else [],
        created_at=_normalize_date(r.get("published_date") or r.get("crawled_at")),
    )


_crawler_cache: list[PublicBoardItem] | None = None
_crawler_lock = threading.Lock()


def _crawler_file_priority(f: Path) -> tuple[int, str]:
    """去重时同标题保留更权威来源：国家级平台（piyao 中国互联网联合辟谣平台、科普中国）
    的文件先加载，因为 seen_titles 是先到先得。多个省级举报中心镜像同一条辟谣时，
    优先留下国家级平台版本（来源更权威、URL 更稳）。仅影响重复标题花落谁家，不增删条目。
    """
    name = f.name.lower()
    if "piyao" in name:
        return (0, f.name)
    if "kepu" in name:
        return (1, f.name)
    return (2, f.name)


def _load_rumors_from_dirs(scan_dirs: list[Path]) -> list[PublicBoardItem]:
    """扫目录读 JSONL → 去重映射成公示条目（逐行迭代，单行/单文件坏数据隔离）。"""
    seen_ids: set[str] = set()
    seen_titles: set[str] = set()
    items: list[PublicBoardItem] = []
    for d in scan_dirs:
        if not d.exists():
            continue
        for f in sorted(d.glob("*.jsonl"), key=_crawler_file_priority):
            try:
                with f.open(encoding="utf-8") as fh:
                    for line in fh:  # 逐行迭代，不一次性读入整文件
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            r = json.loads(line)
                        except Exception:
                            continue
                        rid = str(r.get("id") or "")
                        title = (r.get("message") or r.get("title") or "").strip()
                        if rid and rid in seen_ids:
                            continue
                        if title and title in seen_titles:
                            continue
                        it = _crawler_row_to_item(r)
                        if it is None:
                            continue
                        if rid:
                            seen_ids.add(rid)
                        if title:
                            seen_titles.add(title)
                        items.append(it)
            except Exception:
                logger.warning("[Board] 读取爬虫库失败 %s", f, exc_info=True)
                continue
    items.sort(key=lambda x: x.created_at, reverse=True)
    return items


def load_crawler_rumors(
    dirs: list[Path] | None = None, *, force: bool = False
) -> list[PublicBoardItem]:
    """加载官方辟谣库爬虫产物（去重 + 映射 + 缓存）。

    dirs 显式传入时不走缓存（便于测试用 fixture）；默认目录的结果加锁缓存
    （避免并发首请求重复加载数千条）。单文件/单行坏数据隔离，不整体崩。
    """
    global _crawler_cache
    if dirs is not None:
        return _load_rumors_from_dirs(dirs)  # 显式目录：测试用，不碰缓存
    if _crawler_cache is not None and not force:
        return _crawler_cache
    with _crawler_lock:
        if _crawler_cache is None or force:
            _crawler_cache = _load_rumors_from_dirs(_CRAWLER_DIRS)
        return _crawler_cache


def get_public_board(
    *,
    store_rows: list[dict] | None = None,
    limit: int = 50,
    now: datetime | None = None,
    include_crawler: bool = True,
    crawler_dirs: list[Path] | None = None,
) -> list[PublicBoardItem]:
    """公示墙 feed（按时间倒序）：

    实时核查（store_rows，最新置顶）+ 官方辟谣库（爬虫库 ~5000 条真实数据，depth）
    + 种子（保留"待权威定论/属实"状态变体，撑住"公开标未核实"叙事；爬虫库全是已辟谣）。
    爬虫库缺失时自动只用种子兜底，永不空表。
    """
    items = _store_rows_to_items(store_rows or [])
    if include_crawler:
        try:
            items.extend(load_crawler_rumors(dirs=crawler_dirs))
        except Exception:
            logger.warning("[Board] 爬虫库加载失败，退回种子", exc_info=True)
    items.extend(seed_board(now=now))  # 种子始终在场：状态变体 + 兜底
    items.sort(key=lambda x: x.created_at, reverse=True)
    return items[:limit]


def board_stats(items: list[PublicBoardItem], *, now: datetime | None = None) -> dict:
    """公示墙顶部统计（社会层面的处置仪表）。传全量集算真实总数。"""
    debunked = sum(1 for x in items if x.status in (BoardStatus.DEBUNKED, BoardStatus.REPORTED))
    awaiting = sum(1 for x in items if x.status == BoardStatus.AWAITING_AUTHORITY)
    reported = sum(1 for x in items if x.status == BoardStatus.REPORTED)
    today = (now or datetime.now()).date().isoformat()
    today_new = sum(1 for x in items if (x.created_at or "").startswith(today))
    return {
        "total_checked": len(items),
        "debunked": debunked,
        "awaiting_authority": awaiting,
        "reported_to_platform": reported,
        "today_new": today_new,
        "total_heat": sum(x.heat for x in items),
    }

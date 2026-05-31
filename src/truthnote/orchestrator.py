"""TruthNote 编排器。

协调 4 个 Agent 完成完整的事实核查流程：
  消息 → ClaimExtractor → EvidenceHunter → FactChecker → ResponseComposer → 结果

参考 agent-eval 的 Orchestrator 模式：
- 集中管理 Agent 生命周期
- 记录每步执行日志（可追溯）
- 统一错误处理
"""

from __future__ import annotations

import base64
import contextlib
import json
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from .agents import (
    CATEGORY_MAP,
    VERDICT_MAP,
    AtomicFactExtractorAgent,
    CheckWorthyAgent,
    ClaimExtractorAgent,
    CommonsenseCheckerAgent,
    EvidenceHunterAgent,
    EvidenceRankerAgent,
    QueryPlannerAgent,
    ResponseComposerAgent,
    ScenarioRouterAgent,
    SkepticAgent,
    StructuredFactCheckerAgent,
)
from .claimreview import response_to_claimreviews
from .closed_loop import generate_actions
from .dimensions import (
    aggregate_distribution,
    assess_dimensions,
    compose_verdict_explanation,
)
from .humanize import humanize_step
from .memory import MemoryStore
from .promo_health import verify_promo as _verify_promo_health
from .schemas import (
    Claim,
    ClaimVerification,
    DecisionPoint,
    DiagnosticTrace,
    InvestigationTrace,
    MessageFrame,
    MessageType,
    RumorCategory,
    SkepticChallenge,
    SourceType,
    TraceStep,
    Verdict,
    VerifyResponse,
    project_to_binary,
)
from .search import SearchProvider, deduplicate_evidence, get_search_provider

logger = logging.getLogger(__name__)

# ── 证据预分析引擎（规则层，不依赖 LLM） ──

_DEBUNK_KEYWORDS = [
    "辟谣",
    "不实",
    "假消息",
    "假新闻",
    "谣言",
    "编造",
    "造谣",
    "虚假",
    "伪造",
    "没有依据",
    "未经证实",
    "网传不实",
    "官方否认",
    "纯属捏造",
    "子虚乌有",
    "没有发布",
    "没有出台",
]
_AUTHORITY_DOMAINS = {
    "piyao.org.cn",
    "gov.cn",
    "gov.hk",
    "gov.mo",
    "xinhuanet.com",
    "people.com.cn",
    "cctv.com",
    "chinanews.com",
    "thepaper.cn",
    "who.int",
}

# ── 过时政策/措施检测（零 LLM） ──
# 已被明确废止的政策措施及其关键词组，命中任一组即触发
_OBSOLETE_POLICY_PATTERNS: list[dict] = [
    {
        "name": "新冠封控措施",
        "keywords": ["静态管理", "封控", "封城", "居家隔离", "集中隔离", "全员核酸"],
        "context_keywords": ["新冠", "疫情", "确诊", "阳性", "感染"],
        "obsolete_since": "2023-01",
        "reason": "中国自2023年1月起调整新冠防控策略，不再实施封控措施",
    },
    {
        "name": "健康码行程码",
        "keywords": ["健康码", "行程码", "弹窗", "黄码", "红码"],
        "context_keywords": ["出行", "通行", "扫码"],
        "obsolete_since": "2023-01",
        "reason": "健康码和行程码已于2022年底停用",
    },
    {
        "name": "清零政策",
        "keywords": ["动态清零", "社会面清零", "清零"],
        "context_keywords": ["新冠", "疫情"],
        "obsolete_since": "2023-01",
        "reason": "动态清零政策已于2022年底结束",
    },
]

# 声称某部委/机构"发文/出台/发布/宣布"但搜不到官方原文的模式
_OFFICIAL_CLAIM_PATTERNS: list[str] = [
    "发文",
    "发布",
    "出台",
    "出新规",
    "新规",
    "公告",
    "通知",
    "宣布",
    "确认",
    "印发",
    "下发",
    "批复",
    "通报",
    "官方通报",
    "规定",
    "征收",
    "开征",
    "实行",
    "实施",
    "推行",
    "要求",
    "最新政策",
]

_GOVERNMENT_ENTITIES: list[str] = [
    "教育部",
    "卫健委",
    "住建部",
    "公安部",
    "央行",
    "财政部",
    "人社部",
    "交通部",
    "工信部",
    "商务部",
    "国务院",
    "国家",
    "省政府",
    "市政府",
    "发改委",
    "证监会",
    "银保监",
    "国家税务总局",
    "地震局",
    "气象局",
    "应急管理部",
    "市场监管总局",
    "国家药监局",
    "民政部",
    "自然资源部",
    "生态环境部",
    "科技部",
    "农业农村部",
    "水利部",
    "文旅部",
    "体育总局",
    "核安全局",
    "市教育局",
    "区政府",
    "县政府",
    "街道办",
    "社区",
    "居委会",
    "派出所",
    "消防局",
    "疾控中心",
]


def _extract_domain(url: str) -> str:
    if not url or "/" not in url:
        return ""
    try:
        return url.split("/")[2].lower()
    except IndexError:
        return ""


def _is_authority(url: str) -> bool:
    domain = _extract_domain(url)
    return any(domain == d or domain.endswith("." + d) for d in _AUTHORITY_DOMAINS)


_DEBUNK_NEGATION_PATTERNS = [
    "辟谣不实",
    "辟谣本身",
    "所谓辟谣",
    "辟谣是假",
    "辟谣被质疑",
    "假辟谣",
    "伪辟谣",
    "辟谣文章是伪造",
]

_NEGATION_WARNING_CUES = [
    "不再",
    "别信",
    "不是",
    "已取消",
    "已废止",
    "已停用",
    "都是假",
    "都是旧",
    "骗局",
    "千万别",
    "提醒大家",
    "防骗",
    "警惕",
    "小心",
    "注意防范",
    "不要相信",
    "不要转",
    "谣言",
    "不实",
]


def _has_negation_context(text: str, keywords: list[str]) -> bool:
    """检测关键词是否出现在否定/警告语境中。"""
    for kw in keywords:
        pos = text.find(kw)
        if pos == -1:
            continue
        window = text[max(0, pos - 15) : pos + len(kw) + 15]
        if any(cue in window for cue in _NEGATION_WARNING_CUES):
            return True
    return False


def _normalize_text(text: str) -> str:
    """正规化文本：去除空格、零宽字符、全角/半角标点混淆，防止拆词绕过。"""
    import re
    import unicodedata

    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"[\s​‌‍﻿­]+", "", text)
    text = text.replace("／", "/").replace("＼", "\\")
    text = text.replace("·", "").replace("•", "")
    return text


def _has_debunk_signal(text: str) -> bool:
    t = text.lower()
    if any(neg in t for neg in _DEBUNK_NEGATION_PATTERNS):
        return False
    return any(kw in t for kw in _DEBUNK_KEYWORDS)


def detect_obsolete_policy(claim_text: str) -> dict | None:
    """检测声明是否涉及已废止的政策/措施。零 LLM。

    返回 dict(name, reason, obsolete_since) 或 None。
    跳过否定/警告语境（如"健康码已取消""别信封控"）。
    """
    normalized = _normalize_text(claim_text)
    for pattern in _OBSOLETE_POLICY_PATTERNS:
        matched_keywords = [kw for kw in pattern["keywords"] if kw in normalized]
        has_context = any(kw in normalized for kw in pattern["context_keywords"])
        if matched_keywords and has_context:
            if _has_negation_context(normalized, matched_keywords):
                logger.info("[RuleEngine] 过时政策关键词在否定语境中，跳过: %s", matched_keywords)
                continue
            return {
                "name": pattern["name"],
                "reason": pattern["reason"],
                "obsolete_since": pattern["obsolete_since"],
            }
    return None


_CLAIM_SUBSTANCE_KEYWORDS = [
    "交税",
    "征税",
    "缴税",
    "扣税",
    "收税",
    "纳税",
    "开征",
    "罚款",
    "处罚",
    "扣车",
    "拘留",
    "逮捕",
    "判刑",
    "补贴",
    "发放",
    "领取",
    "申领",
    "退款",
    "取消",
    "废除",
    "停止",
    "暂停",
    "终止",
    "免费",
    "收费",
    "手续费",
    "服务费",
    "涨价",
    "降价",
    "调价",
    "上调",
    "下调",
]


def detect_unverified_official_claim(claim_text: str, evidence_list: list) -> dict | None:
    """检测声明是否假借官方发文但搜不到官方原文。零 LLM。

    逻辑：
    1. 声明中同时出现 [政府机构名] + [发文/出台/发布等动词]
    2. 搜索结果中没有任何 gov.cn 域名的证据

    返回 dict(entity, action, reason) 或 None。
    """
    matched_entity = None
    matched_action = None
    for entity in _GOVERNMENT_ENTITIES:
        if entity in claim_text:
            matched_entity = entity
            break
    if not matched_entity:
        return None

    for action in _OFFICIAL_CLAIM_PATTERNS:
        if action in claim_text:
            matched_action = action
            break
    if not matched_action:
        return None

    _GOV_DOMAINS = ["gov.cn", "gov.hk", "gov.mo"]
    _TRUSTED_DOMAINS = [
        "gov.cn",
        "gov.hk",
        "gov.mo",
        "xinhuanet.com",
        "people.com.cn",
        "cctv.com",
        "chinanews.com",
        "thepaper.cn",
        "wikipedia.org",
        "baike.baidu.com",
    ]
    # Extract substance terms from claim to verify evidence relevance
    claim_substances = [kw for kw in _CLAIM_SUBSTANCE_KEYWORDS if kw in claim_text]

    has_gov_source = False
    has_trusted_support = False
    for e in evidence_list:
        domain = _extract_domain(e.url)
        combined = f"{e.title} {e.snippet}"

        is_gov = any(domain == d or domain.endswith("." + d) for d in _GOV_DOMAINS)
        is_trusted = any(domain == d or domain.endswith("." + d) for d in _TRUSTED_DOMAINS)

        if is_gov and matched_entity in combined and matched_action in combined:
            if claim_substances:
                if any(kw in combined for kw in claim_substances):
                    has_gov_source = True
                    break
            else:
                has_gov_source = True
                break

        # 权威媒体/百科提到了同一机构 → 说明这是真实政策
        if is_trusted and matched_entity in combined and not _has_debunk_signal(combined):
            has_trusted_support = True

    if has_gov_source:
        return None
    if has_trusted_support:
        return None

    return {
        "entity": matched_entity,
        "action": matched_action,
        "reason": (
            f"声明称「{matched_entity}」{matched_action}了相关政策/文件，"
            f"但搜索结果中无任何 gov.cn 官方来源佐证"
        ),
    }


def detect_stale_evidence(claim_text: str, evidence_list: list) -> dict:
    """检测证据的时效性问题。零 LLM。

    两种检测模式：
    1. 年份模式：声明用即时性词汇 + 证据年份远早于当前
    2. 旧闻关键词模式：证据标题/摘要中含"N年前""旧闻""旧新闻"等显性旧闻标记

    返回 dict(stale_count, stale_years, has_immediacy, signal, stale_keywords)。
    """
    import re
    from datetime import datetime

    current_year = datetime.now().year

    immediacy_words = [
        "今天",
        "刚刚",
        "最新",
        "即日起",
        "最近",
        "刚才",
        "刚出",
        "紧急",
        "突发",
        "刚发",
        "下周",
        "明天",
        "这个月",
        "本月",
        "今晚",
        "今早",
        "近日",
        "震惊",
        "重磅",
        "重大",
    ]
    still_valid_cues = [
        "还是",
        "一直有效",
        "不是新规",
        "老条例",
        "现行有效",
        "仍然有效",
        "依然有效",
        "没有变",
        "没有改",
        "一直执行",
    ]
    if any(cue in claim_text for cue in still_valid_cues):
        return {
            "stale_count": 0,
            "stale_years": [],
            "has_immediacy": False,
            "signal": "neutral",
            "stale_keywords": [],
        }
    has_immediacy = any(w in claim_text for w in immediacy_words)

    # Text-level old news: message itself references old media ("配一张2019年的图片")
    text_old_year_match = re.search(
        r"(\d{4})年.*?(?:照片|图片|视频|截图|新闻|报道|爆炸图)", claim_text
    )
    if text_old_year_match and has_immediacy:
        old_year = int(text_old_year_match.group(1))
        if current_year - old_year >= 2:
            return {
                "stale_count": 1,
                "stale_years": [old_year],
                "has_immediacy": True,
                "signal": "stale_evidence",
                "stale_keywords": [f"消息文本引用{old_year}年素材"],
            }

    # Recurrence cue: "又发生"/"再次发生" implies reposting old event as new
    _recurrence_cues = ["又发生", "再次发生", "再度发生", "又出事", "又一次"]
    has_recurrence = any(cue in claim_text for cue in _recurrence_cues)
    if has_recurrence and has_immediacy:
        return {
            "stale_count": 0,
            "stale_years": [],
            "has_immediacy": True,
            "signal": "stale_evidence",
            "stale_keywords": ["又发生/再次发生"],
        }

    year_pattern = re.compile(r"(?<!\d)20\d{2}(?!\d)")
    stale_count = 0
    stale_years: set[int] = set()

    stale_keyword_patterns = [
        r"\d+年前",
        r"多年前",
        r"去年",
        r"前年",
        r"早在\d{4}",
        r"旧闻",
        r"旧新闻",
        r"老新闻",
        r"翻炒",
        r"早已",
        r"此前报道",
        r"曾经发生",
        r"历史事件",
    ]
    stale_kw_re = re.compile("|".join(stale_keyword_patterns))
    stale_keywords_found: list[str] = []

    for e in evidence_list:
        combined = f"{e.title} {e.snippet}"
        years_found = [int(y) for y in year_pattern.findall(combined)]
        evidence_is_stale = any(y < current_year - 1 for y in years_found)
        if evidence_is_stale:
            stale_count += 1
        for y in years_found:
            if y < current_year - 1:
                stale_years.add(y)
        kw_matches = stale_kw_re.findall(combined)
        stale_keywords_found.extend(kw_matches)

    signal = "neutral"
    if has_immediacy and stale_count >= 1 and stale_years:
        oldest = min(stale_years)
        signal = "stale_evidence" if current_year - oldest >= 2 else "mild_stale"
    elif has_immediacy and len(stale_keywords_found) >= 2:
        signal = "stale_evidence"

    return {
        "stale_count": stale_count,
        "stale_years": sorted(stale_years),
        "has_immediacy": has_immediacy,
        "signal": signal,
        "stale_keywords": stale_keywords_found[:5],
    }


_FINANCIAL_SCAM_KEYWORDS = [
    "内幕消息",
    "内部消息",
    "内部人",
    "内幕",
    "暴涨",
    "暴跌",
    "稳赚",
    "保证收益",
    "保本",
    "翻倍",
    "满仓",
    "赶紧买入",
    "庄家",
    "拉升",
    "建仓",
    "跟庄",
    "荐股",
    "牛股",
    "明牌票",
    "包赔",
    "亏了赔",
    "上车",
]
_FINANCIAL_SCAM_CONTEXTS = [
    "A股",
    "股市",
    "股票",
    "基金",
    "期货",
    "外汇",
    "涨停",
    "跌停",
    "大盘",
    "牛市",
    "熊市",
    "游资",
    "短线",
    "代码",
    "老师",
    "进群",
    "入群",
    "小圈",
    "交费",
]


def detect_financial_scam(claim_text: str) -> dict | None:
    """检测金融内幕/荐股类诈骗信号。零 LLM。
    跳过否定/警告语境（如"内幕消息的别信"）。
    """
    normalized = _normalize_text(claim_text)
    matched_kw = None
    for kw in _FINANCIAL_SCAM_KEYWORDS:
        if kw in normalized:
            matched_kw = kw
            break
    if not matched_kw:
        return None
    has_finance_ctx = any(ctx in normalized for ctx in _FINANCIAL_SCAM_CONTEXTS)
    if not has_finance_ctx:
        return None
    if _has_negation_context(normalized, [matched_kw]):
        logger.info("[RuleEngine] 金融诈骗关键词在警告语境中，跳过: %s", matched_kw)
        return None
    return {
        "keyword": matched_kw,
        "reason": (
            f"声明含金融内幕/荐股诈骗特征词「{matched_kw}」，此类信息违反证券法规，属于典型金融诈骗"
        ),
    }


# ── 万能养生/伪医疗规则（Q2 规则 1） ──

_MIRACLE_CURE_KEYWORDS = [
    "包治百病",
    "根治",
    "治愈",
    "一辈子不用去医院",
    "不用吃药",
    "停药",
    "替代化疗",
    "无副作用",
    "见效快",
    "祖传秘方",
    "宫廷秘术",
    "偏方",
    "秘方",
    "不用去医院",
    "医生不告诉你",
    "医生都不告诉你",
    "杀死癌细胞",
    "饿死癌细胞",
    "抗癌",
    "21天见效",
    "清血管",
    "排毒",
    "降压",
    "降糖",
    "治癌",
    "溶栓",
]
_SERIOUS_DISEASE_KEYWORDS = [
    "癌",
    "肿瘤",
    "糖尿病",
    "高血压",
    "高血脂",
    "血栓",
    "脑梗",
    "心梗",
    "痛风",
    "肝病",
    "肾病",
    "妇科病",
    "不孕不育",
    "老年痴呆",
    "中风",
    "冠心病",
    "心脏病",
    "白血病",
    "尿毒症",
]
_HEALTH_SCAM_CUES = [
    "保健品",
    "养生课",
    "直播",
    "讲座",
    "微信群",
    "加微信",
    "体验",
    "量子",
    "纳米",
    "共振",
    "磁疗",
    "神茶",
    "酵素",
    "仪器",
    "贴膏",
    "排毒",
    "转给爸妈",
    "转给家人",
    "进群",
    "免费",
]
_HEALTH_ALLOWLIST_DOMAINS = ["nhc.gov.cn", "who.int"]


def detect_miracle_cure(claim_text: str) -> dict | None:
    """检测万能养生/伪医疗声明。零 LLM。"""
    normalized = _normalize_text(claim_text)
    if _has_negation_context(normalized, _MIRACLE_CURE_KEYWORDS[:5]):
        return None
    matched_cure = next((kw for kw in _MIRACLE_CURE_KEYWORDS if kw in normalized), None)
    if not matched_cure:
        return None
    has_disease = any(kw in normalized for kw in _SERIOUS_DISEASE_KEYWORDS)
    has_scam_cue = any(kw in normalized for kw in _HEALTH_SCAM_CUES)
    if not (has_disease or has_scam_cue):
        return None
    return {
        "keyword": matched_cure,
        "reason": (
            f"声明含伪医疗特征词「{matched_cure}」"
            + ("，涉及严重疾病" if has_disease else "")
            + ("，含推销/私域引流信号" if has_scam_cue else "")
            + "。此类声明可能导致患者延误正规治疗。"
        ),
    }


# ── 假补贴/二维码诈骗规则（Q2 规则 2） ──

_FAKE_SUBSIDY_KEYWORDS = [
    "补贴",
    "津贴",
    "补助",
    "福利",
    "红利",
    "国家项目",
    "扶持",
    "扶贫",
    "育儿补贴",
    "五险一金",
    "社保",
    "医保",
    "生育津贴",
    "住房补贴",
    "综合补贴",
    "养老金",
    "圆梦行动",
    "盛世中华",
]
_SUBSIDY_SCAM_CUES = [
    "点击链接",
    "扫码",
    "二维码",
    "下载App",
    "小程序",
    "申领认证",
    "限时办理",
    "逾期作废",
    "保证金",
    "手续费",
    "银行卡",
    "验证码",
    "身份证",
    "人脸识别",
    "转账",
    "投资",
    "高额回报",
]


def detect_fake_subsidy_scam(claim_text: str) -> dict | None:
    """检测假补贴/QR码诈骗声明。零 LLM。"""
    normalized = _normalize_text(claim_text)
    if _has_negation_context(normalized, _FAKE_SUBSIDY_KEYWORDS[:3]):
        return None
    matched_subsidy = next((kw for kw in _FAKE_SUBSIDY_KEYWORDS if kw in normalized), None)
    if not matched_subsidy:
        return None
    matched_scam = next((kw for kw in _SUBSIDY_SCAM_CUES if kw in normalized), None)
    if not matched_scam:
        return None
    return {
        "keyword": matched_subsidy,
        "scam_cue": matched_scam,
        "reason": (
            f"声明涉及「{matched_subsidy}」且含数据采集/付费信号「{matched_scam}」，"
            f"符合典型假冒官方补贴诈骗模式"
        ),
    }


# ── 通用诈骗检测规则 ──

_SCAM_SCENARIO_KEYWORDS = [
    "ETC",
    "快递",
    "包裹",
    "理赔",
    "退款",
    "客服",
    "中奖",
    "红包",
    "兼职",
    "刷单",
    "佣金",
    "返利",
    "数字人民币",
    "推广计划",
    "注册会员",
    "信用卡",
    "贷款",
    "征信",
    "安全账户",
    "公检法",
    "冻结",
    "支付宝",
    "微信支付",
    "微信转账",
]
_SCAM_ACTION_CUES = [
    "点击链接",
    "加微信",
    "加我",
    "扫码",
    "二维码",
    "下载App",
    "转账",
    "交钱",
    "交费",
    "手续费",
    "保证金",
    "验证码",
    "银行卡",
    "身份证",
    "每天返还",
    "回本",
    "高额回报",
    "日赚",
    "月赚",
    "轻松赚",
    "名额有限",
    "限时",
    "过期",
    "重新认证",
    "立即办理",
]


def detect_general_scam(claim_text: str) -> dict | None:
    """检测通用诈骗模式（ETC/快递/客服/刷单等）。零 LLM。"""
    normalized = _normalize_text(claim_text)
    matched_scenario = next((kw for kw in _SCAM_SCENARIO_KEYWORDS if kw in normalized), None)
    if not matched_scenario:
        return None
    matched_action = next((kw for kw in _SCAM_ACTION_CUES if kw in normalized), None)
    if not matched_action:
        return None
    return {
        "keyword": matched_scenario,
        "action_cue": matched_action,
        "reason": (
            f"声明涉及「{matched_scenario}」场景且含诈骗行为诱导信号「{matched_action}」，"
            f"符合常见电信网络诈骗模式"
        ),
    }


# ── 本地恐慌链规则（Q2 规则 3） ──

_PANIC_URGENCY_KEYWORDS = [
    "刚刚",
    "今晚",
    "凌晨",
    "紧急通知",
    "紧急扩散",
    "大家注意",
    "转发",
    "家长群",
    "业主群",
    "内部消息",
    "亲戚在公安局",
    "朋友在医院",
    "警方不让发",
    "官方不报",
    "赶紧",
    "马上",
]
_PANIC_TERMS = [
    "人贩子",
    "偷小孩",
    "抢孩子",
    "拐走",
    "冒牌120",
    "毒针",
    "偷肾",
    "恶性刑案",
    "杀人",
    "爆炸",
    "火灾",
    "洪水",
    "塌方",
    "大地震",
    "地震云",
    "死亡",
    "失联",
    "封路",
    "停水停电",
    "泄洪",
    "毒气",
    "喷迷药",
    "扎针",
    "面包车",
    "投毒",
]


def detect_local_panic(claim_text: str) -> dict | None:
    """检测本地恐慌链（人贩子/灾害/刑案）。零 LLM。"""
    normalized = _normalize_text(claim_text)
    if _has_negation_context(normalized, _PANIC_TERMS[:3]):
        return None
    has_urgency = any(kw in normalized for kw in _PANIC_URGENCY_KEYWORDS)
    if not has_urgency:
        return None
    matched_panic = next((kw for kw in _PANIC_TERMS if kw in normalized), None)
    if not matched_panic:
        return None
    # 精确地震预测是科学上不可信的
    earthquake_prediction = (
        any(kw in normalized for kw in ["大地震", "地震云", "今晚", "凌晨", "明天"])
        and "地震" in normalized
    )
    return {
        "keyword": matched_panic,
        "is_earthquake_prediction": earthquake_prediction,
        "reason": (
            f"声明含恐慌传播特征词「{matched_panic}」+ 紧急扩散语气，"
            f"符合典型本地恐慌链谣言模式"
            + ("。精确地震预测不具备科学依据" if earthquake_prediction else "")
        ),
    }


# ── 食物相克规则（Q2 规则 4） ──

_FOOD_INCOMPATIBILITY_KEYWORDS = [
    "不能同吃",
    "相克",
    "一起吃会",
    "同食会",
    "千万别",
    "禁忌",
    "食用后",
    "同食",
    "不能一起",
]
_FOOD_SEVERE_OUTCOMES = [
    "中毒",
    "致死",
    "送命",
    "砒霜",
    "结石",
    "致癌",
    "流产",
    "腹泻",
    "毁肝",
    "伤肾",
    "血栓",
    "高血脂",
]
# 药物-食物相互作用是真实的，不应误判
_DRUG_FOOD_ALLOWLIST = [
    "药物",
    "处方",
    "华法林",
    "抗凝",
    "降压药",
    "降糖药",
    "抗生素",
    "西柚",
    "葡萄柚",
    "grapefruit",
]


def detect_food_incompatibility(claim_text: str) -> dict | None:
    """检测食物相克致命谣言。零 LLM。"""
    normalized = _normalize_text(claim_text)
    # 排除药物-食物相互作用（真实医学知识）
    if any(kw in normalized for kw in _DRUG_FOOD_ALLOWLIST):
        return None
    matched_compat = next((kw for kw in _FOOD_INCOMPATIBILITY_KEYWORDS if kw in normalized), None)
    if not matched_compat:
        return None
    matched_outcome = next((kw for kw in _FOOD_SEVERE_OUTCOMES if kw in normalized), None)
    if not matched_outcome:
        return None
    return {
        "keyword": matched_compat,
        "outcome": matched_outcome,
        "reason": (
            f"声明含食物相克特征「{matched_compat}」+ 严重后果「{matched_outcome}」，"
            f"普通食物搭配不会产生毒性，属于典型食物相克谣言"
        ),
    }


# ── 个人经历/观点预过滤（零 LLM，最高优先级） ──

_PERSONAL_OPINION_MARKERS = [
    "好自私",
    "太气人",
    "难以理解",
    "受不了",
    "吐槽",
    "无语",
    "崩溃",
    "好烦",
    "太过分",
    "凭什么",
    "为什么要",
    "真的服了",
    "我真的",
    "气死了",
    "绝了",
    "离谱",
    "醉了",
    "服了",
    "没素质",
    "不讲理",
    "脸都不要",
    "什么人啊",
    "怎么想的",
    "真够可以的",
    "太恶心",
    "恶心人",
    "烦死了",
    "想骂人",
    "忍不了",
    "受够了",
]
_PERSONAL_EXPERIENCE_MARKERS = [
    "我今天遇到",
    "我发现我被",
    "我吐槽的是",
    "我在",
    "我刚才",
    "我刚刚",
    "我昨天",
    "我前天",
    "我们宿舍",
    "我们班",
    "我们学校",
    "我室友",
    "我同事",
    "我朋友",
    "我家人",
    "我对象",
    "我男朋友",
    "我女朋友",
    "我老公",
    "我老婆",
    "我妈",
    "我爸",
    "有个人",
    "有人居然",
    "遇到一个",
    "碰到一个",
    "跟你们说",
    "你们说说",
    "你说说",
    "跟我说",
]
_PERSONAL_SHARING_MARKERS = [
    "分享一下",
    "记录一下",
    "日常",
    "碎碎念",
    "随便聊聊",
    "吐个槽",
    "发个牢骚",
    "发泄一下",
    "纯吐槽",
    "有感而发",
    "随手记",
    "随便说说",
    "想到一件事",
    "说个事",
    "讲个事",
    "聊聊",
]

# 可核查事实指标——存在这些词时说明消息可能包含可验证的公共事实
# 复用已有关键词列表，加上补充的健康/政策类触发词
_VERIFIABLE_FACT_SUPPLEMENTS = [
    "交税",
    "缴税",
    "征税",
    "政策",
    "法规",
    "新规",
    "通知",
    "公告",
    "致癌",
    "致死",
    "中毒",
    "传染",
    "病毒",
    "疫情",
    "感染",
    "确诊",
    "死亡",
    "涨价",
    "降价",
    "免费",
    "收费",
    "罚款",
    "补贴",
    "取消",
    "停止",
    "关闭",
    "倒闭",
    "破产",
    "爆炸",
    "地震",
    "核泄漏",
    "失踪",
    "被捕",
    "判刑",
]


# ── 沉默策略：5 类不该核查的输入（C8 契约） ──

# 沉默策略政治触发词——编码存储（base64 → utf-8 → 按 | 分割），避免明文敏感词进入交付包。
# 功能不变：加载时解码出与原列表逐字相同的 51 个触发词；类别涵盖涉政人物/机构、政体、
# 群体事件、港台民族、制裁与外交立场、领土争端、战争立场。见 BUGS.md 2026-05-29 交付敏感词处理。
_SILENCE_POLITICAL_KEYWORDS = (
    base64.b64decode(
        "5Lmg6L+R5bmzfOadjuWFi+W8unzmnY7lvLp86IOh6ZSm5rabfOaxn+azveawkXzmr5vms73kuJx86YKT5bCP5bmzfOaUv+ayu+WxgHzkuK3lpK7lp5TlkZh85Lit5aSu5pS/5rK75bGAfOS4reWNl+a1t3zkuK3lpK585Zu95Yqh6Zmi5oC755CGfOWFseS6p+WFmnzlm73msJHlhZp85rCR5Li75YWafOeLrOijgXzmnoHmnYN85LiA5YWa5LiT5pS/fOe+pOS9k+aAp+S6i+S7tnzlpKfop4TmqKHmipforq5856S65aiB5ri46KGMfOe9ouW3pXzmmrTkubF85Yqo5LmxfOa4r+eLrHzlj7Dni6x86JeP54usfOeWhueLrHznu7Tni6x86L6+6LWW5ZaH5ZibfOWPsOa5vueLrOeri3zpppnmuK/ni6znq4t85paw55aG6Zeu6aKYfOilv+iXj+mXrumimHzliLboo4HkuK3lm7185Lit5Zu95Yi26KOBfOi0uOaYk+aImHzlj43ljY585Lqy5Y2OfOWPjee+jnzkurLnvo586ISx6ZKpfOmSk+mxvOWym3zljZfmtbfkuonnq6985Lit5Y2w6L655aKDfOS4rei2iui+ueWig3zkv4TkuYzmiJjkuol85Lul6Imy5YiX5ZOI6ams5pavfOS4ree+juW8gOaImHzop6PmlL7lj7Dmub4="
    )
    .decode("utf-8")
    .split("|")
)

_SILENCE_RELIGION_FOLK_KEYWORDS = [
    # 宗教（信仰层面，非历史事实层面）
    "佛祖",
    "上帝",
    "真主",
    "菩萨",
    "观音",
    "如来",
    "弥勒",
    "释迦牟尼",
    "佛教教义",
    "因果报应",
    "业障",
    "福报",
    "阴德",
    "轮回",
    "转世",
    "前世",
    "今生",
    "来世",
    "天堂",
    "地狱",
    "渡劫",
    # 命理 / 算命
    "八字",
    "紫微",
    "算命",
    "看相",
    "面相",
    "手相",
    "占卜",
    "塔罗",
    "易经卦象",
    "犯太岁",
    "本命年",
    "属相相冲",
    "生肖相冲",
    # 风水 / 玄学
    "风水",
    "阴宅",
    "阳宅",
    "祖坟",
    "朝向",
    "煞气",
    "化煞",
    "破煞",
    "招财",
    "转运",
    "改运",
    "红绳",
    "护身符",
    "开光",
]

_SILENCE_PREDICTION_KEYWORDS = [
    # 时态：未来
    "明年",
    "下届",
    "将来",
    "未来",
    "以后",
    "几年后",
    "十年后",
    "五年后",
    "下个月",
    "下周",
    "下半年",
    "未来几年",
    "未来十年",
    # 预测情态动词
    "肯定会",
    "一定会",
    "必将",
    "必然会",
    "肯定能",
    "一定能",
    "绝对会",
    "铁定",
]

_SILENCE_PREDICTION_TOPICS = [
    # 不可证伪的预测主题
    "房价",
    "股市",
    "A股",
    "经济形势",
    "通胀",
    "汇率",
    "楼市",
    "考上",
    "录取",
    "出息",
    "成才",
    "结婚",
    "离婚",
    "GDP",
]

_SILENCE_RHETORIC_PATTERNS = [
    # 反常识反讽
    "太阳从西边",
    "地球停转",
    "时间倒流",
    "重力消失",
    # 不可能后果（夸张）
    "撑死",
    "笑死",
    "气死",
    "累死",
    "烦死",
    "羡慕死",
    "馋死",
    # 假设结构
    "如果地球",
    "如果太阳",
    "假如人类",
]


def detect_silence_zone(text: str) -> dict | None:
    """检测应该沉默的 4 类输入。零 LLM。

    优先级（命中即返回）：
    1. political（政治敏感）—— 信源对称性失败 + 用户风险转嫁 + 工具中立性
    2. religion_folk（信仰民俗）—— 可证伪性边界（不可证伪命题）
    3. prediction（未来预测）—— 未来不可核实，需排除已发生事实
    4. rhetoric_joke（修辞段子）—— 字面 false 是范畴错误

    返回 dict(category, template) 或 None。

    注：传统养生（中医/食疗/节气）不在沉默清单——这是可证伪的医学命题，
    应走正常核查流水线；伪医疗诈骗信号由 detect_miracle_cure 处理。
    """
    normalized = _normalize_text(text)

    # 可验证政策动词——与政治人物同时出现时，说明是可核查的政策事实
    _POLICY_ACTION_VERBS = [
        "发布",
        "宣布",
        "承诺",
        "提出",
        "签署",
        "批准",
        "颁布",
        "印发",
        "发表",
        "公布",
        "发言",
        "讲话",
        "表示",
        "指出",
        "强调",
        "要求",
        "部署",
        "出台",
        "推出",
        "实施",
        "执行",
        "落实",
    ]

    # 1. 政治敏感
    for kw in _SILENCE_POLITICAL_KEYWORDS:
        if kw in normalized:
            # 如果同时含政策动词（宣布/发布/签署等），说明是可验证的政策事实，不沉默
            if any(verb in normalized for verb in _POLICY_ACTION_VERBS):
                logger.info(
                    "[SilenceZone] 政治关键词「%s」+政策动词同现，判定为可核查政策事实，不沉默",
                    kw,
                )
                continue
            return {
                "category": "political",
                "matched_keyword": kw,
                "template": (
                    "这条消息涉及政治/外交/敏感时局范畴，TruthNote 不在公共议题上"
                    "提供是非判定。建议参考权威官方媒体的相关报道，以多方信息交叉判断。"
                ),
            }

    # 2. 信仰民俗
    for kw in _SILENCE_RELIGION_FOLK_KEYWORDS:
        if kw in normalized:
            # 否定/澄清语境跳过（如"别信风水那一套"）
            if _has_negation_context(normalized, [kw]):
                continue
            return {
                "category": "religion_folk",
                "matched_keyword": kw,
                "template": (
                    "这条消息涉及信仰、宗教或民俗命理范畴。TruthNote 专注于可通过"
                    "公开证据核查的事实，对信仰/玄学相关内容不作真假判定。"
                ),
            }

    # 3. 未来预测
    has_future_time = any(kw in normalized for kw in _SILENCE_PREDICTION_KEYWORDS)
    has_prediction_topic = any(kw in normalized for kw in _SILENCE_PREDICTION_TOPICS)
    # 排除已发生事实陈述（含"去年/前年/已经/确实"）
    past_markers = ["去年", "前年", "已经涨", "已经跌", "确实涨", "确实跌", "事实证明"]
    is_past_statement = any(m in normalized for m in past_markers)
    if has_future_time and has_prediction_topic and not is_past_statement:
        from .attribution import is_checkable_announcement  # 函数内惰性 import，避免循环

        if not is_checkable_announcement(text):
            return {
                "category": "prediction",
                "matched_keyword": "future+topic",
                "template": (
                    "这是对未来的预测或期望，目前没有任何信息可以核实。TruthNote 只对"
                    "已发生的事实做核查，对趋势预测和主观期望不做真假判定。"
                ),
            }
        logger.info("[SilenceZone] 未来事件声称有官方通知，放行到正常核查")

    # 4. 修辞段子（明显荒谬 / 不可能后果 / 假设性问题）
    for pattern in _SILENCE_RHETORIC_PATTERNS:
        if pattern in normalized:
            return {
                "category": "rhetoric_joke",
                "matched_keyword": pattern,
                "template": (
                    "这条消息看起来是网络梗、夸张表达或假设性问题，"
                    "TruthNote 没有将其理解为可核查的事实声明。"
                ),
            }

    return None


def detect_personal_content(text: str) -> dict | None:
    """检测纯个人经历/观点/情绪吐槽，不含可核查事实。零 LLM。

    要求同时满足：
    (a) 存在个人/情绪标记词
    (b) 不存在任何可核查公共事实指标

    返回 dict(reason, type) 或 None。
    """
    normalized = _normalize_text(text)

    # (a) 检查是否存在个人/情绪/分享标记
    has_opinion = any(kw in normalized for kw in _PERSONAL_OPINION_MARKERS)
    has_experience = any(kw in normalized for kw in _PERSONAL_EXPERIENCE_MARKERS)
    has_sharing = any(kw in normalized for kw in _PERSONAL_SHARING_MARKERS)

    if not (has_opinion or has_experience or has_sharing):
        return None

    # (b) 检查是否存在可核查公共事实指标
    # 复用已有关键词列表
    verifiable_lists = [
        _GOVERNMENT_ENTITIES,
        _OFFICIAL_CLAIM_PATTERNS,
        _FINANCIAL_SCAM_KEYWORDS,
        _MIRACLE_CURE_KEYWORDS,
        _SCAM_SCENARIO_KEYWORDS,
        _DEBUNK_KEYWORDS,
        _VERIFIABLE_FACT_SUPPLEMENTS,
    ]
    for kw_list in verifiable_lists:
        if any(kw in normalized for kw in kw_list):
            return None

    return {
        "reason": "个人经历/观点分享，不包含可公开核查的事实声明",
        "type": "personal_experience",
    }


# ── UNVERIFIABLE 细分：发展中信息 vs 信息不足 ──

_SPECIFIC_INSTITUTIONS = [
    "北大",
    "清华",
    "复旦",
    "交大",
    "浙大",
    "南大",
    "中科大",
    "武大",
    "华科",
    "人大",
    "北师大",
    "同济",
    "中山大学",
    "厦门大学",
    "四川大学",
    "山东大学",
    "吉林大学",
    "哈工大",
    "西安交大",
    "天津大学",
    "南开大学",
    "东南大学",
    "华东师大",
    "中南大学",
    "华南理工",
    "大连理工",
    "北航",
    "北理工",
    "中国农大",
    "中国矿大",
    "北京大学",
    "清华大学",
    "复旦大学",
    "上海交大",
    "浙江大学",
    "南京大学",
    "中国科学技术大学",
    "武汉大学",
    "华中科技大学",
    "中国人民大学",
    "北京师范大学",
    "阿里",
    "腾讯",
    "百度",
    "华为",
    "小米",
    "京东",
    "美团",
    "字节跳动",
    "拼多多",
    "滴滴",
    "网易",
    "比亚迪",
    "特斯拉",
    "苹果",
    "微软",
    "谷歌",
    "三甲医院",
    "协和",
    "301医院",
    "华西医院",
    "中日友好医院",
    "瑞金医院",
]

_SPECIFIC_POLICY_ACTIONS = [
    "取消",
    "调整",
    "改革",
    "合并",
    "停止",
    "暂停",
    "新增",
    "试点",
    "扩招",
    "缩招",
    "撤销",
    "重组",
    "拆分",
    "升级",
    "降级",
    "转型",
    "搬迁",
    "关停",
    "整改",
    "限制",
    "放开",
    "开放",
    "恢复",
]

_VAGUE_REFERENCE_MARKERS = [
    "某",
    "听说",
    "据说",
    "有人说",
    "好像",
    "似乎",
    "大概",
    "可能是",
    "不知道真假",
    "不确定",
    "有消息称",
    "小道消息",
    "内部消息",
]

_DEVELOPING_DISCUSSION_MARKERS = [
    "有人说",
    "也有人说",
    "还有人说",
    "网上在讨论",
    "热搜",
    "热议",
    "争议",
    "讨论",
    "传",
    "网传",
    "据传",
    "消息称",
    "知情人",
    "多方消息",
]

# 发送者姿态：求证/存疑 vs 断言/传播
_INQUIRY_MARKERS = [
    "想知道",
    "会不会",
    "是不是",
    "真的吗",
    "是真的吗",
    "有人知道吗",
    "求证",
    "求问",
    "请问",
    "不确定",
    "不知道",
    "好像",
    "不要啊",
    "怎么办",
    "该怎么办",
    "有没有人",
    "谁知道",
    "靠谱吗",
    "可信吗",
    "了解吗",
    "？",
    "吗？",
    "呢？",
    "啊？",
]
_ASSERTIVE_SPREAD_MARKERS = [
    "紧急",
    "赶紧转发",
    "马上转发",
    "速转",
    "扩散",
    "赶紧告诉",
    "千万别",
    "一定要",
    "必须",
    "已经确认",
    "官方确认",
    "已经证实",
    "铁定",
    "板上钉钉",
    "！！",
    "转发救人",
    "别说我没提醒",
]


def classify_unverifiable_type(claim_text: str, evidence_list: list) -> str:
    """细分 UNVERIFIABLE 的类型：发展中信息 vs 信息不足。零 LLM。

    返回 "developing" 或 "insufficient"。

    "developing" 信号：
    - 提及具体的命名机构/组织
    - 提及具体的政策/行动动词
    - 多个讨论来源（不是辟谣，是在讨论）
    - 没有找到辟谣证据

    "insufficient" 信号：
    - 模糊引用（"某"、"听说"、"据说"）
    - 单一匿名来源
    - 无具体机构或行动
    """
    normalized = _normalize_text(claim_text)

    # 计算 developing 信号得分
    developing_score = 0

    # 1. 具体机构
    has_specific_institution = any(inst in normalized for inst in _SPECIFIC_INSTITUTIONS)
    # 也检查政府机构
    has_gov_entity = any(ent in normalized for ent in _GOVERNMENT_ENTITIES)
    if has_specific_institution or has_gov_entity:
        developing_score += 2

    # 2. 具体政策/行动
    has_specific_action = any(act in normalized for act in _SPECIFIC_POLICY_ACTIONS)
    if has_specific_action:
        developing_score += 1

    # 3. 多方讨论信号
    discussion_count = sum(1 for m in _DEVELOPING_DISCUSSION_MARKERS if m in normalized)
    if discussion_count >= 2:
        developing_score += 1

    # 4. 证据中无辟谣（人们在讨论，不是在辟谣）
    has_debunk_in_evidence = (
        any(_has_debunk_signal(f"{e.title} {e.snippet}") for e in evidence_list)
        if evidence_list
        else False
    )
    if not has_debunk_in_evidence and evidence_list:
        developing_score += 1

    # 计算 insufficient 信号得分
    insufficient_score = 0

    # 1. 模糊引用
    vague_count = sum(1 for m in _VAGUE_REFERENCE_MARKERS if m in normalized)
    if vague_count >= 1:
        insufficient_score += 2

    # 2. 无具体机构
    if not has_specific_institution and not has_gov_entity:
        insufficient_score += 1

    # 3. 无具体行动
    if not has_specific_action:
        insufficient_score += 1

    # 5. 发送者姿态检测——断言式传播不能判 developing
    inquiry_count = sum(1 for m in _INQUIRY_MARKERS if m in normalized)
    assertive_count = sum(1 for m in _ASSERTIVE_SPREAD_MARKERS if m in normalized)
    is_inquiry = inquiry_count >= 1
    is_assertive = assertive_count >= 1

    # 决策：developing 需要 具体机构+具体行动+非断言式传播
    if (
        developing_score >= 3
        and (has_specific_institution or has_gov_entity)
        and has_specific_action
    ):
        if is_assertive and not is_inquiry:
            # 断言式传播（"紧急！清华取消招生！赶紧转发！"）→ 不给 developing 标签
            return "insufficient"
        return "developing"

    if insufficient_score >= 2:
        return "insufficient"

    # 默认：有具体机构但其他信号不明确
    if has_specific_institution or has_gov_entity:
        if is_assertive and not is_inquiry:
            return "insufficient"
        return "developing"

    return "insufficient"


# ── AI 名人语录/专家代言规则（Q2 规则 5） ──

_CELEBRITY_AUTHORITY_KEYWORDS = [
    "钟南山",
    "张文宏",
    "张伯礼",
    "马云",
    "马化腾",
    "任正非",
    "雷军",
    "李嘉诚",
    "巴菲特",
    "马斯克",
    "比尔盖茨",
    "盖茨",
    "院士",
    "专家",
    "名医",
    "教授",
    "医生",
    "企业家",
    "明星",
    "主持人",
    "官方主播",
    "名人",
]
_CELEBRITY_ATTRIBUTION_CUES = [
    "说",
    "提醒",
    "建议",
    "预测",
    "推荐",
    "代言",
    "亲测",
    "直播",
    "语音",
    "视频",
    "截图",
    "朋友圈卡片",
    "AI换脸",
    "AI合成",
    "克隆声音",
    "数字人",
    "发言",
    "表态",
    "承认",
    "公开",
    "透露",
    "讲话",
    "爆料",
    "证实",
    "研究",
    "最新",
]
_CELEBRITY_HIGH_RISK_DOMAINS = [
    "药",
    "保健品",
    "护肤品",
    "理财",
    "投资",
    "彩票",
    "地震",
    "政策",
    "防疫",
    "养生",
    "补贴",
    "课程",
    "带货",
    "购买链接",
    "二维码",
    "AI",
    "人工智能",
    "替代",
    "失业",
    "觉醒",
    "超过人类",
    "图灵测试",
    "AGI",
    "通用人工智能",
    "看病",
    "治病",
    "预测",
    "领先",
]


def detect_ai_celebrity_quote(claim_text: str, evidence_list: list) -> dict | None:
    """检测 AI 名人语录/专家代言谣言。零 LLM。"""
    normalized = _normalize_text(claim_text)
    matched_celeb = next((kw for kw in _CELEBRITY_AUTHORITY_KEYWORDS if kw in normalized), None)
    if not matched_celeb:
        return None
    has_attribution = any(kw in normalized for kw in _CELEBRITY_ATTRIBUTION_CUES)
    if not has_attribution:
        return None
    has_high_risk = any(kw in normalized for kw in _CELEBRITY_HIGH_RISK_DOMAINS)
    if not has_high_risk:
        return None
    # 检查证据中有无来自该名人本人的权威来源
    has_primary_source = False
    for e in evidence_list:
        if _is_authority(e.url):
            combined = f"{e.title} {e.snippet}"
            if matched_celeb in combined and any(
                kw in combined for kw in ["本人", "亲自", "官方账号", "回应", "声明"]
            ):
                has_primary_source = True
                break
    if has_primary_source:
        return None
    return {
        "celebrity": matched_celeb,
        "reason": (
            f"声明引用「{matched_celeb}」发表高风险领域言论，"
            f"但未找到来自本人或所属机构的原始出处，"
            f"符合典型 AI 换脸/伪造名人语录模式"
        ),
    }


# ── 食品安全谣言检测（零 LLM） ──


def _tn_norm(text: str) -> str:
    return str(text or "").strip().lower()


def _tn_first_keyword(text: str, keywords) -> str | None:
    for keyword in keywords:
        if keyword and keyword in text:
            return keyword
    return None


def _tn_any_keyword(text: str, keywords) -> bool:
    return _tn_first_keyword(text, keywords) is not None


_FOOD_SAFETY_MYTH_NEGATION_CUES = [
    "辟谣",
    "不实",
    "谣言",
    "假消息",
    "假新闻",
    "虚假",
    "伪造",
    "编造",
    "造谣",
    "别信",
    "勿信",
    "不要相信",
    "不要转",
    "别传",
    "勿传",
    "假的",
    "都是假",
    "纯属捏造",
    "子虚乌有",
    "没有依据",
    "无科学依据",
    "没有证据",
    "未经证实",
    "网传不实",
    "官方否认",
    "并非",
    "并不是",
    "不是",
    "不会导致",
    "不会造成",
    "不会引起",
    "不含塑料",
    "不是塑料",
    "非塑料",
    "没有激素",
    "不含激素",
    "不会性早熟",
    "不会不孕",
    "不会绝育",
    "不会致癌",
    "没有致癌",
    "无需恐慌",
    "不必恐慌",
    "科普",
    "澄清",
    "说法不对",
    "说法错误",
]

_FOOD_SAFETY_LEGITIMATE_WARNING_CUES = [
    "抽检",
    "监督抽检",
    "抽样检验",
    "检出",
    "不合格",
    "召回",
    "产品召回",
    "下架",
    "封存",
    "立案调查",
    "行政处罚",
    "批次",
    "生产批号",
    "生产日期",
    "保质期",
    "食品安全标准",
    "风险监测",
    "风险预警",
    "风险评估",
    "菌落总数",
    "大肠菌群",
    "沙门氏菌",
    "金黄色葡萄球菌",
    "李斯特菌",
    "诺如病毒",
    "黄曲霉毒素",
    "赭曲霉毒素",
    "酸价超标",
    "过氧化值超标",
    "镉超标",
    "铅超标",
    "汞超标",
    "砷超标",
    "农残超标",
    "农药残留",
    "兽药残留",
    "重金属超标",
    "非法添加",
    "过期食品",
    "变质",
    "霉变",
    "异物",
    "食物中毒事件",
    "市场监管局",
    "市场监管总局",
    "海关总署",
    "农业农村部",
    "食药监",
    "官方通报",
]

_FOOD_SAFETY_MYTH_ASSERTION_CUES = [
    "会",
    "导致",
    "造成",
    "引起",
    "吃了",
    "吃多了",
    "长期吃",
    "不能吃",
    "千万别吃",
    "别吃",
    "有毒",
    "致癌",
    "不孕",
    "绝育",
    "性早熟",
    "改变基因",
    "都是",
    "全是",
    "全部是",
    "害人",
    "毁孩子",
    "打了",
    "打针",
    "喂了",
    "添加",
    "注入",
    "泡过",
    "抹了",
    "含有",
]

_FOOD_SAFETY_MYTH_PATTERNS = [
    {
        "name": "激素鸡蛋/畜禽激素谣言",
        "keywords": [
            "激素鸡蛋",
            "激素蛋",
            "鸡蛋有激素",
            "鸡蛋含激素",
            "鸡蛋里有激素",
            "吃鸡蛋会性早熟",
            "鸡蛋导致性早熟",
            "鸡蛋让孩子性早熟",
            "避孕药喂鸡",
            "喂避孕药的鸡",
            "鸡吃避孕药",
            "避孕药鸡蛋",
            "避孕药蛋",
            "激素鸡",
            "激素鸭",
            "速成鸡全靠激素",
            "肉鸡都是激素催大的",
        ],
        "context_keywords": [
            "鸡蛋",
            "鸭蛋",
            "蛋",
            "禽蛋",
            "母鸡",
            "肉鸡",
            "鸡",
            "鸭",
            "养殖",
            "蛋黄",
            "蛋清",
        ],
        "risk_keywords": [
            "激素",
            "避孕药",
            "性早熟",
            "早熟",
            "不能吃",
            "有毒",
            "致癌",
            "害孩子",
        ],
    },
    {
        "name": "塑料/人工合成食品谣言",
        "keywords": [
            "塑料大米",
            "塑料米",
            "假大米",
            "人工大米",
            "人工合成大米",
            "合成大米",
            "树脂大米",
            "石蜡大米",
            "塑料紫菜",
            "塑料海带",
            "塑料粉丝",
            "塑料面条",
            "胶水面条",
            "胶水粉条",
            "硅胶虾",
            "假鸡蛋",
            "人造鸡蛋",
            "棉花肉松",
            "纸包子",
            "纸馅包子",
            "塑料做的食品",
            "塑料做的食物",
            "大米是塑料做的",
            "紫菜是塑料做的",
            "海带是塑料做的",
        ],
        "context_keywords": [
            "大米",
            "米饭",
            "米",
            "紫菜",
            "海带",
            "粉丝",
            "粉条",
            "面条",
            "肉松",
            "包子",
            "虾",
            "鸡蛋",
            "食品",
            "食物",
            "农产品",
        ],
        "risk_keywords": [
            "塑料",
            "树脂",
            "石蜡",
            "硅胶",
            "胶水",
            "棉花",
            "纸做",
            "人造",
            "人工合成",
            "合成",
            "造假",
            "不能吃",
            "有毒",
        ],
    },
    {
        "name": "转基因危害谣言",
        "keywords": [
            "转基因导致不孕",
            "转基因会不孕",
            "转基因让人不孕",
            "转基因导致绝育",
            "转基因会绝育",
            "转基因断子绝孙",
            "转基因致癌",
            "转基因会致癌",
            "转基因有毒",
            "转基因改变人的基因",
            "转基因会改变基因",
            "转基因让人变异",
            "转基因食品不能吃",
        ],
        "context_keywords": [
            "转基因",
            "gmo",
            "基因编辑",
            "转基因食品",
            "转基因作物",
            "转基因大豆",
            "转基因玉米",
            "转基因大米",
            "大豆",
            "玉米",
            "大米",
            "食用油",
            "农产品",
        ],
        "risk_keywords": [
            "不孕",
            "不育",
            "绝育",
            "断子绝孙",
            "致癌",
            "有毒",
            "改变基因",
            "基因变异",
            "变异",
            "灭绝",
            "不能吃",
        ],
    },
    {
        "name": "预制菜泛化危害谣言",
        "keywords": [
            "预制菜有毒",
            "预制菜致癌",
            "预制菜不能吃",
            "预制菜都是防腐剂",
            "预制菜全是防腐剂",
            "预制菜是毒药",
            "预制菜毁孩子",
            "预制菜害孩子",
            "预制菜导致白血病",
            "预制菜会白血病",
            "预制菜都是科技与狠活",
            "料理包有毒",
            "料理包致癌",
            "料理包不能吃",
            "中央厨房都是毒",
        ],
        "context_keywords": [
            "预制菜",
            "料理包",
            "中央厨房",
            "半成品菜",
            "预制食品",
            "冷冻菜",
            "复热菜",
            "校园餐",
            "学生餐",
        ],
        "risk_keywords": [
            "有毒",
            "致癌",
            "不能吃",
            "防腐剂",
            "毒药",
            "害孩子",
            "毁孩子",
            "白血病",
            "科技与狠活",
            "垃圾食品",
            "全是添加剂",
        ],
    },
    {
        "name": "果蔬打针/避孕药/催熟谣言",
        "keywords": [
            "西瓜打针",
            "打针西瓜",
            "西瓜注射甜蜜素",
            "西瓜打甜蜜素",
            "西瓜打色素",
            "草莓打避孕药",
            "草莓用了避孕药",
            "草莓打膨大剂",
            "空心草莓是激素",
            "黄瓜抹避孕药",
            "黄瓜用了避孕药",
            "无籽葡萄打避孕药",
            "无籽水果都是避孕药",
            "香蕉催熟有毒",
            "催熟香蕉致癌",
            "蘑菇泡甲醛",
            "甲醛蘑菇",
            "水果打甜蜜素",
            "水果打针增甜",
        ],
        "context_keywords": [
            "西瓜",
            "草莓",
            "黄瓜",
            "葡萄",
            "无籽葡萄",
            "无籽水果",
            "香蕉",
            "蘑菇",
            "水果",
            "蔬菜",
            "果蔬",
            "农产品",
        ],
        "risk_keywords": [
            "打针",
            "避孕药",
            "膨大剂",
            "激素",
            "催熟",
            "甜蜜素",
            "色素",
            "甲醛",
            "有毒",
            "致癌",
            "不能吃",
        ],
    },
]


def detect_food_safety_myth(claim_text: str) -> dict | None:
    """检测食品安全/农产品高频谣言。零 LLM。"""
    text = _tn_norm(claim_text)
    if not text:
        return None
    if _tn_any_keyword(text, _FOOD_SAFETY_MYTH_NEGATION_CUES):
        return None
    if _tn_any_keyword(text, _FOOD_SAFETY_LEGITIMATE_WARNING_CUES):
        return None

    for pattern in _FOOD_SAFETY_MYTH_PATTERNS:
        keyword = _tn_first_keyword(text, pattern["keywords"])
        context = _tn_first_keyword(text, pattern["context_keywords"])
        risk = _tn_first_keyword(text, pattern["risk_keywords"])
        assertion = _tn_first_keyword(text, _FOOD_SAFETY_MYTH_ASSERTION_CUES)

        if keyword and (context or _tn_any_keyword(keyword, pattern["context_keywords"])):
            return {
                "category": "food_safety_myth",
                "keyword": keyword,
                "reason": (
                    f"命中食品/农产品谣言高频模式「{pattern['name']}」："
                    f"出现「{keyword}」，且属于食品安全场景；已跳过辟谣、科普和抽检通报语境。"
                ),
            }

        if context and risk and assertion:
            return {
                "category": "food_safety_myth",
                "keyword": risk,
                "reason": (
                    f"命中食品/农产品谣言高频模式「{pattern['name']}」："
                    f"食品场景「{context}」与危害断言「{risk}」同时出现；"
                    f"已跳过辟谣、科普和真实食品安全预警语境。"
                ),
            }
    return None


# ── 民生政策谣言检测（零 LLM） ──

_LIVELIHOOD_POLICY_NEGATION_CUES = [
    "辟谣",
    "不实",
    "谣言",
    "假消息",
    "假新闻",
    "虚假",
    "编造",
    "造谣",
    "假的",
    "都是假",
    "别信",
    "勿信",
    "不要相信",
    "不要转",
    "别转",
    "官方否认",
    "官方回应",
    "官方澄清",
    "澄清",
    "未发布",
    "没有发布",
    "并未发布",
    "未出台",
    "没有出台",
    "并未出台",
    "未取消",
    "没有取消",
    "并未取消",
    "不会取消",
    "未停用",
    "没有停用",
    "并未停用",
    "不会停用",
    "未作废",
    "没有作废",
    "不会作废",
    "未清零",
    "没有清零",
    "不会清零",
    "并未清零",
    "未暂停",
    "没有暂停",
    "不会暂停",
    "未停发",
    "没有停发",
    "不会停发",
    "网传不实",
    "纯属捏造",
    "子虚乌有",
]

_LIVELIHOOD_POLICY_NEUTRAL_SERVICE_CUES = [
    "怎么停用",
    "如何停用",
    "停用怎么办",
    "申请停用",
    "办理停用",
    "挂失停用",
    "注销社保卡",
    "注销医保卡",
    "怎么注销",
    "如何注销",
    "怎么办理",
    "办理流程",
    "怎么查询",
    "如何查询",
    "在哪里办",
    "去哪办理",
    "报销比例",
    "缴费标准",
    "缴费流程",
    "政策解读",
    "是什么意思",
]

_LIVELIHOOD_POLICY_URGENCY_CUES = [
    "紧急通知",
    "紧急提醒",
    "紧急转发",
    "紧急扩散",
    "马上办理",
    "赶紧办理",
    "赶快办理",
    "尽快办理",
    "立即办理",
    "限时办理",
    "逾期作废",
    "逾期停用",
    "逾期清零",
    "过期作废",
    "月底前",
    "本月底前",
    "今天起",
    "明天起",
    "即日起",
    "本月起",
    "下月起",
    "今年起",
    "2024年起",
    "2025年起",
    "2026年起",
    "最新通知",
    "最新政策",
    "刚刚发布",
    "刚发",
    "重磅",
    "突发",
    "扩散",
    "转发",
    "家长群",
    "业主群",
    "微信群通知",
    "朋友圈通知",
]

_LIVELIHOOD_POLICY_ABSOLUTE_CUES = [
    "取消",
    "全面取消",
    "全部取消",
    "一律取消",
    "不再",
    "全面停用",
    "停用",
    "暂停",
    "暂停使用",
    "作废",
    "失效",
    "冻结",
    "封存",
    "清零",
    "清空",
    "归零",
    "停发",
    "停止发放",
    "禁止",
    "不得",
    "必须",
    "强制",
    "统一",
    "一律",
    "全部",
    "所有",
    "全国",
    "彻底",
    "永久",
    "正式确定",
    "确定了",
    "定了",
]

_LIVELIHOOD_POLICY_ACTIONS = [
    "取消",
    "停用",
    "暂停",
    "作废",
    "失效",
    "冻结",
    "封存",
    "清零",
    "清空",
    "归零",
    "停发",
    "停止发放",
    "不再发放",
    "不再使用",
    "不再办理",
    "禁止",
    "不得",
    "强制",
    "必须",
    "统一认证",
    "重新认证",
    "集中认证",
    "逾期",
]

_LIVELIHOOD_POLICY_RUMOR_PATTERNS = [
    {
        "name": "教育升学政策",
        "keywords": [
            "取消中考",
            "中考取消",
            "不再中考",
            "中考将取消",
            "中考全面取消",
            "中考并入高考",
            "初中直升高中",
            "初中直接升高中",
            "取消普职分流",
            "普职分流取消",
            "取消高考",
            "高考取消",
            "高考取消英语",
            "英语退出高考",
            "取消小学考试",
            "取消期末考试",
            "取消寒暑假",
            "寒暑假取消",
            "义务教育延长到高中",
            "高中纳入义务教育",
            "十二年义务教育全面实施",
            "民办学校全部取消",
            "学区房取消",
            "教师编制取消",
        ],
        "context_keywords": [
            "中考",
            "高考",
            "小升初",
            "小学",
            "初中",
            "高中",
            "义务教育",
            "升学",
            "招生",
            "入学",
            "学籍",
            "普职分流",
            "职高",
            "中职",
            "寒暑假",
            "校外培训",
            "双减",
            "教师编制",
            "教育局",
            "教育部",
            "教育厅",
        ],
    },
    {
        "name": "社保/养老/社保卡政策",
        "keywords": [
            "社保卡停用",
            "社保卡暂停使用",
            "社保卡将停用",
            "社保卡全面停用",
            "社保卡作废",
            "社保卡失效",
            "社保卡冻结",
            "社保卡过期作废",
            "社保卡不认证就停用",
            "社保卡未认证停用",
            "社保卡统一认证",
            "社保账户清零",
            "社保断缴清零",
            "社保缴费年限清零",
            "养老金停发",
            "养老金取消",
            "养老金暂停发放",
            "退休金停发",
            "退休金取消",
            "养老保险取消",
            "社保补缴最后期限",
            "社保补缴即将截止",
            "灵活就业社保取消",
        ],
        "context_keywords": [
            "社保",
            "社保卡",
            "社会保障卡",
            "电子社保卡",
            "养老保险",
            "养老金",
            "退休金",
            "退休",
            "社保账户",
            "个人账户",
            "补缴",
            "断缴",
            "缴费年限",
            "灵活就业",
            "失业保险",
            "工伤保险",
            "人社部",
            "人社局",
            "社保局",
        ],
    },
    {
        "name": "医保/医疗保险政策",
        "keywords": [
            "医保个人账户取消",
            "取消医保个人账户",
            "医保账户取消",
            "医疗保险个人账户取消",
            "医保卡余额清零",
            "医保余额清零",
            "医保账户清零",
            "医保个人账户清零",
            "医保卡余额年底清零",
            "医保卡年底清零",
            "医保卡停用",
            "医保卡暂停使用",
            "医保卡作废",
            "医保卡失效",
            "医保报销取消",
            "门诊报销取消",
            "药店刷医保取消",
            "新农合取消",
            "居民医保取消",
            "职工医保取消",
            "医保断缴清零",
            "医保不交作废",
            "医保账户冻结",
        ],
        "context_keywords": [
            "医保",
            "医保卡",
            "医疗保险",
            "医疗保障",
            "医保账户",
            "个人账户",
            "医保个人账户",
            "门诊共济",
            "统筹账户",
            "医保报销",
            "居民医保",
            "职工医保",
            "新农合",
            "异地就医",
            "电子医保凭证",
            "国家医保局",
            "医保局",
            "医疗保障局",
        ],
    },
]

_LIVELIHOOD_OFFICIAL_DOMAIN_CUES = [
    "gov.cn",
    ".gov.cn",
    "gov.hk",
    "gov.mo",
    "moe.gov.cn",
    "mohrss.gov.cn",
    "nhsa.gov.cn",
    "nhc.gov.cn",
    "mca.gov.cn",
    "moj.gov.cn",
    "mof.gov.cn",
    "samr.gov.cn",
    "www.gov.cn",
]

_LIVELIHOOD_OFFICIAL_SOURCE_CUES = [
    "中国政府网",
    "国务院",
    "教育部",
    "省教育厅",
    "市教育局",
    "区教育局",
    "县教育局",
    "教育考试院",
    "招生考试院",
    "人社部",
    "人力资源和社会保障部",
    "省人社厅",
    "市人社局",
    "区人社局",
    "县人社局",
    "社保局",
    "社会保险事业管理局",
    "国家医保局",
    "国家医疗保障局",
    "省医保局",
    "市医保局",
    "区医保局",
    "县医保局",
    "医疗保障局",
    "财政部",
    "民政部",
]


def _tn_evidence_to_text(evidence) -> str:
    if isinstance(evidence, dict):
        fields = []
        for key in (
            "title",
            "snippet",
            "summary",
            "content",
            "text",
            "url",
            "link",
            "domain",
            "source",
            "site",
            "site_name",
            "publisher",
        ):
            value = evidence.get(key)
            if value is not None:
                fields.append(str(value))
        return " ".join(fields)
    return str(evidence or "")


def _tn_livelihood_groups(text: str) -> set:
    groups = set()
    for pattern in _LIVELIHOOD_POLICY_RUMOR_PATTERNS:
        if _tn_any_keyword(text, pattern["keywords"]) or _tn_any_keyword(
            text, pattern["context_keywords"]
        ):
            groups.add(pattern["name"])
    return groups


def _tn_match_livelihood_policy_keyword(text: str) -> tuple[str | None, str | None]:
    for pattern in _LIVELIHOOD_POLICY_RUMOR_PATTERNS:
        keyword = _tn_first_keyword(text, pattern["keywords"])
        if keyword:
            return keyword, pattern["name"]
        context = _tn_first_keyword(text, pattern["context_keywords"])
        action = _tn_first_keyword(text, _LIVELIHOOD_POLICY_ACTIONS)
        if context and action:
            return f"{context}+{action}", pattern["name"]
    return None, None


def _tn_has_official_livelihood_evidence(evidence_list: list, claim_text: str) -> bool:
    claim_groups = _tn_livelihood_groups(claim_text)
    if not claim_groups:
        return False
    for evidence in evidence_list or []:
        if hasattr(evidence, "url"):
            evidence_text = _tn_norm(f"{evidence.title} {evidence.snippet} {evidence.url}")
        else:
            evidence_text = _tn_norm(_tn_evidence_to_text(evidence))
        if not evidence_text:
            continue
        official_domain = _tn_any_keyword(evidence_text, _LIVELIHOOD_OFFICIAL_DOMAIN_CUES)
        official_source = _tn_any_keyword(evidence_text, _LIVELIHOOD_OFFICIAL_SOURCE_CUES)
        if not (official_domain or official_source):
            continue
        evidence_groups = _tn_livelihood_groups(evidence_text)
        if claim_groups & evidence_groups:
            return True
    return False


def detect_livelihood_policy_rumor(claim_text: str, evidence_list: list) -> dict | None:
    """检测教育/社保/医保类民生政策谣言。零 LLM。"""
    text = _tn_norm(claim_text)
    if not text:
        return None
    if _tn_any_keyword(text, _LIVELIHOOD_POLICY_NEGATION_CUES):
        return None
    if _tn_any_keyword(text, _LIVELIHOOD_POLICY_NEUTRAL_SERVICE_CUES):
        return None
    keyword, group_name = _tn_match_livelihood_policy_keyword(text)
    if not keyword:
        return None
    urgency = _tn_first_keyword(text, _LIVELIHOOD_POLICY_URGENCY_CUES)
    absolute = _tn_first_keyword(text, _LIVELIHOOD_POLICY_ABSOLUTE_CUES)
    if not (urgency or absolute):
        return None
    if _tn_has_official_livelihood_evidence(evidence_list, text):
        return None
    signal = urgency or absolute
    return {
        "category": "livelihood_policy_rumor",
        "keyword": keyword,
        "reason": (
            f"命中民生政策谣言高频模式「{group_name}」："
            f"声明含「{keyword}」并使用紧急/绝对化信号「{signal}」，"
            f"但 evidence_list 未发现对应教育、社保或医保官方证据。"
        ),
    }


# ── AI 生成/合成内容检测（零 LLM） ──

_AI_CONTENT_NEGATION_CUES = [
    "辟谣",
    "不实",
    "谣言",
    "假消息",
    "虚假",
    "编造",
    "造谣",
    "别信",
    "勿信",
    "不要相信",
    "官方否认",
    "官方澄清",
    "并非ai生成",
    "不是ai生成",
    "非ai生成",
    "并非ai合成",
    "不是ai合成",
    "非ai合成",
    "并非合成",
    "不是合成",
    "不是p图",
    "并非p图",
    "不是ps",
    "并非ps",
    "不是换脸",
    "并非换脸",
    "不是摆拍",
    "并非摆拍",
    "没有ai痕迹",
    "证实为真实画面",
    "真实画面不是ai",
    "别信ai造假说",
]

_AI_CONTENT_LEGIT_DISCUSSION_CUES = [
    "技术",
    "模型",
    "论文",
    "研究",
    "产业",
    "应用",
    "工具",
    "教程",
    "课程",
    "培训",
    "算法",
    "算力",
    "版权",
    "监管",
    "伦理",
    "政策",
    "法规",
    "公司发布",
    "功能发布",
    "产品发布",
    "评测",
    "测评",
    "提示词",
    "prompt",
    "模型训练",
    "开源",
    "产品",
    "赛道",
    "招聘",
    "岗位",
    "绘画软件",
    "剪辑软件",
    "怎么用",
    "如何使用",
    "使用方法",
    "使用教程",
    "教学",
    "科普",
    "原理",
    "市场规模",
    "商业化",
    "识别方法",
    "辨别方法",
    "如何识别",
]

_AI_CONTENT_MEDIA_CONTEXTS = [
    "图片",
    "照片",
    "图像",
    "视频",
    "短视频",
    "影像",
    "画面",
    "截图",
    "录屏",
    "录音",
    "语音",
    "音频",
    "直播",
    "监控",
    "航拍",
    "现场图",
    "现场照片",
    "现场视频",
    "新闻画面",
    "灾情图",
    "事故视频",
    "事故照片",
]

_AI_CONTENT_EVENT_CONTEXTS = [
    "地震",
    "洪水",
    "台风",
    "暴雨",
    "山火",
    "火灾",
    "爆炸",
    "车祸",
    "坠机",
    "空难",
    "事故",
    "灾难",
    "灾害",
    "灾情",
    "战争",
    "冲突",
    "袭击",
    "枪击",
    "杀人",
    "刑案",
    "救援",
    "医院",
    "学校",
    "景区",
    "地铁",
    "高铁",
    "机场",
    "桥梁",
    "大楼",
    "现场",
    "事件",
    "发生",
]

_AI_CONTENT_RUMOR_SPREAD_CUES = [
    "网传",
    "热传",
    "疯传",
    "流传",
    "传播",
    "转发",
    "扩散",
    "朋友圈",
    "微信群",
    "群里",
    "短视频平台",
    "视频号",
    "配文称",
    "标题称",
    "声称",
    "称",
    "冒充",
    "伪装",
    "误导",
    "曝光",
    "流出",
    "刚刚",
    "突发",
    "最新",
    "今天",
    "昨日",
    "昨天",
    "近日",
    "本地",
    "当地",
    "某地",
    "一地",
]

_AI_CONTENT_SYNTHETIC_KEYWORDS = [
    "ai生成",
    "ai合成",
    "ai绘图",
    "ai作图",
    "ai图片",
    "ai照片",
    "ai视频",
    "ai造假",
    "ai伪造",
    "ai灾难图片",
    "ai灾害图",
    "ai地震图",
    "ai洪水图",
    "ai火灾图",
    "ai事故图",
    "aigc",
    "生成式ai",
    "人工智能生成",
    "人工智能合成",
    "合成图",
    "合成图片",
    "合成照片",
    "合成视频",
    "伪造图",
    "伪造图片",
    "伪造照片",
    "伪造视频",
    "p图",
    "ps图",
    "换脸",
    "ai换脸",
    "deepfake",
    "深度伪造",
    "数字人",
    "虚拟人",
    "克隆声音",
    "声音克隆",
    "语音合成",
    "配音合成",
    "事故视频是合成",
    "事故视频ai合成",
    "事故视频为ai生成",
    "ai合成事故视频",
    "灾难视频ai合成",
    "灾情图片ai生成",
]

_AI_CONTENT_OLD_MEDIA_KEYWORDS = [
    "旧视频新发",
    "旧图新传",
    "旧照新传",
    "旧闻新炒",
    "老视频新发",
    "老照片新传",
    "旧视频冒充",
    "旧照片冒充",
    "旧图片冒充",
    "旧图冒充",
    "老视频冒充",
    "老照片冒充",
    "老画面冒充",
    "旧画面冒充",
    "旧素材冒充",
    "旧视频配新事件",
    "旧照片配新事件",
    "旧素材配新标题",
    "把旧视频说成",
    "把旧照片说成",
    "把旧图说成",
    "张冠李戴",
    "移花接木",
    "配上新标题",
    "配文嫁接",
    "异地视频冒充本地",
    "外地视频冒充本地",
]

_AI_CONTENT_STAGED_KEYWORDS = [
    "摆拍",
    "摆拍视频",
    "剧本演绎",
    "剧情演绎",
    "自导自演",
    "群演",
    "演员扮演",
    "营销号摆拍",
    "剧情号",
    "虚构剧情",
    "假现场",
    "伪现场",
    "假救援",
    "假事故",
    "假灾情",
    "假采访",
    "假直播",
]


def detect_ai_generated_content(claim_text: str) -> dict | None:
    """检测 AI 生成图片/视频、合成音视频、旧素材嫁接和摆拍内容。零 LLM。"""
    text = _tn_norm(claim_text)
    if not text:
        return None
    if _tn_any_keyword(text, _AI_CONTENT_NEGATION_CUES):
        return None

    has_discussion = _tn_any_keyword(text, _AI_CONTENT_LEGIT_DISCUSSION_CUES)
    has_spread = _tn_any_keyword(text, _AI_CONTENT_RUMOR_SPREAD_CUES)
    has_event = _tn_any_keyword(text, _AI_CONTENT_EVENT_CONTEXTS)
    has_media = _tn_any_keyword(text, _AI_CONTENT_MEDIA_CONTEXTS)

    if has_discussion and not has_spread:
        return None

    keyword = _tn_first_keyword(text, _AI_CONTENT_SYNTHETIC_KEYWORDS)
    rumor_type = "AI生成/合成内容"

    if not keyword:
        keyword = _tn_first_keyword(text, _AI_CONTENT_OLD_MEDIA_KEYWORDS)
        rumor_type = "旧素材嫁接新事件"

    if not keyword:
        keyword = _tn_first_keyword(text, _AI_CONTENT_STAGED_KEYWORDS)
        rumor_type = "摆拍/剧本化内容"

    if not keyword:
        return None

    if not has_media and not _tn_any_keyword(keyword, _AI_CONTENT_MEDIA_CONTEXTS):
        return None

    if not (has_spread or has_event):
        return None

    return {
        "category": "ai_generated_content",
        "keyword": keyword,
        "reason": (
            f"命中{rumor_type}高频模式：出现「{keyword}」，"
            f"且同时具备图片/视频/音频等媒介语境与公共事件或网传扩散语境；"
            f"已跳过普通 AI 技术讨论和明确否定语境。"
        ),
    }


# ── 因果逻辑谬误检测（零 LLM） ──

_CAUSAL_FALLACY_PATTERNS = [
    {
        "name": "post_hoc",
        "markers": ["之后", "后来", "随后", "接着", "然后", "过了"],
        "connectors": ["所以", "因此", "导致", "造成", "引起", "就", "于是"],
        "description": "时间先后≠因果：A发生在B之前不代表A导致了B",
    },
    {
        "name": "correlation_as_causation",
        "markers": ["都", "凡是", "每次", "只要", "一…就"],
        "connectors": ["所以", "说明", "证明", "可见", "就是因为"],
        "description": "相关≠因果：两件事同时发生不代表有因果关系",
    },
]

_CAUSAL_HEALTH_CONTEXTS = [
    "打疫苗",
    "接种",
    "疫苗",
    "吃了",
    "喝了",
    "用了",
    "服用",
    "注射",
    "手术",
    "治疗",
    "检查",
]

_CAUSAL_SEVERE_OUTCOMES = [
    "死亡",
    "去世",
    "死了",
    "猝死",
    "瘫痪",
    "残疾",
    "致癌",
    "白血病",
    "自闭症",
    "不孕",
    "流产",
    "心梗",
    "脑梗",
]


def detect_causal_fallacy(claim_text: str) -> dict | None:
    """检测因果逻辑谬误（post hoc / 相关当因果）。零 LLM。"""
    normalized = _normalize_text(claim_text)
    if _has_negation_context(normalized, _CAUSAL_HEALTH_CONTEXTS[:3]):
        return None
    has_health = any(kw in normalized for kw in _CAUSAL_HEALTH_CONTEXTS)
    has_outcome = any(kw in normalized for kw in _CAUSAL_SEVERE_OUTCOMES)
    if not (has_health and has_outcome):
        return None
    for pattern in _CAUSAL_FALLACY_PATTERNS:
        has_marker = any(m in normalized for m in pattern["markers"])
        has_connector = any(c in normalized for c in pattern["connectors"])
        if has_marker and has_connector:
            return {
                "category": "causal_fallacy",
                "fallacy_type": pattern["name"],
                "reason": (
                    f"声明含因果逻辑谬误特征（{pattern['description']}）："
                    f"健康干预行为与严重后果之间用时序/相关词连接，"
                    f"但时间先后或相关性不等于因果关系。"
                ),
            }
    # 即使没有显式连接词，"打疫苗后死亡"模式也是典型 post hoc
    for m in ["后", "之后"]:
        for ctx in _CAUSAL_HEALTH_CONTEXTS:
            if ctx + m in normalized and has_outcome:
                return {
                    "category": "causal_fallacy",
                    "fallacy_type": "post_hoc",
                    "reason": (
                        f"声明含典型 post hoc 模式：「{ctx}后」+ 严重后果，时间先后不等于因果关系。"
                    ),
                }
    return None


_SCREENSHOT_KEYWORDS = [
    "截图",
    "聊天记录",
    "对话截图",
    "私聊",
    "内部通知截图",
    "朋友圈截图",
    "群截图",
    "截屏",
    "录屏",
]


_OFFICIAL_SCREENSHOT_CUES = [
    "官网",
    "官方网站",
    "数据库",
    "法律法规",
    "政府网",
    "gov.cn",
    "gov.hk",
    "人民网",
    "新华网",
    "央视",
    "公告栏",
    "红头文件",
]


def detect_screenshot_claim(claim_text: str, evidence_list: list) -> dict | None:
    """检测声明是否依赖无法验证的截图/聊天记录。零 LLM。

    逻辑：声明中含截图关键词 + 无权威来源直接辟谣/证实 → 标记无法核实。
    例外：如果截图声称来自官方页面（含官网/数据库等提示词），不判不可验证。
    """
    has_screenshot = any(kw in claim_text for kw in _SCREENSHOT_KEYWORDS)
    if not has_screenshot:
        return None

    if any(cue in claim_text for cue in _OFFICIAL_SCREENSHOT_CUES):
        logger.info("[RuleEngine] 截图声称来自官方页面，跳过不可验证规则")
        return None

    has_authority_debunk = any(
        _is_authority(e.url) and _has_debunk_signal(f"{e.title} {e.snippet}") for e in evidence_list
    )
    if has_authority_debunk:
        return None

    return {
        "reason": (
            "声明依赖截图/聊天记录作为证据，但截图极易伪造，且搜索未找到权威来源直接证实或辟谣"
        ),
    }


def _extract_claim_key_terms(claim_text: str) -> tuple[list[str], list[str]]:
    """从声明文本中提取关键词，用于 same-claim 相关性检查。

    返回 (anchor_terms, generic_terms)：
    - anchor_terms: 4+ 字的中文词组、英文 5+ 字母单词、精确数字+单位
      （专有名词级，如"传媒大学""骨质疏松""碳中和"）
    - generic_terms: 2-3 字的中文词组、短英文
      （通用词，如"失火""大学""政策"）
    """
    import re as _re_terms

    _STOPWORDS = {
        "的",
        "是",
        "在",
        "有",
        "说",
        "了",
        "和",
        "与",
        "或",
        "也",
        "都",
        "就",
        "不",
        "会",
        "要",
        "能",
        "可以",
        "这",
        "那",
        "他",
        "她",
        "它",
        "我",
        "你",
        "们",
        "吗",
        "呢",
        "吧",
        "啊",
        "哦",
        "但",
        "但是",
        "如果",
        "因为",
        "所以",
        "而且",
        "已经",
        "正在",
        "可能",
        "应该",
        "需要",
        "想",
        "看",
        "到",
        "家人",
        "大家",
        "消息",
        "通知",
        "紧急",
        "最新",
        "刚刚",
        "赶紧",
        "转发",
        "一个",
        "什么",
        "怎么",
        "为什么",
        "真的",
        "假的",
    }
    anchor_terms: list[str] = []
    generic_terms: list[str] = []
    anchor_set: set[str] = set()
    generic_set: set[str] = set()

    # 找到所有连续中文字符片段，用滑动窗口提取
    for m in _re_terms.finditer(r"[一-鿿]+", claim_text):
        run = m.group()
        # 4 字滑动窗口 → 锚点词（专有名词级）
        if len(run) >= 4:
            for i in range(len(run) - 3):
                word = run[i : i + 4]
                if word not in _STOPWORDS and word not in anchor_set:
                    anchor_terms.append(word)
                    anchor_set.add(word)
        # 2-3 字片段 → 通用词
        for i in range(len(run) - 1):
            for wlen in (3, 2):
                if i + wlen > len(run):
                    continue
                word = run[i : i + wlen]
                if word in _STOPWORDS or word in generic_set:
                    continue
                generic_terms.append(word)
                generic_set.add(word)

    for m in _re_terms.finditer(r"[A-Za-z]{5,}", claim_text):
        anchor_terms.append(m.group().lower())
    for m in _re_terms.finditer(r"[A-Za-z]{3,4}", claim_text):
        generic_terms.append(m.group().lower())
    for m in _re_terms.finditer(r"\d+[%％万亿元岁分年月日]", claim_text):
        anchor_terms.append(m.group())
    return anchor_terms, generic_terms


def _evidence_matches_claim(
    evidence_text: str, claim_terms: list[str], min_overlap: int = 2
) -> bool:
    """检查证据文本是否与声明有足够的关键词重叠（same-claim match）。"""
    if not claim_terms:
        return True
    overlap = sum(1 for t in claim_terms if t in evidence_text)
    return overlap >= min(min_overlap, len(claim_terms))


def _evidence_matches_claim_strict(
    evidence_text: str,
    anchor_terms: list[str],
    generic_terms: list[str],
) -> bool:
    """严格检查证据是否与声明讨论同一事件。

    规则：
    - 有锚点词时：必须匹配至少 1 个锚点词
    - 无锚点词时：退回到通用词 2 个重叠的宽松匹配
    """
    if anchor_terms:
        anchor_hits = sum(1 for t in anchor_terms if t in evidence_text)
        return anchor_hits >= 1
    # 无锚点词时使用通用词兜底
    if generic_terms:
        generic_hits = sum(1 for t in generic_terms if t in evidence_text)
        return generic_hits >= min(2, len(generic_terms))
    return True


def prescore_evidence(evidence_list: list, claim_text: str = "") -> dict:
    """对证据做规则级预分析，返回信号摘要。

    不调 LLM，纯字符串匹配。返回 dict：
    - debunk_count: 含辟谣关键词的证据数
    - authority_debunk_count: 权威来源 + 辟谣关键词
    - authority_count: 权威来源数
    - signal: "strong_debunk" | "weak_debunk" | "neutral" | "supporting"
    - debunk_snippets: 辟谣证据的摘要（给模型看的预消化文本）
    """
    if claim_text:
        anchor_terms, generic_terms = _extract_claim_key_terms(claim_text)
        # 向后兼容：合并为 flat list 供旧接口使用
        claim_terms = anchor_terms + generic_terms
    else:
        anchor_terms, generic_terms, claim_terms = [], [], []

    debunk_count = 0
    authority_debunk = 0
    authority_count = 0
    debunk_snippets = []

    for e in evidence_list:
        combined = f"{e.title} {e.snippet}"
        is_auth = _is_authority(e.url)
        is_debunk = _has_debunk_signal(combined)

        if is_auth:
            authority_count += 1

        if is_debunk:
            # 辟谣证据必须与当前声明讨论同一事件
            if anchor_terms or generic_terms:
                if not _evidence_matches_claim_strict(combined, anchor_terms, generic_terms):
                    logger.info(
                        "[prescore] 跳过不相关辟谣: %s (锚点词不匹配: %s)",
                        e.title[:40],
                        anchor_terms[:3],
                    )
                    continue
            elif claim_terms and not _evidence_matches_claim(combined, claim_terms):
                logger.debug("[prescore] 跳过不相关辟谣: %s (与声明无关键词重叠)", e.title[:40])
                continue
            debunk_count += 1
            debunk_snippets.append(f"[{'权威' if is_auth else '普通'}] {e.title}")
            if is_auth:
                authority_debunk += 1

    if authority_debunk >= 2:
        signal = "strong_debunk"
    elif debunk_count >= 2 or authority_debunk >= 1:
        signal = "weak_debunk"
    else:
        signal = "neutral"

    return {
        "debunk_count": debunk_count,
        "authority_debunk_count": authority_debunk,
        "authority_count": authority_count,
        "signal": signal,
        "debunk_snippets": debunk_snippets,
    }


def rule_based_verdict(score: dict) -> dict | None:
    """规则兜底：信号足够强时直接出判定，跳过 LLM。

    检查优先级：
    -1. 个人经历/观点分享 → 无法核实（最高优先级）
    0. 截图/聊天记录声明 → 无法核实
    1. 强辟谣信号 → 谣言
    2. 过时政策信号 → 谣言
    3. 官方文件缺失（强分支：有数字+日期 → FALSE/MOSTLY_FALSE）
    3.5~9. 金融诈骗/神医/假补贴/通用诈骗/恐慌链/食物相克/AI名人语录 → 谣言
    3弱. 官方文件缺失（弱分支：无具体数字日期 → UNVERIFIABLE）
    10. 旧闻信号 → 误导性信息

    返回 dict(verdict, confidence, reasoning) 或 None。
    """
    # 规则 -1：个人经历/观点 → 无法核实（最高优先级，不浪费任何资源）
    personal = score.get("personal_content")
    if personal:
        return {
            "verdict": Verdict.UNVERIFIABLE,
            "confidence": 0.90,
            "reasoning": f"[规则判定·个人内容] {personal['reason']}",
        }

    # 规则 0：截图/聊天记录 → 无法核实（优先级最高，截图无法验证真伪）
    screenshot = score.get("screenshot_claim")
    if screenshot:
        return {
            "verdict": Verdict.UNVERIFIABLE,
            "confidence": 0.75,
            "reasoning": f"[规则判定·截图不可验证] {screenshot['reason']}",
        }

    # 规则 1：强辟谣（原有逻辑）
    if score["signal"] == "strong_debunk":
        snippets = "; ".join(score["debunk_snippets"][:3])
        return {
            "verdict": Verdict.FALSE,
            "confidence": 0.92,
            "reasoning": f"[规则判定] {score['authority_debunk_count']} 篇权威辟谣: {snippets}",
        }

    # 规则 2：过时政策（新增）
    obsolete = score.get("obsolete_policy")
    if obsolete:
        return {
            "verdict": Verdict.FALSE,
            "confidence": 0.90,
            "reasoning": (
                f"[规则判定·过时政策] {obsolete['reason']}。"
                f"该政策/措施自 {obsolete['obsolete_since']} 起已废止。"
            ),
        }

    # 规则 3：声称官方发文但无官方来源
    # 有辟谣/矛盾证据 + 精确数字+精确日期 → FALSE
    # 有辟谣/矛盾证据 + (精确数字或日期) → MOSTLY_FALSE
    # 无辟谣/矛盾证据 → 搜索缺失不等于证伪，弱分支延后到规则 9 之后
    unverified_official = score.get("unverified_official_claim")
    has_debunk_or_contradiction = (
        score.get("debunk_count", 0) > 0
        or score.get("authority_debunk_count", 0) > 0
        or score.get("signal") in ("strong_debunk", "weak_debunk")
        or score.get("contradiction_count", 0) > 0
        or score.get("authority_contradiction_count", 0) > 0
    )
    if unverified_official:
        import re as _re

        claim_text_check = score.get("_claim_text", "")
        has_specific_number = bool(_re.search(r"\d+[%％万亿元]", claim_text_check))
        has_specific_date = bool(
            _re.search(r"20\d{2}年(\d{1,2}月(\d{1,2}日)?)?起?|下[月周]起|即日起", claim_text_check)
        )
        if has_debunk_or_contradiction and has_specific_number and has_specific_date:
            return {
                "verdict": Verdict.FALSE,
                "confidence": 0.88,
                "reasoning": (
                    f"[规则判定·伪造官方公告] {unverified_official['reason']}。"
                    f"声明同时含具体日期和数字，且存在明确辟谣或矛盾证据，"
                    f"可判定为伪造。"
                ),
            }
        if has_debunk_or_contradiction and (has_specific_number or has_specific_date):
            return {
                "verdict": Verdict.MOSTLY_FALSE,
                "confidence": 0.78,
                "reasoning": (
                    f"[规则判定·伪造官方公告] {unverified_official['reason']}。"
                    f"声明含具体时间或数字，且存在明确辟谣或矛盾证据，"
                    f"高概率为伪造。"
                ),
            }
        # 无明确辟谣/矛盾时，缺少官方原文不能作为证伪依据；弱分支延后到规则 9 之后

    # 规则 3.5：金融内幕/荐股诈骗 → 谣言
    financial = score.get("financial_scam")
    if financial:
        return {
            "verdict": Verdict.FALSE,
            "confidence": 0.88,
            "reasoning": f"[规则判定·金融诈骗] {financial['reason']}",
        }

    # 规则 5：万能养生/伪医疗 → 谣言
    miracle = score.get("miracle_cure")
    if miracle:
        return {
            "verdict": Verdict.FALSE,
            "confidence": 0.88,
            "reasoning": f"[规则判定·伪医疗] {miracle['reason']}",
        }

    # 规则 6：假补贴/二维码诈骗 → 谣言（SCAM）
    fake_subsidy = score.get("fake_subsidy_scam")
    if fake_subsidy:
        return {
            "verdict": Verdict.FALSE,
            "confidence": 0.90,
            "reasoning": f"[规则判定·假补贴诈骗] {fake_subsidy['reason']}",
        }

    # 规则 6.5：通用诈骗模式（ETC/快递/刷单等）→ 谣言
    general_scam = score.get("general_scam")
    if general_scam:
        return {
            "verdict": Verdict.FALSE,
            "confidence": 0.85,
            "reasoning": f"[规则判定·通用诈骗] {general_scam['reason']}",
        }

    # 规则 7：本地恐慌链 → 谣言（精确地震预测）或 大部分不实（其他恐慌链）
    panic = score.get("local_panic")
    if panic:
        if panic.get("is_earthquake_prediction"):
            return {
                "verdict": Verdict.FALSE,
                "confidence": 0.90,
                "reasoning": f"[规则判定·伪地震预测] {panic['reason']}",
            }
        return {
            "verdict": Verdict.MOSTLY_FALSE,
            "confidence": 0.78,
            "reasoning": (
                f"[规则判定·本地恐慌链] {panic['reason']}。未找到当地官方通报，高概率为恐慌链谣言。"
            ),
        }

    # 规则 8：食物相克 → 谣言
    food = score.get("food_incompatibility")
    if food:
        return {
            "verdict": Verdict.FALSE,
            "confidence": 0.88,
            "reasoning": f"[规则判定·食物相克] {food['reason']}",
        }

    # 规则 9：AI 名人语录/专家代言 → 谣言（名人+高风险产品+无原始出处=典型伪造）
    ai_celeb = score.get("ai_celebrity_quote")
    if ai_celeb:
        return {
            "verdict": Verdict.FALSE,
            "confidence": 0.85,
            "reasoning": f"[规则判定·名人语录] {ai_celeb['reason']}",
        }

    # 规则 9.2：食品安全谣言 → 谣言
    food_myth = score.get("food_safety_myth")
    if food_myth:
        return {
            "verdict": Verdict.FALSE,
            "confidence": 0.88,
            "reasoning": f"[规则判定·食品安全谣言] {food_myth['reason']}",
        }

    # 规则 9.4：民生政策谣言 → 谣言
    livelihood = score.get("livelihood_policy_rumor")
    if livelihood:
        return {
            "verdict": Verdict.FALSE,
            "confidence": 0.85,
            "reasoning": f"[规则判定·民生政策谣言] {livelihood['reason']}",
        }

    # 规则 9.6：AI 生成/合成内容 → 谣言
    ai_content = score.get("ai_generated_content")
    if ai_content:
        return {
            "verdict": Verdict.FALSE,
            "confidence": 0.85,
            "reasoning": f"[规则判定·AI生成内容] {ai_content['reason']}",
        }

    # 规则 9.8：因果逻辑谬误 → 误导性信息
    causal = score.get("causal_fallacy")
    if causal:
        return {
            "verdict": Verdict.MISLEADING,
            "confidence": 0.80,
            "reasoning": f"[规则判定·因果谬误] {causal['reason']}",
        }

    # 规则 3 弱分支（延后）：官方文件缺失但无辟谣/矛盾证据 → UNVERIFIABLE
    # 搜索缺失不等于证伪，在没有明确辟谣或矛盾证据时只能判定为无法核实
    if unverified_official:
        return {
            "verdict": Verdict.UNVERIFIABLE,
            "confidence": 0.65,
            "reasoning": (
                f"[规则判定·官方文件缺失（待核实）] {unverified_official['reason']}。"
                f"声明称「{unverified_official['entity']}」"
                f"{unverified_official['action']}了相关政策，"
                f"但搜索未找到任何官方原文。搜索缺失不等于证伪，"
                f"在没有明确辟谣或矛盾证据时只能判定为无法核实。"
            ),
        }

    # 规则 10：旧闻信号——即时性词汇+证据全是旧年份 → 误导性信息
    # 例外：如果声明含官方机构+发文动词，旧证据大概率来自不相关事件，
    # 不是旧闻翻炒，跳过此规则让 FactChecker 判断。
    stale = score.get("stale_evidence")
    claim_text_for_check = score.get("_claim_text", "")
    has_official_pattern = score.get("unverified_official_claim") is not None or (
        any(e in claim_text_for_check for e in _GOVERNMENT_ENTITIES)
        and any(a in claim_text_for_check for a in _OFFICIAL_CLAIM_PATTERNS)
    )
    if stale and stale["signal"] == "stale_evidence" and not has_official_pattern:
        years_str = "/".join(str(y) for y in stale["stale_years"])
        return {
            "verdict": Verdict.MISLEADING,
            "confidence": 0.82,
            "reasoning": (
                f"[规则判定·旧闻翻炒] 声明使用即时性表述，"
                f"但搜索到的 {stale['stale_count']} 条支持性证据均来自 {years_str} 年，"
                f"与当前时间严重不符。相关事件可能确实发生过，但被移除时间语境后重新传播。"
            ),
        }

    return None


_LLM_AGENTS = {
    "ClaimExtractor",
    "CheckWorthy",
    "CommonsenseChecker",
    "FactChecker",
    "ResponseComposer",
    "ScenarioRouter",
    "QueryPlanner",
    "EvidenceRanker",
    "Skeptic",
    "StructuredFC",
}


@dataclass
class StepLog:
    agent: str
    action: str
    duration_ms: int
    input_summary: str
    output_summary: str
    output_data: dict | None = None
    human_narrative: str = ""
    display: dict | None = None


@dataclass
class VerifyTrace:
    """完整的核查追踪记录。"""

    steps: list[StepLog] = field(default_factory=list)
    total_duration_ms: int = 0
    total_llm_calls: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def add(self, step: StepLog) -> None:
        with self._lock:
            self.steps.append(step)
            self.total_duration_ms += step.duration_ms
            if step.agent in _LLM_AGENTS:
                self.total_llm_calls += 1


def _atom_verification_record(atom: dict, atom_cv: ClaimVerification) -> dict:
    """把单个原子的验证结果打包成 _aggregate_atom_verdicts 消费的中间记录。

    必须携带 relations（= atom_cv.evidence_relations），否则原子内的证据方向标签
    在聚合时被丢弃，下游证实/证伪维度拿不到方向 → C 恒 0（真消息无法救回）。
    key 必须与 _aggregate_atom_verdicts 读取的 key 完全对齐。
    见 BUGS.md 2026-05-29 / CONTRACTS.md C4.4 / C5.2。
    """
    return {
        "atom_id": atom.get("id", "A?"),
        "text": atom["text"],
        "is_core": atom.get("is_core", True),
        "verdict": atom_cv.verdict,
        "confidence": atom_cv.confidence,
        "evidence": atom_cv.evidence_chain,
        "relations": atom_cv.evidence_relations or [],
        "reasoning": atom_cv.reasoning,
    }


def _aggregate_atom_verdicts(
    atom_results: list[dict],
    original_claim: Claim,
) -> ClaimVerification:
    """确定性聚合原子验证结果 → 整条 claim 的判定。

    规则（Oracle 建议）：
    - 任何核心原子被高置信辟谣/矛盾 → FALSE 或 MOSTLY_FALSE
    - 部分核心原子验证 + 部分不支持 → PARTLY_TRUE 或 UNVERIFIABLE
    - 全部核心原子被验证 → TRUE
    - 核心原子证据不足 → UNVERIFIABLE
    """
    if not atom_results:
        return ClaimVerification(
            claim=original_claim,
            verdict=Verdict.UNVERIFIABLE,
            confidence=0.3,
            evidence_chain=[],
            reasoning="[原子化] 无原子验证结果",
        )

    core_atoms = [a for a in atom_results if a.get("is_core", True)]
    all_atoms = atom_results

    core_false = [a for a in core_atoms if a["verdict"] in (Verdict.FALSE, Verdict.MOSTLY_FALSE)]
    core_true = [a for a in core_atoms if a["verdict"] == Verdict.TRUE]
    core_unverifiable = [a for a in core_atoms if a["verdict"] == Verdict.UNVERIFIABLE]
    core_misleading = [a for a in core_atoms if a["verdict"] == Verdict.MISLEADING]

    all_evidence = []
    all_relations: list[dict] = []
    reasoning_parts = []
    for a in all_atoms:
        ev = a.get("evidence", [])
        offset = len(all_evidence)  # 该原子证据在拼接链中的起始位置
        all_evidence.extend(ev)
        for rel in a.get("relations", []):
            idx = rel.get("index")
            # 原子内局部 index 越界 → 丢弃（防 StructuredFC 脏标签重映射到错误证据）
            if not isinstance(idx, int) or not (0 <= idx < len(ev)):
                continue
            remapped = dict(rel)
            remapped["index"] = idx + offset  # 局部 index → 全局拼接链 index
            all_relations.append(remapped)
        reasoning_parts.append(f"[{a['atom_id']}] {a['text'][:30]} → {a['verdict'].value}")

    reasoning_str = "[原子化聚合] " + "; ".join(reasoning_parts)
    # evidence_chain 截断到 10，relations 必须同步丢弃指向截断区外的标签
    evidence_chain = all_evidence[:10]
    evidence_relations = [r for r in all_relations if r["index"] < len(evidence_chain)]

    if core_false:
        if len(core_false) == len(core_atoms):
            verdict = Verdict.FALSE
            conf = 0.88
        else:
            verdict = Verdict.MOSTLY_FALSE
            conf = 0.75
    elif core_misleading:
        verdict = Verdict.MISLEADING
        conf = 0.78
    elif core_true and not core_unverifiable:
        verdict = Verdict.TRUE
        conf = 0.82
    elif core_true and core_unverifiable:
        verdict = Verdict.PARTLY_TRUE
        conf = 0.65
    elif core_unverifiable and len(core_unverifiable) == len(core_atoms):
        verdict = Verdict.UNVERIFIABLE
        conf = 0.50
    else:
        verdict = Verdict.PARTLY_TRUE
        conf = 0.55

    return ClaimVerification(
        claim=original_claim,
        verdict=verdict,
        confidence=conf,
        evidence_chain=evidence_chain,
        evidence_relations=evidence_relations,
        reasoning=reasoning_str,
    )


def _serialize_unverifiable_reason(non_adjudicated, blocked_condition: str) -> dict:
    """把 NonAdjudicatedAction 按统一契约序列化成 unverifiable_reason dict。

    供 SSE done 事件的 claim 对象使用；前端 transformResponse 转 camelCase
    (unverifiableReason{code, codeLabel, detail, blockedCondition, verifyWhere})。

    契约字段（snake_case）：
      code            = primary_blocker 枚举名（如 MISSING_KEY_CONTEXT）——机读
      code_label      = primary_blocker 中文名（如 "缺关键语境"）——展示
      detail          = 这一条专属、不断真伪的具体障碍说明
      blocked_condition = 卡在哪一句（即被核查的声明原文）
      verify_where    = 去哪做一手确认

    INV-U3：detail/code_label 均不得断真伪；本函数只搬运 attribution 已校验过的字段，
    不新增任何能旁路裁决产 verdict 的内容（守 INV-4）。
    """
    blocker = non_adjudicated.primary_blocker
    return {
        "code": blocker.name,
        "code_label": blocker.value,
        "detail": non_adjudicated.claim_specific_detail,
        "blocked_condition": blocked_condition,
        "verify_where": non_adjudicated.verify_where,
        # 机读位：被三无守卫拦下不证伪时含 'falsify_guarded'，前端据此渲染
        # "故意信息不全·非证据不足"，且不得暗示可改判（INV-U5）。
        "secondary_flags": list(non_adjudicated.secondary_flags or []),
    }


def _pick_overall_verdict(verifications: list[ClaimVerification]) -> Verdict:
    if not verifications:
        return Verdict.UNVERIFIABLE
    verdicts = {cv.verdict for cv in verifications}
    # 如果有任何 FALSE/MOSTLY_FALSE，整体偏危险
    if Verdict.FALSE in verdicts:
        return Verdict.FALSE
    if Verdict.MOSTLY_FALSE in verdicts:
        return Verdict.MOSTLY_FALSE
    if Verdict.MISLEADING in verdicts:
        return Verdict.MISLEADING
    # 如果有 UNVERIFIABLE 混合其他结论，整体保守标记
    if Verdict.UNVERIFIABLE in verdicts and len(verdicts) > 1:
        return Verdict.PARTLY_TRUE
    if Verdict.UNVERIFIABLE in verdicts:
        return Verdict.UNVERIFIABLE
    if Verdict.PARTLY_TRUE in verdicts:
        return Verdict.PARTLY_TRUE
    return Verdict.TRUE


# ── INV-3 · Skeptic 降级护栏（CONTRACTS.md C0） ──

# INV-3「通用怀疑」白名单——精确长短语，不含「也可能是」「也许有」之类过短模式
# 避免对含「也可能是某种特殊情况」「也许有 WHO 数据支持」等合法具体反驳误拦截
_SKEPTIC_GENERIC_DOUBT_PATTERNS = [
    "可能被过度简化",
    "原始说法可能更复杂",
    "部分地区/时间段的例外",
    "也许有地区差异",
    "可能存在个别例外",
    "也许有例外情况",
    "可能是过时信息",
    "也许有未覆盖",
    "存在地区差异",
    "可能没有覆盖",
    "可能简化了原始说法",
    "或许有特殊情况",
]

_SKEPTIC_STRONG_CONTRADICTION_RELATIONS = {"直接辟谣", "间接矛盾"}


def _apply_skeptic_invariants(
    verification: ClaimVerification,
    skeptic_result: SkepticChallenge,
) -> tuple[SkepticChallenge, str]:
    """INV-3 守护：阻止 Skeptic 用通用怀疑把已结论的判定降到 UNVERIFIABLE。

    触发拦截需 ALL 同时满足：
    - skeptic_result.revised_verdict == UNVERIFIABLE
    - verification.verdict ∈ {FALSE, MOSTLY_FALSE, PARTLY_TRUE}
      （PARTLY_TRUE 2026-05-28 ILLUSION case 加入：StructuredFC 给出部分属实结论时
       同样不允许通用怀疑无脑降级）
    - verification.evidence_relations 含至少一条「直接辟谣」或「间接矛盾」
    - skeptic_result.challenges 仅含通用怀疑（无具体同主张证据缺口）

    返回 (修正后的 skeptic_result, 触发说明)。
    未拦截时说明为空字符串。

    见 CONTRACTS.md C0 INV-3 / case_213 / BUGS.md 2026-05-28 Bug A。
    """
    if skeptic_result.revised_verdict != Verdict.UNVERIFIABLE:
        return skeptic_result, ""

    if verification.verdict not in (
        Verdict.FALSE,
        Verdict.MOSTLY_FALSE,
        Verdict.PARTLY_TRUE,
    ):
        return skeptic_result, ""

    relations = verification.evidence_relations or []
    strong_count = sum(
        1 for r in relations if r.get("relation") in _SKEPTIC_STRONG_CONTRADICTION_RELATIONS
    )
    if strong_count == 0:
        return skeptic_result, ""

    # P1 #4 v2（adversarial HIGH 修）：PARTLY_TRUE 锁死要求更高。
    # 之前用 confidence < 0.6 判定——会把 LLM ±0.1 抖动直接传到 verdict，
    # 同一 case 跑两次可能 PARTLY_TRUE / UNVERIFIABLE 切换，破坏 determinism。
    # 改用离散 relation 类型：strong_count==1 且全是「间接矛盾」（弱矛盾）放行；
    # 「直接辟谣」即使 1 条也锁死（强矛盾信号足够）。
    if verification.verdict == Verdict.PARTLY_TRUE and strong_count == 1:
        only_indirect = all(
            r.get("relation") == "间接矛盾"
            for r in relations
            if r.get("relation") in _SKEPTIC_STRONG_CONTRADICTION_RELATIONS
        )
        if only_indirect:
            return skeptic_result, ""

    challenges_text = " ".join(skeptic_result.challenges or [])
    is_generic = any(p in challenges_text for p in _SKEPTIC_GENERIC_DOUBT_PATTERNS) or not (
        challenges_text.strip()
    )

    if not is_generic:
        return skeptic_result, ""

    notice = (
        f"[INV-3 拦截] 当前已有强矛盾证据（直接辟谣/间接矛盾 {strong_count} 条），"
        f"不允许用通用怀疑降级 {verification.verdict.value} → 无法核实"
    )
    # P1 #5：用 inv3_blocked 独立字段标记，不再追加 notice 到 challenges。
    # humanize._h_skeptic 把 challenges 前 2 项各取 18 字给用户看，notice
    # 字符串会被截成 "[INV-3 拦截] 当前已有" 污染用户可见预览。
    # 原始 challenges 保留不动，notice 由 orchestrator 的 diag 通道单独传播。
    return (
        SkepticChallenge(
            challenges=list(skeptic_result.challenges or []),
            passed=True,
            revised_verdict=None,
            inv3_blocked=True,
        ),
        notice,
    )


# ── INV-2 · CoverageAuditor ──


def _audit_coverage(
    frame: MessageFrame,
    claims: list,
    verifications: list,
) -> dict:
    """检查 verification_burden 清单是否被 claims/verifications 覆盖。

    返回 dict:
      - burden: list[str] — 必查清单
      - covered: list[str] — 已覆盖的项
      - missing: list[str] — 未覆盖的项
      - central_action_present: bool — central_action_claim 是否在 claims 中
      - satisfied: bool — 是否全部覆盖
      - downgrade_blocked: bool — 是否应阻止过早降级到 UNVERIFIABLE
    """
    burden = list(frame.verification_burden or [])
    covered: list[str] = []
    missing: list[str] = []

    all_text = " ".join([c.text for c in claims] + [v.reasoning for v in verifications]).lower()

    for item in burden:
        # 简单子串匹配：如果 burden 字段名出现在 claim text 或 reasoning 中
        # （后续可升级为 LLM 语义匹配）
        if item and any(token in all_text for token in [item.lower(), item[:2].lower()]):
            covered.append(item)
        else:
            missing.append(item)

    central_action = frame.central_action_claim.strip()
    central_action_present = bool(
        central_action
        and any(
            central_action in c.text or c.text in central_action or c.is_central_action
            for c in claims
        )
    )

    satisfied = bool(burden) and not missing and central_action_present

    # 对推销类，central_action 缺失或 burden 严重不覆盖 → 标记 downgrade_blocked
    downgrade_blocked = frame.message_type == MessageType.HEALTH_PRODUCT_PROMO and (
        not central_action_present or len(missing) >= max(1, len(burden) // 2)
    )

    return {
        "burden": burden,
        "covered": covered,
        "missing": missing,
        "central_action_present": central_action_present,
        "satisfied": satisfied,
        "downgrade_blocked": downgrade_blocked,
        "message_type": frame.message_type.value,
    }


class Orchestrator:
    """核查流水线编排器。"""

    def __init__(
        self,
        search_provider: SearchProvider | None = None,
        memory_store: MemoryStore | None = None,
        *,
        use_advanced: bool = True,
        use_rules: bool = True,
        use_gates: bool = True,
        use_search: bool = True,
        use_debunk_index: bool | None = None,
    ):
        self.searcher = search_provider or get_search_provider()
        self.memory_store = memory_store
        self.extractor = ClaimExtractorAgent()
        self.checkworthy = CheckWorthyAgent()
        self.hunter = EvidenceHunterAgent(self.searcher)
        self.checker = StructuredFactCheckerAgent()
        self.composer = ResponseComposerAgent()
        self.use_advanced = use_advanced
        self.use_rules = use_rules
        self.use_gates = use_gates
        self.use_search = use_search
        # 官方辟谣库本地检索开关：None → 取 .env 的 ENABLE_DEBUNK_INDEX（默认开）。
        # 一键关用于 demo / 评委环节，行为退回纯流式取证。命中只进证据不出判定（INV-4）。
        if use_debunk_index is None:
            from .config import settings as _cfg

            self.use_debunk_index = _cfg.enable_debunk_index
        else:
            self.use_debunk_index = use_debunk_index
        self.commonsense_checker = CommonsenseCheckerAgent()
        self.atomizer = AtomicFactExtractorAgent()
        if use_advanced:
            self.scenario_router = ScenarioRouterAgent()
            self.query_planner = QueryPlannerAgent()
            self.evidence_ranker = EvidenceRankerAgent()
            self.skeptic = SkepticAgent()
        self.trace = VerifyTrace()
        self.on_step: callable | None = None
        self._current_diagnostics: list[DiagnosticTrace] = []
        self._credibility_map: dict[str, str] = {}
        if memory_store:
            self._load_credibility_map(memory_store)

    def _timed(self, agent_name: str, action: str, func, *args, **kwargs):
        """执行并计时，记录到 trace。"""
        t0 = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = int((time.perf_counter() - t0) * 1000)

        input_summary = str(args[0])[:80] if args else ""
        output_data: dict | None = None
        if isinstance(result, list):
            output_summary = f"{len(result)} items"
        elif isinstance(result, tuple):
            output_summary = f"tuple({len(result)} elements)"
        else:
            output_summary = str(result)[:80]
            if hasattr(result, "model_dump"):
                try:
                    output_data = result.model_dump(mode="json")
                except Exception:
                    output_data = None
            elif isinstance(result, dict):
                try:
                    output_data = json.loads(json.dumps(result, default=str, ensure_ascii=False))
                except Exception:
                    output_data = None

        human_narrative = ""
        display: dict | None = None
        try:
            human_narrative, display = humanize_step(
                agent_name=agent_name,
                action=action,
                result=result,
                output_data=output_data,
                output_summary=output_summary,
            )
        except Exception:  # noqa: BLE001
            human_narrative = ""
            display = None

        step = StepLog(
            agent=agent_name,
            action=action,
            duration_ms=elapsed,
            input_summary=input_summary,
            output_summary=output_summary,
            output_data=output_data,
            human_narrative=human_narrative,
            display=display,
        )
        self.trace.add(step)
        if self.on_step:
            with contextlib.suppress(Exception):
                self.on_step(step)
        logger.info("[%s] %s 完成 (%dms)", agent_name, action, elapsed)
        return result

    def _load_credibility_map(self, memory_store: MemoryStore) -> None:
        """从 source_registry 加载域名→等级映射。"""
        try:
            conn = memory_store._conn()
            rows = conn.execute("SELECT domain, credibility_grade FROM source_registry").fetchall()
            conn.close()
            self._credibility_map = {r["domain"]: r["credibility_grade"] for r in rows}
        except Exception:
            logger.warning("[Orchestrator] 加载 credibility_map 失败，使用空映射", exc_info=True)
            self._credibility_map = {}

    def _verify_single_claim(
        self, claim: Claim, scenario_info: dict | None, original_message: str = ""
    ) -> ClaimVerification:
        """对单条 claim 执行完整的搜索→排序→核查→质疑流程。

        线程安全：每次调用创建独立的 hunter 副本，不共享状态。
        补搜逻辑：空证据和 insufficient 都触发补充搜索。
        claim 级记忆：单条 claim 命中记忆且无负面反馈则复用。
        """
        # claim 级记忆查询（使用新的候选召回契约）
        if self.memory_store:
            try:
                candidates = self.memory_store.recall_claim_candidates(claim.text)
            except Exception:
                logger.warning(
                    "[Orchestrator] recall_claim_candidates 失败，跳过记忆", exc_info=True
                )
                candidates = []
            for cand in candidates:
                if not cand.get("reusable"):
                    continue
                case_id = cand.get("case_id")
                if case_id and self.memory_store.has_negative_feedback(case_id):
                    continue
                claim_id = cand.get("claim_id")
                match_type = cand.get("match_type", "exact")
                sim = cand.get("similarity", 1.0)
                logger.info(
                    "[Orchestrator] claim 记忆命中（%s, %.2f）：%s",
                    match_type,
                    sim,
                    claim.text[:30],
                )
                self.trace.add(
                    StepLog(
                        agent="MemoryStore",
                        action=f"claim 记忆命中（{match_type}）",
                        duration_ms=0,
                        input_summary=claim.text[:60],
                        output_summary=f"similarity={sim}",
                    )
                )
                # 优先用 claim_id 精确恢复
                if claim_id:
                    cv = self.memory_store.restore_claim_verification(claim_id)
                    if cv:
                        try:
                            self.memory_store.bump_hit_count(cand["id"])
                        except Exception:
                            logger.warning("[Orchestrator] bump_hit_count 失败", exc_info=True)
                        return cv
                # 兜底：从完整案例中按文本匹配
                if case_id:
                    full_case = self.memory_store.get_full_case(case_id)
                    if full_case:
                        for c in full_case.get("claims", []):
                            try:
                                cv = ClaimVerification.model_validate(c)
                                if cv.claim.text == cand["text"]:
                                    try:
                                        self.memory_store.bump_hit_count(cand["id"])
                                    except Exception:
                                        logger.warning(
                                            "[Orchestrator] bump_hit_count 失败", exc_info=True
                                        )
                                    return cv
                            except Exception:
                                pass

        # ── Step 1.7：常识快速路径 ──
        # 对常识级伪科学（如"运动出汗越多越燃脂"），LLM 训练知识即可判定，
        # 跳过 QueryPlanner / EvidenceHunter / EvidenceRanker，避免 timeout。
        # 安全边界：只允许 FALSE/MOSTLY_FALSE 输出 + confidence >= 0.85 + D2 门兜底。
        try:
            commonsense_result = self._timed(
                "CommonsenseChecker",
                f"常识检查「{claim.text[:15]}」",
                self.commonsense_checker.check,
                claim,
            )
        except Exception:
            commonsense_result = {
                "is_commonsense": False,
                "commonsense_type": "n/a",
                "llm_verdict": None,
                "confidence": 0.0,
                "reasoning": "常识检查异常",
            }
            logger.warning("[Orchestrator] 常识检查调用失败，走完整流水线", exc_info=True)

        _COMMONSENSE_CONFIDENCE_THRESHOLD = 0.85
        if (
            commonsense_result.get("is_commonsense")
            and commonsense_result.get("confidence", 0) >= _COMMONSENSE_CONFIDENCE_THRESHOLD
            and commonsense_result.get("llm_verdict")
        ):
            verdict_str = commonsense_result["llm_verdict"]
            verdict = VERDICT_MAP.get(verdict_str)
            if verdict and verdict in (
                Verdict.FALSE,
                Verdict.MOSTLY_FALSE,
                Verdict.TRUE,
                Verdict.PARTLY_TRUE,
            ):
                logger.info(
                    "[Orchestrator] 常识快速路径命中：%s → %s (%.0f%%)",
                    claim.text[:30],
                    verdict.value,
                    commonsense_result["confidence"] * 100,
                )
                return ClaimVerification(
                    claim=claim,
                    verdict=verdict,
                    confidence=commonsense_result["confidence"],
                    evidence_chain=[],
                    reasoning=(
                        f"[常识快速路径·{commonsense_result.get('commonsense_type', 'n/a')}] "
                        f"{commonsense_result.get('reasoning', '')}"
                    ),
                )

        # ── 原子化尝试：复合声明拆分为独立原子，各自验证后聚合 ──
        try:
            atom_result = self._timed(
                "AtomicFact",
                f"原子化「{claim.text[:15]}」",
                self.atomizer.atomize,
                claim,
            )
        except Exception:
            atom_result = {"should_atomize": False, "atoms": [], "atomization_risk": "high"}
            logger.warning("[Orchestrator] 原子化调用失败，走整条 claim 路径", exc_info=True)

        if (
            atom_result.get("should_atomize")
            and atom_result.get("atomization_risk") != "high"
            and len(atom_result.get("atoms", [])) >= 2
        ):
            atoms = atom_result["atoms"]
            logger.info(
                "[Orchestrator] 原子化成功：%d 个原子，风险=%s",
                len(atoms),
                atom_result.get("atomization_risk"),
            )
            atom_verifications = []
            for atom in atoms:
                atom_claim = Claim(
                    text=atom["text"],
                    category=claim.category,
                    original_context=claim.text,
                )
                try:
                    atom_cv = self._verify_single_atom(atom_claim, scenario_info, original_message)
                    atom_verifications.append(_atom_verification_record(atom, atom_cv))
                except Exception:
                    logger.warning("[Orchestrator] 原子 %s 验证失败", atom.get("id"), exc_info=True)
                    atom_verifications.append(
                        {
                            "atom_id": atom.get("id", "A?"),
                            "text": atom["text"],
                            "is_core": atom.get("is_core", True),
                            "verdict": Verdict.UNVERIFIABLE,
                            "confidence": 0.2,
                            "evidence": [],
                            "relations": [],
                            "reasoning": "原子验证失败",
                        }
                    )
            return _aggregate_atom_verdicts(atom_verifications, claim)

        # ── ClaimMatcher：官方辟谣库本地检索 + 辟谣网站定向预搜索 ──
        # 红线（CONTRACTS.md INV-4）：辟谣库命中只进入 prior_evidence（当高权威证据），
        # 绝不短路最终 verdict。判定仍由下游 6 维度引擎 + 规则裁决出。
        import asyncio as _asyncio

        from .search import enrich_evidence

        prior_evidence: list = []

        # 1) 离线/本地官方辟谣库证据检索（无网络、无 LLM）。
        #    复用 public_board.load_crawler_rumors() 的爬虫库；命中需过同命题守卫才进证据。
        if self.use_debunk_index:
            try:
                from .debunk_index import (
                    confirmed_candidate_to_evidence,
                    retrieve_debunk_candidates,
                    verify_same_claim,
                )

                t0 = time.perf_counter()
                debunk_candidates = retrieve_debunk_candidates(claim.text, top_k=3)
                elapsed = int((time.perf_counter() - t0) * 1000)

                self.trace.add(
                    StepLog(
                        agent="ClaimMatcher",
                        action="官方辟谣库检索",
                        duration_ms=elapsed,
                        input_summary=claim.text[:80],
                        output_summary=f"命中{len(debunk_candidates)}条候选",
                    )
                )

                confirmed_local_hits = 0

                for cand in debunk_candidates:
                    t1 = time.perf_counter()
                    same_claim = verify_same_claim(claim.text, cand)
                    elapsed_verify = int((time.perf_counter() - t1) * 1000)

                    passed = same_claim.label == "same_claim"
                    status = "通过" if passed else "未通过"
                    reason_preview = "；".join(same_claim.reasons[:3])
                    match_input = f"claim={claim.text[:40]} | candidate={cand.claim_text[:40]}"

                    self.trace.add(
                        StepLog(
                            agent="ClaimMatcher",
                            action=f"同命题核对：{status}",
                            duration_ms=elapsed_verify,
                            input_summary=match_input,
                            output_summary=(
                                f"{same_claim.label} score={same_claim.score:.2f}"
                                + (f"；{reason_preview}" if reason_preview else "")
                            )[:240],
                        )
                    )

                    if not passed:
                        # same_topic_different_claim / opposite_claim / no_match：
                        # 仅作调试 trace，不进 prior_evidence，不作辟谣证据。
                        continue

                    ev = confirmed_candidate_to_evidence(cand, same_claim)
                    prior_evidence.append(ev)
                    confirmed_local_hits += 1

                    # 一条确认的官方辟谣通常足够，避免证据淹没。
                    break

                self.trace.add(
                    StepLog(
                        agent="ClaimMatcher",
                        action="官方辟谣库同命题核对",
                        duration_ms=0,
                        input_summary=f"{len(debunk_candidates)} candidates",
                        output_summary=f"通过{confirmed_local_hits}条/候选{len(debunk_candidates)}条",
                    )
                )

            except Exception:
                logger.warning("[ClaimMatcher] 官方辟谣库本地检索失败", exc_info=True)

        # 2) 现有的 live piyao 定向预搜索。与本地库检索并存，喂同一个 prior_evidence。
        if self.use_search:
            try:
                factcheck_queries = [
                    f"{claim.text[:30]} 辟谣 site:piyao.org.cn",
                    f"{claim.text[:30]} 辟谣 核查",
                ]
                for fq in factcheck_queries:
                    t0 = time.perf_counter()
                    try:
                        fc_results = _asyncio.run(self.searcher.search(fq, max_results=2))
                    except RuntimeError:
                        loop = _asyncio.new_event_loop()
                        fc_results = loop.run_until_complete(
                            self.searcher.search(fq, max_results=2)
                        )
                        loop.close()
                    elapsed = int((time.perf_counter() - t0) * 1000)
                    self.trace.add(
                        StepLog(
                            agent="ClaimMatcher",
                            action=f"辟谣预搜「{fq[:20]}」",
                            duration_ms=elapsed,
                            input_summary=fq[:60],
                            output_summary=f"{len(fc_results)} results",
                        )
                    )

                    live_hits_this_query = 0

                    for ev in fc_results:
                        enrich_evidence(ev)
                        if _has_debunk_signal(f"{ev.title} {ev.snippet}"):
                            ev.credibility = "S-辟谣库命中"
                            prior_evidence.append(ev)
                            live_hits_this_query += 1

                    # 仅当本次 live 查询有命中才 break；不因本地库已有命中而提前断掉 live 预搜。
                    if live_hits_this_query:
                        break
            except Exception:
                logger.warning("[ClaimMatcher] 辟谣预搜索失败", exc_info=True)

        if prior_evidence:
            logger.info("[ClaimMatcher] 命中 %d 条辟谣库证据", len(prior_evidence))

        # 每个 claim 用独立的 hunter 实例，避免并行竞态
        hunter = EvidenceHunterAgent(self.searcher)

        # QueryPlanner
        if self.use_advanced:
            query_plan = self._timed(
                "QueryPlanner", f"规划「{claim.text[:15]}」", self.query_planner.plan, claim
            )
            hunter.planned_queries = query_plan.queries
            hunter.scenario_context = scenario_info
        else:
            hunter.planned_queries = []
            hunter.scenario_context = None

        # 第一轮搜索（合并辟谣预搜索结果）
        if self.use_search:
            evidence, _ = self._timed(
                "EvidenceHunter", f"搜索「{claim.text[:15]}」", hunter.hunt, claim
            )
        else:
            evidence = []

        # 合并辟谣预搜索结果（去重后前置）
        # 空 URL 不参与去重：本地辟谣库命中的证据可能无 source_url（candidate.url=""），
        # 否则首条空 URL 占位后，其余空 URL 命中会被静默丢弃（code-review MEDIUM 修）。
        if prior_evidence:
            seen_urls = {e.url for e in evidence if e.url}
            for pe in prior_evidence:
                if not pe.url or pe.url not in seen_urls:
                    evidence.insert(0, pe)
                    if pe.url:
                        seen_urls.add(pe.url)

        # 信源去重：同根域名只保留最高权威度的一条
        if len(evidence) > 3:
            before = len(evidence)
            evidence = deduplicate_evidence(evidence)
            if len(evidence) < before:
                logger.info("[Orchestrator] 信源去重: %d → %d 条", before, len(evidence))

        if self.use_advanced:
            need_supplement = False

            if not evidence:
                # 空证据直接触发补搜
                need_supplement = True
                logger.info("[Orchestrator] 首轮无证据，直接补搜")
            else:
                ranking = self._timed(
                    "EvidenceRanker",
                    f"排序「{claim.text[:15]}」",
                    self.evidence_ranker.rank,
                    claim,
                    evidence,
                    self._credibility_map or None,
                )
                if ranking.ranked_evidence:
                    evidence = ranking.ranked_evidence
                if ranking.sufficiency == "insufficient":
                    need_supplement = True
                    logger.info("[Orchestrator] 证据不充分，启动补充搜索")

            if need_supplement:
                from datetime import datetime as _dt

                supplement_queries = [
                    f"{claim.text} 官方回应",
                    f"{claim.text} 最新消息 {_dt.now().year}",
                    f"{claim.text} 辟谣",
                ]
                hunter2 = EvidenceHunterAgent(self.searcher)
                hunter2.planned_queries = supplement_queries
                hunter2.scenario_context = scenario_info
                extra_evidence, _ = self._timed(
                    "EvidenceHunter", f"补搜「{claim.text[:15]}」", hunter2.hunt, claim
                )
                seen_urls = {e.url for e in evidence}
                for e in extra_evidence:
                    if e.url not in seen_urls:
                        evidence.append(e)

                if evidence:
                    ranking = self._timed(
                        "EvidenceRanker",
                        f"重排「{claim.text[:15]}」",
                        self.evidence_ranker.rank,
                        claim,
                        evidence,
                        self._credibility_map or None,
                    )
                    if ranking.ranked_evidence:
                        evidence = ranking.ranked_evidence
                    if ranking.sufficiency == "insufficient":
                        logger.info("[Orchestrator] 补搜后仍不充分，但仍交给 FactChecker 兜底判定")

        # ── 规则层：证据预分析 + 强信号直接判定 ──
        # 注意：即使无证据也要跑纯文本规则（miracle_cure / local_panic 等不依赖证据）
        raw_claim_text = (
            f"{claim.original_context} {claim.text}" if claim.original_context else claim.text
        )
        claim_full = _normalize_text(raw_claim_text)

        if evidence:
            score = prescore_evidence(evidence, claim_text=claim_full)
        else:
            score = {
                "debunk_count": 0,
                "authority_debunk_count": 0,
                "support_count": 0,
                "neutral_count": 0,
                "signal": "no_evidence",
                "debunk_snippets": [],
            }
        rule_text = _normalize_text(original_message) if original_message else claim_full
        score["_claim_text"] = claim_full
        score["screenshot_claim"] = detect_screenshot_claim(rule_text, evidence or [])
        score["obsolete_policy"] = detect_obsolete_policy(rule_text)
        score["unverified_official_claim"] = detect_unverified_official_claim(
            rule_text, evidence or []
        )
        score["financial_scam"] = detect_financial_scam(rule_text)
        score["miracle_cure"] = detect_miracle_cure(rule_text)
        score["fake_subsidy_scam"] = detect_fake_subsidy_scam(rule_text)
        score["general_scam"] = detect_general_scam(rule_text)
        score["local_panic"] = detect_local_panic(rule_text)
        score["food_incompatibility"] = detect_food_incompatibility(rule_text)
        score["ai_celebrity_quote"] = detect_ai_celebrity_quote(rule_text, evidence or [])
        score["food_safety_myth"] = detect_food_safety_myth(rule_text)
        score["livelihood_policy_rumor"] = detect_livelihood_policy_rumor(rule_text, evidence or [])
        score["ai_generated_content"] = detect_ai_generated_content(rule_text)
        score["causal_fallacy"] = detect_causal_fallacy(rule_text)
        score["stale_evidence"] = detect_stale_evidence(claim_full, evidence or [])
        # 供 _structured_verdict 使用：官方政策声明不适用旧闻规则
        score["suppress_stale_rule"] = score.get("unverified_official_claim") is not None or (
            any(e in claim_full for e in _GOVERNMENT_ENTITIES)
            and any(a in claim_full for a in _OFFICIAL_CLAIM_PATTERNS)
        )

        logger.info(
            "[Orchestrator] 证据预分析: 辟谣=%d 权威辟谣=%d 信号=%s "
            "过时政策=%s 官方文件缺失=%s 金融诈骗=%s 旧闻=%s",
            score["debunk_count"],
            score["authority_debunk_count"],
            score["signal"],
            bool(score["obsolete_policy"]),
            bool(score["unverified_official_claim"]),
            bool(score.get("financial_scam")),
            score["stale_evidence"]["signal"] if score["stale_evidence"] else "N/A",
        )

        # ── 诊断追踪初始化 ──
        diag = DiagnosticTrace(claim_text=claim.text[:80])
        diag.evidence_summary = {
            "total": len(evidence),
            "debunk_count": score["debunk_count"],
            "authority_debunk_count": score["authority_debunk_count"],
            "support_count": score.get("support_count", 0),
            "neutral_count": score.get("neutral_count", 0),
            "signal": score["signal"],
        }

        # 记录哪些规则有信号
        rule_signals = {}
        for rule_name in [
            "obsolete_policy",
            "unverified_official_claim",
            "financial_scam",
            "miracle_cure",
            "fake_subsidy_scam",
            "general_scam",
            "local_panic",
            "food_incompatibility",
            "ai_celebrity_quote",
            "food_safety_myth",
            "livelihood_policy_rumor",
            "ai_generated_content",
            "causal_fallacy",
            "screenshot_claim",
        ]:
            val = score.get(rule_name)
            if val:
                rule_signals[rule_name] = (
                    val.get("reason", str(val))[:120] if isinstance(val, dict) else str(val)[:120]
                )
        stale = score.get("stale_evidence")
        if stale and stale.get("signal") == "stale_evidence":
            rule_signals["stale_evidence"] = f"stale_years={stale.get('stale_years')}"

        diag.add(
            DecisionPoint(
                stage="evidence_prescore",
                verdict_after="",
                fired=True,
                detail=(
                    f"信号={score['signal']}, 辟谣={score['debunk_count']}, "
                    f"权威辟谣={score['authority_debunk_count']}"
                ),
                signals=rule_signals,
            )
        )

        rule_result = rule_based_verdict(score) if self.use_rules else None
        if rule_result:
            logger.info("[Orchestrator] 规则层直接判定: %s", rule_result["verdict"].value)
            diag.add(
                DecisionPoint(
                    stage="rule_engine",
                    verdict_after=rule_result["verdict"].value,
                    fired=True,
                    detail=rule_result["reasoning"][:200],
                    signals={"confidence": rule_result["confidence"]},
                )
            )
            diag.final_verdict = rule_result["verdict"].value
            diag.final_confidence = rule_result["confidence"]
            self._current_diagnostics.append(diag)
            return ClaimVerification(
                claim=claim,
                verdict=rule_result["verdict"],
                confidence=rule_result["confidence"],
                evidence_chain=evidence or [],
                reasoning=rule_result["reasoning"],
            )

        diag.add(
            DecisionPoint(
                stage="rule_engine",
                verdict_after="",
                fired=False,
                detail="无规则命中，交给 LLM FactChecker",
                signals=rule_signals,
            )
        )

        # 无证据且规则也没命中 → 标记无法核实，不浪费 LLM 调用
        if not evidence:
            logger.info("[Orchestrator] 无证据且规则未命中，标记无法核实")
            return ClaimVerification(
                claim=claim,
                verdict=Verdict.UNVERIFIABLE,
                confidence=0.2,
                evidence_chain=[],
                reasoning="搜索未找到可用证据，纯文本规则也未触发强信号，无法做出判定。",
            )

        # ── 模型层：把预消化信号传给 FactChecker ──
        if score["signal"] == "weak_debunk":
            for e in evidence:
                combined = f"{e.title} {e.snippet}"
                if _has_debunk_signal(combined):
                    e.credibility = f"{'S-权威辟谣' if _is_authority(e.url) else 'B-辟谣文章'}"

        # 过滤不相关辟谣证据：含辟谣关键词但锚点词不匹配的证据标记为低相关
        raw_ct = f"{claim.original_context} {claim.text}" if claim.original_context else claim.text
        _anchor, _generic = _extract_claim_key_terms(_normalize_text(raw_ct))
        if _anchor:
            filtered_evidence = []
            for e in evidence:
                combined = f"{e.title} {e.snippet}"
                if _has_debunk_signal(combined) and not _evidence_matches_claim_strict(
                    combined, _anchor, _generic
                ):
                    logger.info("[Orchestrator] 过滤不相关辟谣证据: %s", e.title[:40])
                    continue
                filtered_evidence.append(e)
            if filtered_evidence:
                evidence = filtered_evidence

        verification = self._timed(
            "FactChecker",
            f"核查「{claim.text[:15]}」",
            self.checker.check,
            claim,
            evidence,
            score,
        )

        diag.add(
            DecisionPoint(
                stage="fact_checker",
                verdict_after=verification.verdict.value,
                fired=True,
                detail=verification.reasoning[:200] if verification.reasoning else "",
                signals={"confidence": verification.confidence},
            )
        )

        skeptic_revised_to_harsher = False
        if self.use_advanced:
            skeptic_result = self._timed(
                "Skeptic", f"质疑「{claim.text[:15]}」", self.skeptic.challenge, claim, verification
            )
            # INV-3 护栏：阻止 Skeptic 用通用怀疑把 FALSE-family 降到 UNVERIFIABLE
            skeptic_result, inv3_notice = _apply_skeptic_invariants(verification, skeptic_result)
            if inv3_notice:
                logger.warning("[Orchestrator] %s", inv3_notice)
                # P1 #5 CRITICAL 修（adversarial 抓出）：_timed 在 _apply_skeptic_invariants
                # 之前已经把 raw SkepticChallenge（inv3_blocked=False）写入 trace step。
                # 在这里回写最近一个 Skeptic step 的 output_data + 重新 humanize，
                # 让 inv3_blocked=True 真正流到用户可见 narrative / SSE / 前端。
                #
                # adversarial v2 HIGH 修：多 claim 并发走 ThreadPoolExecutor 共享
                # self.trace.steps，单纯找 agent=="Skeptic" 会串扰到其他 claim
                # 的 step。用 action 字符串精确匹配（_timed 第 4574 行 action 嵌入了
                # claim.text[:15]）锁定当前 claim 的 step。
                target_action = f"质疑「{claim.text[:15]}」"
                for _step in reversed(self.trace.steps):
                    if _step.agent == "Skeptic" and _step.action == target_action:
                        if _step.output_data is None:
                            _step.output_data = {}
                        _step.output_data["inv3_blocked"] = True
                        try:
                            new_narrative, new_display = humanize_step(
                                agent_name=_step.agent,
                                action=_step.action,
                                result=skeptic_result,
                                output_data=_step.output_data,
                                output_summary=_step.output_summary,
                            )
                            _step.human_narrative = new_narrative
                            if new_display is not None:
                                _step.display = new_display
                        except Exception:  # noqa: BLE001
                            logger.debug("[Orchestrator] INV-3 回写后 humanize 失败", exc_info=True)
                        break
                diag.add(
                    DecisionPoint(
                        stage="skeptic_inv3_guard",
                        verdict_before=verification.verdict.value,
                        verdict_after=verification.verdict.value,
                        fired=True,
                        detail=inv3_notice[:200],
                        signals={"invariant": "INV-3", "blocked_downgrade": True},
                    )
                )
            if not skeptic_result.passed:
                if skeptic_result.revised_verdict:
                    verdict_severity = [
                        Verdict.TRUE,
                        Verdict.PARTLY_TRUE,
                        Verdict.UNVERIFIABLE,
                        Verdict.MISLEADING,
                        Verdict.MOSTLY_FALSE,
                        Verdict.FALSE,
                    ]
                    orig_idx = (
                        verdict_severity.index(verification.verdict)
                        if verification.verdict in verdict_severity
                        else 0
                    )
                    rev_idx = (
                        verdict_severity.index(skeptic_result.revised_verdict)
                        if skeptic_result.revised_verdict in verdict_severity
                        else 0
                    )
                    if rev_idx > orig_idx:
                        skeptic_revised_to_harsher = True
                    if rev_idx - orig_idx > 2:
                        capped = verdict_severity[min(orig_idx + 2, len(verdict_severity) - 1)]
                        logger.info(
                            "[Orchestrator] 质疑跳级过大 %s→%s，限制为 %s",
                            verification.verdict.value,
                            skeptic_result.revised_verdict.value,
                            capped.value,
                        )
                        skeptic_result.revised_verdict = capped

                    logger.info(
                        "[Orchestrator] 质疑未通过，%s → %s",
                        verification.verdict.value,
                        skeptic_result.revised_verdict.value,
                    )
                    diag.add(
                        DecisionPoint(
                            stage="skeptic",
                            verdict_before=verification.verdict.value,
                            verdict_after=skeptic_result.revised_verdict.value,
                            fired=True,
                            detail=f"质疑修正: {'; '.join(skeptic_result.challenges)}"[:200],
                            signals={"passed": False, "revised": True},
                        )
                    )
                    verification = ClaimVerification(
                        claim=claim,
                        verdict=skeptic_result.revised_verdict,
                        confidence=verification.confidence * 0.7,
                        evidence_chain=verification.evidence_chain,
                        reasoning=(
                            f"{verification.reasoning}\n"
                            f"[质疑修正] {'; '.join(skeptic_result.challenges)}"
                        ),
                        evidence_relations=verification.evidence_relations,
                    )
                else:
                    logger.info("[Orchestrator] 质疑未通过且无修正建议，降低置信度")
                    diag.add(
                        DecisionPoint(
                            stage="skeptic",
                            verdict_before=verification.verdict.value,
                            verdict_after=verification.verdict.value,
                            fired=True,
                            detail=f"质疑未通过但无修正: {'; '.join(skeptic_result.challenges)}"[
                                :200
                            ],
                            signals={"passed": False, "revised": False},
                        )
                    )
                    verification = ClaimVerification(
                        claim=claim,
                        verdict=verification.verdict,
                        confidence=verification.confidence * 0.7,
                        evidence_chain=verification.evidence_chain,
                        reasoning=(
                            f"{verification.reasoning}\n"
                            f"[质疑未通过] {'; '.join(skeptic_result.challenges)}"
                        ),
                        evidence_relations=verification.evidence_relations,
                    )
            else:
                diag.add(
                    DecisionPoint(
                        stage="skeptic",
                        verdict_before=verification.verdict.value,
                        verdict_after=verification.verdict.value,
                        fired=False,
                        detail="质疑通过，判定不变",
                        signals={"passed": True},
                    )
                )

        verification = self._apply_conservative_gates(
            verification,
            evidence=evidence,
            score=score,
            diag=diag,
            skeptic_revised=skeptic_revised_to_harsher,
        )

        diag.final_verdict = verification.verdict.value
        diag.final_confidence = verification.confidence
        self._current_diagnostics.append(diag)
        return verification

    _VALID_RELATIONS = {"直接辟谣", "间接矛盾", "直接支持", "话题相关", "不相关"}

    @staticmethod
    def _count_relations(
        verification: ClaimVerification, evidence_list: list | None = None
    ) -> dict:
        """统计 StructuredFC 标注的证据关系类别（驱动 D2/TRUE 门）。

        校验：
        - index 必须是合法整数且在 evidence_list 范围内
        - relation 必须在白名单中
        - 当提供 evidence_list 时做交叉验证：
            * 直接辟谣 → 证据原文必须含辟谣关键词
            * 直接支持 → 证据必须 supports_claim=True 或 authority_score>=0.6
            * 间接矛盾 → LLM 逻辑推理，原样信任
        被验伪的标签丢弃。
        """
        rels = verification.evidence_relations or []
        evs = evidence_list or []
        counts = {
            "direct_debunk": 0,
            "indirect_contradict": 0,
            "direct_support": 0,
            "topic_related": 0,
        }
        for la in rels:
            if not isinstance(la, dict):
                continue
            idx = la.get("index")
            rel = la.get("relation")
            if rel not in Orchestrator._VALID_RELATIONS:
                continue
            if evs:
                if not isinstance(idx, int) or idx < 0 or idx >= len(evs):
                    continue
                e = evs[idx]
                ev_text = f"{e.title} {e.snippet}"
                if rel == "直接辟谣":
                    if not _has_debunk_signal(ev_text):
                        continue
                    counts["direct_debunk"] += 1
                elif rel == "间接矛盾":
                    counts["indirect_contradict"] += 1
                elif rel == "直接支持":
                    if e.supports_claim is True or e.authority_score >= 0.6:
                        counts["direct_support"] += 1
                elif rel == "话题相关":
                    counts["topic_related"] += 1
            else:
                # 无 evidence_list 时不做交叉验证（向后兼容）
                if rel == "直接辟谣":
                    counts["direct_debunk"] += 1
                elif rel == "间接矛盾":
                    counts["indirect_contradict"] += 1
                elif rel == "直接支持":
                    counts["direct_support"] += 1
                elif rel == "话题相关":
                    counts["topic_related"] += 1
        return counts

    def _apply_conservative_gates(
        self,
        verification: ClaimVerification,
        *,
        evidence: list,
        score: dict,
        diag: DiagnosticTrace,
        skeptic_revised: bool = False,
    ) -> ClaimVerification:
        """应用 D2 保守门 + TRUE 救援门。

        - 标签优先（经交叉验证）；无可信标签时退回关键词计数兜底
        - Skeptic 已修正过的 verdict 不让 TRUE 救援反弹（避免撤销 Skeptic 的发现）
        - D2 覆盖 FALSE 和 MOSTLY_FALSE
        """
        claim = verification.claim
        rel = self._count_relations(verification, evidence)
        has_labels = bool(verification.evidence_relations)
        has_label_debunk = rel["direct_debunk"] >= 1 or rel["indirect_contradict"] >= 1
        has_label_support = rel["direct_support"] >= 1
        has_keyword_debunk = score.get("debunk_count", 0) >= 1 or score.get("signal") in (
            "strong_debunk",
            "weak_debunk",
        )

        # TRUE 救援门（先于 D2 跑：有支持证据/标签的 FALSE/MOSTLY_FALSE 应升级为 PARTLY_TRUE）
        # 例外：Skeptic 已经把 verdict 从轻判翻为重判，不应被救援反弹（撤销 Skeptic 发现）
        if (
            self.use_gates
            and verification.verdict in (Verdict.FALSE, Verdict.MOSTLY_FALSE)
            and not skeptic_revised
        ):

            def _is_genuine_support(e) -> bool:
                combined = f"{e.title} {e.snippet}".lower()
                if any(kw in combined for kw in _DEBUNK_KEYWORDS):
                    return False
                return (
                    e.authority_score >= 0.70
                    and e.source_type
                    not in (SourceType.UNKNOWN, SourceType.BLOG_FORUM, SourceType.SOCIAL_MEDIA)
                    and e.supports_claim is True
                )

            # 标签支持已经经过交叉验证（_count_relations 内部校验 authority/supports_claim），
            # 无标签时退回纯权威关键词扫描
            if has_label_support:
                genuine_supports: list = []
                has_high_auth_support = False
            else:
                genuine_supports = [e for e in evidence if _is_genuine_support(e)]
                has_high_auth_support = len(genuine_supports) > 0

            # 救援需要：(标签支持 或 高权威支持) AND (无标签辟谣) AND (无关键词辟谣)
            has_any_support = has_label_support or has_high_auth_support
            no_label_debunk = not has_label_debunk if has_labels else True
            no_keyword_debunk = score.get("debunk_count", 0) == 0

            if has_any_support and no_label_debunk and no_keyword_debunk:
                logger.info("[Orchestrator] TRUE 救援门：有支持证据/标签且无辟谣，提升为部分属实")
                support_sources = [
                    f"{e.source}({e.authority_score:.0%})" for e in genuine_supports[:3]
                ]
                diag.add(
                    DecisionPoint(
                        stage="true_rescue_gate",
                        verdict_before=verification.verdict.value,
                        verdict_after=Verdict.PARTLY_TRUE.value,
                        fired=True,
                        detail=(
                            f"label_support={rel['direct_support']}, "
                            f"high_auth_supports=[{', '.join(support_sources)}]; "
                            f"label_debunk=0; "
                            f"keyword_debunk={score.get('debunk_count', 0)}"
                        ),
                        signals={
                            "label_direct_support": rel["direct_support"],
                            "genuine_support_count": len(genuine_supports),
                            "debunk_count": score.get("debunk_count", 0),
                        },
                    )
                )
                verification = ClaimVerification(
                    claim=claim,
                    verdict=Verdict.PARTLY_TRUE,
                    confidence=0.55,
                    evidence_chain=verification.evidence_chain,
                    reasoning=(
                        f"{verification.reasoning}\n"
                        f"[TRUE 救援门] 存在支持证据/标签且无辟谣信号，保守提升为部分属实。"
                    ),
                    evidence_relations=verification.evidence_relations,
                )
            else:
                diag.add(
                    DecisionPoint(
                        stage="true_rescue_gate",
                        verdict_before=verification.verdict.value,
                        verdict_after=verification.verdict.value,
                        fired=False,
                        detail=(
                            f"条件不满足: any_support={has_any_support}, "
                            f"no_label_debunk={no_label_debunk}, "
                            f"no_keyword_debunk={no_keyword_debunk}"
                        ),
                        signals={
                            "label_direct_support": rel["direct_support"],
                            "label_direct_debunk": rel["direct_debunk"],
                            "label_indirect_contradict": rel["indirect_contradict"],
                            "genuine_support_count": len(genuine_supports),
                            "debunk_count": score.get("debunk_count", 0),
                        },
                    )
                )
        else:
            if not self.use_gates:
                skip_detail = "use_gates=False，跳过"
            elif skeptic_revised:
                skip_detail = "Skeptic 已修正 verdict，不允许救援反弹"
            else:
                skip_detail = (
                    f"verdict={verification.verdict.value} 不在 FALSE/MOSTLY_FALSE 中，跳过"
                )
            diag.add(
                DecisionPoint(
                    stage="true_rescue_gate",
                    verdict_before=verification.verdict.value,
                    verdict_after=verification.verdict.value,
                    fired=False,
                    detail=skip_detail,
                    signals={"skeptic_revised": skeptic_revised},
                )
            )

        # D2 保守门（TRUE 救援门之后跑）
        # 判为谣言/基本不实/误导但无辟谣或矛盾证据/标签 → 降级为无法核实
        # 即使 Skeptic 已修正过 verdict，D2 仍然适用
        d2_condition = (
            self.use_gates
            and verification.verdict in (Verdict.FALSE, Verdict.MOSTLY_FALSE, Verdict.MISLEADING)
            and not has_label_debunk
            and not has_keyword_debunk
        )
        if d2_condition:
            logger.info("[Orchestrator] 保守门：LLM 判谣言但无辟谣证据/标签，降级为无法核实")
            diag.add(
                DecisionPoint(
                    stage="d2_gate",
                    verdict_before=verification.verdict.value,
                    verdict_after=Verdict.UNVERIFIABLE.value,
                    fired=True,
                    detail=(
                        "LLM 判谣言但无关系标签(直接辟谣/间接矛盾)且无关键词辟谣信号，"
                        "降级为无法核实"
                    ),
                    signals={
                        "label_direct_debunk": rel["direct_debunk"],
                        "label_indirect_contradict": rel["indirect_contradict"],
                        "debunk_count": score.get("debunk_count", 0),
                        "signal": score.get("signal"),
                    },
                )
            )
            verification = ClaimVerification(
                claim=claim,
                verdict=Verdict.UNVERIFIABLE,
                confidence=0.40,
                evidence_chain=verification.evidence_chain,
                reasoning=(
                    f"{verification.reasoning}\n"
                    f"[保守门] 未找到辟谣证据支撑「谣言」判定，保守降级为无法核实。"
                ),
                evidence_relations=verification.evidence_relations,
            )
        else:
            diag.add(
                DecisionPoint(
                    stage="d2_gate",
                    verdict_before=verification.verdict.value,
                    verdict_after=verification.verdict.value,
                    fired=False,
                    detail=(
                        f"条件不满足: verdict={verification.verdict.value}, "
                        f"label_debunk={has_label_debunk}, keyword_debunk={has_keyword_debunk}"
                    ),
                    signals={
                        "label_direct_debunk": rel["direct_debunk"],
                        "label_indirect_contradict": rel["indirect_contradict"],
                        "debunk_count": score.get("debunk_count", 0),
                        "signal": score.get("signal"),
                    },
                )
            )

        return verification

    def _verify_single_atom(
        self, claim: Claim, scenario_info: dict | None, original_message: str = ""
    ) -> ClaimVerification:
        """对单个原子事实执行搜索→核查流程（不做原子化、不查记忆）。"""
        hunter = EvidenceHunterAgent(self.searcher)

        if self.use_advanced:
            query_plan = self._timed(
                "QueryPlanner", f"原子规划「{claim.text[:15]}」", self.query_planner.plan, claim
            )
            hunter.planned_queries = query_plan.queries
            hunter.scenario_context = scenario_info
        else:
            hunter.planned_queries = []
            hunter.scenario_context = None

        if self.use_search:
            evidence, _ = self._timed(
                "EvidenceHunter", f"原子搜索「{claim.text[:15]}」", hunter.hunt, claim
            )
        else:
            evidence = []

        if self.use_advanced and evidence:
            ranking = self._timed(
                "EvidenceRanker",
                f"原子排序「{claim.text[:15]}」",
                self.evidence_ranker.rank,
                claim,
                evidence,
                self._credibility_map or None,
            )
            if ranking.ranked_evidence:
                evidence = ranking.ranked_evidence

        # 规则层
        rule_text = _normalize_text(claim.text)
        if evidence:
            score = prescore_evidence(evidence, claim_text=rule_text)
        else:
            score = {
                "debunk_count": 0,
                "authority_debunk_count": 0,
                "signal": "no_evidence",
                "debunk_snippets": [],
            }
        score["_claim_text"] = rule_text
        score["screenshot_claim"] = None
        score["obsolete_policy"] = detect_obsolete_policy(rule_text)
        score["unverified_official_claim"] = detect_unverified_official_claim(
            rule_text, evidence or []
        )
        score["financial_scam"] = detect_financial_scam(rule_text)
        score["miracle_cure"] = detect_miracle_cure(rule_text)
        score["fake_subsidy_scam"] = detect_fake_subsidy_scam(rule_text)
        score["general_scam"] = detect_general_scam(rule_text)
        score["local_panic"] = detect_local_panic(rule_text)
        score["food_incompatibility"] = detect_food_incompatibility(rule_text)
        score["ai_celebrity_quote"] = detect_ai_celebrity_quote(rule_text, evidence or [])
        score["food_safety_myth"] = detect_food_safety_myth(rule_text)
        score["livelihood_policy_rumor"] = detect_livelihood_policy_rumor(rule_text, evidence or [])
        score["ai_generated_content"] = detect_ai_generated_content(rule_text)
        score["causal_fallacy"] = detect_causal_fallacy(rule_text)
        score["stale_evidence"] = detect_stale_evidence(rule_text, evidence or [])
        score["suppress_stale_rule"] = False

        rule_result = rule_based_verdict(score)
        if rule_result:
            return ClaimVerification(
                claim=claim,
                verdict=rule_result["verdict"],
                confidence=rule_result["confidence"],
                evidence_chain=evidence or [],
                reasoning=rule_result["reasoning"],
            )

        if not evidence:
            return ClaimVerification(
                claim=claim,
                verdict=Verdict.UNVERIFIABLE,
                confidence=0.2,
                evidence_chain=[],
                reasoning="原子搜索未找到证据",
            )

        verification = self._timed(
            "FactChecker",
            f"原子核查「{claim.text[:15]}」",
            self.checker.check,
            claim,
            evidence,
            score,
        )
        return verification

    def run(self, message: str, context: str = "") -> VerifyResponse:
        """执行完整核查流程。"""
        self.trace = VerifyTrace()
        self._current_diagnostics = []
        t_start = time.perf_counter()
        logger.info("=" * 60)
        logger.info("开始核查：%s", message[:50])
        logger.info("=" * 60)

        # Step 0: 场景路由 + MessageFrame 构建（高级模式）
        # INV-1：MessageFrame 是下游 Agent 必须 consume 的对象
        scenario_info = None
        message_frame = None
        promo_health_result: dict | None = None
        if self.use_advanced:
            scenario_info = self._timed(
                "ScenarioRouter",
                "场景路由 + MessageFrame",
                self.scenario_router.route_with_frame,
                message,
            )
            message_frame = scenario_info.get("message_frame")
            logger.info(
                "[Orchestrator] 场景：%s（%.0f%%）type=%s entity=%s",
                scenario_info["scenario"],
                scenario_info["confidence"] * 100,
                message_frame.message_type.value if message_frame else "—",
                (message_frame.promoted_entity if message_frame else "") or "—",
            )
            # Step 0.2: PromoHealthVerifier（零 LLM 规则）
            # 对 health_product_promo 类型立即跑 FTC 阈值 + BurdenOfProof
            if message_frame is not None:
                promo_health_result = _verify_promo_health(message, message_frame)
                if promo_health_result.get("applied"):
                    logger.info(
                        "[PromoHealth] risk=%s lean=%s ftc_flags=%d missing=%d",
                        promo_health_result["risk_level"],
                        promo_health_result.get("verdict_lean"),
                        len(promo_health_result["ftc"].get("flags_triggered", [])),
                        len(promo_health_result["burden"].get("missing_anchors", [])),
                    )

        # Step 0.4: 沉默策略（C8 契约）——5 类不该核查的输入直接走特定模板
        # 优先级最高，必须在 personal_content 之前（沉默更具体）
        silence_hit = detect_silence_zone(message)
        if silence_hit:
            logger.info(
                "[Orchestrator] 沉默策略命中(%s)，跳过整个核查流水线",
                silence_hit["category"],
            )
            total_ms = int((time.perf_counter() - t_start) * 1000)
            return VerifyResponse(
                original_message=message,
                claims=[],
                overall_verdict=Verdict.UNVERIFIABLE,
                summary=(f"[沉默策略·{silence_hit['category']}] {silence_hit['template']}"),
                friendly_reply=silence_hit["template"],
                evidence_sources=[],
                trace=InvestigationTrace(
                    steps=[
                        TraceStep(
                            agent=s.agent,
                            action=s.action,
                            duration_ms=s.duration_ms,
                            input_summary=s.input_summary,
                            output_summary=s.output_summary,
                            output_data=s.output_data,
                            human_narrative=s.human_narrative,
                            display=s.display,
                        )
                        for s in self.trace.steps
                    ],
                    total_duration_ms=total_ms,
                    total_llm_calls=self.trace.total_llm_calls,
                    scenario=scenario_info["scenario"] if scenario_info else "",
                    scenario_confidence=scenario_info["confidence"] if scenario_info else 0.0,
                ),
            )

        # Step 0.5: 个人内容预过滤（零 LLM，最高优先级）
        # 在消耗任何 LLM 调用前，检测纯个人经历/观点/情绪吐槽
        personal_hit = detect_personal_content(message)
        if personal_hit:
            logger.info("[Orchestrator] 个人内容预过滤命中，跳过整个核查流水线")
            total_ms = int((time.perf_counter() - t_start) * 1000)
            return VerifyResponse(
                original_message=message,
                claims=[],
                overall_verdict=Verdict.UNVERIFIABLE,
                summary=(
                    "这条消息是个人经历分享或观点表达，不包含可通过公开渠道核查的事实声明。"
                    "TruthNote 专注于可核查的事实信息，对此类内容不做真假判定。"
                ),
                friendly_reply=(
                    "这条消息主要是个人感受或经历分享，不涉及需要核查的事实信息。"
                    "每个人都有表达观点的权利，不必较真哦～"
                ),
                evidence_sources=[],
                trace=InvestigationTrace(
                    steps=[
                        TraceStep(
                            agent=s.agent,
                            action=s.action,
                            duration_ms=s.duration_ms,
                            input_summary=s.input_summary,
                            output_summary=s.output_summary,
                            output_data=s.output_data,
                            human_narrative=s.human_narrative,
                            display=s.display,
                        )
                        for s in self.trace.steps
                    ],
                    total_duration_ms=total_ms,
                    total_llm_calls=self.trace.total_llm_calls,
                    scenario=scenario_info["scenario"] if scenario_info else "",
                    scenario_confidence=scenario_info["confidence"] if scenario_info else 0.0,
                ),
            )

        # Step 1: 提取声明
        try:
            claims = self._timed(
                "ClaimExtractor", "提取声明", self.extractor.extract, message, context
            )
        except Exception as e:
            logger.error("[ClaimExtractor] 提取失败：%s", e)
            claims = []

        # INV-1 注入：health_product_promo 类型且 central_action_claim 未被抽到时
        # 强制注入中心行动主张作为优先级最高的 claim
        if (
            message_frame is not None
            and message_frame.message_type == MessageType.HEALTH_PRODUCT_PROMO
            and message_frame.central_action_claim
        ):
            action_text = message_frame.central_action_claim
            already_present = any(
                action_text in c.text or c.text in action_text for c in (claims or [])
            )
            if not already_present:
                logger.info(
                    "[INV-1 注入] ClaimExtractor 漏抽 central_action_claim，补入：%s",
                    action_text[:50],
                )
                central_claim = Claim(
                    text=action_text,
                    category=CATEGORY_MAP.get("健康养生", RumorCategory.HEALTH),
                    original_context=message[:200],
                    is_central_action=True,
                )
                claims = [central_claim] + (claims or [])
            else:
                # 标记已抽到的为中心 claim
                for c in claims or []:
                    if action_text in c.text or c.text in action_text:
                        c.is_central_action = True
                        break

        # Step 1.5: 核查价值过滤
        # INV-1 豁免：is_central_action=True 的 claim 必须保留，不能被 CheckWorthy LLM 过滤
        # （否则 LLM 可能把"购买恒晴药业+双色片"误判为个人经历从而过滤掉，
        # 让 INV-1 强制注入失效）
        if claims:
            raw_count = len(claims)
            central_claims = [c for c in claims if c.is_central_action]
            try:
                claims = self._timed("CheckWorthy", "核查价值过滤", self.checkworthy.filter, claims)
            except Exception as e:
                logger.warning("[CheckWorthy] 过滤失败，保留全部声明：%s", e)
            # INV-1 豁免：被过滤掉的 central_action_claim 必须补回，且排在最前
            for cc in central_claims:
                if not any(
                    cc.text == c.text or cc.text in c.text or c.text in cc.text for c in claims
                ):
                    logger.warning(
                        "[INV-1 豁免] CheckWorthy 把 central_action_claim 过滤掉了，强制补回：%s",
                        cc.text[:40],
                    )
                    claims = [cc] + claims
            logger.info(
                "[CheckWorthy] %d → %d 条声明通过核查价值过滤（含 %d 条 central_action 豁免）",
                raw_count,
                len(claims),
                len(central_claims),
            )

        if not claims:
            logger.info("未提取到可核查声明，尝试纯文本规则引擎兜底")
            # 先检查个人内容（优先于其他规则）
            personal_fallback = detect_personal_content(message)
            if personal_fallback:
                logger.info("[Orchestrator] 无声明 + 个人内容命中，直接返回 UNVERIFIABLE")
                total_ms = int((time.perf_counter() - t_start) * 1000)
                return VerifyResponse(
                    original_message=message,
                    claims=[],
                    overall_verdict=Verdict.UNVERIFIABLE,
                    summary=(
                        "这条消息是个人经历分享或观点表达，不包含可通过公开渠道核查的事实声明。"
                        "TruthNote 专注于可核查的事实信息，对此类内容不做真假判定。"
                    ),
                    friendly_reply=(
                        "这条消息主要是个人感受或经历分享，不涉及需要核查的事实信息。"
                        "每个人都有表达观点的权利，不必较真哦～"
                    ),
                    evidence_sources=[],
                    trace=InvestigationTrace(
                        steps=[
                            TraceStep(
                                agent=s.agent,
                                action=s.action,
                                duration_ms=s.duration_ms,
                                input_summary=s.input_summary,
                                output_summary=s.output_summary,
                                output_data=s.output_data,
                            )
                            for s in self.trace.steps
                        ],
                        total_duration_ms=total_ms,
                        total_llm_calls=self.trace.total_llm_calls,
                        scenario=scenario_info["scenario"] if scenario_info else "",
                        scenario_confidence=scenario_info["confidence"] if scenario_info else 0.0,
                    ),
                )
            # ClaimExtractor 失败时用原文直接跑规则引擎
            fallback_text = _normalize_text(message)
            fallback_score = {
                "debunk_count": 0,
                "authority_debunk_count": 0,
                "support_count": 0,
                "neutral_count": 0,
                "signal": "no_evidence",
                "debunk_snippets": [],
                "_claim_text": fallback_text,
                "personal_content": detect_personal_content(fallback_text),
                "screenshot_claim": detect_screenshot_claim(fallback_text, []),
                "obsolete_policy": detect_obsolete_policy(fallback_text),
                "unverified_official_claim": detect_unverified_official_claim(fallback_text, []),
                "financial_scam": detect_financial_scam(fallback_text),
                "miracle_cure": detect_miracle_cure(fallback_text),
                "fake_subsidy_scam": detect_fake_subsidy_scam(fallback_text),
                "general_scam": detect_general_scam(fallback_text),
                "local_panic": detect_local_panic(fallback_text),
                "food_incompatibility": detect_food_incompatibility(fallback_text),
                "ai_celebrity_quote": detect_ai_celebrity_quote(fallback_text, []),
                "food_safety_myth": detect_food_safety_myth(fallback_text),
                "livelihood_policy_rumor": detect_livelihood_policy_rumor(fallback_text, []),
                "ai_generated_content": detect_ai_generated_content(fallback_text),
                "causal_fallacy": detect_causal_fallacy(fallback_text),
                "stale_evidence": None,
                "suppress_stale_rule": False,
            }
            fallback_rule = rule_based_verdict(fallback_score)
            if fallback_rule:
                logger.info(
                    "[Orchestrator] ClaimExtractor 失败但规则兜底命中: %s",
                    fallback_rule["verdict"].value,
                )
                fallback_verdict = fallback_rule["verdict"]
                fallback_summary = fallback_rule["reasoning"]
                fallback_reply = "这条消息被规则引擎识别为可疑信息，建议谨慎对待。"
            else:
                fallback_verdict = Verdict.UNVERIFIABLE
                fallback_summary = (
                    "这条消息不包含可通过公开渠道核查的事实声明。"
                    "它可能是个人经历分享、观点表达、情绪吐槽或不可验证的私人信息。"
                    "TruthNote 专注于可核查的事实信息，对此类内容不做真假判定。"
                )
                fallback_reply = (
                    "这条消息主要是个人经历或观点分享，不涉及可以核查的具体事实。"
                    "TruthNote 不对个人观点做真假判定——我们只核查能用公开证据验证的事实声明。"
                )
            total_ms = int((time.perf_counter() - t_start) * 1000)
            _fb_proj = project_to_binary(fallback_verdict)
            _fb_badge, _fb_bucket, _fb_subtype = _fb_proj if _fb_proj else (None, None, "")
            return VerifyResponse(
                original_message=message,
                claims=[],
                overall_verdict=fallback_verdict,
                binary_badge=_fb_badge,
                display_bucket=_fb_bucket,
                display_subtype=_fb_subtype,
                summary=fallback_summary,
                friendly_reply=fallback_reply,
                evidence_sources=[],
                trace=InvestigationTrace(
                    steps=[
                        TraceStep(
                            agent=s.agent,
                            action=s.action,
                            duration_ms=s.duration_ms,
                            input_summary=s.input_summary,
                            output_summary=s.output_summary,
                            output_data=s.output_data,
                            human_narrative=s.human_narrative,
                            display=s.display,
                        )
                        for s in self.trace.steps
                    ],
                    total_duration_ms=total_ms,
                    total_llm_calls=self.trace.total_llm_calls,
                    scenario=scenario_info["scenario"] if scenario_info else "",
                    scenario_confidence=scenario_info["confidence"] if scenario_info else 0.0,
                ),
            )

        # Step 2 & 3: 对每条声明搜索→排序→核查→质疑
        if len(claims) == 1:
            try:
                verifications = [self._verify_single_claim(claims[0], scenario_info, message)]
            except Exception as e:
                logger.exception("[Orchestrator] 单 claim 核查异常：%s", claims[0].text[:30])
                verifications = [
                    ClaimVerification(
                        claim=claims[0],
                        verdict=Verdict.UNVERIFIABLE,
                        confidence=0.1,
                        evidence_chain=[],
                        reasoning=f"核查流程异常：{type(e).__name__}",
                    )
                ]
        else:
            # 多 claim 并行处理
            verifications: list[ClaimVerification] = []
            with ThreadPoolExecutor(max_workers=min(len(claims), 3)) as pool:
                futures = [
                    (claim, pool.submit(self._verify_single_claim, claim, scenario_info, message))
                    for claim in claims
                ]
                for claim, fut in futures:
                    try:
                        verifications.append(fut.result())
                    except Exception as e:
                        logger.exception("[Orchestrator] claim 核查异常：%s", claim.text[:30])
                        verifications.append(
                            ClaimVerification(
                                claim=claim,
                                verdict=Verdict.UNVERIFIABLE,
                                confidence=0.1,
                                evidence_chain=[],
                                reasoning=f"核查流程异常：{type(e).__name__}",
                            )
                        )

        # Step 4: 生成回复
        friendly_reply, summary = self._timed(
            "ResponseComposer",
            "生成回复",
            self.composer.compose,
            message,
            verifications,
        )

        overall = _pick_overall_verdict(verifications)

        # Step 4.4: PromoHealth 强制升级（INV-2 落地）
        # 当 PromoHealth 高风险 + 当前 overall 仍保守时，按 verdict_lean 强制升级。
        # 不下调（即不会把 FALSE 降为 MOSTLY_FALSE）；只把过于保守的判定推到更准确的位置。
        if promo_health_result and promo_health_result.get("applied"):
            ph_lean = promo_health_result.get("verdict_lean")
            ph_risk = promo_health_result.get("risk_level")
            ph_lean_verdict = VERDICT_MAP.get(ph_lean) if ph_lean else None
            if ph_lean_verdict and ph_risk == "high":
                # 严重度序：TRUE < PARTLY_TRUE < UNVERIFIABLE < MISLEADING < MOSTLY_FALSE < FALSE
                severity = [
                    Verdict.TRUE,
                    Verdict.PARTLY_TRUE,
                    Verdict.UNVERIFIABLE,
                    Verdict.MISLEADING,
                    Verdict.MOSTLY_FALSE,
                    Verdict.FALSE,
                ]
                cur_idx = severity.index(overall) if overall in severity else 2
                new_idx = severity.index(ph_lean_verdict) if ph_lean_verdict in severity else 2
                if new_idx > cur_idx:
                    logger.warning(
                        "[PromoHealth 升级] %s → %s（risk=%s, %s）",
                        overall.value,
                        ph_lean_verdict.value,
                        ph_risk,
                        promo_health_result.get("reasoning", "")[:80],
                    )
                    overall = ph_lean_verdict

        # Step 4.5: 非裁定道 —— 取代旧 developing/insufficient 干巴模板
        from .attribution import build_non_adjudicated, is_checkable_announcement

        # ── 单条声明级归因（缺口1 修复）──
        # 旧逻辑只在 overall==UNVERIFIABLE 时给整体一份归因；混合判定里夹的
        # 单条 UNVERIFIABLE 声明仍走机械话。这里改成：遍历所有 UNVERIFIABLE 声明，
        # 各自用本条证据链生成归因，挂到 cv.unverifiable_reason（统一契约），
        # 让前端能逐条渲染"卡在哪/去哪查"。绝不产 verdict、不改任何打分（守 INV-4）。
        # 缓存同文本归因，避免一条声明被多处重复算 LLM。
        _na_cache: dict[str, object] = {}

        def _attribution_for(cv) -> object:
            key = cv.claim.text
            cached = _na_cache.get(key)
            if cached is not None:
                return cached
            na = build_non_adjudicated(cv.claim.text, cv.evidence_chain)
            _na_cache[key] = na
            return na

        non_adjudicated = None
        if verifications:
            for cv in verifications:
                if cv.verdict != Verdict.UNVERIFIABLE:
                    continue
                cv_na = _attribution_for(cv)
                cv.unverifiable_reason = _serialize_unverifiable_reason(cv_na, cv.claim.text)

        # ── 整体非裁定道（保持原行为：仅 overall==UNVERIFIABLE 时改写 reply/summary）──
        if overall == Verdict.UNVERIFIABLE and verifications:
            uv = next((v for v in verifications if v.verdict == Verdict.UNVERIFIABLE), None)
            if uv:
                if is_checkable_announcement(uv.claim.text):
                    logger.warning(
                        "[非裁定] 可查官方通知类却停在 UNVERIFIABLE，建议上游 future 闸放行：%s",
                        uv.claim.text[:40],
                    )
                # 复用上面逐条已算好的归因，避免重复 LLM 调用
                non_adjudicated = _attribution_for(uv)
                friendly_reply = (
                    "【未给出真假判定：缺少公开核查所需条件】\n"
                    f"原因（{non_adjudicated.primary_blocker.value}）："
                    f"{non_adjudicated.claim_specific_detail}\n"
                    f"该去哪确认：{non_adjudicated.verify_where}"
                )
                summary = (
                    f"{summary}\n[非裁定·{non_adjudicated.primary_blocker.value}] "
                    f"{non_adjudicated.claim_specific_detail}"
                )
                if self._current_diagnostics:
                    self._current_diagnostics[-1].evidence_summary["primary_blocker"] = (
                        non_adjudicated.primary_blocker.value
                    )

        sources = []
        search_engines: set[str] = set()
        for v in verifications:
            for e in v.evidence_chain:
                if e.url and e.url not in sources:
                    sources.append(e.url)
                if e.source_tag:
                    search_engines.add(e.source_tag)
                elif "tavily" in (e.source or "").lower():
                    search_engines.add("Tavily")
        if not search_engines:
            # 根据配置标注
            from .config import settings as _cfg

            prov = _cfg.search_provider.lower()
            if prov in ("qihoo360", "so360"):
                search_engines.add("360搜索")
            elif prov == "tavily":
                search_engines.add("Tavily")

        total_ms = int((time.perf_counter() - t_start) * 1000)
        logger.info("=" * 60)
        logger.info(
            "核查完成：%d 条声明，总判定 %s，耗时 %dms，%d 次 LLM 调用",
            len(verifications),
            overall.value,
            total_ms,
            self.trace.total_llm_calls,
        )
        logger.info("=" * 60)

        agents_for_tokens = {
            "ClaimExtractor": self.extractor,
            "CheckWorthy": self.checkworthy,
            "CommonsenseChecker": self.commonsense_checker,
            "AtomicFact": self.atomizer,
            "FactChecker": self.checker,
            "ResponseComposer": self.composer,
        }
        if self.use_advanced:
            agents_for_tokens.update(
                {
                    "ScenarioRouter": self.scenario_router,
                    "QueryPlanner": self.query_planner,
                    "EvidenceRanker": self.evidence_ranker,
                    "Skeptic": self.skeptic,
                }
            )
        tokens_by_agent = {
            name: agent.get_token_usage() for name, agent in agents_for_tokens.items()
        }
        total_tokens = sum(tokens_by_agent.values())

        # INV-2 · CoverageAuditor：检查 MessageFrame 必查清单是否覆盖
        coverage_audit = None
        if message_frame is not None:
            coverage_audit = _audit_coverage(message_frame, claims or [], verifications or [])

        # 6 维度评估 + VerdictDistribution（演示输出契约）
        dimensions_out = assess_dimensions(
            frame=message_frame, promo=promo_health_result, verifications=verifications
        )
        pipeline_verdict_for_dist = overall
        pipeline_conf_for_dist = (
            max((v.confidence for v in verifications), default=0.5) if verifications else 0.5
        )
        verdict_dist = aggregate_distribution(
            dimensions_out,
            pipeline_verdict=pipeline_verdict_for_dist,
            pipeline_confidence=pipeline_conf_for_dist,
            verifications=verifications,  # 阶段2 证实/硬证伪后处理（从证据链推导）
        )

        # 综合人话解释（演示输出）
        verdict_explanation = ""
        try:
            verdict_explanation = compose_verdict_explanation(
                overall_verdict=overall,
                distribution=verdict_dist,
                dimensions=dimensions_out,
                frame=message_frame,
                promo=promo_health_result,
            )
        except Exception:  # noqa: BLE001
            logger.warning("[Orchestrator] compose_verdict_explanation 失败", exc_info=True)
            verdict_explanation = ""

        investigation_trace = InvestigationTrace(
            steps=[
                TraceStep(
                    agent=s.agent,
                    action=s.action,
                    duration_ms=s.duration_ms,
                    input_summary=s.input_summary,
                    output_summary=s.output_summary,
                    output_data=s.output_data,
                    human_narrative=s.human_narrative,
                    display=s.display,
                )
                for s in self.trace.steps
            ],
            total_duration_ms=total_ms,
            total_llm_calls=self.trace.total_llm_calls,
            total_tokens_used=total_tokens,
            tokens_by_agent=tokens_by_agent,
            claims_extracted=len(claims) if claims else 0,
            claims_checkworthy=len(verifications),
            scenario=scenario_info["scenario"] if scenario_info else "",
            scenario_confidence=scenario_info["confidence"] if scenario_info else 0.0,
            strategy_hint=scenario_info.get("strategy_hint", "") if scenario_info else "",
            diagnostics=self._current_diagnostics,
            message_frame=message_frame,
            coverage_audit=coverage_audit,
            promo_health=promo_health_result,
            dimensions=dimensions_out,
            verdict_distribution=verdict_dist,
            verdict_explanation=verdict_explanation,
        )

        _proj = project_to_binary(overall)
        _badge, _bucket, _subtype = _proj if _proj else (None, None, "")
        response = VerifyResponse(
            original_message=message,
            claims=verifications,
            overall_verdict=overall,
            summary=summary,
            friendly_reply=friendly_reply,
            evidence_sources=sources,
            search_engines_used=sorted(search_engines),
            trace=investigation_trace,
            binary_badge=_badge,
            display_bucket=_bucket,
            display_subtype=_subtype,
            non_adjudicated=non_adjudicated,
        )

        # 闭环动作层：生成可执行动作 + ClaimReview JSON-LD
        try:
            actions = generate_actions(response)
            response.actions = [a.model_dump() for a in actions]
            response.claimreviews = response_to_claimreviews(response)
        except Exception:
            logger.warning("[Orchestrator] 闭环动作生成失败", exc_info=True)

        return response

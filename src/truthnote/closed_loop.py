"""闭环动作层：从核查结果生成可执行动作 + 持久化。

评委期望的闭环：谣言出现 → 判定 → **用户拿到可执行的动作**。
这个模块补齐最后一步。
"""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, Field

from .config import settings
from .schemas import ClaimVerification, Verdict, VerifyResponse

logger = logging.getLogger(__name__)


class RiskType(StrEnum):
    SCAM = "scam"
    HEALTH_MISINFORMATION = "health_misinformation"
    FAKE_POLICY = "fake_policy"
    PANIC_CHAIN = "panic_chain"
    OLD_NEWS = "old_news"
    AI_FAKE = "ai_fake"
    FINANCIAL_FRAUD = "financial_fraud"
    GENERAL = "general"


class ActionType(StrEnum):
    WARN_USER = "warn_user"
    SHARE_CORRECTION = "share_correction"
    REPORT_SCAM = "report_scam"
    SUBSCRIBE_BACKFILL = "subscribe_backfill"  # 订阅回填：还查不到定论的事，权威结论出来后通知用户
    QUEUE_HUMAN_REVIEW = "queue_human_review"  # 保留向后兼容，不再作为 UNVERIFIABLE 的默认动作
    EXPORT_EVIDENCE = "export_evidence"
    NO_ACTION = "no_action"


class ActionStatus(StrEnum):
    OPEN = "open"
    SENT = "sent"
    RESOLVED = "resolved"
    DISMISSED = "dismissed"


class ClosedLoopAction(BaseModel):
    action_id: str = ""
    claim_text: str = ""
    verdict: Verdict = Verdict.UNVERIFIABLE
    confidence: float = 0.0
    risk_type: RiskType = RiskType.GENERAL
    recommended_action: ActionType = ActionType.WARN_USER
    evidence_urls: list[str] = Field(default_factory=list)
    correction_card: str = Field(default="", description="可分享的纠正卡文本")
    report_links: list[dict] = Field(default_factory=list, description="举报渠道链接")
    official_channels: list[dict] = Field(default_factory=list, description="官方核实渠道")
    subscription: dict = Field(
        default_factory=dict,
        description="订阅回填配置（仅 SUBSCRIBE_BACKFILL 非空）：topic/watch_sources/channel/note",
    )
    status: ActionStatus = ActionStatus.OPEN
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())


class DispositionReceipt(BaseModel):
    """处置回执：用户对某个闭环动作做了处置（发送/已处置/忽略）后的可渲染确认凭证。

    这是"把结果真正给用户用起来"的可见证据——用户点了「举报」「分享纠正卡」「已处置」之后，
    系统回一张带受理编号 + 时间 + 下一步的回执卡，闭环在用户侧"看得见地"合上。
    """

    receipt_id: str = Field(default="", description="受理编号 TN-RC-xxxxxxxx，同一处置稳定可复现")
    action_id: str = ""
    status: ActionStatus = ActionStatus.OPEN
    disposition_label: str = Field(default="", description="处置动作的人话标签，如「已发送纠正卡」")
    message: str = Field(default="", description="人话回执正文")
    next_step: str = Field(default="", description="下一步引导")
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())


_REPORT_LINKS: dict[RiskType, list[dict]] = {
    RiskType.SCAM: [
        {"name": "国家反诈中心", "url": "https://www.12321.cn/", "description": "电信网络诈骗举报"},
        {
            "name": "12321 网络不良信息举报",
            "url": "https://www.12321.cn/",
            "description": "网络不良信息举报中心",
        },
        {"name": "110 报警", "url": "tel:110", "description": "涉及资金损失请立即报警"},
    ],
    RiskType.HEALTH_MISINFORMATION: [
        {
            "name": "中国互联网联合辟谣平台",
            "url": "https://www.piyao.org.cn/",
            "description": "国家网信办主管",
        },
        {"name": "卫健委 12320 热线", "url": "tel:12320", "description": "卫生健康咨询投诉"},
    ],
    RiskType.FAKE_POLICY: [
        {"name": "中国政府网", "url": "https://www.gov.cn/", "description": "查询权威政策原文"},
        {
            "name": "中国互联网联合辟谣平台",
            "url": "https://www.piyao.org.cn/",
            "description": "政策类谣言举报",
        },
    ],
    RiskType.PANIC_CHAIN: [
        {
            "name": "中国互联网联合辟谣平台",
            "url": "https://www.piyao.org.cn/",
            "description": "恐慌谣言举报",
        },
        {"name": "当地公安 110", "url": "tel:110", "description": "紧急情况报警"},
    ],
    RiskType.FINANCIAL_FRAUD: [
        {"name": "证监会 12386 热线", "url": "tel:12386", "description": "证券期货投诉举报"},
        {"name": "银保监 12378 热线", "url": "tel:12378", "description": "银行保险消费投诉"},
        {"name": "国家反诈中心", "url": "https://www.12321.cn/", "description": "金融诈骗举报"},
    ],
    RiskType.AI_FAKE: [
        {
            "name": "中国互联网联合辟谣平台",
            "url": "https://www.piyao.org.cn/",
            "description": "AI伪造内容举报",
        },
    ],
    RiskType.OLD_NEWS: [
        {
            "name": "中国互联网联合辟谣平台",
            "url": "https://www.piyao.org.cn/",
            "description": "旧闻翻炒举报",
        },
    ],
    RiskType.GENERAL: [
        {
            "name": "中国互联网联合辟谣平台",
            "url": "https://www.piyao.org.cn/",
            "description": "综合谣言举报",
        },
    ],
}

_OFFICIAL_CHANNELS: dict[RiskType, list[dict]] = {
    RiskType.SCAM: [
        {"name": "国家反诈中心 App", "description": "下载安装可拦截诈骗电话/短信"},
        {"name": "微信/支付宝安全中心", "description": "账号被盗或资金异常时使用"},
    ],
    RiskType.HEALTH_MISINFORMATION: [
        {"name": "国家卫健委", "url": "https://www.nhc.gov.cn/", "description": "权威健康信息"},
        {"name": "WHO 世卫组织", "url": "https://www.who.int/zh", "description": "国际卫生权威"},
        {"name": "当地三甲医院官网", "description": "就医请选择正规医疗机构"},
    ],
    RiskType.FAKE_POLICY: [
        {"name": "中国政府网", "url": "https://www.gov.cn/", "description": "国务院政策文件查询"},
        {
            "name": "国家税务总局",
            "url": "https://www.chinatax.gov.cn/",
            "description": "税务政策查询",
        },
        {"name": "人社部", "url": "https://www.mohrss.gov.cn/", "description": "社保/就业政策查询"},
    ],
    RiskType.PANIC_CHAIN: [
        {
            "name": "中国地震台网",
            "url": "https://www.ceic.ac.cn/",
            "description": "地震信息权威发布",
        },
        {"name": "应急管理部", "url": "https://www.mem.gov.cn/", "description": "灾害应急信息"},
        {"name": "当地政府官网", "description": "本地突发事件以当地政府通报为准"},
    ],
    RiskType.FINANCIAL_FRAUD: [
        {"name": "中国证监会", "url": "https://www.csrc.gov.cn/", "description": "证券监管信息"},
        {"name": "中国人民银行", "url": "https://www.pbc.gov.cn/", "description": "货币/金融政策"},
    ],
    RiskType.AI_FAKE: [
        {"name": "相关机构官方账号", "description": "名人言论以本人或所属机构官方发布为准"},
    ],
    RiskType.OLD_NEWS: [
        {
            "name": "新华社",
            "url": "https://www.xinhuanet.com/",
            "description": "权威新闻时间线核实",
        },
    ],
    RiskType.GENERAL: [
        {
            "name": "中国互联网联合辟谣平台",
            "url": "https://www.piyao.org.cn/",
            "description": "综合辟谣查询",
        },
    ],
}


def _infer_risk_type(cv: ClaimVerification) -> RiskType:
    """从 reasoning 推断风险类型。"""
    r = cv.reasoning.lower()
    if "诈骗" in r or "scam" in r:
        return RiskType.SCAM
    if "伪医疗" in r or "养生" in r or "食物相克" in r:
        return RiskType.HEALTH_MISINFORMATION
    if "官方" in r or "政策" in r or "过时政策" in r:
        return RiskType.FAKE_POLICY
    if "恐慌" in r or "地震" in r:
        return RiskType.PANIC_CHAIN
    if "旧闻" in r or "翻炒" in r:
        return RiskType.OLD_NEWS
    if "名人" in r or "ai" in r:
        return RiskType.AI_FAKE
    if "金融" in r or "荐股" in r:
        return RiskType.FINANCIAL_FRAUD
    return RiskType.GENERAL


def _recommend_action(verdict: Verdict, risk_type: RiskType) -> ActionType:
    """根据判定+风险类型推荐动作。

    UNVERIFIABLE（还查不到定论）→ 订阅回填，而不是丢进"人工复核"队列空表。
    命题人校准：把唯一真边界（时序未知/刚发生）变成体面的产品动作——
    "权威结论出来后通知你"，而不是演一张永远空着的追踪仪表盘。
    """
    if verdict == Verdict.TRUE:
        return ActionType.NO_ACTION
    if risk_type == RiskType.SCAM:
        return ActionType.REPORT_SCAM
    if verdict == Verdict.UNVERIFIABLE:
        return ActionType.SUBSCRIBE_BACKFILL
    if verdict in (Verdict.FALSE, Verdict.MOSTLY_FALSE):
        return ActionType.SHARE_CORRECTION
    return ActionType.WARN_USER


def _generate_correction_card(
    original_message: str,
    cv: ClaimVerification,
    risk_type: RiskType,
) -> str:
    """生成可分享的纠正卡文本（爸妈能看懂的版本）。"""
    verdict_label = {
        Verdict.FALSE: "经核查不实",
        Verdict.MOSTLY_FALSE: "大部分内容不实",
        Verdict.MISLEADING: "信息存在误导",
        Verdict.PARTLY_TRUE: "部分内容属实",
        Verdict.UNVERIFIABLE: "暂无法核实",
        Verdict.TRUE: "经核查属实",
    }

    risk_tip = {
        RiskType.SCAM: "这可能是诈骗信息，请勿点击链接或转账。",
        RiskType.HEALTH_MISINFORMATION: "健康问题请咨询正规医疗机构。",
        RiskType.FAKE_POLICY: "政策信息请以政府官网为准。",
        RiskType.PANIC_CHAIN: "此类恐慌消息多为编造，请勿转发。",
        RiskType.FINANCIAL_FRAUD: "投资有风险，请勿相信所谓内幕消息。",
        RiskType.AI_FAKE: "名人言论请以官方渠道为准，谨防 AI 伪造。",
        RiskType.OLD_NEWS: "这是旧闻翻炒，事件发生时间与描述不符。",
        RiskType.GENERAL: "建议多方核实后再转发。",
    }

    label = verdict_label.get(cv.verdict, "待核查")
    tip = risk_tip.get(risk_type, "建议谨慎对待。")

    sources = []
    for e in cv.evidence_chain[:3]:
        if e.url:
            sources.append(e.url)

    report_links = _REPORT_LINKS.get(risk_type, _REPORT_LINKS[RiskType.GENERAL])
    official_channels = _OFFICIAL_CHANNELS.get(risk_type, _OFFICIAL_CHANNELS[RiskType.GENERAL])

    card = "【TruthNote 核查卡】\n\n"
    card += f"原消息：{original_message[:80]}{'...' if len(original_message) > 80 else ''}\n\n"
    card += f"核查结果：{label}\n"
    card += f"提示：{tip}\n"
    if sources:
        card += "\n📎 参考来源：\n"
        for i, url in enumerate(sources, 1):
            card += f"  {i}. {url}\n"
    if official_channels:
        card += "\n✅ 官方核实渠道：\n"
        for ch in official_channels[:3]:
            url_part = f" {ch['url']}" if ch.get("url") else ""
            card += f"  · {ch['name']}{url_part} — {ch.get('description', '')}\n"
    if report_links and cv.verdict in (Verdict.FALSE, Verdict.MOSTLY_FALSE):
        card += "\n🚨 举报渠道：\n"
        for rl in report_links[:2]:
            card += f"  · {rl['name']} {rl['url']} — {rl.get('description', '')}\n"
    card += "\n—— TruthNote AI 核查引擎"
    return card


def _build_subscription(cv: ClaimVerification, risk_type: RiskType) -> dict:
    """为「还查不到定论」的声明构造订阅回填配置。

    把唯一真边界（时序未知）变成体面动作：用户订阅这条，权威结论出现后回填通知。
    watch_sources 复用该风险类型对应的官方核实渠道（订阅就盯这些权威源）。
    """
    official = _OFFICIAL_CHANNELS.get(risk_type, _OFFICIAL_CHANNELS[RiskType.GENERAL])
    watch_sources = [
        {"name": ch["name"], "url": ch.get("url", "")} for ch in official[:3] if ch.get("name")
    ]
    topic = cv.claim.text.strip()[:60] or "该事件"
    return {
        "topic": topic,
        "watch_sources": watch_sources,
        "channel": "extension_notification",  # 前端默认走插件角标/弹窗通知，可被用户改成邮箱等
        "note": "权威结论出现后回填通知你；在此之前我们不臆测真假。",
    }


def _generate_subscribe_card(
    original_message: str,
    cv: ClaimVerification,
    risk_type: RiskType,
) -> str:
    """生成订阅回填卡文本（无法核实时的体面动作，而不是冷冰冰的"信息不足"）。"""
    official_channels = _OFFICIAL_CHANNELS.get(risk_type, _OFFICIAL_CHANNELS[RiskType.GENERAL])

    card = "【TruthNote 订阅回填】\n\n"
    card += f"原消息：{original_message[:80]}{'...' if len(original_message) > 80 else ''}\n\n"
    card += "核查结果：暂无权威定论（事件可能刚发生，权威源尚未表态）\n"
    card += "我们不臆测真假——这是诚实的边界，不是失败。\n\n"
    card += "🔔 已为你订阅这条：权威结论一出现，第一时间通知你。\n"
    if official_channels:
        card += "\n👀 持续盯的权威源：\n"
        for ch in official_channels[:3]:
            url_part = f" {ch['url']}" if ch.get("url") else ""
            card += f"  · {ch['name']}{url_part}\n"
    card += "\n—— TruthNote AI 核查引擎"
    return card


def generate_actions(response: VerifyResponse) -> list[ClosedLoopAction]:
    """从 VerifyResponse 生成闭环动作列表。"""
    actions = []
    for cv in response.claims:
        risk_type = _infer_risk_type(cv)
        action_type = _recommend_action(cv.verdict, risk_type)
        card = ""
        subscription: dict = {}
        if action_type in (
            ActionType.SHARE_CORRECTION,
            ActionType.WARN_USER,
            ActionType.REPORT_SCAM,
        ):
            card = _generate_correction_card(response.original_message, cv, risk_type)
        elif action_type == ActionType.SUBSCRIBE_BACKFILL:
            card = _generate_subscribe_card(response.original_message, cv, risk_type)
            subscription = _build_subscription(cv, risk_type)
        urls = [e.url for e in cv.evidence_chain if e.url][:5]
        report_links = _REPORT_LINKS.get(risk_type, _REPORT_LINKS[RiskType.GENERAL])
        official_channels = _OFFICIAL_CHANNELS.get(risk_type, _OFFICIAL_CHANNELS[RiskType.GENERAL])
        actions.append(
            ClosedLoopAction(
                action_id=f"act_{hash(cv.claim.text) & 0xFFFFFFFF:08x}",
                claim_text=cv.claim.text,
                verdict=cv.verdict,
                confidence=cv.confidence,
                risk_type=risk_type,
                recommended_action=action_type,
                evidence_urls=urls,
                correction_card=card,
                report_links=report_links,
                official_channels=official_channels,
                subscription=subscription,
            )
        )
    return actions


_DISPOSITION_LABELS: dict[ActionStatus, str] = {
    ActionStatus.SENT: "已发送",
    ActionStatus.RESOLVED: "已处置",
    ActionStatus.DISMISSED: "已忽略",
    ActionStatus.OPEN: "待处理",
}


def build_disposition_receipt(
    action_id: str,
    status: ActionStatus,
    *,
    claim_text: str = "",
    recommended_action: ActionType | None = None,
) -> DispositionReceipt:
    """构造处置回执：用户处置某动作后，回一张可渲染的受理凭证。

    receipt_id 由 (action_id, status) 确定性派生，同一处置回执号稳定（demo 可复现）。
    回执是闭环在用户侧"看得见地"合上的那一下——不是又一个静默的状态字段。
    """
    seed = f"{action_id}|{status.value}"
    receipt_id = f"TN-RC-{hash(seed) & 0xFFFFFFFF:08x}"
    label = _DISPOSITION_LABELS.get(status, status.value)

    # 处置标签按推荐动作细化（让回执说人话）
    verb = "处置"
    if recommended_action == ActionType.SHARE_CORRECTION:
        verb = "分享纠正卡"
    elif recommended_action == ActionType.REPORT_SCAM:
        verb = "举报"
    elif recommended_action == ActionType.SUBSCRIBE_BACKFILL:
        verb = "订阅回填"
    elif recommended_action == ActionType.WARN_USER:
        verb = "提醒"
    disposition_label = f"{label}·{verb}"

    claim_part = f"「{claim_text[:40]}」" if claim_text else "这条核查"
    if status == ActionStatus.RESOLVED:
        if recommended_action == ActionType.REPORT_SCAM:
            message = f"{claim_part}的举报已记录，受理编号 {receipt_id}。感谢你完成这次辟谣闭环。"
            next_step = "如涉及资金损失，请同时拨打 110；举报进展可在历史记录里查看。"
        else:
            message = f"{claim_part}已标记为已处置，受理编号 {receipt_id}。"
            next_step = "对方若仍有疑问，可把证据链逐条转给他核对。"
    elif status == ActionStatus.SENT:
        message = f"{claim_part}的纠正卡已生成并标记为已发送，受理编号 {receipt_id}。"
        next_step = "纠正卡支持一键复制转发；可附上官方核实渠道增强说服力。"
    elif status == ActionStatus.DISMISSED:
        message = f"已忽略{claim_part}，受理编号 {receipt_id}。"
        next_step = "如改变主意，可在历史记录里重新处置这一条。"
    else:
        message = f"{claim_part}已受理，编号 {receipt_id}。"
        next_step = "选择「分享纠正卡 / 举报 / 订阅回填」中的一项继续。"

    return DispositionReceipt(
        receipt_id=receipt_id,
        action_id=action_id,
        status=status,
        disposition_label=disposition_label,
        message=message,
        next_step=next_step,
    )


class ClosedLoopStore:
    """SQLite 持久化闭环动作。"""

    def __init__(self, db_path: str | Path | None = None):
        self.db_path = Path(db_path) if db_path else Path(settings.search_cache_db)
        self._init_db()

    def _init_db(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self.db_path))
        conn.execute(
            """CREATE TABLE IF NOT EXISTS closed_loop_actions (
                action_id TEXT PRIMARY KEY,
                claim_text TEXT NOT NULL,
                verdict TEXT NOT NULL,
                confidence REAL,
                risk_type TEXT,
                recommended_action TEXT,
                evidence_urls TEXT,
                correction_card TEXT,
                status TEXT DEFAULT 'open',
                created_at TEXT NOT NULL
            )"""
        )
        conn.commit()
        conn.close()

    def save(self, action: ClosedLoopAction) -> None:
        conn = sqlite3.connect(str(self.db_path))
        conn.execute(
            """INSERT OR REPLACE INTO closed_loop_actions
            (action_id, claim_text, verdict, confidence, risk_type,
             recommended_action, evidence_urls, correction_card, status, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                action.action_id,
                action.claim_text,
                action.verdict.value,
                action.confidence,
                action.risk_type.value,
                action.recommended_action.value,
                json.dumps(action.evidence_urls, ensure_ascii=False),
                action.correction_card,
                action.status.value,
                action.created_at,
            ),
        )
        conn.commit()
        conn.close()

    def update_status(self, action_id: str, status: ActionStatus) -> None:
        conn = sqlite3.connect(str(self.db_path))
        conn.execute(
            "UPDATE closed_loop_actions SET status = ? WHERE action_id = ?",
            (status.value, action_id),
        )
        conn.commit()
        conn.close()

    def get_action(self, action_id: str) -> dict | None:
        """按 action_id 取单条动作（回执需要 claim_text + recommended_action 说人话）。"""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT * FROM closed_loop_actions WHERE action_id = ?",
            (action_id,),
        ).fetchone()
        conn.close()
        return dict(row) if row else None

    def get_open_actions(self) -> list[dict]:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM closed_loop_actions WHERE status = 'open' ORDER BY created_at DESC"
        ).fetchall()
        conn.close()
        return [dict(r) for r in rows]

"""非裁定道：primary_blocker 决策树 + 两层归因生成（判定重构 v2）。

主线落地：把本文件 cp 到 src/truthnote/attribution.py 即可（无需改名）。

设计见 docs/施工_判定重构/README.md。三条不变量：
  INV-U1 判定层对外二元、非裁定不给徽章；INV-U2 非裁定必带 blocker+非空 detail；
  INV-U3 detail 不得断真伪。

⚠️ import 说明（为什么 from .orchestrator 不循环）：
  orchestrator.py 对本模块的所有 import 都是**函数内惰性 import**（Step 4.5 / 沉默补丁），
  orchestrator 模块加载时不会触发本模块，故本模块在 top-level `from .orchestrator import` 安全：
  无论谁先被 import，另一方都能先完整加载。详见 README §集成-import 安全性。
"""

import json
import logging
import re

from .llm import chat_text  # 真实 LLM 接口：chat_text(prompt:str) -> str
from .orchestrator import (  # 复用既有信号常量/函数；惰性链保证不循环
    _GOVERNMENT_ENTITIES,
    _SPECIFIC_INSTITUTIONS,
    _has_debunk_signal,
    _normalize_text,
)
from .schemas import BlockerType, NonAdjudicatedAction

logger = logging.getLogger(__name__)

# ── 信号标记表 ──────────────────────────────────────────────────
_SCREENSHOT_MARKERS = ["截图", "图片显示", "如图", "网传一张", "群里发的图", "聊天记录", "通知截图"]
_PRIVATE_CHANNEL_MARKERS = [
    "同事说",
    "朋友说",
    "朋友转",
    "朋友圈",
    "群里说",
    "群里",
    "群消息",
    "群聊",
    "微信群",
    "内部通知",
    "hr说",
    "听同事",
    "据内部",
    "不让外传",
    "别外传",
    "班主任群",
    "家长群",
    "物业群",
    "业主群",
    "私聊",
    "内部消息",
]
_FUTURE_MARKERS = [
    "明天",
    "后天",
    "下周",
    "下个月",
    "下月",
    "明年",
    "未来",
    "即将",
    "将于",
    "拟",
    "计划于",
]
# 声称有官方通知 → 即便未来时态也可查（查通知是否存在）
_OFFICIAL_NOTICE_MARKERS = [
    "已发布通知",
    "已发通知",
    "刚发通知",
    "发布公告",
    "官微",
    "官方通知",
    "发文",
    "通告",
    "〔",
    "号文",
    "正式通知",
    "已公告",
]
_QR_LINK_MARKERS = ["二维码", "扫码", "点这个链接", "点击领取", "长按识别", "点开链接"]
# 与 C8.2 信仰民俗重叠，作 secondary 信号（C8 在 Step0.4 已先拦，这里兜 C8 漏网）
_NON_FALSIFIABLE_MARKERS = [
    "财运",
    "运势",
    "风水",
    "命理",
    "转运",
    "因果报应",
    "前世",
    "缘分",
    "旺夫",
    "招财",
]

_DATE_RE = re.compile(
    r"\d{1,2}月\d{1,2}[日号]|\d{4}年|\d{1,2}[:：]\d{2}|昨[天晚]|今[天早晚]|前[天晚]"
)
_NAMED_SUBJECT_RE = re.compile(
    r"[一-龥]{2,8}(银行|医院|学校|公司|大学|集团|药业|疾控|教委|教育局|供水|物业)"
)
# '某银行'式模糊指代 —— 必须先于 _NAMED_SUBJECT_RE 判，否则 '某'+'银行' 会被误当点名
_VAGUE_SUBJECT_RE = re.compile(
    r"某[一-龥]{0,4}(银行|医院|学校|公司|大学|集团|药业|地方|市|区|部门|机构|品牌|电商|平台|企业)"
)


# ── 特征抽取（零 LLM）────────────────────────────────────────────
def _looks_has_named_subject(t: str) -> bool:
    """是否点名了具体主体（『某银行』式模糊指代不算点名）。"""
    if _VAGUE_SUBJECT_RE.search(t):
        return False
    if any(i in t for i in _SPECIFIC_INSTITUTIONS) or any(e in t for e in _GOVERNMENT_ENTITIES):
        return True
    return bool(_NAMED_SUBJECT_RE.search(t))


def _compute_missing_fields(t: str) -> list[str]:
    """缺哪些**关键且阻断**核查的要素 —— 只列真的缺、且足以阻断核查的（主体 / 链接目标）。
    ⚠️ 不计『缺时间』：几乎每条都没显式时间戳，计进去会让 MISSING_KEY_CONTEXT 吞掉一切；
    时效问题由 C3 时间契约处理。"""
    missing: list[str] = []
    if _VAGUE_SUBJECT_RE.search(t) or (
        any(v in t for v in ("听说", "据说", "网传", "有人说")) and not _looks_has_named_subject(t)
    ):
        missing.append("具体主体（哪家/哪地/谁）")
    if any(m in t for m in _QR_LINK_MARKERS):
        missing.append("实际链接或二维码目标（选中文本未包含）")
    return missing


def _has_conflicting_authority(evidence_list: list) -> bool:
    """证据池里是否同时存在高权威『支持』与高权威『辟谣』（真冲突，保守判定）。"""
    if not evidence_list or len(evidence_list) < 2:
        return False
    has_support = any(
        getattr(e, "supports_claim", False) and getattr(e, "authority_score", 0) >= 0.6
        for e in evidence_list
    )
    has_debunk = any(
        _has_debunk_signal(f"{getattr(e, 'title', '')} {getattr(e, 'snippet', '')}")
        and getattr(e, "authority_score", 0) >= 0.6
        for e in evidence_list
    )
    return has_support and has_debunk


def extract_claim_features(claim_text: str, evidence_list: list) -> dict:
    """确定性抽取核查障碍特征。零 LLM。供决策树 + 兜底文案 + 校验层共用。"""
    t = _normalize_text(claim_text)
    return {
        "has_subject": _looks_has_named_subject(t),
        "has_date_time": bool(_DATE_RE.search(claim_text)),
        "has_screenshot": any(m in t for m in _SCREENSHOT_MARKERS),
        "has_private_channel": any(m in t for m in _PRIVATE_CHANNEL_MARKERS),
        "has_future_marker": any(m in t for m in _FUTURE_MARKERS),
        "claims_official_notice": any(m in t for m in _OFFICIAL_NOTICE_MARKERS),
        "has_qr_or_link_ref": any(m in t for m in _QR_LINK_MARKERS),
        "non_falsifiable": any(m in t for m in _NON_FALSIFIABLE_MARKERS),
        "has_conflicting_evidence": _has_conflicting_authority(evidence_list),
        "missing_fields": _compute_missing_fields(t),
    }


def is_checkable_announcement(text: str) -> bool:
    """声称有官方通知/公告/发文 + 点名机构 → 即便未来时态也可查（查通知是否存在）。
    用于：① 沉默策略未来预测分支放行；② UNVERIFIABLE future 细分前放行到正常流水线。"""
    t = _normalize_text(text)
    return any(m in t for m in _OFFICIAL_NOTICE_MARKERS) and _looks_has_named_subject(t)


# ── 决策树（替换 classify_unverifiable_type）────────────────────
def classify_blocker(claim_text: str, evidence_list: list) -> tuple[BlockerType, list[str]]:
    """非裁定道：按 primary_blocker 优先级决策树返回 (主障碍, 副标签[])。零 LLM。"""
    f = extract_claim_features(claim_text, evidence_list)
    flags: list[str] = []
    if f["has_screenshot"]:
        flags.append("unsourced_screenshot")
    if f["has_private_channel"]:
        flags.append("private_channel")
    if f["has_future_marker"]:
        flags.append("future_time_reference")
    if f["has_qr_or_link_ref"]:
        flags.append("inaccessible_target")

    checkable_notice = f["claims_official_notice"] and f["has_subject"]

    # 1. 非事实命题（C8 漏网兜底）
    if f["non_falsifiable"]:
        return BlockerType.NO_CHECKABLE_FACT, flags
    # 2. 缺关键语境（例外：声称官方通知+点名 → 可查，不算缺）
    if f["missing_fields"] and not checkable_notice:
        return BlockerType.MISSING_KEY_CONTEXT, flags
    # 3. 来源/载体不可溯源（例外同上）
    if f["has_screenshot"] and not checkable_notice:
        return BlockerType.SOURCE_ARTIFACT_AUTH, flags
    # 4. 私域/超本地不可及
    if f["has_private_channel"]:
        return BlockerType.NON_PUBLIC_EVIDENCE, flags
    # 5. 未决/真未来预测（声称官方通知的未来事件要排除——那是可查的）
    if f["has_future_marker"] and not f["claims_official_notice"]:
        return BlockerType.NOT_YET_SETTLED, flags
    # 6. 权威证据真冲突
    if f["has_conflicting_evidence"]:
        return BlockerType.CONFLICTING_AUTH, flags
    # 7. 可公开查但本轮没搜到证据（含 HIGH#1：可查官方通知类）：点名了公共主体
    #    （机构/政府/企业）、不带私域红旗 —— 「能公开查、只是这轮证据没覆盖」，区别于
    #    性质上就私域的事。可查官方通知（点名机构+官方公告/发文/号文）也落这里——
    #    它本就该出现在公开渠道，绝不能用 NON_PUBLIC_EVIDENCE 的「本就不会出现在公开
    #    渠道」自相矛盾描述（修 HIGH#1，INV-U4 在判定层落地）。
    #    走到这里 has_private_channel 已被规则4 拦掉；此处再显式判一次仅为可读。
    if f["has_subject"] and not f["has_private_channel"]:
        return BlockerType.NO_PUBLIC_EVIDENCE_FOUND, flags
    # 兜底：无点名公共主体的私域/超本地（电梯困人、小区水管、机场即时运营态等）→ 绝不判谣言
    return BlockerType.NON_PUBLIC_EVIDENCE, flags


def classify_unverifiable_type(claim_text: str, evidence_list: list) -> str:
    """[DEPRECATED] 返回旧二分字符串，内部转调 classify_blocker。
    迁移期保留，避免一次性改爆调用点（orchestrator.py:5540）。新代码一律用 classify_blocker。"""
    blocker, _ = classify_blocker(claim_text, evidence_list)
    return "developing" if blocker == BlockerType.NOT_YET_SETTLED else "insufficient"


# ── 两层归因生成（骨架兜底 + LLM 填细节 + 展示前校验）─────────────
# 禁止出现的"缺席话术" + 真伪断言 —— 会把真假判断偷渡回来（INV-U3）
_ABSENCE_PHRASES = [
    "没有官方",
    "未见",
    "查不到",
    "搜不到",
    "暂无证据",
    "缺乏依据",
    "缺乏证据",
    "没有找到",
    "无证据支持",
    "未查到",
    "网上没有",
    "找不到来源",
    "不属实",
    "是假的",
    "为假",
    "属实",
    "是真的",
    "为真",
    "确认为",
]

_BLOCKER_SKELETON: dict[BlockerType, str] = {
    BlockerType.NO_CHECKABLE_FACT: "这条内容不是可用公开证据判真伪的事实命题，属于{slot}范畴。",
    BlockerType.MISSING_KEY_CONTEXT: (
        "这条消息缺少核查所需的关键要素（{slot}），无法定位到可查证的具体事实。"
    ),
    BlockerType.SOURCE_ARTIFACT_AUTH: (
        "这条消息依赖无法独立溯源的{slot}，真实性需回到原始出处确认。"
    ),
    BlockerType.NON_PUBLIC_EVIDENCE: (
        "这条消息属于{slot}，按其性质本就不会出现在公开渠道——公开搜索没有覆盖不等于事件不存在。"
    ),
    BlockerType.NOT_YET_SETTLED: (
        "这条消息指向{slot}，相关事实当前尚未落定，需待其正式发生或公布后才能核对。"
    ),
    BlockerType.CONFLICTING_AUTH: (
        "公开渠道对这条消息存在{slot}，权威来源结论暂不一致，目前无法给出单一判定。"
    ),
    BlockerType.NO_PUBLIC_EVIDENCE_FOUND: (
        "这条消息点名了{slot}、属于本可公开查证的事，但本轮公开检索尚未覆盖到对应记录——"
        "证据不足只是这一轮没拿到，并不说明事情真假。"
    ),
}

_BLOCKER_VERIFY_WHERE: dict[BlockerType, str] = {
    BlockerType.NO_CHECKABLE_FACT: "这类问题宜回到价值/信仰讨论，而非事实核查。",
    BlockerType.MISSING_KEY_CONTEXT: "补全缺失要素后，可查对应主体的官方公告或权威通报。",
    BlockerType.SOURCE_ARTIFACT_AUTH: "请回到截图/转述的原始来源（官方账号、原始发文）核对。",
    BlockerType.NON_PUBLIC_EVIDENCE: (
        "请向直接相关方一手确认（如 HR 正式邮件、物业公告、当事单位、出警记录）。"
    ),
    BlockerType.NOT_YET_SETTLED: "请以届时官方正式发布为准，勿提前作为既成事实传播。",
    BlockerType.CONFLICTING_AUTH: "请以最高权威来源的最新结论为准，留意各来源发布时间与口径。",
    BlockerType.NO_PUBLIC_EVIDENCE_FOUND: (
        "可直接到该主体的官网/官方账号、主管部门或权威媒体报道里核对，或换更精确的关键词再查一次。"
    ),
}

_SLOT_DEFAULT: dict[BlockerType, str] = {
    BlockerType.NO_CHECKABLE_FACT: "信仰/价值/修辞",
    BlockerType.MISSING_KEY_CONTEXT: "关键要素",
    BlockerType.SOURCE_ARTIFACT_AUTH: "截图或转述内容",
    BlockerType.NON_PUBLIC_EVIDENCE: "私域、内部或超本地信息",
    BlockerType.NOT_YET_SETTLED: "尚未发生或仍在进行中的事件",
    BlockerType.CONFLICTING_AUTH: "相互冲突的权威证据",
    BlockerType.NO_PUBLIC_EVIDENCE_FOUND: "具体主体（机构/政府/企业）",
}


def fallback_detail(blocker: BlockerType, features: dict) -> str:
    """确定性兜底文案（不调 LLM）。INV-U2：LLM 失败也不回退到空壳。"""
    if blocker == BlockerType.MISSING_KEY_CONTEXT and features.get("missing_fields"):
        slot = "、".join(features["missing_fields"])
    else:
        slot = _SLOT_DEFAULT[blocker]
    return _BLOCKER_SKELETON[blocker].format(slot=slot)


def validate_attribution_detail(detail: str, blocker: BlockerType, features: dict) -> bool:
    """展示前校验 LLM 细节。不合格 → 调用方回退 fallback_detail。
    a) 禁缺席话术/真伪断言（INV-U3）  b) 禁谎称缺失（防编理由）  c) 非空且合理长度"""
    if not detail or len(detail) < 6:
        return False
    if any(p in detail for p in _ABSENCE_PHRASES):
        return False
    if features.get("has_date_time") and any(
        x in detail for x in ("没有时间", "缺少时间", "未注明时间", "没说时间")
    ):
        return False
    return not (
        features.get("has_subject")
        and any(x in detail for x in ("没有主体", "缺少主体", "未指明", "没说是哪"))
    )


_ATTRIBUTION_LLM_PROMPT = (
    "已知该声明被归为「{blocker_cn}」类"
    "（公开渠道无法判定真伪的结构性原因）。\n"
    "请只做一件事：用一句话指出【这一条声明里】具体是哪个要素导致公开渠道判不了。\n"
    "可用的结构性事实（只能据此写，不得编造）：{features_json}\n"
    "禁止：改变类型判定；输出「无法核实/不确定/查不到/没有官方/暂无证据」等空泛或缺席词；\n"
    "断言该声明为真或为假。只描述「为什么这一条查不了」。直接输出一句话，不要解释。\n"
    "\n"
    "声明：{claim_text}"
)


def _safe_features(features: dict) -> dict:
    """只暴露给 LLM 布尔/列表信号，避免它拿到原文外的东西编。"""
    return {k: v for k, v in features.items() if k != "missing_fields"} | {
        "missing_fields": features.get("missing_fields", [])
    }


def generate_attribution_detail(
    claim_text: str, blocker: BlockerType, features: dict, llm_fn=None
) -> str:
    """两层生成：LLM 在类型框架内填一句 → 校验 → 不过则确定性兜底。
    llm_fn：可注入的 str->str 调用（测试注 lambda）；默认走 llm.chat_text。"""
    llm_fn = llm_fn or chat_text
    try:
        raw = llm_fn(
            _ATTRIBUTION_LLM_PROMPT.format(
                blocker_cn=blocker.value,
                features_json=json.dumps(_safe_features(features), ensure_ascii=False),
                claim_text=claim_text,
            )
        )
        lines = (raw or "").strip().splitlines()
        detail = lines[0].strip() if lines else ""
        if validate_attribution_detail(detail, blocker, features):
            return detail
        logger.info("[attribution] LLM 细节未过校验，回退确定性兜底：%s", detail[:40])
    except Exception:
        logger.exception("[attribution] LLM 细节生成异常，回退确定性兜底")
    return fallback_detail(blocker, features)


def build_non_adjudicated(
    claim_text: str, evidence_list: list, llm_fn=None
) -> NonAdjudicatedAction:
    """组装非裁定输出。INV-U2 保证 primary_blocker + detail 必非空。
    llm_fn 默认 None → 内部用 llm.chat_text；orchestrator 直接调即可，无需传参。"""
    features = extract_claim_features(claim_text, evidence_list)
    blocker, flags = classify_blocker(claim_text, evidence_list)
    detail = generate_attribution_detail(claim_text, blocker, features, llm_fn)
    return NonAdjudicatedAction(
        action_kind="not_a_fact_claim"
        if blocker == BlockerType.NO_CHECKABLE_FACT
        else "needs_first_hand_confirmation",
        primary_blocker=blocker,
        secondary_flags=flags,
        claim_specific_detail=detail,
        verify_where=_BLOCKER_VERIFY_WHERE[blocker],
    )

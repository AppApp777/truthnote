"""TruthNote 核查流水线（面向外部调用的简单接口）。

内部使用 Orchestrator 多 Agent 架构执行。
记忆层：claim 级复用——先提取 claim，逐条查记忆，命中的秒回、未命中的走完整流程。
反馈闭环：有负面反馈的案例不复用记忆，强制重新核查。
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta

from .memory import MemoryStore
from .orchestrator import Orchestrator
from .schemas import (
    CaseLifecycle,
    ClaimVerification,
    InvestigationTrace,
    LifecycleState,
    RiskLevel,
    TraceStep,
    Verdict,
    VerifyResponse,
)
from .search import get_search_provider

logger = logging.getLogger(__name__)

_memory_store: MemoryStore | None = None

_RISK_MAP = {
    "健康养生": RiskLevel.LOW,
    "AI名人语录": RiskLevel.LOW,
    "政策法规": RiskLevel.MEDIUM,
    "旧闻翻炒": RiskLevel.MEDIUM,
    "金融财经": RiskLevel.MEDIUM,
    "伪造截图": RiskLevel.MEDIUM,
    "诈骗套路": RiskLevel.HIGH,
    "灾难恐慌": RiskLevel.HIGH,
}

_INTERVENTION_MAP = {
    RiskLevel.LOW: "private_warn",
    RiskLevel.MEDIUM: "private_warn_timeout_group",
    RiskLevel.HIGH: "group_alert",
}


def _infer_risk_level(result: VerifyResponse) -> RiskLevel:
    levels = []
    for cv in result.claims:
        cat = cv.claim.category.value
        levels.append(_RISK_MAP.get(cat, RiskLevel.MEDIUM))
    if not levels:
        return RiskLevel.MEDIUM
    priority = {RiskLevel.HIGH: 3, RiskLevel.MEDIUM: 2, RiskLevel.LOW: 1}
    return max(levels, key=lambda x: priority[x])


_ESCALATION_TIMEOUT = {
    RiskLevel.LOW: 600,
    RiskLevel.MEDIUM: 300,
    RiskLevel.HIGH: 0,
}


def _generate_self_correction_script(result: VerifyResponse) -> str:
    """预写发信人自纠话术：让 ta 从"被纠正者"变成"负责人"。"""
    verdict = result.overall_verdict.value
    claim_text = result.claims[0].claim.text if result.claims else "相关内容"
    return (
        f"刚才我转发的那条消息，我又查了一下，"
        f"发现{claim_text[:20]}这个说法{verdict}。"
        f"大家别信哈，是我没核实就转了，抱歉～"
    )


def _generate_escalation_message(result: VerifyResponse) -> str:
    """超时后的群发补充消息：客观陈述，不点名。"""
    summaries = []
    for cv in result.claims:
        summaries.append(f"「{cv.claim.text[:25]}」→ {cv.verdict.value}")
    claims_str = "；".join(summaries)
    return (
        f"📋 信息核查提醒：群里最近有条消息需要注意——{claims_str}。"
        f"建议大家以官方渠道信息为准，不轻信转发内容。"
    )


def _build_lifecycle(
    memory: MemoryStore | None, case_id: int, result: VerifyResponse
) -> CaseLifecycle:
    risk = _infer_risk_level(result)
    intervention = _INTERVENTION_MAP[risk]
    if result.overall_verdict == Verdict.TRUE:
        intervention = "positive_confirm"

    timeout_sec = _ESCALATION_TIMEOUT[risk]
    escalation_msg = ""
    correction_script = ""
    deadline = None

    if result.overall_verdict != Verdict.TRUE:
        escalation_msg = _generate_escalation_message(result)
        correction_script = _generate_self_correction_script(result)
        if timeout_sec > 0:
            deadline = datetime.now() + timedelta(seconds=timeout_sec)

    lc = CaseLifecycle(
        case_id=case_id,
        risk_level=risk,
        intervention_type=intervention,
        self_correction_script=correction_script,
        escalation_message=escalation_msg,
        escalation_deadline=deadline,
        escalation_timeout_sec=timeout_sec,
    )
    lc.advance(LifecycleState.DETECTED, "消息接收")
    lc.advance(LifecycleState.EXTRACTED, f"{len(result.claims)} 条声明")
    lc.advance(LifecycleState.VERIFIED, "证据搜索完成")
    lc.advance(LifecycleState.JUDGED, f"判定: {result.overall_verdict.value}")
    lc.advance(LifecycleState.INTERVENED, f"策略: {intervention}")

    if intervention == "group_alert":
        lc.advance(LifecycleState.ESCALATED, "高风险，立即群内警告")
        lc.advance(LifecycleState.MEMORIZED, f"case_id={case_id}")
    elif intervention == "positive_confirm":
        lc.advance(LifecycleState.MEMORIZED, f"case_id={case_id}")
    else:
        lc.advance(LifecycleState.TRACKING, f"追踪中，{timeout_sec}s 后升级")

    if memory:
        try:
            memory.create_lifecycle(case_id, risk.value)
            for t in lc.transitions:
                memory.advance_lifecycle(case_id, t.to_state.value, t.detail)
            memory.update_lifecycle_intervention(case_id, intervention)
            if deadline and timeout_sec > 0:
                memory.start_tracking(
                    case_id,
                    escalation_deadline=time.time() + timeout_sec,
                    escalation_message=escalation_msg,
                    self_correction_script=correction_script,
                )
        except Exception as e:
            logger.warning("[Pipeline] 生命周期持久化失败：%s", e)

    return lc


def check_escalations() -> list[dict]:
    """检查所有到期未自纠的追踪案例，执行升级。"""
    memory = get_memory_store()
    pending = memory.get_pending_escalations()
    escalated = []
    for case in pending:
        case_id = case["case_id"]
        memory.advance_lifecycle(case_id, "escalated", "超时未自纠，群内补发")
        memory.advance_lifecycle(case_id, "memorized", f"case_id={case_id}")
        escalated.append(
            {
                "case_id": case_id,
                "message": case.get("original_message", "")[:60],
                "escalation_message": case.get("escalation_message", ""),
            }
        )
        logger.info("[Escalation] case_id=%d 超时升级", case_id)
    return escalated


def report_self_correction(case_id: int) -> bool:
    """发信人主动自纠 → 跳过升级，直接进入记忆。"""
    memory = get_memory_store()
    lc = memory.get_lifecycle(case_id)
    if not lc or lc["current_state"] not in ("tracking", "intervened"):
        return False
    memory.mark_self_corrected(case_id)
    memory.advance_lifecycle(case_id, "memorized", "发信人已自纠，跳过升级")
    logger.info("[SelfCorrection] case_id=%d 发信人自纠", case_id)
    return True


def get_memory_store() -> MemoryStore | None:
    global _memory_store
    if _memory_store is None:
        try:
            _memory_store = MemoryStore()
        except Exception:
            logger.warning("[Pipeline] MemoryStore 初始化失败", exc_info=True)
            return None
    return _memory_store


def _try_case_exact_recall(memory: MemoryStore, message: str) -> VerifyResponse | None:
    """案例级精确召回：只匹配完整消息原文。

    不做模糊匹配——模糊匹配在 claim 级由 orchestrator 处理，
    避免多声明消息被单条旧声明劫持。
    """
    case_row = memory.recall_case_exact(message)
    if not case_row:
        return None

    case_id = case_row["id"]
    if memory.has_negative_feedback(case_id):
        logger.info("[Pipeline] 案例精确命中但有负面反馈，跳过")
        return None

    full_case = memory.get_full_case(case_id)
    if not full_case:
        return None

    claims = []
    for c in full_case.get("claims", []):
        try:
            claims.append(ClaimVerification.model_validate(c))
        except Exception as e:
            logger.warning("[Pipeline] 记忆中的 claim 校验失败：%s", e)
    if not claims:
        logger.warning("[Pipeline] 案例精确命中但所有 claims 校验失败，跳过记忆")
        return None
    evidence_sources = full_case.get("evidence_sources", [])

    memory_trace = InvestigationTrace(
        steps=[
            TraceStep(
                agent="MemoryStore",
                action="案例精确命中",
                duration_ms=0,
                input_summary=message[:60],
                output_summary=f"case_id={case_id}",
            )
        ],
        total_duration_ms=0,
        total_llm_calls=0,
    )

    try:
        overall = Verdict(full_case["overall_verdict"])
    except (ValueError, KeyError):
        logger.warning(
            "[Pipeline] 案例 overall_verdict 无效：%r，跳过记忆", full_case.get("overall_verdict")
        )
        return None

    logger.info("[Pipeline] 案例精确命中，秒拦（case_id=%d）", case_id)

    lc = CaseLifecycle(
        case_id=case_id,
        repeat_blocked=True,
        current_state=LifecycleState.REPEAT_BLOCKED,
    )
    lc.advance(LifecycleState.DETECTED, "消息接收")
    lc.advance(LifecycleState.REPEAT_BLOCKED, f"记忆秒拦，原始案例 #{case_id}")

    return VerifyResponse(
        original_message=message,
        claims=claims,
        overall_verdict=overall,
        summary=f"[重复拦截 · 秒拦] {full_case.get('summary', '')}",
        friendly_reply=full_case.get("friendly_reply", ""),
        evidence_sources=evidence_sources,
        trace=memory_trace,
        lifecycle=lc,
    )


def verify_message(
    message: str,
    context: str = "",
    *,
    use_memory: bool = True,
    use_rules: bool = True,
    use_gates: bool = True,
    use_search: bool = True,
    use_debunk_index: bool | None = None,
    on_step: callable | None = None,
) -> VerifyResponse:
    """核查一条群聊消息。同步接口。

    记忆策略：
    1. 先整条消息查记忆（快路径：单 claim 消息秒回）
    2. 整条未命中 → 交给 orchestrator 提取 claims → 逐条查记忆
    3. 命中的 claim 复用缓存，未命中的走完整核查
    4. 有负面反馈的案例不复用
    """
    memory = get_memory_store() if use_memory else None

    # 快路径：整条消息精确命中记忆（不做模糊匹配）
    if memory:
        try:
            fast_hit = _try_case_exact_recall(memory, message)
            if fast_hit:
                return fast_hit
        except Exception:
            logger.warning("[Pipeline] 案例精确召回异常，跳过记忆", exc_info=True)

    # 完整核查（orchestrator 内部做 claim 级模糊记忆）
    orchestrator = Orchestrator(
        search_provider=get_search_provider(),
        memory_store=memory,
        use_rules=use_rules,
        use_gates=use_gates,
        use_search=use_search,
        use_debunk_index=use_debunk_index,
    )
    if on_step:
        orchestrator.on_step = on_step
    result = orchestrator.run(message, context)

    # 保存记忆并回填 case_id + 生命周期
    case_id = 0
    if memory:
        try:
            case_id = memory.save_case(result)
            result.disposition.case_id = case_id
        except Exception as e:
            logger.warning("[Pipeline] 保存记忆失败：%s", e)

    result.lifecycle = _build_lifecycle(memory, case_id, result)

    return result

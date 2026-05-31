"""Step humanizer — 把 Agent 输出翻译成人话 + 结构化 display payload。

调用契约：
    humanize_step(agent_name, action, result, output_data, output_summary)
        -> (human_narrative: str, display: dict | None)

human_narrative：评委友好的一句话「小T xx：...」。
display：{"template": "...", "data": {...}}，对应 render-engine.js 的模板注册表。

设计原则：
- 任何分支异常都不应阻塞 pipeline（兜底返回 zh+summary）
- 不做 LLM 调用、不读 IO、纯本地映射
- 与 docs/extension_event_contract.md 的 schema 对齐
"""

from __future__ import annotations

from typing import Any

# ============ Agent 中文名 ============
AGENT_ZH: dict[str, str] = {
    "ScenarioRouter": "场景分诊",
    "ClaimExtractor": "拆声明",
    "CheckWorthy": "筛声明",
    "CommonsenseChecker": "常识审",
    "AtomicFact": "原子化",
    "ClaimMatcher": "记忆对照",
    "QueryPlanner": "规划查询",
    "EvidenceHunter": "搜证据",
    "EvidenceRanker": "证据排序",
    "FactChecker": "对一对",
    "StructuredFactChecker": "结构化核查",
    "Skeptic": "质疑",
    "ResponseComposer": "写报告",
    "MemoryStore": "存档",
    "PromoHealth": "推销话术核验",
}


def _zh(agent_name: str) -> str:
    return AGENT_ZH.get(agent_name, agent_name or "处理")


def _claim_to_dict(c: Any) -> dict:
    """Claim 对象（含 BaseModel/Mock）→ 渲染端可用 dict。"""
    if hasattr(c, "model_dump"):
        d = c.model_dump(mode="json")
        text = d.get("text") or ""
        category = d.get("category") or ""
        return {
            "text": text,
            "category": str(category) if category else "",
            "is_central_action": bool(d.get("is_central_action", False)),
        }
    if isinstance(c, dict):
        return {
            "text": c.get("text", ""),
            "category": str(c.get("category", "")) if c.get("category") else "",
            "is_central_action": bool(c.get("is_central_action", False)),
        }
    return {"text": str(c), "category": "", "is_central_action": False}


def _evidence_to_dict(e: Any) -> dict:
    if hasattr(e, "model_dump"):
        d = e.model_dump(mode="json")
        return {
            "title": d.get("title") or "(无标题)",
            "url": d.get("url") or "",
            "snippet": d.get("snippet") or "",
            "source": d.get("source") or "",
            "authority_score": d.get("authority_score"),
        }
    if isinstance(e, dict):
        return {
            "title": e.get("title") or "(无标题)",
            "url": e.get("url") or "",
            "snippet": e.get("snippet") or "",
            "source": e.get("source") or "",
            "authority_score": e.get("authority_score"),
        }
    return {"title": str(e), "url": "", "snippet": "", "source": ""}


# ============ 各 Agent 处理函数 ============


def _h_scenario_router(*, action, result, output_data, output_summary):
    data = output_data or {}
    scenario = data.get("scenario") or "未知场景"
    frame = data.get("message_frame") or {}
    entity = frame.get("promoted_entity") or ""
    flags = frame.get("red_flags") or []
    msg_type = frame.get("message_type") or ""
    type_label = {
        "health_product_promo": "健康产品推销",
        "financial_scam": "金融诈骗",
        "political_rumor": "政治谣言",
        "health_advice": "健康建议",
        "personal_experience": "个人经历",
        "fact_assertion": "事实声明",
        "other": "其他",
    }.get(msg_type, "")

    parts = [f"小T 识别为「{scenario}」"]
    if type_label:
        parts.append(f"消息类型：{type_label}")
    if entity:
        parts.append(f"被推销对象「{entity}」")
    if flags:
        flag_preview = "、".join(flags[:3])
        parts.append(f"检测到 {len(flags)} 个话术红旗（{flag_preview}）")
    return ("，".join(parts), None)


def _h_claim_extractor(*, action, result, output_data, output_summary):
    items = []
    if isinstance(result, list):
        items = [_claim_to_dict(c) for c in result]
    if not items:
        return ("小T 没拆出可核查的具体声明", None)
    preview = "、".join(f"「{c['text'][:16]}」" for c in items[:2])
    central = sum(1 for c in items if c.get("is_central_action"))
    extra = f"（{central} 条核心）" if central else ""
    narrative = f"小T 拆出 {len(items)} 条事实声明{extra}：{preview}" + (
        "..." if len(items) > 2 else ""
    )
    return (narrative, {"template": "claim_list", "data": {"claims": items}})


def _h_checkworthy(*, action, result, output_data, output_summary):
    items = []
    if isinstance(result, list):
        items = [_claim_to_dict(c) for c in result]
    if not items:
        return ("小T 没有值得核查的声明", None)
    return (
        f"小T 筛后保留 {len(items)} 条值得核查的声明",
        {"template": "claim_list", "data": {"claims": items}},
    )


def _h_commonsense(*, action, result, output_data, output_summary):
    data = output_data or (result if isinstance(result, dict) else {})
    is_cs = bool(data.get("is_commonsense"))
    cs_type = data.get("commonsense_type") or ""
    try:
        conf = float(data.get("confidence") or 0)
    except (TypeError, ValueError):
        conf = 0.0
    if is_cs:
        return (
            f"小T 用常识就能判断：{cs_type}（置信 {conf:.0%}），跳过外部搜索",
            None,
        )
    return ("小T 这条不是常识级伪科学，进入正常搜证据流程", None)


def _h_atomic_fact(*, action, result, output_data, output_summary):
    data = output_data or {}
    atoms = data.get("atoms") or data.get("atom_facts") or []
    if not atoms:
        return (f"小T 把声明原子化（{output_summary})", None)
    return (f"小T 把这条声明拆成 {len(atoms)} 个原子事实分别核查", None)


def _h_query_planner(*, action, result, output_data, output_summary):
    queries: list[str] = []
    if hasattr(result, "queries"):
        queries = list(result.queries or [])
    elif output_data:
        queries = output_data.get("queries") or []
    if not queries:
        return ("小T 在准备搜索词...", None)
    preview = "、".join(f"「{q}」" for q in queries[:3])
    return (
        f"小T 规划了 {len(queries)} 个搜索关键词：{preview}",
        {"template": "generic", "data": {"queries": queries}},
    )


def _h_evidence_hunter(*, action, result, output_data, output_summary):
    evidence_items: list = []
    if isinstance(result, tuple) and result:
        evidence_items = result[0] or []
    elif isinstance(result, list):
        evidence_items = result
    total = len(evidence_items)
    top = [_evidence_to_dict(e) for e in evidence_items[:5]]
    # 注：EvidenceHunterAgent.hunt() 返回 (evidence_list, engine_name_str)，
    # 不在结果里携带 queries；planned_queries 在 hunter 实例上，这里拿不到。
    if total == 0:
        return ("小T 没搜到相关证据", None)
    return (
        f"小T 找到 {total} 条证据",
        {
            "template": "evidence_list",
            "data": {
                "total": total,
                "kept": total,
                "top_results": top,
            },
        },
    )


def _h_evidence_ranker(*, action, result, output_data, output_summary):
    ranked = []
    sufficiency = ""
    if hasattr(result, "ranked_evidence"):
        ranked = list(result.ranked_evidence or [])
        sufficiency = getattr(result, "sufficiency", "") or ""
    elif output_data:
        ranked = output_data.get("ranked_evidence") or []
        sufficiency = output_data.get("sufficiency", "") or ""
    suff_zh = {"sufficient": "充分", "insufficient": "不足", "conflicting": "冲突"}.get(
        sufficiency, sufficiency
    )
    top = [_evidence_to_dict(e) for e in ranked[:5]]
    narrative_extra = f"，证据{suff_zh}" if suff_zh else ""
    if not ranked:
        return (f"小T 排序后没有有效证据保留{narrative_extra}", None)
    return (
        f"小T 排序后保留 {len(ranked)} 条最相关证据{narrative_extra}",
        {
            "template": "evidence_list",
            "data": {"total": len(ranked), "kept": len(ranked), "top_results": top},
        },
    )


def _h_fact_checker(*, action, result, output_data, output_summary):
    return _verification_card("FactChecker", result, output_data, "交叉验证")


def _h_structured_fc(*, action, result, output_data, output_summary):
    return _verification_card("StructuredFactChecker", result, output_data, "结构化核查")


def _h_skeptic(*, action, result, output_data, output_summary):
    # Skeptic 输入往往是 dict（challenges/passed/revised_verdict/inv3_blocked）
    data = output_data or (result if isinstance(result, dict) else {})
    challenges = data.get("challenges") or []
    passed = data.get("passed")
    revised = data.get("revised_verdict") or ""
    inv3_blocked = bool(data.get("inv3_blocked"))
    if isinstance(revised, dict):
        revised = revised.get("value", "")
    if challenges:
        preview = "、".join(c[:18] for c in challenges[:2])
        suffix = f"，提了 {len(challenges)} 个质疑（{preview}）"
    else:
        suffix = "，无显著质疑"
    # P1 #5：INV-3 拦截单独提示给用户，不再混进 challenges 预览
    inv3_suffix = "（INV-3 守护：已有强矛盾证据，不允许通用怀疑降级）" if inv3_blocked else ""
    if revised:
        return (f"小T 质疑后修订判定 → {revised}{suffix}{inv3_suffix}", None)
    if passed is True:
        return (f"小T 质疑通过{suffix}{inv3_suffix}", None)
    return (f"小T 质疑检验{suffix}{inv3_suffix}", None)


def _h_response_composer(*, action, result, output_data, output_summary):
    reply = ""
    summary_text = ""
    if isinstance(result, tuple) and result:
        reply = result[0] if isinstance(result[0], str) else ""
        if len(result) > 1 and isinstance(result[1], str):
            summary_text = result[1]
    if not reply:
        return ("小T 写完报告", None)
    return (
        f"小T 写了一段温和回复（{len(reply)} 字）",
        {
            "template": "reply_draft",
            "data": {"reply": reply, "summary": summary_text, "tone": "gentle"},
        },
    )


def _h_memory_store(*, action, result, output_data, output_summary):
    return (f"小T 把本次核查存档（{output_summary or 'ok'}）", None)


def _h_claim_matcher(*, action, result, output_data, output_summary):
    data = output_data or {}
    matched = data.get("matched") or data.get("hit")
    if matched:
        return ("小T 在记忆库里找到了相似旧 case，直接复用结论", None)
    return ("小T 在记忆库里没找到匹配的旧 case，走完整核查", None)


def _verification_card(agent_name: str, result, output_data, action_zh: str):
    """FactChecker / StructuredFactChecker 共用渲染。
    输入 result 是 ClaimVerification（单条）或 list[ClaimVerification]。
    """
    verifs = []
    if hasattr(result, "claim") and hasattr(result, "verdict"):
        verifs = [result]
    elif isinstance(result, list):
        verifs = [v for v in result if hasattr(v, "verdict")]
    if not verifs:
        return (f"小T 完成{action_zh}", None)
    grid = []
    for v in verifs:
        verdict = getattr(v.verdict, "value", str(v.verdict))
        claim_text = getattr(getattr(v, "claim", None), "text", "") or ""
        grid.append(
            {
                "claim": claim_text,
                "verdict": verdict,
                "confidence": getattr(v, "confidence", None),
                "reasoning": (getattr(v, "reasoning", "") or "")[:140],
            }
        )
    if len(verifs) == 1:
        v = verifs[0]
        verdict = getattr(v.verdict, "value", str(v.verdict))
        conf = getattr(v, "confidence", 0) or 0
        narrative = f"小T {action_zh}：判定「{verdict}」（置信 {conf:.0%}）"
    else:
        narrative = f"小T 对 {len(verifs)} 条声明做了{action_zh}"
    return (narrative, {"template": "verification_grid", "data": {"verifications": grid}})


_HANDLERS = {
    "ScenarioRouter": _h_scenario_router,
    "ClaimExtractor": _h_claim_extractor,
    "CheckWorthy": _h_checkworthy,
    "CommonsenseChecker": _h_commonsense,
    "AtomicFact": _h_atomic_fact,
    "ClaimMatcher": _h_claim_matcher,
    "QueryPlanner": _h_query_planner,
    "EvidenceHunter": _h_evidence_hunter,
    "EvidenceRanker": _h_evidence_ranker,
    "FactChecker": _h_fact_checker,
    "StructuredFactChecker": _h_structured_fc,
    "Skeptic": _h_skeptic,
    "ResponseComposer": _h_response_composer,
    "MemoryStore": _h_memory_store,
}


def humanize_step(
    agent_name: str,
    action: str,
    result: Any,
    output_data: dict | None,
    output_summary: str,
) -> tuple[str, dict | None]:
    """主入口。返回 (human_narrative, display | None)。"""
    fn = _HANDLERS.get(agent_name)
    if fn is not None:
        try:
            return fn(
                action=action,
                result=result,
                output_data=output_data,
                output_summary=output_summary,
            )
        except Exception:  # noqa: BLE001
            pass
    zh = _zh(agent_name)
    if output_summary:
        return (f"小T {zh}：{output_summary}", None)
    return (f"小T {zh}", None)


__all__ = ["humanize_step", "AGENT_ZH"]

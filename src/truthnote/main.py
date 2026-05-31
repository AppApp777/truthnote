from __future__ import annotations

import asyncio
import contextlib
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from .benchmark import router as benchmark_router
from .config import settings
from .pipeline import (
    check_escalations,
    get_memory_store,
    report_self_correction,
    verify_message,
)
from .schemas import VerifyRequest, VerifyResponse

logging.basicConfig(
    level=logging.DEBUG if settings.debug else logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)

BASE_DIR = Path(__file__).resolve().parent.parent.parent
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"

_MAX_ESCALATION_LOG = 500
_escalation_log: list[dict] = []
logger = logging.getLogger(__name__)


async def _escalation_checker():
    """后台定时器：每 30 秒检查到期的追踪案例并升级。"""
    while True:
        await asyncio.sleep(30)
        try:
            escalated = check_escalations()
            if escalated:
                _escalation_log.extend(escalated)
                if len(_escalation_log) > _MAX_ESCALATION_LOG:
                    _escalation_log[:] = _escalation_log[-_MAX_ESCALATION_LOG:]
                logger.info("[Escalation] 本轮升级 %d 条", len(escalated))
        except Exception:
            logger.warning("[Escalation] 检查失败", exc_info=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    task = asyncio.create_task(_escalation_checker())
    yield
    task.cancel()


app = FastAPI(
    title="TruthNote — 群聊求真小纸条",
    version="0.5.0",
    description="多 Agent 协作的事实核查系统",
    lifespan=lifespan,
)

_ALLOWED_ORIGINS = [
    "chrome-extension://*",
    "http://localhost:*",
    "http://127.0.0.1:*",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_ALLOWED_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization"],
)

STATIC_DIR.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

app.include_router(benchmark_router)


@app.get("/health")
async def health():
    search_display = settings.search_provider
    if settings.search_provider.lower() in ("qihoo360", "so360"):
        search_display = "360搜索 (so.com)" if not settings.qihoo_api_key else "360智搜 API"
    return {
        "status": "ok",
        "version": "0.5.0",
        "architecture": "multi-agent (8 agents)",
        "agents": [
            "ScenarioRouter",
            "ClaimExtractor",
            "QueryPlanner",
            "EvidenceHunter",
            "EvidenceRanker",
            "FactChecker",
            "Skeptic",
            "ResponseComposer",
        ],
        "search": search_display,
        "search_engine": "360搜索"
        if settings.search_provider.lower() in ("qihoo360", "so360")
        else settings.search_provider,
    }


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(request, "index.html")


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request):
    return templates.TemplateResponse(request, "dashboard.html")


@app.get("/benchmark", response_class=HTMLResponse)
async def benchmark_page(request: Request):
    return templates.TemplateResponse(request, "benchmark.html")


_MAX_MESSAGE_LENGTH = 2000


@contextlib.contextmanager
def _override_models(default_model: str = "", strong_model: str = ""):
    """临时切换模型配置。空字符串 = 不覆盖，保持 .env 默认值。"""
    if not default_model and not strong_model:
        yield
        return
    from . import agents

    old_env_default = os.environ.get("DEFAULT_MODEL", "")
    old_env_strong = os.environ.get("STRONG_MODEL", "")
    old_fast = agents._FAST_MODEL
    old_strong = agents._STRONG_MODEL
    try:
        if default_model:
            os.environ["DEFAULT_MODEL"] = default_model
            agents._FAST_MODEL = default_model
        if strong_model:
            os.environ["STRONG_MODEL"] = strong_model
            agents._STRONG_MODEL = strong_model
        yield
    finally:
        os.environ["DEFAULT_MODEL"] = old_env_default
        os.environ["STRONG_MODEL"] = old_env_strong
        agents._FAST_MODEL = old_fast
        agents._STRONG_MODEL = old_strong


@app.get("/api/models")
async def list_models():
    """返回可用模型列表（前端下拉用）。"""
    return {
        "default_models": [
            {"id": "", "name": "DeepSeek V4 Flash（默认）"},
            {"id": "qwen-max", "name": "Qwen 3.7 Max"},
            {"id": "glm-5.1", "name": "GLM 5.1"},
            {"id": "gemini-2.5-flash", "name": "Gemini 2.5 Flash"},
        ],
        "strong_models": [
            {"id": "", "name": "Qwen 3.7 Max（默认）"},
            {"id": "glm-5.1", "name": "GLM 5.1"},
            {"id": "deepseek-v4-flash", "name": "DeepSeek V4 Flash"},
            {"id": "gemini-2.5-flash", "name": "Gemini 2.5 Flash"},
        ],
    }


@app.post("/api/verify", response_model=VerifyResponse)
def verify(req: VerifyRequest):
    if len(req.message) > _MAX_MESSAGE_LENGTH:
        from fastapi import HTTPException

        raise HTTPException(status_code=400, detail=f"消息长度不能超过 {_MAX_MESSAGE_LENGTH} 字")
    with _override_models(req.default_model, req.strong_model):
        # 判定复用缓存已定不用（召回不靠谱）：demo 端点显式关记忆，每条走完整重跑、
        # 永不秒回旧判定、现场零写库。见 docs/记忆线_demo安全与C10话术.md §五。
        result = verify_message(req.message, req.context, use_memory=False)
    return result


@app.post("/api/verify_stream")
async def verify_stream(req: VerifyRequest):
    """SSE 流式核查——每完成一步就推送进度，最后推送完整结果。"""
    import json
    import queue
    import threading
    import time as _time
    import uuid as _uuid

    from fastapi.responses import StreamingResponse

    if len(req.message) > _MAX_MESSAGE_LENGTH:
        from fastapi import HTTPException

        raise HTTPException(status_code=400, detail=f"消息长度不能超过 {_MAX_MESSAGE_LENGTH} 字")

    req_default = req.default_model
    req_strong = req.strong_model
    request_id = req.request_id or _uuid.uuid4().hex[:12]
    step_queue: queue.Queue = queue.Queue()
    t0 = _time.time()

    def _wrap(payload):
        payload["request_id"] = request_id
        payload["t_ms"] = int((_time.time() - t0) * 1000)
        return payload

    def on_step(step):
        payload = {
            "type": "step",
            "agent": step.agent,
            "action": step.action,
            "duration_ms": step.duration_ms,
            "output_summary": step.output_summary,
            "output_data": step.output_data,
        }
        # 评委友好的人话 + 渲染模板（对齐 docs/extension_event_contract.md）
        human_narrative = getattr(step, "human_narrative", "") or ""
        display = getattr(step, "display", None)
        if human_narrative:
            payload["human_narrative"] = human_narrative
        if display:
            payload["display"] = display
        step_queue.put(_wrap(payload))

    def run_pipeline():
        logger.info("[verify_stream] start request_id=%s msg_len=%d", request_id, len(req.message))
        # 注入 _pipeline_progress.queue 让 llm_token / llm_start / llm_done 进 SSE
        from .agents import _pipeline_progress

        _prev_queue = _pipeline_progress.queue
        _pipeline_progress.queue = step_queue
        try:
            with _override_models(req_default, req_strong):
                # 判定复用缓存已定不用（召回不靠谱）：插件实际打的端点，显式关记忆，
                # 现场每条都走完整流式取证、不秒回。见 docs/记忆线_demo安全与C10话术.md §五。
                result = verify_message(req.message, req.context, on_step=on_step, use_memory=False)
            step_queue.put(_wrap({"type": "done", "result": result.model_dump(mode="json")}))
            logger.info(
                "[verify_stream] done request_id=%s elapsed_ms=%d",
                request_id,
                int((_time.time() - t0) * 1000),
            )
        except Exception as e:
            step_queue.put(_wrap({"type": "error", "message": str(e)}))
            logger.exception("[verify_stream] error request_id=%s", request_id)
        finally:
            _pipeline_progress.queue = _prev_queue

    thread = threading.Thread(target=run_pipeline, daemon=True)
    thread.start()

    import asyncio

    async def event_generator():
        yield f"data: {json.dumps(_wrap({'type': 'started'}), ensure_ascii=False)}\n\n"
        while True:
            try:
                # 同步 queue.get 在 worker 线程跑，避免阻塞 asyncio loop
                item = await asyncio.to_thread(step_queue.get, True, 0.5)
            except queue.Empty:
                if not thread.is_alive():
                    break
                yield f"data: {json.dumps(_wrap({'type': 'heartbeat'}), ensure_ascii=False)}\n\n"
                continue
            # _pipeline_progress 推的格式是 {event, data}；展开成 {type, ...data}
            if isinstance(item, dict) and "event" in item and "data" in item:
                flat = {"type": item["event"], **(item.get("data") or {})}
                item = _wrap(flat)
            yield f"data: {json.dumps(item, ensure_ascii=False, default=str)}\n\n"
            if item.get("type") in ("done", "error"):
                break

    return StreamingResponse(event_generator(), media_type="text/event-stream")


class FeedbackRequest(BaseModel):
    case_id: int
    feedback_type: str
    content: str = ""


_NEGATIVE_FEEDBACK_TYPES = {
    "incorrect",
    "disagree",
    "不对",
    "有误",
    "判错了",
    "错误",
    "wrong",
}


@app.post("/api/feedback")
async def submit_feedback(req: FeedbackRequest):
    memory = get_memory_store()
    if not memory:
        return {"status": "error", "message": "记忆系统不可用"}
    memory.save_feedback(req.case_id, req.feedback_type, req.content)
    invalidated = 0
    if req.feedback_type in _NEGATIVE_FEEDBACK_TYPES:
        invalidated = memory.invalidate_memory_by_case(req.case_id)
    return {"status": "ok", "memory_invalidated": invalidated}


class DispositionRequest(BaseModel):
    case_id: int
    action: str  # "copy" | "report" | "track" | "clarify"


@app.post("/api/disposition")
async def record_disposition(req: DispositionRequest):
    memory = get_memory_store()
    if not memory:
        return {"status": "error", "message": "记忆系统不可用"}
    memory.save_feedback(req.case_id, f"disposition:{req.action}")
    return {"status": "ok", "action": req.action}


@app.get("/api/stats")
async def stats():
    memory = get_memory_store()
    if not memory:
        return {"error": "记忆系统不可用"}
    return memory.get_stats()


@app.get("/api/cases")
async def list_cases():
    memory = get_memory_store()
    if not memory:
        return []
    return memory.get_cases()


@app.get("/api/report_urls")
async def report_urls():
    return {
        "urls": [
            {"name": "中国互联网联合辟谣平台", "url": "https://www.piyao.org.cn/"},
            {"name": "12321 网络不良信息举报", "url": "https://www.12321.cn/"},
            {"name": "360 安全举报", "url": "https://jubao.360.cn/"},
            {"name": "国家网信办举报中心", "url": "https://www.12377.cn/"},
        ]
    }


@app.get("/api/lifecycles")
async def list_lifecycles():
    memory = get_memory_store()
    if not memory:
        return []
    return memory.get_all_lifecycles()


@app.get("/api/lifecycle/{case_id}")
async def get_lifecycle(case_id: int):
    memory = get_memory_store()
    if not memory:
        return {"error": "记忆系统不可用"}
    lc = memory.get_lifecycle(case_id)
    if not lc:
        return {"error": "not found"}
    return lc


_VALID_LIFECYCLE_STATES = {
    "tracking",
    "escalated",
    "self_corrected",
    "resolved",
    "expired",
}


class LifecycleAdvanceRequest(BaseModel):
    case_id: int
    new_state: str
    detail: str = ""


@app.post("/api/lifecycle/advance")
async def advance_lifecycle(req: LifecycleAdvanceRequest):
    if req.new_state not in _VALID_LIFECYCLE_STATES:
        from fastapi import HTTPException

        raise HTTPException(
            status_code=400,
            detail=f"无效状态: {req.new_state}，允许: {', '.join(sorted(_VALID_LIFECYCLE_STATES))}",
        )
    memory = get_memory_store()
    if not memory:
        return {"status": "error", "message": "记忆系统不可用"}
    memory.advance_lifecycle(req.case_id, req.new_state, req.detail)
    return {"status": "ok", "case_id": req.case_id, "new_state": req.new_state}


class SelfCorrectionRequest(BaseModel):
    case_id: int


@app.post("/api/self_correction")
async def self_correction(req: SelfCorrectionRequest):
    """发信人主动自纠：跳过升级，直接进入记忆。"""
    ok = report_self_correction(req.case_id)
    if not ok:
        return {"status": "error", "message": "案例不在追踪中"}
    return {"status": "ok", "case_id": req.case_id, "action": "self_corrected"}


@app.get("/api/tracking")
async def tracking_cases():
    """返回当前正在追踪的案例（含自纠话术和升级消息）。"""
    memory = get_memory_store()
    if not memory:
        return []
    return memory.get_tracking_cases()


@app.get("/api/escalation_log")
async def escalation_log():
    """返回已升级的案例日志（含应发的群消息）。"""
    return _escalation_log


@app.post("/api/check_escalations")
async def manual_check_escalations():
    """手动触发升级检查（Demo 用，不等定时器）。"""
    escalated = check_escalations()
    _escalation_log.extend(escalated)
    if len(_escalation_log) > _MAX_ESCALATION_LOG:
        _escalation_log[:] = _escalation_log[-_MAX_ESCALATION_LOG:]
    return {"escalated_count": len(escalated), "cases": escalated}


class RecheckRequest(BaseModel):
    case_id: int


@app.post("/api/recheck")
def recheck_case(req: RecheckRequest):
    """闭环复查：取原始消息重新核查，对比新旧判定，回写原案例 + 失效旧记忆。"""
    memory = get_memory_store()
    if not memory:
        return {"error": "记忆系统不可用"}
    old_case = memory.get_case_by_id(req.case_id)
    if not old_case:
        return {"error": "案例不存在", "case_id": req.case_id}

    new_result = verify_message(old_case["original_message"], use_memory=False)
    new_verdict = new_result.overall_verdict.value
    old_verdict = old_case["overall_verdict"]
    changed = new_verdict != old_verdict

    if changed:
        memory.invalidate_memory_by_case(req.case_id)
        memory.update_case_verdict(req.case_id, new_verdict, new_result.summary)
        logger.info(
            "[Recheck] case_id=%d 判定变更 %s→%s，旧记忆已失效",
            req.case_id,
            old_verdict,
            new_verdict,
        )

    return {
        "case_id": req.case_id,
        "original_message": old_case["original_message"][:100],
        "old_verdict": old_verdict,
        "new_verdict": new_verdict,
        "changed": changed,
        "new_result": new_result,
    }


@app.get("/api/case/{case_id}")
async def get_case_detail(case_id: int):
    """单案例详情（含声明、证据、反馈）。"""
    memory = get_memory_store()
    if not memory:
        return {"error": "记忆系统不可用"}
    case = memory.get_case_by_id(case_id)
    if not case:
        return {"error": "not found"}
    return case


# ── 闭环动作 + ClaimReview 端点 ──


@app.post("/api/verify_full")
def verify_full(req: VerifyRequest):
    """完整核查：返回判定 + 闭环动作 + ClaimReview JSON-LD。"""
    if len(req.message) > _MAX_MESSAGE_LENGTH:
        from fastapi import HTTPException

        raise HTTPException(status_code=400, detail=f"消息长度不能超过 {_MAX_MESSAGE_LENGTH} 字")
    result = verify_message(req.message, req.context)
    return {
        "verification": result.model_dump(),
        "actions": result.actions,
        "claimreviews": result.claimreviews,
    }


@app.get("/api/actions")
async def list_open_actions():
    """返回所有待处理的闭环动作。"""
    from .closed_loop import ClosedLoopStore

    store = ClosedLoopStore()
    return store.get_open_actions()


class ActionUpdateRequest(BaseModel):
    action_id: str
    status: str  # "sent" | "resolved" | "dismissed"


@app.post("/api/actions/update")
async def update_action_status(req: ActionUpdateRequest):
    """更新闭环动作状态，并返回可渲染的处置回执（闭环在用户侧合上的那一下）。"""
    from .closed_loop import (
        ActionStatus,
        ActionType,
        ClosedLoopStore,
        build_disposition_receipt,
    )

    valid = {s.value for s in ActionStatus}
    if req.status not in valid:
        from fastapi import HTTPException

        raise HTTPException(status_code=400, detail=f"无效状态，允许: {', '.join(sorted(valid))}")
    store = ClosedLoopStore()
    row = store.get_action(req.action_id)  # 取处置前的动作信息，回执才能说人话
    store.update_status(req.action_id, ActionStatus(req.status))

    rec_action = None
    claim_text = ""
    if row:
        claim_text = row.get("claim_text", "")
        raw = row.get("recommended_action")
        if raw:
            try:
                rec_action = ActionType(raw)
            except ValueError:
                rec_action = None
    receipt = build_disposition_receipt(
        req.action_id,
        ActionStatus(req.status),
        claim_text=claim_text,
        recommended_action=rec_action,
    )
    return {
        "status": "ok",
        "action_id": req.action_id,
        "new_status": req.status,
        "receipt": receipt.model_dump(),
    }


class ImageVerifyRequest(BaseModel):
    image_url: str = ""
    image_path: str = ""
    context: str = ""


@app.post("/api/verify_image")
def verify_image_endpoint(req: ImageVerifyRequest):
    """图片核查：提取图片文字 → 送入核查流水线。"""
    from .image import verify_image

    source = req.image_url or req.image_path
    if not source:
        from fastapi import HTTPException

        raise HTTPException(status_code=400, detail="必须提供 image_url 或 image_path")
    result = verify_image(source, context=req.context)
    if result.get("verify_result"):
        result["verify_result"] = result["verify_result"].model_dump(mode="json")
    return result


class ImageSourceRequest(BaseModel):
    image_url: str = ""
    page_url: str = ""  # 图片所在网页（备用，暂未使用）
    allow_live: bool = True  # demo 台上可设 False 强制只走缓存，绝不现场联网裸调


@app.post("/api/verify_image_source")
def verify_image_source_endpoint(req: ImageSourceRequest):
    """图片溯源（旧图新用 / 张冠李戴反查）：百度识图 + 缓存 → 推理卡片可渲染对象。

    信任边界：image_url 是外部输入，reverse_search_image 内部 `_is_safe_public_url`
    做 SSRF 防护（拒内网/环回/非 http(s)），下载有大小上限、失败 fail-soft。
    溯源**不判真假**（overall_verdict 恒中性"无法核实"），只给"该图在全网哪些网页出现过"。
    """
    from .reverse_image import build_source_card, reverse_search_image

    if not req.image_url:
        from fastapi import HTTPException

        raise HTTPException(status_code=400, detail="必须提供 image_url")
    result = reverse_search_image(image_url=req.image_url, allow_live=req.allow_live)
    return build_source_card(result, image_url=req.image_url)


@app.post("/api/claimreview")
def export_claimreview(req: VerifyRequest):
    """核查并导出标准 ClaimReview JSON-LD（可对接 360 搜索生态等平台）。"""
    if len(req.message) > _MAX_MESSAGE_LENGTH:
        from fastapi import HTTPException

        raise HTTPException(status_code=400, detail=f"消息长度不能超过 {_MAX_MESSAGE_LENGTH} 字")
    result = verify_message(req.message, req.context)
    return {
        "@context": "https://schema.org",
        "@graph": result.claimreviews,
    }


# ── 社会层面闭环 · 谣言公示墙（公开辟谣大厅）──
@app.get("/api/board")
def get_board(limit: int = 50):
    """公示墙 feed + 统计：把零散核查汇成社会层面的实时辟谣公示墙。

    数据 = 真实核查库（ClosedLoopStore，如有）+ 种子样本（保证不空表）。
    脱敏：库里本就只存谣言文本 + 判定 + 证据，无任何用户身份。
    """
    from .closed_loop import ClosedLoopStore
    from .public_board import board_stats, get_public_board

    rows: list[dict] = []
    try:
        rows = ClosedLoopStore().get_open_actions()
    except Exception:
        logger.warning("[Board] 读取核查库失败，仅用种子样本", exc_info=True)
    # 全量算统计（官方辟谣库可达数千条），feed 只返回前 limit 条
    full = get_public_board(store_rows=rows, limit=1_000_000)
    feed = full[:limit]
    return {
        "items": [it.model_dump() for it in feed],
        "stats": board_stats(full),
    }


@app.get("/board", response_class=HTMLResponse)
def board_page():
    """公开辟谣大厅雏形页面（社会层面闭环 demo）。"""
    from fastapi.responses import FileResponse

    page = BASE_DIR / "社会闭环雏形" / "公开辟谣大厅.html"
    if page.exists():
        return FileResponse(str(page))
    return HTMLResponse("<h1>公示墙雏形未找到</h1>", status_code=404)

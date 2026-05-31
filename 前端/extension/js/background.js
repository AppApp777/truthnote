// TruthNote background.js — SSE 流式核查 + 全链路日志落盘
const MAX_TRACE_ENTRIES = 200;       // chrome.storage 里最多保留多少次核查
const MAX_EVENTS_PER_TRACE = 500;    // 单次核查最多保留多少事件

// ===== 运行时设置：设置页 options.js 写进 chrome.storage.local 的 "tn_settings" =====
// 只放产品配置：后端地址 / 国产分层模型 / 鉴权 token。兜底默认 = 历史硬编码值，
// 读不到/出错一律退回这里，保证开箱即用。
const TN_SETTINGS_KEY = "tn_settings";
// 右键菜单 id（顶部声明，供下方 rebuildContextMenus / onClicked 共用；放顶部避免与早期调用产生 TDZ）
const TN_IMAGE_MENU_ID = "truthnote-verify-image";
const TN_TEXT_MENU_ID = "truthnote-verify-text";
const TN_DEFAULTS = {
  backendBaseUrl: "http://localhost:8000",
  backendToken: "",
  defaultModel: "",
  strongModel: "",
  contextMenuCheck: true, // 右键「核查选中文字」菜单开关（设置页可关；缺省=开）
  heartbeatTimeoutMs: 10000 // 10 秒没收到事件视为挂死（界面暂无此项，缺省即用兜底）
};
let TN_SETTINGS = Object.assign({}, TN_DEFAULTS);

// 读 storage 合并兜底；任何缺失/异常都落到默认值。MV3 worker 会休眠重启，故入口处还会再现读一次。
async function loadSettings() {
  let stored = {};
  try {
    if (chrome.storage && chrome.storage.local && typeof chrome.storage.local.get === "function") {
      const got = await chrome.storage.local.get(TN_SETTINGS_KEY);
      stored = (got && got[TN_SETTINGS_KEY]) || {};
    }
  } catch (_) { stored = {}; }
  const merged = Object.assign({}, TN_DEFAULTS, stored);
  const hb = Number(merged.heartbeatTimeoutMs); // 非正数一律退回默认（防设置页写脏）
  merged.heartbeatTimeoutMs = (Number.isFinite(hb) && hb > 0) ? hb : TN_DEFAULTS.heartbeatTimeoutMs;
  TN_SETTINGS = merged;
  return TN_SETTINGS;
}
loadSettings().then(() => { rebuildContextMenus(); }); // 初始化先读一次（worker 启动即有内存值）+ 按当前设置建右键菜单

// 设置页保存后即时刷新内存值，无需重载扩展。注册前判存在——回归 harness 的 fake chrome 没有 onChanged。
if (chrome.storage && chrome.storage.onChanged && chrome.storage.onChanged.addListener) {
  chrome.storage.onChanged.addListener((changes, area) => {
    // 设置改了→刷新内存值，再按新 contextMenuCheck 重建右键菜单（开关即时生效）
    if (area === "local" && changes && changes[TN_SETTINGS_KEY]) loadSettings().then(() => { rebuildContextMenus(); });
  });
}

// 后端地址：始终返回去尾斜杠的合法字符串（兜底默认）
function backendBase() {
  const u = String(TN_SETTINGS.backendBaseUrl || TN_DEFAULTS.backendBaseUrl).trim().replace(/\/+$/, "");
  return u || TN_DEFAULTS.backendBaseUrl;
}
// 鉴权头：token 非空才加 Authorization；空则不加（返回空对象）。绝不打印 token、不落 trace。
function authHeaders() {
  const t = String(TN_SETTINGS.backendToken || "").trim();
  return t ? { "Authorization": "Bearer " + t } : {};
}

function genRequestId() {
  return Date.now().toString(36) + "-" + Math.random().toString(36).slice(2, 8);
}

// 展示层 verdict → 三色徽章侧（true/false/unverified）。
// ⚠️ 这是「呈现归类」不是「打分」：把绝对化证伪产物（误导性/大部分不实）归到 false/warn 侧，
// 否则改判产物会被前端又折叠回「无法核实」，证伪诉求白做。
//   · 误导性信息 = 真假混杂，落 false 侧（banner 走「不实」），不再折叠进 unverified；
//   · 大部分不实 / 谣言 = false 侧；
//   · 无法核实 = unverified（唯一进 unverified 的口子：不给二元徽章）。
function mapVerdict(verdict) {
  if (!verdict) return "unverified";
  const v = String(verdict).toLowerCase();
  if (v === "属实" || v === "true" || v === "confirmed") return "true";
  if (v === "谣言" || v === "大部分不实" || v === "误导性信息"
      || v === "false" || v === "debunked" || v === "mostly_false" || v === "misleading") return "false";
  if (v === "部分属实" || v === "partly_true") return "unverified";
  if (v === "无法核实" || v === "unverifiable" || v === "unverified") return "unverified";
  return "unverified";
}

// 后端 Step4.5 非裁定道把 NonAdjudicatedAction 序列化进 done 事件的 claim 对象，
// 字段 unverifiable_reason{code, code_label, detail, blocked_condition, verify_where}（snake_case）。
// 这里转 camelCase；缺字段一律给空串，渲染端按"有值才渲染"兜底，绝不报错。
function toUnverifiableReason(r) {
  if (!r || typeof r !== "object") return null;
  const reason = {
    code: r.code || "",
    codeLabel: r.code_label || "",
    detail: r.detail || "",
    blockedCondition: r.blocked_condition || "",
    verifyWhere: r.verify_where || ""
  };
  // 全空（后端没给）→ 返回 null，渲染端跳过，不占首屏
  if (!reason.codeLabel && !reason.detail && !reason.blockedCondition && !reason.verifyWhere) return null;
  return reason;
}

function transformResponse(data) {
  const claims = (data.claims || []).map(c => ({
    text: c.claim?.text || c.claim || "",
    verdict: mapVerdict(c.verdict),
    verdictRaw: c.verdict || "",
    confidence: c.confidence || 0,
    evidence: (c.evidence_chain || []).map(e => ({
      title: e.title || "",
      snippet: e.snippet || "",
      url: e.url || ""
    })),
    reasoning: c.reasoning || "",
    unverifiableReason: toUnverifiableReason(c.unverifiable_reason)
  }));
  const counts = { true: 0, false: 0, unverified: 0 };
  claims.forEach(c => counts[c.verdict]++);
  // 闭环动作（后端 closed_loop.py 的 actions[]）→ 归一成前端 camelCase；与真契约字段一致
  const actions = (data.actions || []).map(a => ({
    claimText: a.claim_text || "",
    verdict: a.verdict || "",
    recommendedAction: a.recommended_action || "",
    correctionCard: a.correction_card || "",
    reportLinks: Array.isArray(a.report_links) ? a.report_links : [],
    officialChannels: Array.isArray(a.official_channels) ? a.official_channels : [],
    subscription: a.subscription && Object.keys(a.subscription).length ? a.subscription : null
  }));
  return {
    original: data.original_message || "",
    claims,
    counts,
    summary: data.summary || "",
    friendlyReply: data.friendly_reply || "",
    reasoningChain: data.reasoning_chain || [],
    headlineNote: data.headline_note || "",
    actions
  };
}

// ===== 日志落盘 =====
async function loadTraces() {
  const o = await chrome.storage.local.get("tn_traces");
  return Array.isArray(o.tn_traces) ? o.tn_traces : [];
}

async function saveTraces(traces) {
  await chrome.storage.local.set({ tn_traces: traces });
}

async function appendTraceEvent(requestId, event) {
  const traces = await loadTraces();
  let trace = traces.find(t => t.request_id === requestId);
  if (!trace) {
    trace = {
      request_id: requestId,
      started_at: Date.now(),
      message: event.message_preview || "",
      tab_url: event.tab_url || "",
      status: "running",
      events: [],
      result: null,
      error: null
    };
    traces.unshift(trace);
  }
  trace.events.push(event);
  if (trace.events.length > MAX_EVENTS_PER_TRACE) {
    trace.events = trace.events.slice(-MAX_EVENTS_PER_TRACE);
  }
  if (event.type === "done") {
    trace.status = "done";
    trace.finished_at = Date.now();
    trace.result = event.result || null;
  } else if (event.type === "error") {
    trace.status = "error";
    trace.finished_at = Date.now();
    trace.error = event.message || "";
  } else if (event.type === "timeout") {
    trace.status = "timeout";
    trace.finished_at = Date.now();
  }
  while (traces.length > MAX_TRACE_ENTRIES) traces.pop();
  await saveTraces(traces);
}

// ===== SSE 流式核查（POST /api/verify_stream） =====
async function streamVerify(text, context, tabId, tabUrl) {
  await loadSettings(); // worker 重启后 onChanged 不补发历史，入口处现读一次
  const requestId = genRequestId();
  const startEvent = {
    type: "request",
    request_id: requestId,
    timestamp: Date.now(),
    message_preview: text.slice(0, 80),
    tab_url: tabUrl
  };
  await appendTraceEvent(requestId, startEvent);
  notifyTab(tabId, { type: "TN_STREAM", event: startEvent });

  let timeoutTimer = null;
  const resetTimeout = () => {
    clearTimeout(timeoutTimer);
    timeoutTimer = setTimeout(async () => {
      const ev = { type: "timeout", request_id: requestId, timestamp: Date.now(), message: `超过 ${TN_SETTINGS.heartbeatTimeoutMs}ms 未收到事件` };
      await appendTraceEvent(requestId, ev);
      notifyTab(tabId, { type: "TN_STREAM", event: ev });
    }, TN_SETTINGS.heartbeatTimeoutMs);
  };

  try {
    const resp = await fetch(`${backendBase()}/api/verify_stream`, {
      method: "POST",
      headers: Object.assign(
        { "Content-Type": "application/json", "Accept": "text/event-stream" },
        authHeaders()
      ),
      body: JSON.stringify({
        message: text,
        context: context || "",
        request_id: requestId,
        default_model: TN_SETTINGS.defaultModel || "", // 空串=不覆盖（main.py VerifyRequest）
        strong_model: TN_SETTINGS.strongModel || ""
      })
    });
    if (!resp.ok || !resp.body) {
      const errEv = { type: "error", request_id: requestId, timestamp: Date.now(), message: `HTTP ${resp.status}` };
      await appendTraceEvent(requestId, errEv);
      notifyTab(tabId, { type: "TN_STREAM", event: errEv });
      return;
    }
    resetTimeout();
    const reader = resp.body.getReader();
    const decoder = new TextDecoder("utf-8");
    let buf = "";
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      let idx;
      while ((idx = buf.indexOf("\n\n")) !== -1) {
        const chunk = buf.slice(0, idx);
        buf = buf.slice(idx + 2);
        const line = chunk.split("\n").find(l => l.startsWith("data:"));
        if (!line) continue;
        const json = line.slice(5).trim();
        if (!json) continue;
        let ev;
        try { ev = JSON.parse(json); }
        catch (e) { console.warn("[TN] SSE parse failed:", json.slice(0, 120), e); continue; }
        ev.timestamp = ev.timestamp || Date.now();
        ev.request_id = ev.request_id || requestId;
        resetTimeout();
        await appendTraceEvent(requestId, ev);
        notifyTab(tabId, { type: "TN_STREAM", event: ev });
        if (ev.type === "done") {
          ev.data = transformResponse(ev.result || {});
          notifyTab(tabId, { type: "TN_STREAM_DONE", request_id: requestId, data: ev.data });
        }
      }
    }
    clearTimeout(timeoutTimer);
  } catch (err) {
    clearTimeout(timeoutTimer);
    const ev = { type: "error", request_id: requestId, timestamp: Date.now(), message: err?.message || String(err) };
    await appendTraceEvent(requestId, ev);
    notifyTab(tabId, { type: "TN_STREAM", event: ev });
  }
}

// ===== 图片溯源（右键图片 → 反查"旧图新用"，POST /api/verify_image_source） =====
async function streamImageSource(srcUrl, tabId, pageUrl) {
  await loadSettings(); // 同 streamVerify：入口现读一次
  const requestId = genRequestId();
  const startEvent = { type: "request", request_id: requestId, timestamp: Date.now(), message_preview: "[图片溯源] " + (srcUrl || "").slice(0, 80), tab_url: pageUrl || "" };
  await appendTraceEvent(requestId, startEvent);
  // 图片右键由 background 发起，content 需要被动开面板（文字流是 content 自己开的）
  notifyTab(tabId, { type: "TN_IMAGE_START", request_id: requestId, srcUrl });

  // 一次性 POST /api/verify_image_source → 适配成 transformResponse 形状 → 复用文字渲染链
  try {
    const resp = await fetch(`${backendBase()}/api/verify_image_source`, {
      method: "POST",
      headers: Object.assign({ "Content-Type": "application/json" }, authHeaders()),
      body: JSON.stringify({ image_url: srcUrl, page_url: pageUrl || "", request_id: requestId })
    });
    if (!resp.ok) {
      const errEv = { type: "error", request_id: requestId, timestamp: Date.now(), message: `HTTP ${resp.status}` };
      await appendTraceEvent(requestId, errEv);
      notifyTab(tabId, { type: "TN_STREAM", event: errEv });
      return;
    }
    const raw = await resp.json();
    const data = transformImageSource(raw, srcUrl);
    notifyTab(tabId, { type: "TN_STREAM_DONE", request_id: requestId, data });
  } catch (err) {
    const ev = { type: "error", request_id: requestId, timestamp: Date.now(), message: "无法连接服务：" + (err?.message || err) };
    await appendTraceEvent(requestId, ev);
    notifyTab(tabId, { type: "TN_STREAM", event: ev });
  }
}

// 真后端适配器：把图片溯源端点的返回塑成前端渲染链认的形状（transformResponse 同形）。
// 兼容两种后端契约：① verify_stream 风格(claims[].evidence_chain) 直接复用 transformResponse；
//                  ② build_source_card 风格(overall_verdict + claims[].evidence[].source_url) 手工映射。
function transformImageSource(raw, srcUrl) {
  raw = raw || {};
  // verify_stream 风格：交给现成 transformResponse
  if (Array.isArray(raw.claims) && raw.claims.some(c => c && (c.evidence_chain || c.claim))) {
    const data = transformResponse(raw);
    if (!data.original) data.original = "[图片溯源] " + (srcUrl || "");
    return data;
  }
  // build_source_card 风格：手工映射 evidence[].source_url → 渲染链的 evidence[].url
  const srcClaims = Array.isArray(raw.claims) ? raw.claims : [];
  const claims = srcClaims.map(c => ({
    text: c.text || c.claim || "图片溯源",
    verdict: mapVerdict(c.verdict || raw.overall_verdict),
    verdictRaw: c.verdict || raw.overall_verdict || "",
    confidence: c.confidence || 0,
    evidence: (c.evidence || []).map(e => ({
      title: e.title || e.source || "来源网页",
      snippet: e.snippet || "",
      url: e.source_url || e.url || ""
    })),
    reasoning: c.reasoning || ""
  }));
  const counts = { true: 0, false: 0, unverified: 0 };
  claims.forEach(c => counts[c.verdict]++);
  const actions = (raw.actions || []).map(a => ({
    claimText: a.claim_text || "",
    verdict: a.verdict || "",
    recommendedAction: a.recommended_action || "",
    correctionCard: a.correction_card || "",
    reportLinks: Array.isArray(a.report_links) ? a.report_links : [],
    officialChannels: Array.isArray(a.official_channels) ? a.official_channels : [],
    subscription: a.subscription && Object.keys(a.subscription).length ? a.subscription : null
  }));
  return {
    original: raw.original_message || ("[图片溯源] " + (srcUrl || "")),
    claims,
    counts,
    summary: raw.summary || "",
    friendlyReply: raw.friendly_reply || "",
    reasoningChain: raw.reasoning_chain || [],
    headlineNote: raw.headline_note || "",
    actions
  };
}

function notifyTab(tabId, message) {
  if (!tabId) return;
  try {
    chrome.tabs.sendMessage(tabId, message, () => { void chrome.runtime.lastError; });
  } catch (_) { /* tab closed */ }
}

// ===== 消息路由 =====
chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
  if (msg.type === "VERIFY") {
    const tabId = sender.tab?.id;
    const tabUrl = sender.tab?.url || "";
    streamVerify(msg.text, msg.context, tabId, tabUrl);
    sendResponse({ ok: true, streaming: true });
    return;
  }

  if (msg.type === "GET_SELECTION") {
    chrome.tabs.query({ active: true, currentWindow: true }, tabs => {
      if (!tabs[0]) return sendResponse({ text: "" });
      chrome.tabs.sendMessage(tabs[0].id, { type: "GET_SELECTION" }, res => {
        sendResponse(res || { text: "" });
      });
    });
    return true;
  }

  if (msg.type === "TN_GET_TRACES") {
    loadTraces().then(traces => sendResponse({ ok: true, traces }));
    return true;
  }

  if (msg.type === "TN_CLEAR_TRACES") {
    saveTraces([]).then(() => sendResponse({ ok: true }));
    return true;
  }
});

// ===== 右键菜单：图片溯源（始终在）+ 文字核查（受 contextMenuCheck 开关控制）=====
// 统一用 removeAll 再按当前设置重建——幂等，worker 重启 / 开关切换都收敛到正确状态，且不会重复 id 报错。
// 菜单 id 常量在文件顶部声明（TN_IMAGE_MENU_ID / TN_TEXT_MENU_ID）。
function rebuildContextMenus() {
  if (!chrome.contextMenus || typeof chrome.contextMenus.removeAll !== "function") return;
  chrome.contextMenus.removeAll(() => {
    void chrome.runtime.lastError;
    // 图片溯源（旧图新用反查）——独立功能，始终注册
    chrome.contextMenus.create({
      id: TN_IMAGE_MENU_ID,
      title: "用小T溯源这张图（查旧图新用）",
      contexts: ["image"]
    }, () => { void chrome.runtime.lastError; });
    // 文字核查——仅当设置页「右键菜单核查」未关（缺省=开）
    if (TN_SETTINGS.contextMenuCheck !== false) {
      chrome.contextMenus.create({
        id: TN_TEXT_MENU_ID,
        title: "用小T核查选中的这句话",
        contexts: ["selection"]
      }, () => { void chrome.runtime.lastError; });
    }
  });
}

chrome.runtime.onInstalled.addListener(details => {
  if (details.reason === "install") {
    chrome.tabs.create({ url: "pages/onboarding.html" });
  }
  rebuildContextMenus(); // 安装/更新时按当前设置建菜单
});

// 右键菜单点击：图片→溯源；选中文字→核查（发 START_VERIFY 给 content.js，走和选中触点同一条核查通道）
if (chrome.contextMenus && chrome.contextMenus.onClicked) {
  chrome.contextMenus.onClicked.addListener((info, tab) => {
    if (!tab || !tab.id) return;
    if (info.menuItemId === TN_IMAGE_MENU_ID) {
      streamImageSource(info.srcUrl || "", tab.id, info.pageUrl || "");
    } else if (info.menuItemId === TN_TEXT_MENU_ID) {
      const text = (info.selectionText || "").trim();
      if (text && chrome.tabs && typeof chrome.tabs.sendMessage === "function") {
        chrome.tabs.sendMessage(tab.id, { type: "START_VERIFY", text }, () => { void chrome.runtime.lastError; });
      }
    }
  });
}

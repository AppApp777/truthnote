(() => {
  "use strict";

  const VERDICT_LABEL = { true: "查到了", false: "不对", unverified: "小T没查到" };
  const STEPS = [
    { key: "extract", label: "提取声明", done: "提取好了。" },
    { key: "search", label: "搜索证据", done: "找..." },
    { key: "verify", label: "交叉验证", done: "对一对。" },
    { key: "compose", label: "生成报告", done: "写。" }
  ];

  // 数字转中文（计数用，2→两），面板标题更顺嘴："2 条。看。" → "两条"
  function cnCount(n) {
    const cn = ["零", "一", "两", "三", "四", "五", "六", "七", "八", "九", "十"];
    return (n >= 0 && n <= 10) ? cn[n] : String(n);
  }
  function panelTitle(n) { return cnCount(n) + "条声明"; }

  let trigger = null;
  let fadeTimer = null;
  let widget = null;
  let panel = null;
  let widgetState = "idle";
  let lastResult = null;
  // 路演放大：把侧栏条框等比放大到全屏竖屏，让会场看清（比例不变，纯 transform: scale）
  let panelZoomed = false;
  let zoomBackdrop = null;
  let lastText = "";
  let currentRequestId = null;
  let streamEventBuffer = []; // 保存本次核查所有 step 事件，供面板二次展开时回放
  let explanationTypewriter = null; // 综合解释打字机句柄（要在 closePanel 时清掉）

  // ===== 打字机 + 极简 markdown 渲染 =====
  function escapeHTML(s) {
    return String(s).replace(/[&<>"']/g, ch => ({
      "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
    }[ch]));
  }

  // 把 verdict_explanation 的 markdown 转成简单 HTML（## / ### / **bold** / - 列表）
  function mdToHTML(md) {
    const lines = String(md || "").split("\n");
    const out = [];
    for (let raw of lines) {
      let l = escapeHTML(raw);
      l = l.replace(/^### (.+)$/, '<div class="tn-expl-h3">$1</div>');
      l = l.replace(/^## (.+)$/, '<div class="tn-expl-h2">$1</div>');
      l = l.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
      if (/^- /.test(raw)) {
        l = '<div class="tn-expl-li">• ' + l.slice(2) + '</div>';
      } else if (raw.trim() === "") {
        l = '<div class="tn-expl-br"></div>';
      } else if (!/^<div /.test(l)) {
        l = '<div class="tn-expl-p">' + l + '</div>';
      }
      out.push(l);
    }
    return out.join("");
  }

  // 字符级打字机：textContent 一个一个加，结束后切换到 innerHTML（带格式）
  function typewriter(el, plainText, htmlAtEnd, speed) {
    if (!el) return null;
    el.textContent = "";
    el.classList.add("tn-typing");
    let i = 0;
    const len = plainText.length;
    const tick = speed || 22;
    const timer = setInterval(() => {
      if (i >= len) {
        clearInterval(timer);
        el.classList.remove("tn-typing");
        if (htmlAtEnd != null) {
          el.innerHTML = htmlAtEnd;
        }
        return;
      }
      // 中文一字一闪，英文 / 数字加速 2 字一闪
      const ch = plainText[i++];
      el.appendChild(document.createTextNode(ch));
    }, tick);
    return { stop: () => clearInterval(timer) };
  }

  // ===== 初始化常驻挂件 =====
  function initWidget() {
    if (widget) return;

    widget = document.createElement("div");
    widget.className = "tn-widget tn-widget--idle";

    const pill = document.createElement("div");
    pill.className = "tn-widget-pill";

    const eyeL = document.createElement("span");
    eyeL.className = "tn-widget-eye";
    eyeL.id = "tn-eye-l";
    const eyeR = document.createElement("span");
    eyeR.className = "tn-widget-eye";
    eyeR.id = "tn-eye-r";
    const label = document.createElement("span");
    label.className = "tn-widget-label";
    label.id = "tn-widget-label";
    label.textContent = "";

    pill.appendChild(eyeL);
    pill.appendChild(eyeR);
    pill.appendChild(label);

    const progress = document.createElement("div");
    progress.className = "tn-widget-progress";
    progress.id = "tn-widget-progress";
    progress.style.display = "none";
    const bar = document.createElement("div");
    bar.className = "tn-widget-progress-bar";
    bar.id = "tn-progress-bar";
    progress.appendChild(bar);

    widget.appendChild(pill);
    widget.appendChild(progress);

    // 待机悬停提示：挂件待机态只有两只眼睛、没字，新用户（评委）不知这黑块是干嘛的。
    // 悬停给一句 on-brand 黑底直角提示，纯 CSS 控显隐、只在 .tn-widget--idle 出现
    //（checking/ready 时挂件自己有文案，不打架）；pointer-events:none 不抢交互。
    const tip = document.createElement("div");
    tip.className = "tn-widget-tip";
    tip.textContent = "选中可疑文字，小T 帮你查";
    widget.appendChild(tip);

    pill.addEventListener("click", e => {
      e.stopPropagation();
      if (widgetState === "ready" || widgetState === "checking") {
        togglePanel();
      }
    });

    document.body.appendChild(widget);
    setWidgetState("idle");
  }

  function setWidgetState(state, text) {
    widgetState = state;
    if (!widget) return;

    widget.className = "tn-widget tn-widget--" + state;

    const eyeL = widget.querySelector("#tn-eye-l");
    const eyeR = widget.querySelector("#tn-eye-r");
    const label = widget.querySelector("#tn-widget-label");
    const progressEl = widget.querySelector("#tn-widget-progress");

    eyeL.className = "tn-widget-eye";
    eyeR.className = "tn-widget-eye";

    switch (state) {
      case "idle":
        label.textContent = "";
        progressEl.style.display = "none";
        break;
      case "checking":
        eyeL.classList.add("tn-widget-eye--scan");
        label.textContent = text || "查...";
        progressEl.style.display = "";
        break;
      case "ready":
        label.textContent = text || "看。";
        progressEl.style.display = "none";
        break;
      case "error":
        eyeL.classList.add("tn-widget-eye--error");
        label.textContent = text || "没查成。";
        progressEl.style.display = "none";
        break;
    }
  }

  function setProgress(pct) {
    const bar = widget?.querySelector("#tn-progress-bar");
    if (bar) bar.style.transform = "scaleX(" + (Math.max(0, Math.min(100, pct)) / 100) + ")";
  }

  // ===== 结果面板 =====
  function togglePanel() {
    if (panel) {
      closePanel();
    } else {
      openPanel();
    }
  }

  function closePanel() {
    if (explanationTypewriter) {
      explanationTypewriter.stop();
      explanationTypewriter = null;
    }
    exitPanelZoom();              // 关面板时一并退出放大态（清遮罩 + 监听）
    if (panel && panel.parentNode) {
      panel.parentNode.removeChild(panel);
    }
    panel = null;
  }

  // ===== 路演放大（全屏竖屏，比例不变）=====
  // 原理：给 .tn-panel 套 transform: scale(var(--tn-zoom))，等比放大整张卡（文字是矢量，放大依旧清晰），
  // 居中铺满竖直方向。缩放因子按视口高/面板自然高算一次，窗口变化时重算。
  function togglePanelZoom() {
    if (!panel) return;
    panelZoomed ? exitPanelZoom() : enterPanelZoom();
    syncZoomButton();
  }

  function enterPanelZoom() {
    if (!panel || panelZoomed) return;
    panelZoomed = true;
    // 半透明遮罩：盖住页面、突出面板；点遮罩也能退出
    if (!zoomBackdrop) {
      zoomBackdrop = document.createElement("div");
      zoomBackdrop.className = "tn-zoom-backdrop";
      zoomBackdrop.addEventListener("click", () => { exitPanelZoom(); syncZoomButton(); });
      document.body.appendChild(zoomBackdrop);
    }
    panel.classList.add("tn-panel--zoom");
    fitPanelZoom();
    window.addEventListener("resize", fitPanelZoom);
  }

  function exitPanelZoom() {
    window.removeEventListener("resize", fitPanelZoom);
    if (zoomBackdrop) { zoomBackdrop.remove(); zoomBackdrop = null; }
    if (panel) {
      panel.classList.remove("tn-panel--zoom");
      panel.style.removeProperty("--tn-zoom");
    }
    panelZoomed = false;
  }

  // 量面板的自然尺寸（offsetWidth/Height 不受 transform 影响），算出"铺满竖直方向"的等比因子
  function fitPanelZoom() {
    if (!panel || !panelZoomed) return;
    const h = panel.offsetHeight || 480;
    const w = panel.offsetWidth || 340;
    const s = Math.min((window.innerHeight * 0.94) / h, (window.innerWidth * 0.96) / w);
    panel.style.setProperty("--tn-zoom", Math.max(1, s).toFixed(3));
  }

  // 面板每次重建（renderCheckingPanel 清了 innerHTML）后，让放大按钮文案与当前状态一致
  function syncZoomButton() {
    if (!panel) return;
    const btn = panel.querySelector(".tn-panel-zoom");
    if (btn) {
      btn.textContent = panelZoomed ? "还原" : "放大";
      btn.title = panelZoomed ? "还原大小" : "放大到全屏（路演用）";
    }
  }

  function openPanel() {
    closePanel();

    panel = document.createElement("div");
    panel.className = "tn-panel";

    if (widgetState === "checking") {
      renderCheckingPanel();
    } else if (widgetState === "ready" && lastResult) {
      // ready 状态：先渲染完整的过程（流水线 + 所有过程卡），再追加最终结论卡
      renderCheckingPanel();
      // 把头部标题改成最终态
      const titleEl = panel.querySelector(".tn-panel-title");
      if (titleEl) {
        const n = (lastResult.claims || []).length;
        titleEl.textContent = panelTitle(n);
      }
      updatePanelStep(4);
      appendFinalCard(lastResult);
    } else if (widgetState === "error") {
      renderErrorPanel();
    }

    document.body.appendChild(panel);
  }

  function renderCheckingPanel() {
    if (!panel) return;

    const header = buildPanelHeader("查...");

    const body = document.createElement("div");
    body.className = "tn-panel-loading";

    // 顶部 4 步流水线（圆点 + 标签）
    const steps = document.createElement("ul");
    steps.className = "tn-panel-steps";
    steps.id = "tn-panel-steps";
    const STAGE_LABELS = (window.TNRender && window.TNRender.STAGE_LABELS) || ["提取声明", "搜证据", "交叉验证", "写报告"];
    STAGE_LABELS.forEach((label, i) => {
      const li = document.createElement("li");
      li.className = "tn-panel-step";
      li.setAttribute("data-stage", String(i));
      const icon = document.createElement("span");
      icon.className = "tn-panel-step-icon tn-panel-step-icon--pending";
      const txt = document.createElement("span");
      txt.textContent = label;
      li.appendChild(icon);
      li.appendChild(txt);
      steps.appendChild(li);
    });

    // 卡片瀑布容器 —— 每个 step 事件来都 append 一张
    const cards = document.createElement("div");
    cards.className = "tn-panel-cards";
    cards.id = "tn-panel-cards";

    body.appendChild(steps);
    body.appendChild(cards);

    panel.appendChild(header);
    panel.appendChild(body);

    // 渲染已经收到的事件历史（如果用户中途展开面板，要把之前的卡片补上）
    streamEventBuffer.forEach(ev => appendEventCard(ev));
  }

  function renderResultPanel(data) {
    if (!panel) return;
    panel.innerHTML = "";

    const n = (data.claims || []).length;
    const header = buildPanelHeader(panelTitle(n));

    // 原文
    const orig = document.createElement("div");
    orig.className = "tn-panel-original";
    orig.textContent = "\u201C" + (data.original || lastText || "") + "\u201D";

    // 声明卡片
    const claimsContainer = document.createElement("div");
    claimsContainer.className = "tn-panel-claims";

    (data.claims || []).forEach(c => {
      const card = document.createElement("div");
      card.className = "tn-panel-claim tn-panel-claim--" + c.verdict;

      const line = document.createElement("div");
      line.className = "tn-panel-claim-line tn-panel-claim-line--" + c.verdict;

      const body = document.createElement("div");
      body.className = "tn-panel-claim-body";

      const tag = document.createElement("span");
      tag.className = "tn-panel-tag tn-panel-tag--" + c.verdict;
      tag.textContent = VERDICT_LABEL[c.verdict] || "小T没查到";

      const text = document.createElement("div");
      text.className = "tn-panel-claim-text";
      text.textContent = c.text;

      body.appendChild(tag);
      body.appendChild(text);

      if (c.evidence && c.evidence[0] && c.evidence[0].snippet) {
        const ev = document.createElement("div");
        ev.className = "tn-panel-claim-evidence";
        ev.textContent = c.evidence[0].snippet;
        // url 可能来自真后端/图片溯源（外部不可信输入）→ 过 safeUrl 白名单，挡 javascript:/data: 注入，
        // 与 renderFinalResult 一致。没过白名单就不渲染链接（只留 snippet 文本）。
        const _safeUrl = (window.TNRender && window.TNRender._utils && window.TNRender._utils.safeUrl) || (u => u);
        const _evHref = c.evidence[0].url ? _safeUrl(c.evidence[0].url) : "";
        if (_evHref) {
          const link = document.createElement("a");
          link.className = "tn-panel-claim-link";
          link.href = _evHref;
          link.target = "_blank";
          link.rel = "noopener";
          link.textContent = " → 看看";
          ev.appendChild(link);
        }
        body.appendChild(ev);
      } else if (c.reasoning) {
        const ev = document.createElement("div");
        ev.className = "tn-panel-claim-evidence";
        ev.textContent = c.reasoning;
        body.appendChild(ev);
      }

      card.appendChild(line);
      card.appendChild(body);
      claimsContainer.appendChild(card);
    });

    // 综合解释（评委杀器：6 维度 + PromoHealth + MessageFrame 编织的人话）
    const explainText = data.trace && data.trace.verdict_explanation;
    let explainEl = null;
    if (explainText) {
      explainEl = document.createElement("div");
      explainEl.className = "tn-panel-explanation";
      const head = document.createElement("div");
      head.className = "tn-panel-explanation-head";
      head.innerHTML = '<span class="tn-panel-eye"></span><span class="tn-panel-eye"></span><span class="tn-panel-explanation-title">小T 说</span>';
      explainEl.appendChild(head);
      const body = document.createElement("div");
      body.className = "tn-panel-explanation-body";
      explainEl.appendChild(body);
      // 启动打字机：plain text 一个个蹦，结束后切换到 markdown 渲染后的 HTML
      const plain = String(explainText).replace(/\*\*(.+?)\*\*/g, "$1");
      const html = mdToHTML(explainText);
      if (explanationTypewriter) explanationTypewriter.stop();
      explanationTypewriter = typewriter(body, plain, html, 22);
    }

    // 总结栏
    const summary = document.createElement("div");
    summary.className = "tn-panel-summary";
    const counts = document.createElement("div");
    counts.className = "tn-panel-counts";
    const ct = data.counts || {};
    if (ct.true > 0) counts.appendChild(makeCountItem("true", ct.true, "查到"));
    if (ct.false > 0) counts.appendChild(makeCountItem("false", ct.false, "不对"));
    if (ct.unverified > 0) counts.appendChild(makeCountItem("unverified", ct.unverified, "没查到"));
    const actions = document.createElement("div");
    actions.className = "tn-panel-actions";
    const fbBtn = document.createElement("button");
    fbBtn.className = "tn-panel-btn-ghost";
    fbBtn.textContent = "小T错了？";
    fbBtn.addEventListener("click", () => closePanel());
    actions.appendChild(fbBtn);
    summary.appendChild(counts);
    summary.appendChild(actions);

    panel.appendChild(header);
    panel.appendChild(orig);
    panel.appendChild(claimsContainer);
    if (explainEl) panel.appendChild(explainEl);
    panel.appendChild(summary);
  }

  function renderErrorPanel() {
    if (!panel) return;
    panel.innerHTML = "";

    const header = buildPanelHeader("没查成。");
    const body = document.createElement("div");
    body.style.cssText = "padding:24px 16px;text-align:center;";

    const eyes = document.createElement("div");
    eyes.style.cssText = "display:flex;gap:8px;justify-content:center;margin-bottom:16px;";
    const eL = document.createElement("span");
    eL.style.cssText = "width:16px;height:4px;background:#000;display:inline-block;";
    const eR = document.createElement("span");
    eR.style.cssText = "width:16px;height:16px;background:#000;display:inline-block;";
    eyes.appendChild(eL);
    eyes.appendChild(eR);

    const msg = document.createElement("div");
    msg.style.cssText = "font-size:13px;color:#666;margin-bottom:16px;";
    msg.textContent = "...没查成。再来？";

    const btn = document.createElement("button");
    btn.style.cssText = "background:#000;color:#fff;border:none;padding:6px 20px;font-size:13px;cursor:pointer;font-family:inherit;";
    btn.textContent = "再查";
    btn.addEventListener("click", () => {
      closePanel();
      if (lastText) startVerify(lastText);
    });

    body.appendChild(eyes);
    body.appendChild(msg);
    body.appendChild(btn);
    panel.appendChild(header);
    panel.appendChild(body);
  }

  function buildPanelHeader(titleText) {
    const header = document.createElement("div");
    header.className = "tn-panel-header";

    const left = document.createElement("div");
    left.style.cssText = "display:flex;align-items:center;gap:8px;";
    const eyes = document.createElement("div");
    eyes.className = "tn-panel-eyes";
    eyes.innerHTML = '<span class="tn-panel-eye"></span><span class="tn-panel-eye"></span>';
    const title = document.createElement("span");
    title.className = "tn-panel-title";
    title.textContent = titleText || "";
    left.appendChild(eyes);
    left.appendChild(title);

    // 右侧操作区：放大（路演看清）+ 关闭
    const ctrls = document.createElement("div");
    ctrls.style.cssText = "display:flex;align-items:center;gap:4px;";

    const zoomBtn = document.createElement("button");
    zoomBtn.className = "tn-panel-zoom";
    zoomBtn.type = "button";
    zoomBtn.textContent = panelZoomed ? "还原" : "放大";
    zoomBtn.title = panelZoomed ? "还原大小" : "放大到全屏（路演用）";
    zoomBtn.addEventListener("click", e => {
      e.stopPropagation();
      togglePanelZoom();
    });

    const closeBtn = document.createElement("button");
    closeBtn.className = "tn-panel-close";
    closeBtn.textContent = "×";
    closeBtn.addEventListener("click", e => {
      e.stopPropagation();
      closePanel();
    });

    ctrls.appendChild(zoomBtn);
    ctrls.appendChild(closeBtn);
    header.appendChild(left);
    header.appendChild(ctrls);
    return header;
  }

  function makeCountItem(type, count, label) {
    const span = document.createElement("span");
    span.className = "tn-panel-count-item";
    const dot = document.createElement("span");
    dot.className = "tn-panel-dot tn-panel-dot--" + type;
    span.appendChild(dot);
    span.appendChild(document.createTextNode(count + label));
    return span;
  }

  // ===== 核查流程（SSE 流式，通过 TNRender 引擎渲染） =====
  function startVerify(text) {
    lastText = text;
    lastResult = null;
    currentRequestId = null;
    streamEventBuffer = [];

    // 挂件"接住"反馈：先膨胀闪一下 + 标签显「收到」200ms
    triggerWidgetCatchAnimation();
    setWidgetState("checking", "收到");
    setProgress(3);

    if (panel) {
      panel.innerHTML = "";
      renderCheckingPanel();
    }

    setTimeout(() => {
      if (widgetState === "checking") setWidgetState("checking", "提取声明...");
    }, 280);

    chrome.runtime.sendMessage({ type: "VERIFY", text }, () => {
      // background 已转流式，结果通过 TN_STREAM / TN_STREAM_DONE 推回来
    });
  }

  // 图片溯源：背景已发起反查（不再 sendMessage），content 只负责把面板开成 loading 态等流回来
  function startImageVerify(srcUrl, requestId) {
    lastText = srcUrl ? ("图片溯源：" + srcUrl) : "图片溯源";
    lastResult = null;
    currentRequestId = requestId;
    streamEventBuffer = [];

    triggerWidgetCatchAnimation();
    setWidgetState("checking", "溯源中...");
    setProgress(3);

    if (panel) {
      panel.innerHTML = "";
      renderCheckingPanel();
    } else {
      openPanel();
    }
  }

  function triggerWidgetCatchAnimation() {
    if (!widget) return;
    widget.classList.remove("tn-widget--catch");
    // 强制 reflow 重启动画
    void widget.offsetWidth;
    widget.classList.add("tn-widget--catch");
    setTimeout(() => widget && widget.classList.remove("tn-widget--catch"), 420);
  }

  // SSE 事件 → 更新挂件 + 面板（全部走 TNRender）
  function handleStreamEvent(ev) {
    if (!ev) return;
    if (ev.request_id) currentRequestId = ev.request_id;

    if (ev.type === "started") {
      setWidgetState("checking", "收到");
      setProgress(5);
      return;
    }
    if (ev.type === "heartbeat") {
      // 静默心跳，仅用于"还活着"信号
      return;
    }

    if (ev.type === "step") {
      streamEventBuffer.push(ev);
      const stage = (window.TNRender && window.TNRender.stageOfEvent(ev));
      const agentMeta = window.TNRender ? window.TNRender.resolveAgent(ev.agent) : null;
      const label = agentMeta ? (agentMeta.zh + "...") : ((ev.agent || "处理") + "...");
      setWidgetState("checking", label);
      if (stage != null && stage >= 0) {
        setProgress(Math.min(95, (stage + 1) * 22));
        updatePanelStep(stage);
      }
      appendEventCard(ev);
      return;
    }

    if (ev.type === "timeout") {
      streamEventBuffer.push(ev);
      setWidgetState("error", "可能卡了");
      appendEventCard(ev);
      return;
    }

    if (ev.type === "error") {
      streamEventBuffer.push(ev);
      setWidgetState("error", "没查成。");
      if (panel) {
        panel.innerHTML = "";
        renderErrorPanel();
      }
      return;
    }
  }

  function handleStreamDone(data) {
    setProgress(100);
    lastResult = data;
    const n = (data.claims || []).length;
    // 挂件标签前置真/假（命题人要"确定真/假"）；派生逻辑复用 TNRender.binaryLean，与结论卡一致
    const lean = window.TNRender && window.TNRender.binaryLean
      ? window.TNRender.binaryLean(data.counts)
      : null;
    // 空结果（真后端可能返回 0 声明）单独兜底，别误显示"证据暂不足"
    const widgetLabel = n === 0 ? "没查到。" : (lean ? (lean.zh + "。看。") : (n + "条。看。"));
    setWidgetState("ready", widgetLabel);
    // 不再清空 panel —— 保留所有过程卡，只在末尾追加最终结论
    if (panel) {
      const titleEl = panel.querySelector(".tn-panel-title");
      if (titleEl) titleEl.textContent = panelTitle(n);
      updatePanelStep(4);
      appendFinalCard(data);
    }
  }

  function appendFinalCard(data) {
    if (!panel || !window.TNRender) return;
    const container = panel.querySelector("#tn-panel-cards");
    if (!container) return;
    // 防止重复追加：先移除旧 final 卡
    const old = container.querySelector(".tn-card--final");
    if (old) old.remove();
    const card = window.TNRender.renderFinalResult(data);
    if (card) {
      // 追加前先量：用户此刻是否贴着底部跟读（容差 80px）。
      // 只有"本来就在底部"才把最终结论卡滚进视野；若用户已向上滚去看过程卡，
      // 就停在原处不拽走（挂件标签已前置真/假，知道查完了可自己滚下来）。
      const nearBottom =
        container.scrollHeight - container.scrollTop - container.clientHeight < 80;
      container.appendChild(card);
      if (nearBottom) {
        setTimeout(() => {
          card.scrollIntoView({ behavior: "smooth", block: "nearest" });
        }, 60);
      }
    }
  }

  // 用渲染引擎渲一张卡，追加到面板的 #tn-panel-cards 容器
  let _cardSeq = 0;
  function appendEventCard(ev) {
    if (!panel) return;
    const container = panel.querySelector("#tn-panel-cards");
    if (!container) return;
    if (!window.TNRender) return;
    const card = window.TNRender.render(ev);
    if (card) {
      card.style.setProperty("--i", String(_cardSeq++ % 8)); // 0-7 循环，驱动阴影微澜错相位
      container.appendChild(card);
      // 核实过程中【不】自动下滑：新卡追加在底部，视口停在用户正在看的位置不动。
      // 往下 append DOM 不会改变 scrollTop，浏览器天然保持眼前内容不跳——用户自己控制读到哪。
      // （命题人/用户要求：别在用户读某张过程卡时把视图拽到底部。请勿"修回"自动滚到底。）
    }
  }

  function updatePanelStep(activeIndex) {
    if (!panel) return;
    const steps = panel.querySelectorAll(".tn-panel-step");
    steps.forEach((step, i) => {
      const icon = step.querySelector(".tn-panel-step-icon");
      step.className = "tn-panel-step";
      icon.className = "tn-panel-step-icon";
      icon.textContent = "";
      if (i < activeIndex) {
        step.classList.add("tn-panel-step--done");
        icon.classList.add("tn-panel-step-icon--done");
        icon.textContent = "■";
      } else if (i === activeIndex) {
        step.classList.add("tn-panel-step--active");
        icon.classList.add("tn-panel-step-icon--active");
      } else {
        icon.classList.add("tn-panel-step-icon--pending");
      }
    });
    const statusEl = panel.querySelector("#tn-panel-status");
    if (statusEl && STEPS[activeIndex]) {
      statusEl.textContent = STEPS[activeIndex].done;
    }
  }

  // ===== 选中文字触点 =====
  document.addEventListener("mouseup", e => {
    try {
      if (e.target.closest && e.target.closest(".tn-trigger, .tn-widget, .tn-panel")) return;

      removeTrigger();

      const sel = window.getSelection();
      if (!sel || sel.rangeCount === 0) return;
      const text = sel.toString().trim();
      if (!text || text.length < 4) return;

      const range = sel.getRangeAt(0);
      const rect = range.getBoundingClientRect();
      if (rect.width === 0 && rect.height === 0) return;

      trigger = document.createElement("div");
      trigger.className = "tn-trigger";

      const eye1 = document.createElement("span");
      eye1.className = "tn-trigger-eye";
      const eye2 = document.createElement("span");
      eye2.className = "tn-trigger-eye";
      const label = document.createElement("span");
      label.textContent = "查";
      trigger.appendChild(eye1);
      trigger.appendChild(eye2);
      trigger.appendChild(label);

      const capturedText = text;
      trigger.addEventListener("click", ev => {
        ev.stopPropagation();
        ev.preventDefault();
        removeTrigger();
        initWidget();
        startVerify(capturedText);
        openPanel();
      });

      document.body.appendChild(trigger);
      // 贴边自动回收：必须在 append 进 DOM 后调用（要量真实宽高）。
      placeTrigger(trigger, rect);

      fadeTimer = setTimeout(() => {
        if (trigger) {
          trigger.style.opacity = "0";
          trigger.style.transition = "opacity 150ms";
          setTimeout(removeTrigger, 160);
        }
      }, 3000);
    } catch (_) {}
  }, true);

  document.addEventListener("mousedown", e => {
    if (e.target.closest && !e.target.closest(".tn-trigger, .tn-widget, .tn-panel")) {
      removeTrigger();
    }
  }, true);

  function removeTrigger() {
    clearTimeout(fadeTimer);
    if (trigger && trigger.parentNode) {
      trigger.parentNode.removeChild(trigger);
    }
    trigger = null;
  }

  // 触点贴边自动回收：选区靠近视口右/下边缘时，贴右下的默认位置会把触点挤出屏幕看不见。
  // 必须在元素 append 进 DOM 后调用（offsetWidth/Height 要真实布局）。规则：
  //   右边塞不下 → 翻到选区左侧；左侧也不够 → 硬夹到右边界内（留 6px 边距）。
  //   下边塞不下 → 翻到选区上方。最后统一 clamp 进 [边距, 视口-尺寸-边距]。
  function placeTrigger(elm, rect) {
    const M = 6;
    const tw = elm.offsetWidth || 50;
    const th = elm.offsetHeight || 24;
    const vw = window.innerWidth;
    const vh = window.innerHeight;
    let left = rect.right + 4;
    let top = rect.bottom + 4;
    if (left + tw > vw - M) {
      left = rect.left - tw - 4;          // 翻到选区左侧
      if (left < M) left = vw - tw - M;   // 左侧也不够 → 夹到右边界内
    }
    if (top + th > vh - M) {
      top = rect.top - th - 4;            // 翻到选区上方
    }
    elm.style.left = Math.max(M, Math.min(left, vw - tw - M)) + "px";
    elm.style.top = Math.max(M, Math.min(top, vh - th - M)) + "px";
  }

  // ===== Popup / 平台适配器 / background 消息接口 =====
  chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
    if (msg.type === "GET_SELECTION") {
      sendResponse({ text: window.getSelection()?.toString().trim() || "" });
    }
    if (msg.type === "START_VERIFY" && msg.text) {
      initWidget();
      startVerify(msg.text);
    }
    // 图片溯源由 background 发起（右键图片），content 不像选词那样自己开面板 → 这里被动开 loading 面板，
    // 后续 TN_STREAM / TN_STREAM_DONE 走和文字核查完全一样的渲染通道。
    if (msg.type === "TN_IMAGE_START") {
      initWidget();
      startImageVerify(msg.srcUrl || "", msg.request_id || null);
    }
    if (msg.type === "TN_STREAM" && msg.event) {
      handleStreamEvent(msg.event);
    }
    if (msg.type === "TN_STREAM_DONE" && msg.data) {
      handleStreamDone(msg.data);
    }
  });

  // 平台适配器通过 postMessage 委托核查
  window.addEventListener("message", e => {
    // 只接受本页面自己发来的消息，挡掉任意网页伪造 postMessage 强制触发核查（防滥用/盗刷后端配额）
    if (e.source !== window) return;
    if (e.data && e.data.type === "TN_START_VERIFY" && e.data.text) {
      initWidget();
      startVerify(e.data.text);
      openPanel();
    }
  });

  // ===== 页面加载后立刻显示挂件（常驻） =====
  function boot() {
    if (document.body) {
      initWidget();
    } else {
      document.addEventListener("DOMContentLoaded", () => initWidget());
    }
  }
  boot();
})();

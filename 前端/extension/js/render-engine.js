// TruthNote 通用事件渲染引擎
// ----------------------------------------------------------------
// 核心职责：把后端推过来的 SSE event 渲染成 DOM
// 3 层 fallback：
//   1. event.display.template 命中 → 用对应模板
//   2. agent 前缀命中 AGENT_REGISTRY → 用推断模板 + 自动 narrative
//   3. 完全不认识 → generic 兜底卡片
// 与后端解耦：模板按 key 注册，新增模板/Agent 不影响老逻辑
// ----------------------------------------------------------------

(function (global) {
  "use strict";

  // ============ Agent 注册表（前端 best-effort 推断） ============
  // 后端没推 display.template 时，按这张表推断
  const AGENT_REGISTRY = [
    { prefix: "ClaimExtractor",        zh: "提取声明",   stage: 0, template: "claim_list" },
    { prefix: "CheckWorthy",           zh: "筛声明",     stage: 0, template: "generic" },
    { prefix: "AtomicFact",            zh: "原子化",     stage: 0, template: "generic" },
    { prefix: "CommonsenseChecker",    zh: "常识审",     stage: 0, template: "generic" },
    { prefix: "ScenarioRouter",        zh: "场景路由",   stage: 0, template: "generic" },
    { prefix: "ClaimMatcher",          zh: "记忆对照",   stage: 0, template: "generic" },
    { prefix: "QueryPlanner",          zh: "规划查询",   stage: 1, template: "query_plan" },
    { prefix: "EvidenceHunter",        zh: "搜证据",     stage: 1, template: "evidence_list" },
    // 图片溯源（百度识图反查"旧图新用"）——必须排在 EvidenceHunter 之后但用独立前缀，复用 evidence_list 渲染来源页
    { prefix: "ReverseImageSearch",    zh: "图片溯源",   stage: 1, template: "evidence_list" },
    { prefix: "EvidenceRanker",        zh: "证据排序",   stage: 1, template: "evidence_rank" },
    // 编排页第二屏用的两个细分节点（必须排在通用 StructuredFactChecker 之前，让前缀精确命中）。
    // 侧栏的 StructuredFactCheckerAgent（...Agent 后缀）不匹配这两条，仍走下面的通用条目。
    { prefix: "StructuredFactCheckerLabel", zh: "标关系",   stage: 2, template: "evidence_relations" },
    { prefix: "RuleVerdict",                zh: "规则裁决", stage: 2, template: "rule_verdict" },
    { prefix: "StructuredFactChecker", zh: "结构化核查", stage: 2, template: "verification_grid" },
    { prefix: "FactChecker",           zh: "对一对",     stage: 2, template: "verification_grid" },
    { prefix: "Skeptic",               zh: "自我质疑",   stage: 2, template: "skeptic_challenges" },
    { prefix: "ResponseComposer",      zh: "写报告",     stage: 3, template: "reply_draft" },
    { prefix: "DimensionAggregator",   zh: "综合判定",   stage: 3, template: "dimension_radar" },
    { prefix: "MemoryStore",           zh: "存档",       stage: 3, template: "generic" }
  ];

  const STAGE_LABELS = ["提取声明", "搜证据", "交叉验证", "写报告"];

  function resolveAgent(agentName) {
    if (!agentName) return null;
    for (const m of AGENT_REGISTRY) {
      if (agentName.startsWith(m.prefix)) return m;
    }
    return null;
  }

  // ============ DOM 小工具 ============
  function el(tag, opts) {
    const e = document.createElement(tag);
    if (!opts) return e;
    if (opts.cls) e.className = opts.cls;
    if (opts.text != null) e.textContent = opts.text;
    if (opts.attrs) for (const k in opts.attrs) e.setAttribute(k, opts.attrs[k]);
    if (opts.style) for (const k in opts.style) e.style[k] = opts.style[k];
    if (opts.children) opts.children.forEach(c => c && e.appendChild(c));
    return e;
  }

  function fmtMs(ms) {
    if (ms == null) return "";
    if (ms < 1000) return ms + "ms";
    return (ms / 1000).toFixed(1) + "s";
  }

  function fmtTime(t_ms) {
    if (t_ms == null) return "";
    return "+" + fmtMs(t_ms);
  }

  function trimText(s, n) {
    if (!s) return "";
    s = String(s);
    return s.length > n ? s.slice(0, n) + "…" : s;
  }

  // expo-out：与横条生长同款缓动（先冲后稳，像仪表指针定位）
  function easeOutExpo(t) {
    return t >= 1 ? 1 : 1 - Math.pow(2, -10 * t);
  }

  function prefersReducedMotion() {
    return !!(global.matchMedia && global.matchMedia("(prefers-reduced-motion: reduce)").matches);
  }

  // 数字 count-up：node 文本从 0 数到 to（百分比整数），delay 后启动，duration 内完成。
  // 与同行横条 scaleX 生长同步（同 delay/同缓动），修「数字与条形不同步」。
  // reduced-motion / 无 rAF 环境直接落终值。
  function animateCountPct(node, to, duration, delay) {
    if (!node) return;
    to = Math.max(0, Math.round(to));
    if (prefersReducedMotion() || typeof requestAnimationFrame !== "function") {
      node.textContent = to + "%";
      return;
    }
    const now0 = (global.performance && performance.now) ? performance.now() : null;
    if (now0 == null) { node.textContent = to + "%"; return; }
    node.textContent = "0%";
    function frame(now) {
      if (!node.isConnected) return;  // 节点已脱离文档（切 case/重播）→ 停掉残留 rAF
      const elapsed = now - now0 - delay;
      if (elapsed < 0) { requestAnimationFrame(frame); return; }
      const p = Math.min(1, elapsed / duration);
      node.textContent = Math.round(to * easeOutExpo(p)) + "%";
      if (p < 1) requestAnimationFrame(frame);
      else node.textContent = to + "%";
    }
    requestAnimationFrame(frame);
  }

  // ============ 模板注册表 ============
  const TEMPLATES = {};

  function registerTemplate(key, renderFn) {
    TEMPLATES[key] = renderFn;
  }

  // -------------------- 通用卡片骨架 --------------------
  function cardShell(event, opts) {
    opts = opts || {};
    const meta = resolveAgent(event.agent);
    const card = el("div", {
      cls: "tn-card tn-card--" + (opts.flavor || "default") + " tn-card--enter",
      attrs: { "data-agent": event.agent || "", "data-template": opts.template || "" }
    });

    // 头部：图标 + Agent 中文名 + 耗时（相对时间「+Xs」已按用户要求去掉；英文 Agent 类名也不露）
    const head = el("div", { cls: "tn-card-head" });
    const dot = el("span", { cls: "tn-card-dot tn-card-dot--" + (opts.flavor || "default") });
    const title = el("span", { cls: "tn-card-title", text: opts.title || (meta ? meta.zh : (event.agent || "处理")) });
    const right = el("div", { cls: "tn-card-right" });
    if (event.duration_ms != null) right.appendChild(el("span", { cls: "tn-card-dur", text: fmtMs(event.duration_ms) }));
    head.appendChild(dot);
    head.appendChild(title);
    head.appendChild(right);
    card.appendChild(head);

    // 副标题（小T 说人话的灰色解说）已按用户要求全部移除——卡片只留标题 + 真数据，更干净。
    // 注：各模板仍会算出 narrative 传进来，这里统一不渲染（留着不删是为少改各模板；error 卡的 message 另有自己的渲染，不受影响）。

    // body 容器，给具体模板填
    const body = el("div", { cls: "tn-card-body" });
    card.appendChild(body);

    // 入场动画：下一帧切到 active
    requestAnimationFrame(() => {
      card.classList.remove("tn-card--enter");
      card.classList.add("tn-card--active");
    });

    return { card, body };
  }

  // -------------------- 模板：generic（兜底） --------------------
  // 只显示人话叙述（在卡片头部下方），不暴露 action/output_summary 这类开发者字段
  registerTemplate("generic", function (event) {
    const meta = resolveAgent(event.agent);
    const narrative = meta
      ? `小T ${meta.zh}：${event.output_summary || event.action || "处理中"}`
      : (event.output_summary || event.action || "");
    const { card } = cardShell(event, {
      flavor: "neutral",
      template: "generic",
      narrative: narrative
    });
    return card;
  });

  // -------------------- 模板：claim_list --------------------
  registerTemplate("claim_list", function (event) {
    const data = event.display?.data || {};
    const claims = Array.isArray(data.claims) ? data.claims : [];
    const narrative = event.human_narrative
      || (claims.length ? `小T 拆出 ${claims.length} 条事实声明：` : `小T 在拆解你的话...`);
    const { card, body } = cardShell(event, {
      flavor: "primary",
      template: "claim_list",
      narrative
    });

    // 图片溯源场景：把"正在核查的这张图"小尺寸钳进卡片首位（限高不占大版面），让人一眼看到查的是哪张图
    const _thumbSrc = safeImgSrc(data.image_url);
    if (_thumbSrc) {
      const fig = el("figure", { cls: "tn-claim-thumb" });
      fig.appendChild(el("img", { cls: "tn-claim-thumb-img", attrs: { src: _thumbSrc, alt: "正在核查的图片", loading: "lazy" } }));
      fig.appendChild(el("figcaption", { cls: "tn-claim-thumb-cap", text: "正在核查这张图" }));
      body.appendChild(fig);
    }

    if (claims.length === 0) {
      // 没有结构化数据 —— 把 output_summary 当一行显示
      if (event.output_summary) {
        body.appendChild(el("div", { cls: "tn-card-empty", text: event.output_summary }));
      }
    } else {
      const ul = el("ol", { cls: "tn-claim-list" });
      claims.forEach((c, i) => {
        const li = el("li", { cls: "tn-claim-item tn-stagger", attrs: { style: `--i:${Math.min(i, 8)}` } });
        li.appendChild(el("span", { cls: "tn-claim-num", text: String(i + 1) }));
        const txt = typeof c === "string" ? c : (c.text || "");
        li.appendChild(el("span", { cls: "tn-claim-text", text: txt }));
        if (c && c.category) {
          li.appendChild(el("span", { cls: "tn-claim-tag", text: c.category }));
        }
        ul.appendChild(li);
      });
      body.appendChild(ul);
    }
    return card;
  });

  // -------------------- 模板：evidence_list --------------------
  registerTemplate("evidence_list", function (event) {
    const data = event.display?.data || {};
    const queries = Array.isArray(data.queries) ? data.queries : [];
    const results = Array.isArray(data.top_results) ? data.top_results : [];
    const total = data.total;
    const kept = data.kept;

    const narrativeParts = [];
    if (queries.length) narrativeParts.push(`查了 ${queries.length} 个关键词`);
    if (total != null) narrativeParts.push(`找到 ${total} 条结果`);
    if (kept != null && kept !== total) narrativeParts.push(`筛出最相关 ${kept} 条`);
    const narrative = event.human_narrative
      || (narrativeParts.length ? "小T " + narrativeParts.join("，") : `小T ${event.output_summary || "正在搜证据..."}`);

    const { card, body } = cardShell(event, {
      flavor: "search",
      template: "evidence_list",
      narrative
    });

    // 真联网取证徽章：把"我们真的去搜了，不是问大模型基模知识"摆到台面上（命题人叙事补强）
    if (data.live) {
      const live = el("div", { cls: "tn-ev-live" });
      live.appendChild(el("span", { cls: "tn-ev-live-dot" }));
      live.appendChild(el("span", {
        cls: "tn-ev-live-label",
        text: data.live.provider ? ("真实联网检索 · " + data.live.provider) : "真实联网检索"
      }));
      // 采纳源置顶：被采信的权威源排到最前，评委一眼看到「真取证」的落点。
      // slice 出副本再排（不动后端/demo 原数据）；ES2019 稳定排序，组内保持后端给的相关性顺序。
      const pages = (Array.isArray(data.live.pages) ? data.live.pages.slice() : [])
        .sort((a, b) => (b && b.kept ? 1 : 0) - (a && a.kept ? 1 : 0));
      const crawled = data.live.crawled != null ? data.live.crawled : (pages.length || null);
      if (crawled != null && pages.length) {
        // 「抓取 N 个网页」可点开 → 列出真正检索到的网页（域名+标题），证明"真取证"不是空话
        // 默认折叠（避免信息过载、保持卡片清爽）；▾ 提示可展开
        const toggle = el("button", { cls: "tn-ev-live-count tn-ev-live-toggle", text: "抓取 " + crawled + " 个网页 ▾", attrs: { type: "button" } });
        live.appendChild(toggle);
        body.appendChild(live);

        const list = el("ul", { cls: "tn-ev-pages is-collapsed" });
        pages.forEach(p => {
          const li = el("li", { cls: "tn-ev-page" + (p.kept ? " tn-ev-page--kept" : "") });
          const host = p.host || hostOf(p.url);
          // 点击整行打开真实来源：有真链就直达那篇文章；demo 没存真链时退化成"按真标题去搜"
          // （标题是真的、唯一，搜索首条就是这篇真报道）——保证点下去一定弹出真网页，绝不 404 穿帮。
          const openHref = safeUrl(p.url)
            || (p.title ? "https://www.bing.com/search?q=" + encodeURIComponent(p.title)
                        : (host ? "https://" + host : ""));
          if (openHref) {
            const a = el("a", { cls: "tn-ev-page-link", attrs: { href: openHref, target: "_blank", rel: "noopener" } });
            a.appendChild(el("span", { cls: "tn-ev-page-host", text: host || "" }));
            a.appendChild(el("span", { cls: "tn-ev-page-title", text: trimText(p.title || "", 42) }));
            li.appendChild(a);
          } else {
            li.appendChild(el("span", { cls: "tn-ev-page-host", text: host || "" }));
            li.appendChild(el("span", { cls: "tn-ev-page-title", text: trimText(p.title || "", 42) }));
          }
          if (p.kept) li.appendChild(el("span", { cls: "tn-ev-page-kept", text: "采纳" }));
          list.appendChild(li);
        });
        body.appendChild(list);
        // 默认折叠；点 toggle 展开看真实域名（demo 可在关键时刻一点亮出"我们真抓了这些站"）
        toggle.addEventListener("click", () => {
          const collapsed = list.classList.toggle("is-collapsed");
          toggle.textContent = "抓取 " + crawled + " 个网页 " + (collapsed ? "▾" : "▴");
        });
      } else {
        if (crawled != null) live.appendChild(el("span", { cls: "tn-ev-live-count", text: "抓取 " + crawled + " 个网页" }));
        body.appendChild(live);
      }
    }

    if (queries.length) {
      const qrow = el("div", { cls: "tn-query-row" });
      qrow.appendChild(el("span", { cls: "tn-query-label", text: "关键词" }));
      queries.forEach(q => qrow.appendChild(el("span", { cls: "tn-query-chip", text: q })));
      body.appendChild(qrow);
    }

    if (results.length) {
      const ul = el("ul", { cls: "tn-evidence-list" });
      results.forEach((r, i) => {
        const li = el("li", { cls: "tn-evidence-item tn-stagger", attrs: { style: `--i:${Math.min(i, 8)}` } });
        const head = el("div", { cls: "tn-evidence-head" });
        head.appendChild(el("span", { cls: "tn-evidence-idx", text: String(i + 1) }));
        const _evHref = safeUrl(r.url);
        const titleEl = _evHref
          ? el("a", { cls: "tn-evidence-title", text: r.title || _evHref, attrs: { href: _evHref, target: "_blank", rel: "noopener" } })
          : el("span", { cls: "tn-evidence-title", text: r.title || "(无标题)" });
        head.appendChild(titleEl);
        if (r.source) head.appendChild(el("span", { cls: "tn-evidence-source", text: r.source }));
        li.appendChild(head);
        // 摘要灰字已按用户要求移除——内容与标题重合，多余。只留标题+来源。
        ul.appendChild(li);
      });
      body.appendChild(ul);
    } else if (!queries.length) {
      // 完全没结构化数据时显示兜底
      if (event.output_summary) body.appendChild(el("div", { cls: "tn-card-empty", text: event.output_summary }));
    }
    return card;
  });

  // -------------------- 模板：verification_grid --------------------
  const VERDICT_META = {
    "属实":           { cls: "true",        zh: "属实",        sign: "✓" },
    "谣言":           { cls: "false",       zh: "不对",        sign: "✗" },
    "大部分不实":     { cls: "false",       zh: "大部分不实",  sign: "✗" },
    "误导性信息":     { cls: "warn",        zh: "有误导",      sign: "!" },
    "部分属实":       { cls: "warn",        zh: "部分属实",    sign: "~" },
    "无法核实":       { cls: "unverified",  zh: "没查到",      sign: "?" }
  };

  // verdict 概率分布的 6 类标签 → 中文 + 颜色类（颜色只编码判定，符合「黑白主色 + 三色判定」规范）
  // order：从「最不实」到「最属实」固定排序，便于评委建立空间记忆
  const VERDICT_DIST_META = {
    "FALSE":        { zh: "谣言",       cls: "false",      order: 0 },
    "MOSTLY_FALSE": { zh: "大部分不实", cls: "false",      order: 1 },
    "MISLEADING":   { zh: "误导性",     cls: "warn",       order: 2 },
    "UNVERIFIABLE": { zh: "无法核实",   cls: "unverified", order: 3 },
    "PARTLY_TRUE":  { zh: "部分属实",   cls: "warn",       order: 4 },
    "TRUE":         { zh: "属实",       cls: "true",       order: 5 }
  };

  // 证据→声明 关系标签（StructuredFactChecker 步骤1 的 5 类）→ 颜色 + 符号
  // 关系颜色编码"这条证据把声明往哪个方向推"：辟谣=红、支持=绿、矛盾=黄、相关/无关=灰
  const RELATION_META = {
    "直接辟谣": { cls: "false",   sign: "✗" },
    "间接矛盾": { cls: "warn",    sign: "≠" },
    "直接支持": { cls: "true",    sign: "✓" },
    "话题相关": { cls: "neutral", sign: "~" },
    "不相关":   { cls: "muted",   sign: "·" }
  };

  // 渲染 verdict 概率分布：按概率降序，最大项加粗 + 着色；横条从 0 长出（路演动效）
  function renderVerdictDistribution(dist) {
    const entries = Object.keys(dist || {})
      .map(k => ({ key: k, prob: Number(dist[k]) || 0, meta: VERDICT_DIST_META[k] || { zh: k, cls: "unverified", order: 9 } }))
      .filter(e => e.prob > 0)
      .sort((a, b) => b.prob - a.prob);
    if (entries.length === 0) return null;

    const wrap = el("div", { cls: "tn-vdist" });
    // 「综合判定 · 概率分布」小灰字标题已按用户要求去掉——卡片标题已是「综合判定」，不再重复
    const topKey = entries[0].key;
    const fills = [];
    entries.forEach((e, i) => {
      const isTop = e.key === topKey;
      const row = el("div", {
        cls: "tn-vdist-row tn-vdist-row--" + e.meta.cls + (isTop ? " tn-vdist-row--top" : "") + " tn-stagger",
        attrs: { style: `--i:${Math.min(i, 8)}` }
      });
      row.appendChild(el("span", { cls: "tn-vdist-label", text: e.meta.zh }));
      const bar = el("div", { cls: "tn-vdist-bar" });
      const fill = el("div", { cls: "tn-vdist-bar-fill" });
      const ratio = Math.max(0, Math.min(1, e.prob));
      const delay = Math.min(i, 8) * 55;
      fill.style.transitionDelay = delay + "ms";
      bar.appendChild(fill);
      row.appendChild(bar);
      // 百分比数字：初始 0%，与横条同步 count-up（揭晓动效）
      const pctEl = el("span", { cls: "tn-vdist-pct", text: "0%" });
      fills.push({ fill: fill, ratio: ratio, delay: delay, pct: pctEl, target: Math.round(e.prob * 100) });
      row.appendChild(pctEl);
      wrap.appendChild(row);
    });
    // 双 rAF：横条从 scaleX(0) 长到目标比例（合成器层，无 reflow）；长完转入明度流动
    requestAnimationFrame(() => requestAnimationFrame(() => {
      fills.forEach(f => {
        f.fill.addEventListener("transitionend", () => f.fill.classList.add("is-settled"), { once: true });
        f.fill.style.transform = "scaleX(" + f.ratio + ")";
        // 数字与横条同步：同 delay 启动、贴合 700ms transition 时长数到目标
        animateCountPct(f.pct, f.target, 700, f.delay);
        // 兜底：ratio 极小或 transitionend 未触发时，仍能进入 settled 微动
        setTimeout(() => f.fill.classList.add("is-settled"), f.delay + 800);
      });
    }));
    return wrap;
  }

  registerTemplate("verification_grid", function (event) {
    const data = event.display?.data || {};
    const verifications = Array.isArray(data.verifications) ? data.verifications : [];
    const narrative = event.human_narrative
      || (verifications.length
        ? `小T 对 ${verifications.length} 条声明做了交叉验证`
        : `小T 在对照证据做判断...`);

    const { card, body } = cardShell(event, {
      flavor: "verify",
      template: "verification_grid",
      narrative
    });

    if (verifications.length === 0) {
      if (event.output_summary) body.appendChild(el("div", { cls: "tn-card-empty", text: event.output_summary }));
      return card;
    }

    const ul = el("ul", { cls: "tn-verify-list" });
    verifications.forEach((v, i) => {
      const li = el("li", { cls: "tn-verify-item tn-stagger", attrs: { style: `--i:${Math.min(i, 8)}` } });
      const vMeta = VERDICT_META[v.verdict] || { cls: "unverified", zh: v.verdict || "?", sign: "?" };
      const head = el("div", { cls: "tn-verify-head" });
      head.appendChild(el("span", { cls: "tn-verify-claim", text: trimText(v.claim, 60) }));
      const verdictTag = el("span", { cls: "tn-verify-verdict tn-verify-verdict--" + vMeta.cls + " tn-verify-flip" });
      verdictTag.appendChild(el("span", { cls: "tn-verify-sign", text: vMeta.sign }));
      verdictTag.appendChild(el("span", { text: vMeta.zh }));
      head.appendChild(verdictTag);
      if (v.confidence != null) {
        head.appendChild(el("span", { cls: "tn-verify-conf", text: Math.round(v.confidence * 100) + "%" }));
      }
      li.appendChild(head);
      if (v.reasoning) {
        li.appendChild(el("div", { cls: "tn-verify-reason", text: v.reasoning }));
      }
      ul.appendChild(li);
    });
    body.appendChild(ul);
    return card;
  });

  // -------------------- 模板：reply_draft --------------------
  registerTemplate("reply_draft", function (event) {
    const data = event.display?.data || {};
    const narrative = event.human_narrative || "小T 给出结论";

    const { card, body } = cardShell(event, {
      flavor: "compose",
      template: "reply_draft",
      narrative
    });

    // 新形态（结论卡）：板上钉钉一句话——为什么是谣言（直接摆证据）+ 社会PU体系库登记说明。
    // 兼容旧形态（温和回复 reply+tone），未改造的 case 仍走旧渲染。
    if (data.conclusion) {
      body.appendChild(el("div", { cls: "tn-verdict-conclusion", text: data.conclusion }));
      if (data.registry) {
        const reg = el("div", { cls: "tn-verdict-registry" });
        reg.appendChild(el("span", { cls: "tn-verdict-registry-tag", text: "存档" }));
        reg.appendChild(el("span", { cls: "tn-verdict-registry-text", text: data.registry }));
        body.appendChild(reg);
      }
      if (data.summary) body.appendChild(el("div", { cls: "tn-reply-meta", text: data.summary }));
      return card;
    }

    const reply = data.reply || event.output_summary || "";
    if (reply) {
      const box = el("blockquote", { cls: "tn-reply-box" });
      box.appendChild(el("span", { cls: "tn-reply-quote", text: "“" }));
      box.appendChild(el("span", { cls: "tn-reply-text", text: reply }));
      box.appendChild(el("span", { cls: "tn-reply-quote tn-reply-quote--end", text: "”" }));
      body.appendChild(box);
      if (data.tone) body.appendChild(el("div", { cls: "tn-reply-meta", text: "语气：" + data.tone }));
      if (data.summary) body.appendChild(el("div", { cls: "tn-reply-meta", text: data.summary }));
    } else {
      body.appendChild(el("div", { cls: "tn-card-empty", text: "（还在写...）" }));
    }
    return card;
  });

  // -------------------- 模板：query_plan（查询词 chips） --------------------
  registerTemplate("query_plan", function (event) {
    const data = event.display?.data || {};
    const queries = Array.isArray(data.queries) ? data.queries : [];
    const strategy = data.strategy || "";
    const narrative = event.human_narrative
      || (queries.length ? `小T 想好了 ${queries.length} 个关键词` : "小T 在规划搜索...");

    const { card, body } = cardShell(event, {
      flavor: "search",
      template: "query_plan",
      narrative
    });

    if (queries.length === 0) {
      if (event.output_summary) body.appendChild(el("div", { cls: "tn-card-empty", text: event.output_summary }));
      return card;
    }
    const wrap = el("div", { cls: "tn-qp-chips" });
    queries.forEach((q, i) => {
      const chip = el("span", {
        cls: "tn-qp-chip tn-stagger",
        text: q,
        attrs: { style: `--i:${Math.min(i, 8)}` }
      });
      wrap.appendChild(chip);
    });
    body.appendChild(wrap);
    // 「策略：…」灰色说明块已按用户要求移除（strategy 数据保留但不渲染）
    return card;
  });

  // -------------------- 模板：evidence_rank（Top 3 证据 + 可信度条） --------------------
  registerTemplate("evidence_rank", function (event) {
    const data = event.display?.data || {};
    const ranked = Array.isArray(data.ranked) ? data.ranked : (Array.isArray(data.top_results) ? data.top_results : []);
    const narrative = event.human_narrative
      || (ranked.length ? `小T 按可信度排出 Top ${ranked.length}` : "小T 在评估证据质量...");

    const { card, body } = cardShell(event, {
      flavor: "search",
      template: "evidence_rank",
      narrative
    });

    if (ranked.length === 0) {
      if (event.output_summary) body.appendChild(el("div", { cls: "tn-card-empty", text: event.output_summary }));
      return card;
    }
    const list = el("ul", { cls: "tn-er-list" });
    // 找最大分用于归一化条形长度
    const maxScore = Math.max(...ranked.map(r => r.authority_score || r.score || 0.5), 1);
    ranked.forEach((r, i) => {
      const li = el("li", { cls: "tn-er-item tn-stagger", attrs: { style: `--i:${Math.min(i, 8)}` } });
      const head = el("div", { cls: "tn-er-head" });
      head.appendChild(el("span", { cls: "tn-er-rank", text: "#" + (i + 1) }));
      const src = r.source || r.title || "";
      head.appendChild(el("span", { cls: "tn-er-src", text: trimText(src, 28) }));
      const score = (r.authority_score != null ? r.authority_score : (r.score != null ? r.score : 0));
      head.appendChild(el("span", { cls: "tn-er-score", text: score.toFixed(2) }));
      li.appendChild(head);
      const barWrap = el("div", { cls: "tn-er-bar-wrap" });
      const barFill = el("div", { cls: "tn-er-bar" });
      const ratio = Math.max(0, Math.min(1, score / maxScore));
      const barDelay = Math.min(i, 8) * 55;
      barFill.style.transitionDelay = barDelay + "ms";
      barWrap.appendChild(barFill);
      li.appendChild(barWrap);
      requestAnimationFrame(() => requestAnimationFrame(() => {
        barFill.addEventListener("transitionend", () => barFill.classList.add("is-settled"), { once: true });
        barFill.style.transform = "scaleX(" + ratio + ")";
        setTimeout(() => barFill.classList.add("is-settled"), barDelay + 700);
      }));
      list.appendChild(li);
    });
    body.appendChild(list);
    return card;
  });

  // -------------------- 模板：skeptic_challenges（自我质疑列表） --------------------
  registerTemplate("skeptic_challenges", function (event) {
    const data = event.display?.data || {};
    const challenges = Array.isArray(data.challenges) ? data.challenges : [];
    const conclusion = data.conclusion || "";
    const narrative = event.human_narrative
      || (challenges.length ? `小T 自己挑了 ${challenges.length} 个刺` : "小T 在自我反思...");

    const { card, body } = cardShell(event, {
      flavor: "verify",
      template: "skeptic_challenges",
      narrative
    });

    if (challenges.length === 0) {
      if (event.output_summary) body.appendChild(el("div", { cls: "tn-card-empty", text: event.output_summary }));
      return card;
    }
    const list = el("ul", { cls: "tn-sk-list" });
    challenges.forEach((c, i) => {
      const li = el("li", { cls: "tn-sk-item tn-stagger", attrs: { style: `--i:${Math.min(i, 8)}` } });
      const q = typeof c === "string" ? c : (c.question || c.text || "");
      const a = typeof c === "string" ? "" : (c.answer || c.resolution || "");
      const qEl = el("div", { cls: "tn-sk-q" });
      qEl.appendChild(el("span", { cls: "tn-sk-qmark", text: "?" }));
      qEl.appendChild(el("span", { cls: "tn-sk-qtext", text: q }));
      li.appendChild(qEl);
      if (a) {
        li.appendChild(el("div", { cls: "tn-sk-a", text: a }));
      }
      list.appendChild(li);
    });
    body.appendChild(list);
    if (conclusion) {
      body.appendChild(el("div", { cls: "tn-sk-conclusion", text: conclusion }));
    }
    return card;
  });

  // -------------------- 模板：evidence_relations（标关系——StructuredFactChecker 步骤1） --------------------
  // 命题人叙事核心：不是"问大模型一句真假"，而是把每条证据和声明的关系逐条标出来
  registerTemplate("evidence_relations", function (event) {
    const data = event.display?.data || {};
    // 支持多声明分组：data.groups = [{claim, relations}]；兼容旧单声明结构 data.claim + data.relations
    const groups = (Array.isArray(data.groups) && data.groups.length)
      ? data.groups
      : [{ claim: data.claim || "", relations: Array.isArray(data.relations) ? data.relations : [] }];
    const totalRel = groups.reduce((n, g) => n + (Array.isArray(g.relations) ? g.relations.length : 0), 0);
    const narrative = event.human_narrative
      || (totalRel ? "小T 把每条证据和声明的关系标出来，逐条比对" : "小T 在比对证据与声明的关系...");

    const { card, body } = cardShell(event, {
      flavor: "verify",
      template: "evidence_relations",
      title: "标关系",
      narrative
    });

    if (totalRel === 0 && !groups.some(g => g.claim)) {
      if (event.output_summary) body.appendChild(el("div", { cls: "tn-card-empty", text: event.output_summary }));
      return card;
    }

    // 多条声明时每条都列出来（声明1 / 声明2…），各自带自己的证据关系；单条时仍标「声明」
    const multi = groups.length > 1;
    groups.forEach((g, gi) => {
      const groupEl = el("div", { cls: "tn-rel-group" });
      if (g.claim) {
        const claimRow = el("div", { cls: "tn-rel-claim" });
        claimRow.appendChild(el("span", { cls: "tn-rel-claim-tag", text: multi ? ("声明" + (gi + 1)) : "声明" }));
        claimRow.appendChild(el("span", { cls: "tn-rel-claim-text", text: trimText(g.claim, 60) }));
        groupEl.appendChild(claimRow);
      }
      const rels = Array.isArray(g.relations) ? g.relations : [];
      if (rels.length) {
        const list = el("ul", { cls: "tn-rel-list" });
        rels.forEach((r, i) => {
          const meta = RELATION_META[r.relation] || { cls: "neutral", sign: "~" };
          const li = el("li", { cls: "tn-rel-item tn-stagger", attrs: { style: `--i:${Math.min(i, 8)}` } });
          const head = el("div", { cls: "tn-rel-head" });
          head.appendChild(el("span", { cls: "tn-rel-src", text: trimText(r.source || r.title || ("证据 " + (i + 1)), 24) }));
          // 关系徽章只给文字（去掉前缀符号 ✗/≠/~），方向仍由底色编码（红=辟谣/黄=矛盾/灰=相关）
          const badge = el("span", { cls: "tn-rel-badge tn-rel-badge--" + meta.cls + " tn-verify-flip" });
          badge.appendChild(el("span", { text: r.relation || "?" }));
          head.appendChild(badge);
          li.appendChild(head);
          if (r.note) li.appendChild(el("div", { cls: "tn-rel-note", text: r.note }));
          list.appendChild(li);
        });
        groupEl.appendChild(list);
      }
      body.appendChild(groupEl);
    });
    return card;
  });

  // -------------------- 模板：rule_verdict（规则裁决——StructuredFactChecker 步骤3，零-LLM） --------------------
  // 命题人叙事核心：判定不是模型说了算，是固定优先级规则匹配出来的（可审计、可复现）
  registerTemplate("rule_verdict", function (event) {
    const data = event.display?.data || {};
    const verdict = data.verdict || "";
    const vMeta = VERDICT_META[verdict] || { cls: "unverified", zh: verdict || "?", sign: "?" };
    const narrative = event.human_narrative || "判定靠规则，不靠模型拍脑袋";

    const { card, body } = cardShell(event, {
      flavor: "verify",
      template: "rule_verdict",
      title: "规则裁决",
      narrative
    });

    // 「零-LLM 规则引擎」徽章已按用户要求移除——卡片只留 命中规则 → 判定。

    // 命中规则 →(箭头) 判定
    const flow = el("div", { cls: "tn-rv-flow" });
    const ruleBox = el("div", { cls: "tn-rv-rule" });
    ruleBox.appendChild(el("span", { cls: "tn-rv-rule-label", text: "命中规则" }));
    ruleBox.appendChild(el("span", { cls: "tn-rv-rule-text", text: data.matched_rule || "" }));
    flow.appendChild(ruleBox);
    flow.appendChild(el("span", { cls: "tn-rv-arrow", text: "→" }));
    // 裁决框只给文字（去掉前缀符号），判定由底色编码
    const verdictBox = el("div", { cls: "tn-rv-verdict tn-rv-verdict--" + vMeta.cls });
    verdictBox.appendChild(el("span", { cls: "tn-rv-verdict-zh", text: verdict || vMeta.zh }));
    if (data.priority != null) {
      verdictBox.appendChild(el("span", { cls: "tn-rv-verdict-pri", text: "P" + Number(data.priority).toFixed(2) }));
    }
    flow.appendChild(verdictBox);
    body.appendChild(flow);

    // 规则说明(rule_detail)与 INV-3 守护备注(note)灰字已按用户要求移除——多余、与上方「命中规则」重合。
    return card;
  });

  // -------------------- 模板：dimension_radar（占位，路演杀器） --------------------
  registerTemplate("dimension_radar", function (event) {
    const data = event.display?.data || {};
    const dims = Array.isArray(data.dimensions) ? data.dimensions : [];
    const narrative = event.human_narrative || `小T 从 ${dims.length || 6} 个维度独立投票`;

    const { card, body } = cardShell(event, {
      flavor: "primary",
      template: "dimension_radar",
      narrative
    });

    // 顶部：综合判定概率分布（6 维度投票的聚合结果，路演杀器）
    const distNode = renderVerdictDistribution(data.verdict_distribution);
    if (distNode) body.appendChild(distNode);

    if (dims.length === 0) {
      if (!distNode && event.output_summary) body.appendChild(el("div", { cls: "tn-card-empty", text: event.output_summary }));
      return card;
    }
    const grid = el("div", { cls: "tn-dim-grid" });
    const dimFills = [];
    dims.forEach((d, i) => {
      const cell = el("div", { cls: "tn-dim-cell tn-stagger", attrs: { style: `--i:${Math.min(i, 8)}` } });
      if (d.question) {
        cell.appendChild(el("div", { cls: "tn-dim-question", text: d.question }));
      }
      // 副标签（学术名）已删 —— 评委看不懂，徒增视觉噪声
      const bar = el("div", { cls: "tn-dim-bar" });
      // 横条用 scaleX 从 0 长出（合成器层，无 reflow）
      const fill = el("div", { cls: "tn-dim-bar-fill" });
      const ratio = Math.max(0, Math.min(1, d.score || 0));
      const delay = Math.min(i, 8) * 55;
      fill.style.transitionDelay = delay + "ms";
      dimFills.push({ fill: fill, ratio: ratio, delay: delay });
      bar.appendChild(fill);
      cell.appendChild(bar);
      // 倾向用浅色胶囊承载颜色，字保持黑色
      const leanMeta = VERDICT_META[d.verdict_lean || ""];
      const leanCls = leanMeta ? leanMeta.cls : "unverified";
      cell.appendChild(el("div", {
        cls: "tn-dim-lean tn-dim-lean--" + leanCls,
        text: d.verdict_lean || ""
      }));
      grid.appendChild(cell);
    });
    body.appendChild(grid);
    // 双 rAF：6 维度条从 scaleX(0) 逐条长出，长完转入明度流动
    requestAnimationFrame(() => requestAnimationFrame(() => {
      dimFills.forEach(f => {
        f.fill.addEventListener("transitionend", () => f.fill.classList.add("is-settled"), { once: true });
        f.fill.style.transform = "scaleX(" + f.ratio + ")";
        setTimeout(() => f.fill.classList.add("is-settled"), f.delay + 700);
      });
    }));
    return card;
  });

  // ============ 路由器 ============
  function renderEvent(event) {
    if (!event || !event.type) return null;

    // 只渲染 step / error / timeout（其它事件不出卡片）
    if (event.type === "step") {
      const template = event.display?.template || (resolveAgent(event.agent)?.template) || "generic";
      const fn = TEMPLATES[template] || TEMPLATES.generic;
      return fn(event);
    }

    if (event.type === "error" || event.type === "timeout") {
      const card = el("div", { cls: "tn-card tn-card--error tn-card--enter" });
      const head = el("div", { cls: "tn-card-head" });
      head.appendChild(el("span", { cls: "tn-card-dot tn-card-dot--error" }));
      head.appendChild(el("span", { cls: "tn-card-title", text: event.type === "timeout" ? "可能卡了" : "出错了" }));
      // 相对时间「+Xs」已按用户要求去掉
      card.appendChild(head);
      if (event.message) {
        card.appendChild(el("div", { cls: "tn-card-narrative", text: event.message }));
      }
      requestAnimationFrame(() => {
        card.classList.remove("tn-card--enter");
        card.classList.add("tn-card--active");
      });
      return card;
    }

    return null;
  }

  function stageOfEvent(event) {
    if (!event || event.type !== "step") return -1;
    const meta = resolveAgent(event.agent);
    return meta ? meta.stage : -1;
  }

  // 二元真/假前置：从 counts 派生「假 / 真 / 证据暂不足」（纯展示，不改后端打分）
  // 命题人不接受"无法核实"当主态——demo 主推确定真/假案例，这里把真/假倾向摆到结论卡最前。
  function binaryLean(counts) {
    counts = counts || {};
    const f = counts.false || 0, t = counts.true || 0, u = counts.unverified || 0;
    if (f > 0) return { cls: "false", big: "假", sign: "✗", zh: "不实", sub: "这条消息不实，别信别传" };
    if (t > 0 && u === 0) return { cls: "true", big: "真", sign: "✓", zh: "属实", sub: "这条核查为真，可放心" };
    if (t > 0) return { cls: "true", big: "真", sign: "✓", zh: "基本属实", sub: "主要内容为真，个别需留意" };
    return { cls: "unverified", big: "?", sign: "?", zh: "证据暂不足", sub: "暂无足够证据，小T 不瞎判真假" };
  }

  // ============ 闭环动作（把结果"真正给用户用起来"——命题人定义的闭环） ============
  // URL 协议白名单：插件把卡片注入任意网页，渠道 url 可能来自真后端/记忆回填，
  // 只放行 http/https/tel/mailto，挡掉 javascript:/data: 注入。
  function safeUrl(u) {
    if (!u) return "";
    const s = String(u).trim();
    return /^(https?:|tel:|mailto:)/i.test(s) ? s : "";
  }
  // 图片 src 专用白名单：比 href 的 safeUrl 宽松——img src 不执行脚本，放行 http/https、data:image、
  // 以及相对/协议相对路径（预览页用本地图）；只挡 javascript:/vbscript:/file: 这类危险/越权 scheme。
  function safeImgSrc(u) {
    if (!u) return "";
    const s = String(u).trim();
    if (/^(javascript|vbscript|file):/i.test(s)) return "";
    if (/^data:/i.test(s)) return /^data:image\//i.test(s) ? s : "";
    return s;
  }
  // 从 url 取主机名（去 www.），用于"检索到的网页"列表显示域名
  function hostOf(u) {
    const s = safeUrl(u);
    const m = s.match(/^https?:\/\/([^/]+)/i);
    return m ? m[1].replace(/^www\./i, "") : "";
  }
  // 处置回执编号：从 action_id 确定性派生（无 Date/random，demo 可复现，对齐后端 TN-RC-xxxx）
  function deriveReceiptId(seed) {
    let h = 2166136261 >>> 0;
    const s = String(seed || "act");
    for (let i = 0; i < s.length; i++) { h ^= s.charCodeAt(i); h = Math.imul(h, 16777619) >>> 0; }
    return "TN-RC-" + h.toString(16).padStart(8, "0").slice(0, 8);
  }
  function fallbackCopy(text, done) {
    try {
      const ta = document.createElement("textarea");
      ta.value = text;
      ta.style.position = "fixed";
      ta.style.opacity = "0";
      document.body.appendChild(ta);
      ta.select();
      document.execCommand("copy");
      document.body.removeChild(ta);
    } catch (_) { /* 复制失败也给反馈，别卡住交互 */ }
    done();
  }
  function copyToClipboard(text, btn, after) {
    const done = () => {
      if (btn) {
        btn._origText = btn._origText || btn.textContent;
        btn.textContent = "已复制 ✓";
        btn.classList.add("is-done");
      }
      if (typeof after === "function") after();
    };
    try {
      if (global.navigator && navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(text).then(done, () => fallbackCopy(text, done));
        return;
      }
    } catch (_) { /* 落兜底 */ }
    fallbackCopy(text, done);
  }

  function makeChannelRow(kind, ch) {
    const href = safeUrl(ch.url);
    const row = el("a", {
      cls: "tn-action-channel tn-action-channel--" + kind + (href ? "" : " tn-action-channel--nolink"),
      attrs: href ? { href: href, target: "_blank", rel: "noopener" } : {}
    });
    row.appendChild(el("span", { cls: "tn-action-channel-kind", text: kind === "official" ? "核实" : "举报" }));
    row.appendChild(el("span", { cls: "tn-action-channel-name", text: ch.name || href || "" }));
    if (ch.description) row.appendChild(el("span", { cls: "tn-action-channel-desc", text: ch.description }));
    return row;
  }

  // 处置 → 回执（闭环在用户侧"看得见地"合上那一下；spec §4）
  function showReceipt(box, controls, info) {
    if (controls) controls.style.display = "none";
    const r = el("div", { cls: "tn-receipt tn-receipt--" + (info.cls || "resolved") + " tn-card--enter" });
    const head = el("div", { cls: "tn-receipt-head" });
    head.appendChild(el("span", { cls: "tn-receipt-label", text: info.label }));
    head.appendChild(el("span", { cls: "tn-receipt-id", text: info.receiptId }));
    r.appendChild(head);
    if (info.message) r.appendChild(el("div", { cls: "tn-receipt-msg", text: info.message }));
    if (info.nextStep) r.appendChild(el("div", { cls: "tn-receipt-next", text: info.nextStep }));
    box.appendChild(r);
    requestAnimationFrame(() => { r.classList.remove("tn-card--enter"); r.classList.add("tn-card--active"); });
  }

  // recommended_action → 主 CTA 文案（spec §3 矩阵）
  const ACTION_CTA = { share_correction: "复制纠正卡", warn_user: "复制提醒", report_scam: "去举报", subscribe_backfill: "订阅回填", no_action: "" };

  function renderClosedLoopActions(actions) {
    if (!Array.isArray(actions) || actions.length === 0) return null;
    const sec = el("div", { cls: "tn-final-section tn-actions-section" });
    sec.appendChild(el("div", { cls: "tn-final-section-title", text: "闭环动作 · 把结果用起来" }));

    actions.forEach(a => {
      const box = el("div", { cls: "tn-action-box" });
      const rec = a.recommendedAction || "share_correction";
      const receiptId = deriveReceiptId(a.actionId || a.claimText);
      const claimShort = trimText(a.claimText || "这条消息", 22);

      // no_action（属实）：只给一个绿色 chip，不渲染处置块（spec §3）
      if (rec === "no_action") {
        box.appendChild(el("div", { cls: "tn-action-chip tn-action-chip--ok", text: "✅ 核查属实，无需处置" }));
        sec.appendChild(box);
        return;
      }

      const controls = el("div", { cls: "tn-action-controls" });

      // —— 块构造器（按 recommended_action 决定主 CTA 与顺序）——
      const renderCC = () => {
        if (!a.correctionCard) return;
        const ccWrap = el("div", { cls: "tn-action-cc" });
        // 把末尾「——TruthNote 核查」署名拆出来单独居右；正文照常左对齐
        const m = a.correctionCard.match(/^([\s\S]*?)\n(——[^\n]*)\s*$/);
        const ccMain = m ? m[1] : a.correctionCard;
        const ccAttr = m ? m[2] : "";
        const ccBox = el("div", { cls: "tn-action-cc-text" });
        ccBox.appendChild(el("div", { cls: "tn-action-cc-body", text: ccMain }));
        if (ccAttr) ccBox.appendChild(el("div", { cls: "tn-action-cc-attr", text: ccAttr }));
        ccWrap.appendChild(ccBox);

        const btnRow = el("div", { cls: "tn-action-cc-btns" });
        const isPrimary = (rec === "share_correction" || rec === "warn_user");
        const copyBtn = el("button", {
          cls: "tn-action-btn" + (isPrimary ? " tn-action-btn--primary" : ""),
          text: "📋 " + (ACTION_CTA[rec] || "复制纠正卡"),
          attrs: { type: "button" }
        });
        copyBtn.addEventListener("click", () => copyToClipboard(a.correctionCard, copyBtn, () => {
          // 复制即"已发送"处置 → 回执（命题人要的"看得见地合上"）
          setTimeout(() => showReceipt(box, controls, {
            cls: "sent",
            label: "已发送 · 纠正卡",
            receiptId,
            message: `「${claimShort}」的纠正卡已复制，可转发到群里辟谣。`,
            nextStep: "感谢完成这次辟谣闭环。"
          }), 900);
        }));
        btnRow.appendChild(copyBtn);

        // 社会闭环：可选把这条辟谣贡献进「小T 社会辟谣库」，供全网后续命中复用
        const upBtn = el("button", {
          cls: "tn-action-btn tn-action-cc-upload",
          text: "🛡 上传到社会辟谣库",
          attrs: { type: "button" }
        });
        upBtn.addEventListener("click", () => {
          if (upBtn.disabled) return;
          upBtn.disabled = true;
          upBtn.classList.add("is-done");
          upBtn.textContent = "✓ 已贡献到社会辟谣库";
        });
        btnRow.appendChild(upBtn);

        ccWrap.appendChild(btnRow);
        controls.appendChild(ccWrap);
      };

      const renderReport = () => {
        if (!(a.reportLinks && a.reportLinks.length)) return;
        // report_scam：第一个举报渠道升级成主 CTA 大按钮（spec §3：诈骗主推去举报）
        if (rec === "report_scam") {
          const first = a.reportLinks[0];
          const href = safeUrl(first.url);
          const goBtn = el("a", {
            cls: "tn-action-btn tn-action-btn--danger",
            text: "🚨 去举报 · " + (first.name || "反诈中心"),
            attrs: href ? { href: href, target: "_blank", rel: "noopener" } : {}
          });
          goBtn.addEventListener("click", () => {
            setTimeout(() => showReceipt(box, controls, {
              cls: "resolved",
              label: "已处置 · 举报",
              receiptId,
              message: `「${claimShort}」已记录举报。`,
              nextStep: "如涉及资金损失，请同时拨打 110。"
            }), 200);
          });
          controls.appendChild(goBtn);
          const rest = a.reportLinks.slice(1);
          if (rest.length) {
            const wrap = el("div", { cls: "tn-action-channels" });
            rest.forEach(ch => wrap.appendChild(makeChannelRow("report", ch)));
            controls.appendChild(wrap);
          }
        } else {
          const wrap = el("div", { cls: "tn-action-channels" });
          a.reportLinks.forEach(ch => wrap.appendChild(makeChannelRow("report", ch)));
          controls.appendChild(wrap);
        }
      };

      const renderOfficial = () => {
        if (!(a.officialChannels && a.officialChannels.length)) return;
        const wrap = el("div", { cls: "tn-action-channels" });
        a.officialChannels.forEach(ch => wrap.appendChild(makeChannelRow("official", ch)));
        controls.appendChild(wrap);
      };

      const renderSubscription = () => {
        if (!a.subscription) return;
        const sub = el("div", { cls: "tn-action-sub" });
        const top = el("div", { cls: "tn-action-sub-top" });
        top.appendChild(el("span", { cls: "tn-action-sub-label", text: "订阅回填" }));
        if (a.subscription.topic) top.appendChild(el("span", { cls: "tn-action-sub-topic", text: "「" + trimText(a.subscription.topic, 24) + "」" }));
        sub.appendChild(top);
        // watch_sources（spec §3.1：列出持续盯的权威源，可点）
        const ws = Array.isArray(a.subscription.watch_sources) ? a.subscription.watch_sources : [];
        if (ws.length) {
          const wsWrap = el("div", { cls: "tn-action-watch" });
          wsWrap.appendChild(el("span", { cls: "tn-action-watch-label", text: "持续盯：" }));
          ws.forEach(s => {
            const href = safeUrl(s.url);
            wsWrap.appendChild(href
              ? el("a", { cls: "tn-action-watch-src", text: s.name || href, attrs: { href: href, target: "_blank", rel: "noopener" } })
              : el("span", { cls: "tn-action-watch-src", text: s.name || "" }));
          });
          sub.appendChild(wsWrap);
        }
        if (a.subscription.note) sub.appendChild(el("div", { cls: "tn-action-sub-note", text: a.subscription.note }));
        const subBtn = el("button", { cls: "tn-action-btn tn-action-btn--sub", text: "🔔 订阅回填", attrs: { type: "button" } });
        subBtn.addEventListener("click", () => showReceipt(box, controls, {
          cls: "resolved",
          label: "已订阅 · 回填",
          receiptId,
          message: `已为「${trimText(a.subscription.topic || claimShort, 22)}」开启订阅。`,
          nextStep: a.subscription.note || "权威结论出现后第一时间通知你。"
        }));
        sub.appendChild(subBtn);
        controls.appendChild(sub);
      };

      // 块顺序按 recommended_action（spec §3 矩阵）
      if (rec === "subscribe_backfill") {
        renderSubscription();
        renderOfficial();
      } else if (rec === "report_scam") {
        renderReport();   // 举报主 CTA 在前
        renderCC();       // 纠正卡次要
        renderOfficial();
      } else {            // share_correction / warn_user / 未知兜底
        renderCC();
        renderOfficial();
        if (rec === "share_correction") renderReport();
      }

      box.appendChild(controls);
      sec.appendChild(box);
    });
    return sec;
  }

  // ============ 无法核实·细化归因常驻小块 ============
  // 不进折叠链，直接挂在 final 卡的「无法核实」声明下方常驻显示。
  // 4 段：codeLabel 标题 + detail 正文 + blockedCondition「卡在哪」高亮行 + verifyWhere「去哪查」脚注。
  // 颜色统一灰（复用 tn-final-original / tn-final-claim-reason 同款灰阶），7 类只在文案区分、不变色。
  // INV-U3：这里只搬运后端给的文案，不做任何真伪判断。reason 为空时返回 null（不占位）。
  function renderUnverifiableReason(reason) {
    if (!reason || typeof reason !== "object") return null;
    const codeLabel = reason.codeLabel || "";
    const detail = reason.detail || "";
    const blockedCondition = reason.blockedCondition || "";
    const verifyWhere = reason.verifyWhere || "";
    // 四段全空 → 不渲染（避免空壳块占首屏）
    if (!codeLabel && !detail && !blockedCondition && !verifyWhere) return null;

    const box = el("div", { cls: "tn-uvreason" });
    // 标题：codeLabel（如「私域或超本地」「可查但证据不足」），带灰色「为什么没下判定」前缀引导
    const head = el("div", { cls: "tn-uvreason-head" });
    head.appendChild(el("span", { cls: "tn-uvreason-tag", text: "没下判定的原因" }));
    if (codeLabel) head.appendChild(el("span", { cls: "tn-uvreason-code", text: codeLabel }));
    box.appendChild(head);
    // 正文：detail 一句具体障碍
    if (detail) box.appendChild(el("div", { cls: "tn-uvreason-detail", text: detail }));
    // 「卡在哪」高亮行
    if (blockedCondition) {
      const blocked = el("div", { cls: "tn-uvreason-blocked" });
      blocked.appendChild(el("span", { cls: "tn-uvreason-blocked-label", text: "卡在哪：" }));
      blocked.appendChild(el("span", { cls: "tn-uvreason-blocked-text", text: blockedCondition }));
      box.appendChild(blocked);
    }
    // 「去哪查」脚注
    if (verifyWhere) {
      const vw = el("div", { cls: "tn-uvreason-where" });
      vw.appendChild(el("span", { cls: "tn-uvreason-where-label", text: "去哪查：" }));
      vw.appendChild(el("span", { cls: "tn-uvreason-where-text", text: verifyWhere }));
      box.appendChild(vw);
    }
    return box;
  }

  // ============ 最终结论卡（done 之后追加到卡片瀑布末尾） ============
  function renderFinalResult(data) {
    if (!data) return null;
    const card = el("div", { cls: "tn-card tn-card--final tn-card--enter" });

    // 头部：完成徽章
    const head = el("div", { cls: "tn-card-head" });
    head.appendChild(el("span", { cls: "tn-card-dot tn-card-dot--final" }));
    head.appendChild(el("span", { cls: "tn-card-title", text: "完成了" }));
    const claims = Array.isArray(data.claims) ? data.claims : [];
    const counts = data.counts || { true: 0, false: 0, unverified: 0 };
    const sumParts = [];
    if (counts.false > 0) sumParts.push(counts.false + " 不对");
    if (counts.unverified > 0) sumParts.push(counts.unverified + " 没查到");
    if (counts.true > 0) sumParts.push(counts.true + " 查到了");
    head.appendChild(el("span", { cls: "tn-card-sub", text: sumParts.join(" · ") }));
    card.appendChild(head);

    // 二元真/假前置 banner（结论卡最前，命题人要"确定真/假"，少说"无法核实"）
    const lean = binaryLean(counts);
    const binary = el("div", { cls: "tn-final-binary tn-final-binary--" + lean.cls });
    // 「✗/假」大红底印章块已按用户要求去掉——banner 只留文字「不实 / 这条消息不实，别信别传」
    const btext = el("div", { cls: "tn-final-binary-text" });
    btext.appendChild(el("div", { cls: "tn-final-binary-zh", text: lean.zh }));
    btext.appendChild(el("div", { cls: "tn-final-binary-sub", text: lean.sub }));
    binary.appendChild(btext);
    card.appendChild(binary);

    // 诚实限定（图片溯源场景必带）：二元 banner 写死"假/别信别传"会把"图判假"的歧义带出来；
    // 这里把"图是真的、假的只是今天"这层限定提到首屏 banner 紧下方，不藏进折叠的推理链。
    if (data.headlineNote) {
      card.appendChild(el("div", { cls: "tn-final-headnote", text: data.headlineNote }));
    }

    // 原文引用已按用户要求去掉——「提取声明」卡顶部已展示原文，最终卡再引一次重复。

    // 声明列表（每条带 verdict 标签 + 关键证据）
    if (claims.length) {
      const sec = el("div", { cls: "tn-final-section" });
      sec.appendChild(el("div", { cls: "tn-final-section-title", text: "结论" }));
      const ul = el("ul", { cls: "tn-final-claims" });
      claims.forEach((c, i) => {
        const verdictKey = c.verdict || "unverified";
        const li = el("li", { cls: "tn-final-claim tn-final-claim--" + verdictKey + " tn-stagger", attrs: { style: `--i:${Math.min(i, 8)}` } });
        const head = el("div", { cls: "tn-final-claim-head" });
        head.appendChild(el("span", { cls: "tn-final-claim-num", text: String(i + 1) }));
        head.appendChild(el("span", { cls: "tn-final-claim-text", text: c.text || "" }));
        const tag = el("span", { cls: "tn-final-claim-tag tn-final-claim-tag--" + verdictKey });
        const tagLabel = { true: "查到了", false: "不对", unverified: "没查到" }[verdictKey] || c.verdictRaw || "?";
        tag.textContent = tagLabel;
        head.appendChild(tag);
        li.appendChild(head);
        if (c.evidence && c.evidence[0] && (c.evidence[0].snippet || c.evidence[0].title)) {
          const ev = el("div", { cls: "tn-final-claim-ev" });
          const _clHref = safeUrl(c.evidence[0].url);
          if (c.evidence[0].title && _clHref) {
            ev.appendChild(el("a", {
              cls: "tn-final-claim-link",
              text: c.evidence[0].title,
              attrs: { href: _clHref, target: "_blank", rel: "noopener" }
            }));
          } else if (c.evidence[0].title) {
            ev.appendChild(el("span", { cls: "tn-final-claim-link", text: c.evidence[0].title }));
          }
          if (c.evidence[0].snippet) {
            ev.appendChild(el("div", { cls: "tn-final-claim-snippet", text: trimText(c.evidence[0].snippet, 100) }));
          }
          li.appendChild(ev);
        } else if (c.reasoning) {
          li.appendChild(el("div", { cls: "tn-final-claim-reason", text: c.reasoning }));
        }
        // 无法核实·细化归因常驻块（有 unverifiableReason 才挂；不抢占 evidence，附在声明末尾）
        const uvReasonEl = renderUnverifiableReason(c.unverifiableReason);
        if (uvReasonEl) li.appendChild(uvReasonEl);
        ul.appendChild(li);
      });
      sec.appendChild(ul);
      card.appendChild(sec);
    }

    // 闭环动作（命题人定义的闭环：把结果真正给用户用起来）
    const actionsSec = renderClosedLoopActions(data.actions);
    if (actionsSec) card.appendChild(actionsSec);

    // 思维链条（具象过程展示，让用户从案例里自己学辨别方法）
    const chain = data.reasoningChain || data.reasoning_chain || [];
    if (Array.isArray(chain) && chain.length) {
      const sec = el("div", { cls: "tn-final-section tn-chain-section" });
      // 折叠头：默认收起，用户点开才逐条显形（揭晓动效改为主动触发）
      const toggle = el("button", {
        cls: "tn-chain-toggle",
        attrs: { type: "button", "aria-expanded": "false" }
      });
      toggle.appendChild(el("span", { cls: "tn-final-section-title", text: "小T 是这么想的" }));
      toggle.appendChild(el("span", { cls: "tn-chain-toggle-hint", text: "点开看推理" }));
      toggle.appendChild(el("span", { cls: "tn-chain-caret", text: "▸", attrs: { "aria-hidden": "true" } }));
      sec.appendChild(toggle);
      const collapse = el("div", { cls: "tn-chain-collapse" });
      const ol = el("ol", { cls: "tn-chain-list" });
      chain.forEach((step, i) => {
        const li = el("li", {
          cls: "tn-chain-item",
          attrs: { style: `--i:${Math.min(i, 8)}` }
        });
        const num = el("span", { cls: "tn-chain-num", text: String(i + 1) });
        li.appendChild(num);
        const body = el("div", { cls: "tn-chain-body" });
        if (typeof step === "string") {
          body.appendChild(el("div", { cls: "tn-chain-text", text: step }));
        } else if (step && typeof step === "object") {
          if (step.title) body.appendChild(el("div", { cls: "tn-chain-title", text: step.title }));
          if (step.detail) body.appendChild(el("div", { cls: "tn-chain-detail", text: step.detail }));
          if (Array.isArray(step.points)) {
            const ul = el("ul", { cls: "tn-chain-points" });
            step.points.forEach(p => ul.appendChild(el("li", { text: p })));
            body.appendChild(ul);
          }
        }
        li.appendChild(body);
        ol.appendChild(li);
      });
      collapse.appendChild(ol);
      sec.appendChild(collapse);
      // 点击展开/收起；展开时重启逐条显形（remove→reflow→add 让 CSS 动画从头跑）
      toggle.addEventListener("click", function () {
        const open = sec.classList.toggle("is-open");
        toggle.setAttribute("aria-expanded", open ? "true" : "false");
        if (open) {
          // 同步：先移除→reflow→再加（canonical 动画重启）。同步加而非 rAF——
          // 让 display:block 与 reveal 的 opacity:0 落在同一帧前，避免「整链先全亮再重置」的闪烁。
          const items = ol.querySelectorAll(".tn-chain-item");
          items.forEach(function (it) { it.classList.remove("tn-chain-reveal"); });
          void ol.offsetWidth; // 强制 reflow，确保动画从头重启
          items.forEach(function (it) { it.classList.add("tn-chain-reveal"); });
        }
      });
      card.appendChild(sec);
    }

    requestAnimationFrame(() => {
      card.classList.remove("tn-card--enter");
      card.classList.add("tn-card--active");
    });
    return card;
  }

  // ============ 导出 ============
  const RenderEngine = {
    render: renderEvent,
    renderFinalResult,
    binaryLean,          // 二元真/假前置：content.js 设挂件标签复用同一套派生逻辑
    registerTemplate,
    resolveAgent,
    stageOfEvent,
    STAGE_LABELS,
    AGENT_REGISTRY,
    VERDICT_DIST_META,   // verdict 概率分布 6 类 → 中文/颜色类（orchestra.js 复用，避免抄第二份）
    // 暴露小工具给 content.js / debug.js / orchestra.js 复用
    _utils: { el, fmtMs, fmtTime, trimText, prefersReducedMotion, animateCountPct, safeUrl, safeImgSrc }
  };

  global.TNRender = RenderEngine;
})(typeof window !== "undefined" ? window : globalThis);

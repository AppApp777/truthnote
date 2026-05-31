// TruthNote 设置页逻辑 options.js
// 设计：① 全程读写已有节点的 textContent/value，绝不拼 innerHTML（沿用 render-engine 安全纪律）
//       ② chrome.* 缺失时自动降级——双击 pages/options.html(file://) 也能预览界面效果，不报错
//       ③ 所有事件 addEventListener 绑定（CSP 禁内联）
// 形态：消费级为主 + 高级设置折叠（接入自己的服务/模型）。两组都即时保存、串行化写入。
//   · 消费级偏好（布尔）：核查方式/显示/隐私（含「允许上传核查数据」opt-in，默认关）
//   · 服务配置（字符串）：服务地址/API 密钥/基础模型/关键模型 —— 与 background.js 共用同一 tn_settings key
//     （background 读这几个键发请求；本页只负责让用户填、存本机，不做后端能力）
// 安全：API 密钥只进 chrome.storage.local（绝不 sync），不 console.log、不外传。
"use strict";
(function () {
  const HAS_CHROME =
    typeof chrome !== "undefined" &&
    chrome.storage &&
    chrome.storage.local &&
    typeof chrome.storage.local.get === "function";

  const STORAGE_KEY = "tn_settings";

  // 消费级偏好默认值（行为偏好，被使用页面/背景消费）
  const PREF_DEFAULTS = {
    selectTrigger: true,      // 选中文字后弹"查"触点
    contextMenuCheck: true,   // 右键菜单核查
    showWidget: true,         // 右下角常驻挂件
    reasoningExpanded: false, // 推理详情默认展开
    uiMotion: true,           // 界面动效
    allowDataUpload: false    // 允许上传核查数据进闭环库（隐私优先 · 默认关 · opt-in）
  };
  // 服务配置默认值（字符串；留空 = 用小T云端默认，background 自行兜底）
  const SERVICE_DEFAULTS = {
    backendBaseUrl: "",
    backendToken: "",
    defaultModel: "",
    strongModel: ""
  };

  // 开关 DOM id ↔ 偏好键
  const TOGGLES = [
    { id: "tgSelectTrigger", key: "selectTrigger" },
    { id: "tgContextMenu", key: "contextMenuCheck" },
    { id: "tgWidget", key: "showWidget" },
    { id: "tgReasoning", key: "reasoningExpanded" },
    { id: "tgMotion", key: "uiMotion" },
    { id: "tgDataUpload", key: "allowDataUpload" }
  ];
  // 文本输入 DOM id ↔ 服务配置键
  const SERVICE_INPUTS = [
    { id: "cfgUrl", key: "backendBaseUrl" },
    { id: "cfgToken", key: "backendToken" },
    { id: "cfgBaseModel", key: "defaultModel" },
    { id: "cfgStrongModel", key: "strongModel" }
  ];

  const $ = (id) => document.getElementById(id);
  const refs = {};
  ["clearBtn", "saveFlash", "feVersion", "lnkHelp", "cfgToken", "cfgTokenReveal", "advToggle", "advBody", "advIcon", "upgradeBtn", "boardLink"]
    .forEach((id) => { refs[id] = $(id); });

  let prefs = Object.assign({}, PREF_DEFAULTS);
  let service = Object.assign({}, SERVICE_DEFAULTS);

  // 只挑已知布尔偏好键
  function pickPrefs(stored) {
    const p = Object.assign({}, PREF_DEFAULTS);
    stored = stored || {};
    Object.keys(PREF_DEFAULTS).forEach((k) => { if (typeof stored[k] === "boolean") p[k] = stored[k]; });
    return p;
  }
  // 只挑已知字符串服务键
  function pickService(stored) {
    const s = Object.assign({}, SERVICE_DEFAULTS);
    stored = stored || {};
    Object.keys(SERVICE_DEFAULTS).forEach((k) => { if (typeof stored[k] === "string") s[k] = stored[k]; });
    return s;
  }

  // ---- 开关 ----
  function setSwitch(btn, on) {
    if (!btn) return;
    btn.setAttribute("aria-checked", on ? "true" : "false");
    btn.classList.toggle("is-on", on);
  }
  const getSwitch = (btn) => btn && btn.getAttribute("aria-checked") === "true";

  // 仿真密钥：API 密钥框默认显示一串"已配置"圆点（纯展示效果，不进 service/storage、background 不会真发）
  const TN_DEMO_KEY = "tn_sk_live_a93f12c7e4b8602d5f1a9c3e";

  function applyToForm() {
    TOGGLES.forEach((t) => setSwitch($(t.id), prefs[t.key] === true));
    SERVICE_INPUTS.forEach((si) => {
      const el = $(si.id);
      if (!el) return;
      // API 密钥：没真填过就回填仿真密钥（密码框渲染成一串圆点的效果）；其余字段空就留空
      if (si.key === "backendToken") el.value = service.backendToken || TN_DEMO_KEY;
      else el.value = service[si.key] || "";
    });
  }

  // 即时保存：读现有 tn_settings 合并（保留未在本页出现的键，如 background 的 heartbeatTimeoutMs），
  // 写入串行化（_writeChain 排队）——多次连改时每次"读-改-写"完整跑完再跑下一次，消除并发覆盖。
  let _writeChain = Promise.resolve();
  function persist() {
    if (!HAS_CHROME) return _writeChain;
    _writeChain = _writeChain.then(async () => {
      try {
        const got = await chrome.storage.local.get(STORAGE_KEY);
        const existing = (got && got[STORAGE_KEY]) || {};
        const merged = Object.assign({}, existing, prefs, service);
        const payload = {};
        payload[STORAGE_KEY] = merged;
        await chrome.storage.local.set(payload);
      } catch (e) { /* 预览/异常下静默 */ }
    });
    return _writeChain;
  }

  function onToggle(t) {
    const btn = $(t.id);
    const next = !getSwitch(btn);
    setSwitch(btn, next);           // 开关滑动本身就是"已保存"反馈（成熟产品的即时保存范式）
    prefs[t.key] = next;
    persist();
  }

  // ---- 保存提示（仅清空历史这类动作用）----
  function flash(text) {
    if (!refs.saveFlash) return;
    refs.saveFlash.textContent = text || "已保存 ✓";
    refs.saveFlash.hidden = false;
    refs.saveFlash.classList.remove("is-show");
    void refs.saveFlash.offsetWidth;
    refs.saveFlash.classList.add("is-show");
    setTimeout(() => { refs.saveFlash.hidden = true; }, 1600);
  }

  // ---- 清空核查历史（两次点击确认，不用原生 confirm）----
  let clearArmed = false, clearTimer = null;
  function disarmClear() {
    clearArmed = false;
    clearTimeout(clearTimer);
    if (refs.clearBtn) { refs.clearBtn.textContent = "清空核查历史"; refs.clearBtn.classList.remove("is-armed"); }
  }
  function clearHistory() {
    if (!clearArmed) {
      clearArmed = true;
      if (refs.clearBtn) { refs.clearBtn.textContent = "再点一次确认清空"; refs.clearBtn.classList.add("is-armed"); }
      clearTimer = setTimeout(disarmClear, 3000);
      return;
    }
    disarmClear();
    if (HAS_CHROME && chrome.runtime && typeof chrome.runtime.sendMessage === "function") {
      try { chrome.runtime.sendMessage({ type: "TN_CLEAR_TRACES" }, () => { void chrome.runtime.lastError; }); } catch (e) { /* SW 不在也无妨 */ }
    }
    flash("已清空 ✓");
  }

  // ---- 高级设置折叠 ----
  function toggleAdvanced() {
    if (!refs.advBody) return;
    refs.advBody.classList.toggle("is-collapsed");
    const collapsed = refs.advBody.classList.contains("is-collapsed");
    if (refs.advToggle) refs.advToggle.setAttribute("aria-expanded", collapsed ? "false" : "true");
    if (refs.advIcon) refs.advIcon.textContent = collapsed ? "▾" : "▴";
  }

  // ---- 升级会员（前端效果，不接真支付）----
  let upgradeTimer = null;
  function onUpgrade() {
    if (!refs.upgradeBtn) return;
    clearTimeout(upgradeTimer);
    refs.upgradeBtn.textContent = "敬请期待 ✓";
    upgradeTimer = setTimeout(() => { refs.upgradeBtn.textContent = "升级会员"; }, 1600);
  }

  // ---- 使用帮助 → 引导页 ----
  function openHelp() {
    if (HAS_CHROME && chrome.tabs && typeof chrome.tabs.create === "function" && chrome.runtime && chrome.runtime.getURL) {
      try { chrome.tabs.create({ url: chrome.runtime.getURL("pages/onboarding.html") }); } catch (e) { /* noop */ }
    }
  }

  // ---- 社会辟谣大厅（真相公示墙）= 闭环社会谣言库的可视入口 ----
  function openBoard() {
    // 真扩展环境：新标签页打开打包页
    if (HAS_CHROME && chrome.tabs && typeof chrome.tabs.create === "function" && chrome.runtime && chrome.runtime.getURL) {
      try { chrome.tabs.create({ url: chrome.runtime.getURL("pages/piyao-board.html") }); return; } catch (e) { /* 落到降级 */ }
    }
    // 降级（file:// 双击预览 / 无扩展 API）：同目录相对跳转，让预览也能点过去
    try { window.location.href = "piyao-board.html"; } catch (e) { /* noop */ }
  }

  // ---- 初始化 ----
  async function init() {
    // 版本
    let feVer = "1.1.0";
    if (HAS_CHROME && chrome.runtime && typeof chrome.runtime.getManifest === "function") {
      try { feVer = chrome.runtime.getManifest().version || feVer; } catch (e) { /* noop */ }
    }
    if (refs.feVersion) refs.feVersion.textContent = feVer;

    // 读已存设置
    if (HAS_CHROME) {
      try {
        const got = await chrome.storage.local.get(STORAGE_KEY);
        const stored = (got && got[STORAGE_KEY]) || {};
        prefs = pickPrefs(stored);
        service = pickService(stored);
      } catch (e) { prefs = Object.assign({}, PREF_DEFAULTS); service = Object.assign({}, SERVICE_DEFAULTS); }
    }
    applyToForm();

    // 事件：开关即时保存
    TOGGLES.forEach((t) => { const btn = $(t.id); if (btn) btn.addEventListener("click", () => onToggle(t)); });
    // 服务配置：失焦(change)时保存，避免每个键击都写
    SERVICE_INPUTS.forEach((si) => {
      const el = $(si.id);
      if (el) el.addEventListener("change", () => { service[si.key] = el.value.trim(); persist(); });
    });
    // 密钥显隐
    if (refs.cfgTokenReveal && refs.cfgToken) {
      refs.cfgTokenReveal.addEventListener("click", () => {
        const showing = refs.cfgToken.type === "text";
        refs.cfgToken.type = showing ? "password" : "text";
        refs.cfgTokenReveal.textContent = showing ? "显示" : "隐藏";
      });
    }
    // 升级会员（前端效果）
    if (refs.upgradeBtn) refs.upgradeBtn.addEventListener("click", onUpgrade);
    // 高级折叠
    if (refs.advToggle) refs.advToggle.addEventListener("click", toggleAdvanced);
    // 清空历史
    if (refs.clearBtn) refs.clearBtn.addEventListener("click", clearHistory);
    // 「社会辟谣体系」内联索引 → 打开真相公示墙（键盘可达）
    if (refs.boardLink) {
      refs.boardLink.addEventListener("click", openBoard);
      refs.boardLink.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); openBoard(); } });
    }
    // 使用帮助
    if (refs.lnkHelp) {
      refs.lnkHelp.addEventListener("click", openHelp);
      refs.lnkHelp.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); openHelp(); } });
    }
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
  else init();
})();

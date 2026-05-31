/**
 * TruthNote 平台适配引擎
 * 检测当前网站 → 加载适配器 → 注入核查按钮 → 委托给常驻挂件
 */
(() => {
  "use strict";

  const TN_ATTR = "data-tn-injected";

  // ⛔ 评论区「一键查询·查」按钮注入已按用户要求暂停（知乎/微博等平台适配器）——该设定有问题，先取消。
  //    改回 true 即可恢复；适配器/选择器逻辑、manifest 注册全部原样保留，方便随时重启。
  //    主链路「选中文字 → 查」走 content.js，与本开关无关、不受影响。
  const PLATFORM_INJECT_ENABLED = false;

  let activeAdapter = null;
  const adapters = [];

  window.__tnRegisterAdapter = function(adapter) {
    adapters.push(adapter);
  };

  function init() {
    if (!PLATFORM_INJECT_ENABLED) return;   // 评论区按钮注入已暂停（先取消，保留代码）
    const host = location.hostname;
    for (const adapter of adapters) {
      if (adapter.match(host)) {
        activeAdapter = adapter;
        break;
      }
    }
    if (!activeAdapter) return;

    startObserving();
    scanAndInject();
  }

  function startObserving() {
    let timer = null;
    const observer = new MutationObserver(() => {
      clearTimeout(timer);
      timer = setTimeout(scanAndInject, 300);
    });
    observer.observe(document.body, { childList: true, subtree: true });
  }

  function scanAndInject() {
    if (!activeAdapter) return;
    try {
      const items = document.querySelectorAll(activeAdapter.contentSelector);
      items.forEach(el => {
        if (el.getAttribute(TN_ATTR)) return;
        el.setAttribute(TN_ATTR, "1");

        const text = activeAdapter.getText(el);
        if (!text || text.trim().length < 10) return;

        const anchor = activeAdapter.getAnchor(el);
        if (!anchor) return;

        const btn = createTnButton();
        const capturedText = text.trim();
        btn.addEventListener("click", e => {
          e.stopPropagation();
          e.preventDefault();
          window.postMessage({ type: "TN_START_VERIFY", text: capturedText }, "*");
        });

        activeAdapter.inject(anchor, btn);
      });
    } catch (_) {}
  }

  function createTnButton() {
    const btn = document.createElement("span");
    btn.className = "tn-platform-btn";
    btn.setAttribute("role", "button");

    const eye1 = document.createElement("span");
    eye1.className = "tn-pbtn-eye";
    const eye2 = document.createElement("span");
    eye2.className = "tn-pbtn-eye";
    const label = document.createElement("span");
    label.className = "tn-pbtn-label";
    label.textContent = "查";

    btn.appendChild(eye1);
    btn.appendChild(eye2);
    btn.appendChild(label);
    return btn;
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", () => setTimeout(init, 500));
  } else {
    setTimeout(init, 500);
  }
})();

// X / Twitter 适配器
window.__tnRegisterAdapter({
  name: "twitter",
  match: host => host.includes("x.com") || host.includes("twitter.com"),

  // data-testid 是 X 上最稳定的选择器
  contentSelector: '[data-testid="tweet"]',

  getText(el) {
    const textEl = el.querySelector('[data-testid="tweetText"]');
    return textEl ? textEl.textContent?.trim() : null;
  },

  getAnchor(el) {
    // 推文底部的互动栏（转发/点赞/回复按钮所在行）
    const actionBar = el.querySelector('[role="group"]');
    return actionBar || el;
  },

  inject(anchor, btn) {
    btn.style.cssText += ";margin-left:8px;border-radius:9999px;padding:2px 10px;";
    anchor.appendChild(btn);
  }
});

// 小红书适配器
window.__tnRegisterAdapter({
  name: "xiaohongshu",
  match: host => host.includes("xiaohongshu.com"),

  contentSelector: [
    '[class*="note-content"]',
    '[class*="content"]',
    '[class*="desc"]',
    '.note-text',
    '#detail-desc',
    'div[class*="note"] div[class*="content"]'
  ].join(","),

  getText(el) {
    return el.textContent?.trim()?.slice(0, 500) || null;
  },

  getAnchor(el) {
    return el.querySelector('[class*="interact"], [class*="engage"], [class*="action"]') || el;
  },

  inject(anchor, btn) {
    anchor.appendChild(btn);
  }
});

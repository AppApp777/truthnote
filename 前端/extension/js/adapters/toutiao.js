// 今日头条适配器
window.__tnRegisterAdapter({
  name: "toutiao",
  match: host => host.includes("toutiao.com"),

  contentSelector: [
    'article',
    '.article-content',
    '[class*="article-body"]',
    '[class*="feed-card"]',
    '.comment-list .comment-item'
  ].join(","),

  getText(el) {
    return el.textContent?.trim()?.slice(0, 500) || null;
  },

  getAnchor(el) {
    return el.querySelector('[class*="action"], [class*="interact"], footer') || el;
  },

  inject(anchor, btn) {
    anchor.appendChild(btn);
  }
});

// 豆瓣适配器
window.__tnRegisterAdapter({
  name: "douban",
  match: host => host.includes("douban.com"),

  contentSelector: [
    '.topic-richtext',
    '.short-content',
    '.review-content',
    '.comment-item',
    '.status-saying',
    '#link-report',
    '.note-container'
  ].join(","),

  getText(el) {
    return el.textContent?.trim()?.slice(0, 500) || null;
  },

  getAnchor(el) {
    return el.querySelector('.action, .comment-action, .operation') || el;
  },

  inject(anchor, btn) {
    anchor.appendChild(btn);
  }
});

// Bilibili 评论区适配器
window.__tnRegisterAdapter({
  name: "bilibili",
  match: host => host.includes("bilibili.com"),

  contentSelector: [
    '.reply-item',
    '.comment-item',
    '[class*="reply-item"]',
    '.list-item.reply-wrap'
  ].join(","),

  getText(el) {
    const textEl = el.querySelector(
      '.reply-content, .text-con, [class*="reply-content"], [class*="root-reply"]'
    );
    return textEl ? textEl.textContent?.trim() : null;
  },

  getAnchor(el) {
    return el.querySelector(
      '.reply-info, [class*="reply-operation"], [class*="action"], .operation'
    ) || el;
  },

  inject(anchor, btn) {
    anchor.appendChild(btn);
  }
});

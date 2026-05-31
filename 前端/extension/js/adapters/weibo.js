// 微博适配器
window.__tnRegisterAdapter({
  name: "weibo",
  match: host => host.includes("weibo.com"),

  // 微博帖子容器（多种选择器兼容新旧版）
  contentSelector: [
    'article[class*="Feed"]',
    'div[class*="card-wrap"]',
    'div[class*="weibo-main"]',
    '.card[action-type="feed_list_item"]',
    'div[mid]'
  ].join(","),

  getText(el) {
    const textEl = el.querySelector(
      '[class*="detail_wbtext"], [class*="Feed_body"] [class*="wbpro-feed"], .WB_text, [node-type="feed_list_content"]'
    );
    return textEl ? textEl.textContent : el.textContent?.slice(0, 500);
  },

  getAnchor(el) {
    return el.querySelector(
      '[class*="toolbar"], [class*="card-act"], .WB_feed_handle, [class*="Feed_interact"]'
    ) || el;
  },

  inject(anchor, btn) {
    anchor.appendChild(btn);
  }
});

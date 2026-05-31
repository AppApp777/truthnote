// 百度贴吧适配器
window.__tnRegisterAdapter({
  name: "tieba",
  match: host => host.includes("tieba.baidu.com"),

  contentSelector: [
    '.d_post_content',
    '.p_content',
    '.j_d_post_content',
    '#post_content',
    '.lzl_content_main'
  ].join(","),

  getText(el) {
    return el.textContent?.trim()?.slice(0, 500) || null;
  },

  getAnchor(el) {
    const postEl = el.closest('.l_post, .p_postlist, .j_l_post');
    if (postEl) {
      return postEl.querySelector('.post-tail-wrap, .p_tail, .core_reply_tail') || el;
    }
    return el;
  },

  inject(anchor, btn) {
    anchor.appendChild(btn);
  }
});

// 微信公众号文章适配器
window.__tnRegisterAdapter({
  name: "wechat",
  match: host => host.includes("mp.weixin.qq.com"),

  // 公众号文章是静态页面，文章内容在 #js_content 里
  contentSelector: '#js_content > section, #js_content > p, #js_content > div',

  getText(el) {
    const text = el.textContent?.trim();
    if (!text || text.length < 15) return null;
    return text;
  },

  getAnchor(el) {
    return el;
  },

  inject(anchor, btn) {
    btn.style.cssText += ";display:block;margin:4px 0;width:fit-content;";
    if (anchor.nextSibling) {
      anchor.parentNode.insertBefore(btn, anchor.nextSibling);
    } else {
      anchor.parentNode.appendChild(btn);
    }
  }
});

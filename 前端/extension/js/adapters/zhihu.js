// 知乎适配器
window.__tnRegisterAdapter({
  name: "zhihu",
  match: host => host.includes("zhihu.com"),

  contentSelector: [
    '.AnswerItem',
    '.ContentItem',
    '[data-za-detail-view-path-module="AnswerItem"]',
    '.CommentContent',
    '.Post-RichTextContainer',
    '.RichContent'
  ].join(","),

  getText(el) {
    const textEl = el.querySelector('.RichText, .CopyrightRichText, [class*="RichText"]');
    return textEl ? textEl.textContent : el.textContent?.slice(0, 500);
  },

  getAnchor(el) {
    return el.querySelector(
      '.ContentItem-actions, .AnswerItem-extraInfo, [class*="BottomActions"], [class*="ContentItem-action"]'
    ) || el;
  },

  inject(anchor, btn) {
    anchor.appendChild(btn);
  }
});

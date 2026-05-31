const $ = id => document.getElementById(id);

document.addEventListener("DOMContentLoaded", () => {
  chrome.runtime.sendMessage({ type: "GET_SELECTION" }, res => {
    const text = res?.text?.trim();
    if (text && text.length >= 4) {
      $("stateEmpty").style.display = "none";
      $("stateSelection").style.display = "";
      $("selectedText").textContent = "\u201C" + text.slice(0, 80) + (text.length > 80 ? "..." : "") + "\u201D";

      $("btnCheck").addEventListener("click", () => {
        chrome.tabs.query({ active: true, currentWindow: true }, tabs => {
          if (tabs[0]) {
            chrome.tabs.sendMessage(tabs[0].id, {
              type: "START_VERIFY",
              text: text
            });
          }
        });
        window.close();
      });
    }
  });

});

document.addEventListener("DOMContentLoaded", () => {
  function goTo(step) {
    document.querySelectorAll(".slide").forEach(s => s.classList.remove("active"));
    document.querySelector(`[data-step="${step}"]`).classList.add("active");
  }

  document.querySelectorAll("[data-goto]").forEach(btn => {
    btn.addEventListener("click", () => {
      goTo(btn.getAttribute("data-goto"));
    });
  });

  document.querySelectorAll("[data-finish]").forEach(btn => {
    btn.addEventListener("click", () => {
      window.close();
    });
  });
});

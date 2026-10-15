// Light and dark, as on kasra.page: light by default, a stored choice wins,
// the OS setting is ignored. Loaded in <head> so the theme is set before the
// first paint; the button is wired once the page has loaded.
(() => {
  const DEFAULT_THEME = "light";
  const root = document.documentElement;
  const stored = localStorage.getItem("theme");
  root.setAttribute("data-theme", stored === "light" || stored === "dark" ? stored : DEFAULT_THEME);

  window.addEventListener("DOMContentLoaded", () => {
    const button = document.querySelector("[data-theme-toggle]");
    if (!button) return;
    const label = button.querySelector(".theme__label");
    const next = () => (root.getAttribute("data-theme") === "dark" ? "light" : "dark");
    const render = () => {
      label.textContent = next();
      button.setAttribute("aria-label", `Switch to ${next()} theme`);
    };
    render();
    button.addEventListener("click", () => {
      const theme = next();
      root.setAttribute("data-theme", theme);
      localStorage.setItem("theme", theme);
      render();
    });
  });
})();

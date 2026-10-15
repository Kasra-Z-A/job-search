// Light and dark, as on kasra.page: light by default, a stored choice wins,
// the OS setting is ignored. Loaded in <head> so the theme is set before the
// first paint; the button is wired once the page has loaded.
//
// The choice is a cookie on kasra.page, shared with the website and the other
// demos, so following a link keeps the theme. localStorage is the fallback
// where that cookie can't be set (localhost). Same logic as the website's
// Base.astro; keep the two in step.
(() => {
  const DEFAULT_THEME = "light";
  const root = document.documentElement;

  const read = () => {
    const match = document.cookie.match(/(?:^|;\s*)theme=(light|dark)(?:;|$)/);
    if (match) return match[1];
    try {
      const stored = localStorage.getItem("theme");
      if (stored === "light" || stored === "dark") return stored;
    } catch {}
    return null;
  };

  const store = (theme) => {
    try {
      localStorage.setItem("theme", theme);
    } catch {}
    if (/(^|\.)kasra\.page$/.test(location.hostname)) {
      document.cookie = `theme=${theme}; Domain=kasra.page; Path=/; Max-Age=31536000; SameSite=Lax; Secure`;
    }
  };

  const stored = read();
  if (stored) store(stored);
  root.setAttribute("data-theme", stored ?? DEFAULT_THEME);

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
      store(theme);
      render();
    });
  });
})();

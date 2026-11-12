// Day / night switch (see app/theme_switch.py): store the choice where Streamlit reads its theme,
// then reload so Streamlit and the platform's CSS both pick it up.
const OWN = "labTheme";

export default function ({ data, parentElement }) {
  const key = `stActiveTheme-${window.location.pathname}-v2`;
  const btn = parentElement.querySelector(".ts-btn");
  const isDark = data.current === "dark";
  let own = null;
  try { own = localStorage.getItem(OWN); } catch (e) { /* storage blocked: system setting */ }
  // A page opened at another address has no stored theme yet: follow the last choice once.
  if (own && localStorage.getItem(key) !== JSON.stringify(own)) {
    localStorage.setItem(key, JSON.stringify(own));
    if ((own === "Dark") !== isDark) { window.location.reload(); return; }
  }
  const next = isDark ? "Light" : "Dark";
  btn.querySelector(".ts-icon").textContent = isDark ? "light_mode" : "dark_mode";
  btn.querySelector(".ts-label").textContent = data.labels[next];
  btn.title = data.tips[next];
  btn.onclick = () => {
    try {
      localStorage.setItem(OWN, next);
      localStorage.setItem(key, JSON.stringify(next));
    } catch (e) { return; }
    window.location.reload();
  };
}

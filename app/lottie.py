"""Lottie animations (app/static/lottie): lottie-web's SVG player, bundled so it works offline.

    lottie.play("routing", height=160, text={...})

The player (lottie_light.min.js, MIT, see LICENSE-lottie-web.md) runs in the page through a
components-v2 component; with "reduce motion" set in the system it shows one still frame.
"""

import json
from functools import cache
from pathlib import Path

import streamlit as st

DIR = Path(__file__).parent / "static" / "lottie"

JS = """
export default function ({ data, parentElement }) {
  const box = parentElement.querySelector(".lt-box");
  box.style.height = `${data.height}px`;
  box.replaceChildren();
  const still = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const anim = globalThis.lottie.loadAnimation({
    container: box, renderer: "svg", loop: data.loop, autoplay: !still,
    animationData: data.animation,
    rendererSettings: { preserveAspectRatio: "xMidYMid meet" },
  });
  if (still) anim.goToAndStop(data.still_frame, true);
  return () => anim.destroy();
}
"""


@cache
def _component():
    player = (DIR / "lottie_light.min.js").read_text(encoding="utf-8")
    return st.components.v2.component(
        "lal_lottie",
        html='<div class="lt-box"></div>',
        css=".lt-box { width: 100%; } .lt-box svg { display: block; }",
        js=player + "\n" + JS,  # the player defines globalThis.lottie first
    )


@cache
def animation(name: str) -> dict:
    return json.loads((DIR / f"{name}.json").read_text(encoding="utf-8"))


def _with_text(data: dict, text: dict[str, str]) -> dict:
    """A copy of the animation with {placeholders} in its text layers filled in."""
    out = json.loads(json.dumps(data))
    for layer in out.get("layers", []):
        if layer.get("ty") != 5:
            continue
        for key in layer["t"]["d"]["k"]:
            key["s"]["t"] = key["s"]["t"].format_map(text)
    return out


def play(
    name: str,
    height: int = 160,
    loop: bool = True,
    key: str | None = None,
    still_frame: int = 8,
    text: dict[str, str] | None = None,
) -> None:
    data = animation(name)
    if text:
        data = _with_text(data, text)
    _component()(
        key=key or f"lottie_{name}",
        data={"animation": data, "height": height, "loop": loop, "still_frame": still_frame},
    )

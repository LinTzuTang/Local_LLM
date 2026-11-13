"""Write app/static/lottie/core.json: the home page's Lottie animation (our own drawing).

The platform in one picture: a core (this machine) breathing in the middle and sending out a
ripple now and then; models and data as points of light circling it on three tilted orbits,
dimmed when they pass behind; and an unbroken boundary around everything, turning slowly:
every model and every byte stays inside this machine. 8 s seamless loop at 30 fps, no text.

    uv run python scripts/make_lottie.py
"""

import json
import math
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "app" / "static" / "lottie" / "core.json"
W, H, FPS, END = 240, 200, 30, 240
CX, CY = 120, 100
STEP = 3  # frames between position keyframes on the orbits (linear in between)
# tilt (degrees), half axes, frames per revolution (must divide END), dots as phase fractions
ORBITS = [
    (-18, 84, 26, 240, [0.0]),
    (42, 70, 24, 120, [0.15, 0.65]),
    (102, 58, 22, 80, [0.4]),
]
RING_AT = (0, 80, 160)  # frames at which the core sends a ripple
RING_LEN = 72


def rgb(hex_colour: str) -> list[float]:
    h = hex_colour.lstrip("#")
    return [round(int(h[i : i + 2], 16) / 255, 4) for i in (0, 2, 4)] + [1]


TEAL, PALE, WARM, INK = rgb("#9DBBE8"), rgb("#DCE3EE"), rgb("#EFD9A6"), rgb("#1B2840")
DOT_COLOURS = [PALE, TEAL, WARM]


def static(value):
    return {"a": 0, "k": value}


def eased(frames):
    """Value keyframes [(frame, [..]), ...] with a soft ease in and out."""
    keys = []
    for i, (t, v) in enumerate(frames):
        key = {"t": t, "s": v}
        if i < len(frames) - 1:
            n = len(v)
            key["i"], key["o"] = {"x": [0.42] * n, "y": [1] * n}, {"x": [0.58] * n, "y": [0] * n}
        keys.append(key)
    return {"a": 1, "k": keys}


def held(frames):
    """Value keyframes that jump (no interpolation)."""
    return {"a": 1, "k": [{"t": t, "s": v, "h": 1} for t, v in frames]}


def linear_path(points):
    """Spatial keyframes [(frame, [x, y, 0]), ...], straight lines in between."""
    keys = []
    for i, (t, v) in enumerate(points):
        key = {"t": t, "s": v}
        if i < len(points) - 1:
            key["i"], key["o"] = {"x": 1, "y": 1}, {"x": 0, "y": 0}
            key["to"], key["ti"] = [0, 0, 0], [0, 0, 0]
        keys.append(key)
    return {"a": 1, "k": keys}


def transform(position, scale=None, opacity=None, rotation=None):
    return {
        "o": opacity or static(100),
        "r": rotation or static(0),
        "p": position if isinstance(position, dict) else static([*position, 0]),
        "a": static([0, 0, 0]),
        "s": scale or static([100, 100, 100]),
    }


def group(*items, rotation=0):
    tr = {"ty": "tr", "p": static([0, 0]), "a": static([0, 0]), "s": static([100, 100]),
          "r": static(rotation), "o": static(100)}  # fmt: skip
    return {"ty": "gr", "it": [*items, tr]}


def circle(size):
    return {"ty": "el", "p": static([0, 0]), "s": static([size, size])}


def ellipse(rx, ry):
    return {"ty": "el", "p": static([0, 0]), "s": static([2 * rx, 2 * ry])}


def rect(w, h, radius):
    return {"ty": "rc", "p": static([0, 0]), "s": static([w, h]), "r": static(radius)}


def fill(colour, opacity=None):
    return {"ty": "fl", "c": static(colour), "o": opacity or static(100), "r": 1}


def stroke(colour, width, opacity=100, dash=None):
    out = {"ty": "st", "c": static(colour), "o": static(opacity), "w": static(width),
           "lc": 2, "lj": 2}  # fmt: skip
    if dash:
        gap, offset = dash
        out["d"] = [{"n": "d", "v": static(gap)}, {"n": "g", "v": static(gap)},
                    {"n": "o", "v": offset}]  # fmt: skip
    return out


def layer(name, ks, shapes):
    return {"ddd": 0, "ty": 4, "nm": name, "sr": 1, "ks": ks, "ao": 0, "shapes": shapes,
            "ip": 0, "op": END, "st": 0, "bm": 0}  # fmt: skip


def orbit_point(tilt, rx, ry, angle):
    x, y = rx * math.cos(angle), ry * math.sin(angle)
    a = math.radians(tilt)
    return [CX + x * math.cos(a) - y * math.sin(a), CY + x * math.sin(a) + y * math.cos(a), 0]


def dot_layers(tilt, rx, ry, rev, phase, colour):
    """A point of light on one orbit: one layer in front of the core, one behind it (dimmer),
    switched by where on the orbit it is."""
    frames = list(range(0, END + 1, STEP))
    points, front = [], []
    for f in frames:
        angle = 2 * math.pi * (f / rev + phase)
        points.append((f, orbit_point(tilt, rx, ry, angle)))
        front.append(math.sin(angle) >= 0)

    def shown(side):
        changes = [(f, [100 if front[i] == side else 0]) for i, f in enumerate(frames)
                   if i == 0 or front[i] != front[i - 1]]  # fmt: skip
        return held(changes)

    path = linear_path(points)
    halo = group(circle(16), fill(colour, static(14)))
    front_layer = layer("dot front", transform(path, opacity=shown(True)),
                        [halo, group(circle(7), fill(colour))])  # fmt: skip
    back_layer = layer("dot back", transform(path, scale=static([82, 82, 100]),
                                             opacity=shown(False)),
                       [group(circle(7), fill(colour, static(55)))])  # fmt: skip
    return front_layer, back_layer


def build():
    below, above = [], []  # behind the core, in front of it
    # boundary: a dashed circle, unbroken, turning slowly
    turn = {"a": 1, "k": [{"t": 0, "s": [0], "i": {"x": [1], "y": [1]}, "o": {"x": [0], "y": [0]}},
                          {"t": END, "s": [24]}]}  # fmt: skip
    below.append(layer("boundary", transform((CX, CY)),
                       [group(circle(186), stroke(TEAL, 1.6, 38, dash=(6, turn)))]))  # fmt: skip
    # orbits (faint) and their dots
    for tilt, rx, ry, _rev, _phases in ORBITS:
        below.append(layer("orbit", transform((CX, CY), rotation=static(tilt)),
                           [group(ellipse(rx, ry), stroke(TEAL, 1.2, 22))]))  # fmt: skip
    for k, (tilt, rx, ry, rev, phases) in enumerate(ORBITS):
        for phase in phases:
            front, back = dot_layers(tilt, rx, ry, rev, phase, DOT_COLOURS[k])
            above.append(front)
            below.append(back)
    # ripples from the core
    for t0 in RING_AT:
        t1 = t0 + RING_LEN
        scale = eased([(t0, [40, 40, 100]), (t1, [150, 150, 100])])
        opacity = eased([(t0, [55]), (t1, [0])])
        ring = layer("ripple", transform((CX, CY), scale=scale, opacity=opacity),
                     [group(circle(60), stroke(TEAL, 1.8))])  # fmt: skip
        ring["ip"], ring["op"] = t0, t1
        below.append(ring)
    # the core: a soft glow, a breathing chip, a pinpoint inside
    glow_o = eased([(0, [10]), (60, [20]), (120, [10]), (180, [20]), (END, [10])])
    breath = eased([(0, [100, 100, 100]), (60, [107, 107, 100]), (120, [100, 100, 100]),
                    (180, [107, 107, 100]), (END, [100, 100, 100])])  # fmt: skip
    core = [
        layer("glow", transform((CX, CY), opacity=glow_o), [group(circle(96), fill(TEAL))]),
        layer("chip", transform((CX, CY), scale=breath),
              [group(rect(38, 38, 10), fill(INK), stroke(TEAL, 2)),
               group(rect(22, 22, 5), stroke(TEAL, 1.2, 55)),
               group(circle(6), fill(PALE))]),  # fmt: skip
    ]
    layers = above + core + below  # Lottie draws the last layer first
    for i, lyr in enumerate(layers, start=1):
        lyr["ind"] = i
    return {"v": "5.7.4", "fr": FPS, "ip": 0, "op": END, "w": W, "h": H, "nm": "core",
            "ddd": 0, "assets": [], "layers": layers}  # fmt: skip


if __name__ == "__main__":
    OUT.write_text(json.dumps(build(), separators=(",", ":")))
    print(f"wrote {OUT} ({OUT.stat().st_size} bytes)")

"""The Scores page draws every kind and every test without a server (no Ollama or gateway)."""

from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[1] / "app"
# The page runs inside the platform (app/ first on the path, as main.py has it), not from views/,
# where views/voice_assistant.py would shadow the voice_assistant module.
SCRIPT = f"""
import runpy, sys
sys.path.insert(0, {str(APP)!r})
runpy.run_path({str(APP / "views" / "scores.py")!r}, run_name="__main__")
"""


def test_every_kind_and_test_draws():
    at = AppTest.from_string(SCRIPT, default_timeout=180)
    at.run()
    for kind in ("llm", "speech", "ocr"):
        at.segmented_control[0].set_value(kind).run()
        assert not at.exception, (kind, at.exception[0].value)
        if kind == "ocr":  # one test: no picker
            assert not at.selectbox, kind
            continue
        options = at.selectbox[0].options
        assert options, kind
        for option in options:
            at.selectbox[0].set_value(option).run()
            assert not at.exception, (kind, option, at.exception[0].value)

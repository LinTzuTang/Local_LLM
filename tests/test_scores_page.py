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
    for kind in ("llm", "speech", "ocr", "external"):
        at.segmented_control[0].set_value(kind).run()
        assert not at.exception, (kind, at.exception[0].value)
        if kind == "ocr":  # one test: no picker
            assert len(at.selectbox) == 0, kind
            continue
        if kind == "external":  # AA's leaderboards: a second switch
            try:
                from local_llm import aa_media

                boards = ["ours", "llm", *((aa_media.load() or {}).get("categories") or {})]
            except ImportError:
                boards = ["ours", "llm"]
            for board in boards:  # the values (options are the shown labels)
                at.segmented_control[1].set_value(board).run()
                assert not at.exception, (kind, board, at.exception[0].value)
            continue
        options = at.selectbox[0].options
        assert options, kind
        for option in options:
            at.selectbox[0].set_value(option).run()
            assert not at.exception, (kind, option, at.exception[0].value)


def test_compare_models():
    """Language models, "compare models": a summary row per headline test and a column per chosen
    model; a model's own name is the key across tests, whatever setting or machine it ran on."""
    import sys

    sys.path.insert(0, str(APP))
    import scores_llm

    assert scores_llm.model_key("**Gemma 4 31B**（GPU 主機）") == "Gemma 4 31B"
    assert scores_llm.model_key("Claude Opus 5.5 (cloud)") == "Claude Opus 5.5"
    assert scores_llm.model_key("Breeze 2 8B／TAIDE LX 8B") is None
    found, _ = scores_llm.model_index(scores_llm.catalog())
    assert not [m for m in found if "消融" in m or "abliterated" in m]  # abliterated: own test

    at = AppTest.from_string(SCRIPT, default_timeout=180)
    at.run()
    at.segmented_control[1].set_value("compare").run()
    assert not at.exception, at.exception[0].value
    chosen = ["Gemma 4 31B", "Qwen3.8 27B", "Inkling 2 Mini 26B-A5B"]
    at.multiselect[0].set_value(chosen).run()
    assert not at.exception, at.exception[0].value
    summary = at.dataframe[0].value
    assert list(summary.columns)[1:] == chosen
    assert len(summary) == len(scores_llm.SUMMARY)
    at.multiselect[0].set_value([]).run()
    assert not at.exception and at.info

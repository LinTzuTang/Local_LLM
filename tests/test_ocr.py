import base64
import io

from PIL import Image

from local_llm import moe_bench, ocr


def test_plain_drops_markup_case_and_space():
    md = "## 標題\n\n| A | B |\n|---|---|\n| 1 | 2 |\n\n$x^2$ <b>Bold</b>"
    assert ocr.plain(md) == "標題ab12x2bold"
    assert ocr.distance("Hello  World", "hello world") == 0
    assert ocr.distance("", "abc") == 1


def test_omnidoc_truth_keeps_content_in_reading_order():
    page = {"layout_dets": [
        {"category_type": "text_block", "order": 2, "text": "第二段"},
        {"category_type": "header", "order": 0, "text": "頁首"},
        {"category_type": "title", "order": 1, "text": "標題"},
        {"category_type": "table", "order": 3, "html": "<table><tr><td>1</td></tr></table>"},
        {"category_type": "text_block", "order": 4, "text": "略過", "ignore": True},
    ]}  # fmt: skip
    assert ocr.omnidoc_truth(page) == "標題\n第二段\n<table><tr><td>1</td></tr></table>"


def png(width: int, height: int) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (width, height), "white").save(buf, format="PNG")
    return buf.getvalue()


def size_of(url: str) -> tuple[int, int]:
    raw = base64.b64decode(url.split(",", 1)[1])
    with Image.open(io.BytesIO(raw)) as img:
        return img.size


def test_jev_request_puts_the_original_first_and_pads_thin_crops():
    body = ocr.jev_body(png(300, 40), "一行文字")
    assert body["state"] == "一行文字"
    assert size_of(body["files"][0]) == (300, 40)  # the original, untouched
    assert size_of(body["files"][1])[0] == 300  # the render, as wide as the original
    assert body["questions"]["consistency"]["criteria"] == {"0": None, "1": None}
    assert size_of(ocr.data_url(png(2000, 5))) == (2000, 11)  # over 200:1 -> padded


def test_sample_image_and_engines():
    with Image.open(io.BytesIO(ocr.sample_image())) as img:
        assert img.size == (980, 420)
    assert set(ocr.PROMPTS) <= set(ocr.ENGINES)
    assert all({"label", "runtime", "size_gb", "licence"} <= set(e) for e in ocr.ENGINES.values())


def test_moe_answer_letter_and_pairs():
    assert moe_bench.answer_letter("B") == "B"
    assert moe_bench.answer_letter("<think>A 或 C？</think>答案是 D。") == "D"
    assert moe_bench.answer_letter("Based on the options") is None  # no letter standing alone
    assert moe_bench.MODELS["qwen3:30b-a3b"] == ("A", "MoE")
    assert {kind for _, kind in moe_bench.MODELS.values()} == {"MoE", "dense"}
    assert moe_bench.long_prompt(500).endswith("請用一句話總結上面這份文件的主題。")

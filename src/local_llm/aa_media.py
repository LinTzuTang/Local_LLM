"""Artificial Analysis (artificialanalysis.ai) leaderboards for the Scores page's "External
benchmarks" view: language models, speech to text (recorded files and streaming), text to speech
(English and Mandarin), speech to speech, text to image, text to video and music.

Each board is one AA page; its numbers sit in the page data (the same Next.js stream aa_scores
reads). `fetch_all` reads the pages (one a second, nine pages, plus about ten language-model pages
for parameter counts), keeps AA's top of each board by AA's main measure and writes one file;
`load` returns the saved copy. The file lives in storage/ (not in git: the numbers are AA's, shown
with a link to the source):

    uv run python -m local_llm.aa_media            # download and write the file
    uv run python -m local_llm.aa_media --cache D  # reuse / fill a folder of saved pages

File: storage/llm_eval/aa/aa_media.json
    {"fetched": date, "categories": {id: {"name", "url", "about", "metrics", "total", "rows",
     "open_rows", "fetched"}}, "failed": [ids], "errors": ["id: reason"]}
"total" is how many models AA ranks on the board; "rows" are the first 25 of them, best first;
"open_rows" the first 10 open-weights models (empty when there are none or AA does not say).
Language models ("llm") keep 30 each of "rows", "open_rows" and "small_rows" (open, 4B-40B).
Values are AA's own: fractions stay 0..1 (unit "pct"), Elo and indexes are plain numbers, prices
are US dollars in the unit the metric names. A board that cannot be read is listed under "failed"
and keeps its copy from the previous file, if there was one.
"""

import datetime
import json
import re
import sys
import time
import urllib.request
from collections.abc import Callable
from pathlib import Path

from local_llm.aa_scores import page_data

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "storage" / "llm_eval" / "aa" / "aa_media.json"
SITE = "https://artificialanalysis.ai/"
TOP = 25  # rows kept per board
OPEN_TOP = 10  # open-weights rows kept per board
_DECODER = json.JSONDecoder()


def fetch(page: str, cache: Path | None = None) -> str:
    """One AA page ("speech-to-text/streaming"), cached as speech-to-text_streaming.html."""
    saved = cache / f"{page.replace('/', '_')}.html" if cache else None
    if saved and saved.exists():
        return saved.read_text(encoding="utf-8", errors="ignore")
    req = urllib.request.Request(SITE + page, headers={"User-Agent": "Mozilla/5.0 research-script"})
    html = urllib.request.urlopen(req, timeout=60).read().decode("utf-8", errors="ignore")
    if saved:
        saved.parent.mkdir(parents=True, exist_ok=True)
        saved.write_text(html, encoding="utf-8")
    time.sleep(1)  # one page a second: be gentle
    return html


# ---- reading the page data ----


def objects(data: str, test: Callable[[dict], bool]) -> list[dict]:
    """Every record {"id": ...} in the stream that passes `test`, once per id, in stream order.
    The same record often appears in several places (a chart, a table, a selector); some lists
    only point at it, so collecting the records themselves is the reliable way to get them all."""
    seen: dict = {}
    for m in re.finditer(r'\{"id":\s*"', data):
        try:
            o = _DECODER.raw_decode(data, m.start())[0]
        except ValueError:
            continue
        if test(o):
            seen.setdefault(o["id"], o)
    return list(seen.values())


def chunk(data: str, ref: str):
    """The value of stream row `ref` (the rows are "<hex id>:<json>" lines)."""
    m = re.search(rf"(?:^|\n){re.escape(ref)}:", data)
    if not m:
        raise ValueError(f"no row {ref} in the page data")
    return _DECODER.raw_decode(data, m.end())[0]


def board(data: str, prop: str, key: str | None = None) -> list[dict]:
    """A board of an arena leaderboard (Elo). The component's `prop` holds the rows, or "$Q<row>"
    pointing at them; a map of boards ([[key, rows], ...]) gives the overall one under the null
    key (the others are categories, accents, languages or genres). With `key` (a language such as
    "zh"), a map that has that key gives that board instead."""
    for m in re.finditer(rf'"{re.escape(prop)}":', data):
        try:
            value = _DECODER.raw_decode(data, m.end())[0]
            for _ in range(5):
                if isinstance(value, str) and value.startswith("$Q"):
                    value = chunk(data, value[2:])
                elif isinstance(value, list) and value and isinstance(value[0], list):
                    boards = dict((k, v) for k, v in value)
                    value = boards[key] if key in boards else boards.get(None)
                else:
                    break
        except (ValueError, TypeError):
            continue  # another use of the same name, not the leaderboard
        if (
            isinstance(value, list)
            and value
            and isinstance(value[0], dict)
            and "values" in value[0]
        ):
            return [r["values"] for r in value]
    raise ValueError(f"no leaderboard under {prop!r}")


def ranked(rows: list[dict], field: str, lower_better: bool) -> list[dict]:
    """Rows that have the main measure, best first (AA's order)."""
    rows = [r for r in rows if r.get(field) is not None]
    return sorted(rows, key=lambda r: r[field] if lower_better else -r[field])


def open_weights(v: dict) -> bool | None:
    """AA's open-weights flag; boards that only link the weights say so with the link."""
    if "openWeights" in v:
        return v["openWeights"]
    if "openWeightsUrl" in v:
        return v["openWeightsUrl"] is not None
    return None


# ---- one parser per board: page data -> rows (all ranked rows; the caller keeps the top) ----


def stt_rows(data: str) -> list[dict]:
    """Each model's AA-WER index is its representative endpoint's; speed and price come from that
    same endpoint when the page lists it (some, e.g. models without a public API, have none)."""
    models = objects(data, lambda o: isinstance(o.get("representativeHostModel"), dict))
    hosts = {h["id"]: h for h in objects(data, lambda o: "aaWerIndex" in o and "host" in o)}
    rows = []
    for m in models:
        rep = m["representativeHostModel"]
        host = hosts.get(rep.get("id"), {})
        rows.append({
            "model": m["name"], "creator": (m.get("creator") or {}).get("name"),
            "open_weights": m.get("openWeights"), "wer": rep.get("aaWerIndex"),
            "speed": (host.get("performance") or {}).get("medianSpeedFactor"),
            "price": host.get("pricePer1kMinutes"),
        })  # fmt: skip
    return ranked(rows, "wer", lower_better=True)


def stt_streaming_rows(data: str) -> list[dict]:
    """The streaming page lists endpoints and names no model creator: "creator" is the endpoint's
    provider (the developer for all but third-party hosts such as Together AI)."""
    rows = []
    for h in objects(data, lambda o: "aaWerStreamingIndex" in o):
        rows.append({
            "model": h.get("displayName") or h["name"],
            "creator": (h.get("host") or {}).get("name"),
            "open_weights": (h.get("model") or {}).get("openWeights"),
            "wer": h["aaWerStreamingIndex"], "final_s": h.get("timeToFinalTranscriptSeconds"),
            "price": h.get("pricePer1kMinutes"),
        })  # fmt: skip
    return ranked(rows, "wer", lower_better=True)


def s2s_rows(data: str) -> list[dict]:
    """The index (v2, the one the page charts) is on the arena's overall results; latency and cost
    are on the endpoint the result names."""
    hosts = {h["id"]: h for h in objects(data, lambda o: "bbaScore" in o and "host" in o)}
    rows = []
    for a in objects(
        data, lambda o: o.get("sliceKey") == "overall" and "stsQualityIndexV2Score" in o
    ):
        host = hosts.get(a.get("stsHostModelId"), {})
        rows.append({
            "model": a["name"], "creator": a.get("creatorName"),
            "open_weights": (host.get("model") or {}).get("openSource"),
            "index": a["stsQualityIndexV2Score"], "ttfa": host.get("timeToFirstAudioSeconds"),
            "price": host.get("costPerHourOfInputAudio"),
        })  # fmt: skip
    return ranked(rows, "index", lower_better=False)


def elo_rows(prop: str, price: str | None, key: str | None = None) -> Callable[[str], list[dict]]:
    def parse(data: str) -> list[dict]:
        rows = []
        for v in board(data, prop, key):
            row = {"model": v["name"], "creator": (v.get("creator") or {}).get("name"),
                   "open_weights": open_weights(v), "elo": v.get("elo")}  # fmt: skip
            if price:
                row["price"] = v.get(price)
            rows.append(row)
        return ranked(rows, "elo", lower_better=False)

    return parse


# Language models: the leaderboard page carries every variant (reasoning level) of every model,
# with the Intelligence Index and AA's evaluation scores keyed by evaluation slug.
LLM_SCORES = {
    "gdpval": "gdpval-aa-normalized",
    "terminal_bench": "terminalbench-4-1",
    "lcr": "artificial-analysis-long-context-reasoning",
    "non_hallucination": "omniscience-non-hallucination",
}
SMALL = (4, 40)  # billions of parameters (total, for MoE): the size we run locally


def llm_rows(data: str) -> list[dict]:
    """One row per model, best first by the Intelligence Index. A model that AA lists at several
    reasoning levels (or with and without reasoning) keeps only its highest-scoring variant, so it
    takes one place. "parameters" is filled in later (the leaderboard page does not carry it);
    "size_class" is AA's own size group (tiny / small 4B-40B / medium / large)."""
    start = data.index("[", data.index('"initialModels":'))
    best: dict = {}
    for m in _DECODER.raw_decode(data, start)[0]:
        if m.get("intelligenceIndex") is None:
            continue
        release = (m.get("release") or {}).get("slug") or m["id"]
        if release not in best or m["intelligenceIndex"] > best[release]["intelligenceIndex"]:
            best[release] = m
    rows = []
    for release, m in best.items():
        scores = m.get("scores") or {}
        rows.append({
            "model": m["name"], "creator": m.get("modelCreatorName"),
            "open_weights": m.get("isOpenWeights"), "parameters": None,
            "size_class": m.get("paramClass"), "intelligence_index": m["intelligenceIndex"],
            **{field: scores.get(slug) for field, slug in LLM_SCORES.items()},
            "speed": m.get("outputSpeed"), "price_in": m.get("price1mInputTokens"),
            "price_out": m.get("price1mOutputTokens"), "slug": release, "_id": m["id"],
        })  # fmt: skip
    return ranked(rows, "intelligence_index", lower_better=False)


def add_parameters(rows: list[dict], cache: Path | None = None) -> None:
    """Fill "parameters" (billions, total for MoE) of the open-weights rows from AA's model pages
    (aa_scores reads them). A model page also lists comparable models, so most rows are found
    without their own page; a page that cannot be read leaves its rows at None."""
    from local_llm import aa_scores  # the language-model pages and their reader

    known: dict = {}
    for row in rows:
        if not row["open_weights"] or row["_id"] in known:
            continue
        try:
            for m in aa_scores.variants(aa_scores.fetch(row["slug"], cache)).values():
                known.setdefault(m.get("id"), m.get("parameters"))
        except Exception:  # noqa: BLE001, S112  (page gone or changed: parameters stay None)
            continue
    for row in rows:
        row["parameters"] = known.get(row["_id"])


def llm_lists(rows: list[dict], cache: Path | None = None) -> dict:
    """All models, open-weights models, and open models of 4B-40B (by AA's parameter count; AA's
    "small" group when the count is missing), 30 each."""
    add_parameters(rows, cache)
    open_rows = [r for r in rows if r["open_weights"]]

    def small(r):
        if r["parameters"] is None:
            return r["size_class"] == "small"
        return SMALL[0] <= r["parameters"] <= SMALL[1]

    lists = {"rows": rows[:30], "open_rows": open_rows[:30],
             "small_rows": [r for r in open_rows if small(r)][:30]}  # fmt: skip
    for kept in lists.values():
        for r in kept:
            r.pop("_id", None)
    return lists


def _m(field, zh, en, unit, lower_better):
    return {
        "field": field,
        "name": {"zh": zh, "en": en},
        "unit": unit,
        "lower_better": lower_better,
    }


ELO = _m("elo", "Arena Elo（人盲測投票）", "Arena Elo (blind human votes)", "elo", False)

# id -> page, parser, and what the Scores page says about the board. "about" follows AA's own pages
# (the board and its methodology page); unit is pct (a 0..1 fraction), elo, index (0-100), x
# (times real time), usd, s or tps (tokens per second).
CATEGORIES = {
    "llm": {
        "page": "leaderboards/models",
        "parse": llm_rows,
        "name": {"zh": "語言模型", "en": "Language models"},
        "about": {
            "zh": "AA 綜合指數（Intelligence Index）：AA 自己跑的多項英文評測加權（知識工作、"
            "寫程式、終端機任務、科學、知識與幻覺等），全部走各家 API、用完整精度，思考開到"
            "該版本的設定。同一模型有多個推理等級或思考開關時，只留指數最高的那一版。三份清單："
            "總排名、開源權重、開源且 4B–40B（AA 的參數數，MoE 算總參數；AA 沒給參數時用 AA 的"
            "「small」分組）。子分數：GDPval-AA（44 種職業的工作成品，評審兩兩比較的 Elo）、"
            "Terminal-Bench 4.1（終端機工程任務）、AA-LCR（長文件推理）、AA-Omniscience 不幻覺率"
            "（答不出來時不硬給錯答案的比例，越高越好）。速度是每秒輸出 token 數，價格是每百萬"
            "token。我們地端跑的是 4-bit 量化，分數通常比 AA 低。",
            "en": "AA's Intelligence Index: a weighted set of AA's own English evaluations "
            "(knowledge work, coding, terminal tasks, science, knowledge and hallucination), "
            "all run through each provider's API at full precision, reasoning as the variant is "
            "set. A model listed at several reasoning levels keeps only its highest-scoring "
            "variant. Three lists: all models, open weights, and open weights at 4B-40B (AA's "
            "parameter count, total for MoE; AA's 'small' group when the count is missing). "
            "Sub-scores: GDPval-AA (work products for 44 occupations, pairwise-judged Elo), "
            "Terminal-Bench 4.1 (engineering tasks in a terminal), AA-LCR (long-document "
            "reasoning), AA-Omniscience non-hallucination (share of unanswerable cases where the "
            "model does not guess wrong; higher is better). Speed in output tokens per second, "
            "prices per 1M tokens. We run 4-bit quantized models locally, which usually score "
            "lower.",
        },
        "metrics": [
            _m("intelligence_index", "綜合指數", "Intelligence Index", "index", False),
            _m("gdpval", "GDPval-AA", "GDPval-AA", "elo", False),
            _m("terminal_bench", "Terminal-Bench 4.1", "Terminal-Bench 4.1", "pct", False),
            _m("lcr", "長文推理 AA-LCR", "Long context AA-LCR", "pct", False),
            _m("non_hallucination", "不幻覺率", "Non-hallucination rate", "pct", False),
            _m("speed", "輸出速度", "Output speed", "tps", False),
            _m("price_in", "輸入價格（每百萬 token）", "Input price per 1M tokens", "usd", True),
            _m("price_out", "輸出價格（每百萬 token）", "Output price per 1M tokens", "usd", True),
        ],
    },
    "stt": {
        "page": "speech-to-text/non-streaming",
        "parse": stt_rows,
        "name": {"zh": "語音轉文字（錄好的檔案）", "en": "Speech to text (recorded files)"},
        "about": {
            "zh": "AA-WER v2.2：三個英文資料集約 8 小時——AA 自己的語音助理對話 "
            "AA-AgentTalk（不公開，占 50%）、歐洲議會演講 VoxPopuli（25%）、"
            "企業法說會 Earnings22（25%），用英文的正規化規則（以 Whisper 的為基礎）"
            "比對後算字錯率。只有英文，沒有中文。速度是 1 秒能處理幾秒錄音"
            "（近 7 天中位數，含網路），價格換算成每千分鐘錄音。",
            "en": "AA-WER v2.2: about 8 hours from three English datasets - AA's own voice-agent "
            "conversations AA-AgentTalk (held out, 50%), European Parliament speeches "
            "VoxPopuli (25%) and earnings calls Earnings22 (25%) - scored after an English "
            "text normalizer built on Whisper's. English only; no Chinese. Speed is seconds "
            "of audio per second (7-day median, network included); price per 1,000 minutes.",
        },
        "metrics": [
            _m("wer", "字錯率（AA-WER）", "Word error rate (AA-WER)", "pct", True),
            _m("speed", "速度（幾倍即時）", "Speed factor (x real time)", "x", False),
            _m("price", "每千分鐘錄音價格", "Price per 1,000 min of audio", "usd", True),
        ],
    },
    "stt_streaming": {
        "page": "speech-to-text/streaming",
        "parse": stt_streaming_rows,
        "name": {"zh": "語音轉文字（即時串流）", "en": "Speech to text (streaming)"},
        "about": {
            "zh": "同樣三個英文資料集（AA-AgentTalk 50%、VoxPopuli 25%、Earnings22 25%），聲音以"
            "真實速度每 20 毫秒送一段；偵測到說完（SileroVAD）就要求定稿，算定稿文字的字錯率，"
            "並量說完到收到定稿的秒數（含網路延遲）。只有英文。「開發商」欄是 AA 列的 API 供應商。",
            "en": "The same three English datasets (AA-AgentTalk 50%, VoxPopuli 25%, Earnings22 "
            "25%), streamed in real time in 20 ms chunks; at the end of speech (SileroVAD) a "
            "final transcript is forced. Scores the final transcript's word error rate and "
            "the seconds from end of speech to that final (network included). English only. "
            "The creator column is the API provider AA lists.",
        },
        "metrics": [
            _m("wer", "字錯率（AA-WER 串流）", "Word error rate (AA-WER Streaming)", "pct", True),
            _m("final_s", "說完到定稿", "Time to final transcript", "s", True),
            _m("price", "每千分鐘錄音價格", "Price per 1,000 min of audio", "usd", True),
        ],
    },
    "tts": {
        "page": "text-to-speech/leaderboard/provider-voice",
        "parse": elo_rows("data", "pricePer1mCharacters"),
        "name": {"zh": "文字轉語音", "en": "Text to speech"},
        "about": {
            "zh": "Arena 盲測（各家自己的聲音）：同一段文字由兩個模型各唸一次，每個模型"
            "用 8 個美式與英式口音的聲音，不顯示模型名稱，由住在美國或英國、母語是"
            "英文的評審選比較喜歡的，再用 Bradley-Terry 算成 Elo。這個榜只有英文；"
            "AA 另有用同一組複製聲音比的榜（Controlled Voice），有普通話等 9 種語言。"
            "價格是開發商 API 每百萬字元的費用。",
            "en": "Arena with each provider's own voices: two models read the same text (8 US and "
            "UK English voices per model), names hidden; native English raters in the US or "
            "UK pick the one they prefer, fitted to Elo with Bradley-Terry. This board is "
            "English only; AA's separate Controlled Voice board (same cloned voices) also "
            "covers 9 other languages including Mandarin. Price per 1M characters on the "
            "creator's API.",
        },
        "metrics": [ELO, _m("price", "每百萬字元價格", "Price per 1M characters", "usd", True)],
    },
    "tts_zh": {
        "page": "text-to-speech/leaderboard/controlled-voice",
        "parse": elo_rows("data", "pricePer1mCharacters", key="zh"),
        "name": {"zh": "文字轉語音（普通話）", "en": "Text to speech (Mandarin)"},
        "about": {
            "zh": "Controlled Voice Arena 的普通話榜：每個模型都用同一組複製的聲音（一男一女，"
            "由母語者專業錄製的參考聲音），只比支援普通話的模型；評審篩選為第一語言是該語言、"
            "盡量住在原生國家的人，盲選比較喜歡的，用 Bradley-Terry 算成 Elo。比數夠多才公布"
            "這個語言的榜。AA 寫的是普通話（Mandarin Chinese），沒有說明是台灣還是大陸口音。"
            "價格是開發商 API 每百萬字元的費用。",
            "en": "The Mandarin board of AA's Controlled Voice Arena: every model speaks with the "
            "same cloned voices (one male, one female reference recorded by native speakers), "
            "only models that support the language; raters are screened for it as their first "
            "language, living in the country of origin where possible, and pick blind, fitted "
            "to Elo with Bradley-Terry. Published once there are enough comparisons. AA says "
            "Mandarin Chinese without naming an accent. Price per 1M characters on the "
            "creator's API.",
        },
        "metrics": [ELO, _m("price", "每百萬字元價格", "Price per 1M characters", "usd", True)],
    },
    "s2s": {
        "page": "speech-to-speech",
        "parse": s2s_rows,
        "name": {"zh": "語音對語音（即時語音對話）", "en": "Speech to speech (live voice)"},
        "about": {
            "zh": "語音對語音指數＝四項各占 25%：語音推理（Big Bench Audio，Big Bench Hard 的題目"
            "用合成語音唸出來）、代理任務（τ-Voice，模擬航空／零售／電信客服，只有英文）、"
            "Arena 偏好（真人和兩個匿名模型完成同一情境後選喜歡的）、任務成功率。四項都有結果"
            "才有指數，所以目前只有封閉模型。延遲是 Big Bench Audio 上第一段聲音出來的秒數；"
            "成本是固定 40 題依輸入語音長度換算的每小時費用。",
            "en": "Speech to Speech Index = four parts at 25% each: speech reasoning (Big Bench "
            "Audio, Big Bench Hard questions read by synthetic voices), agentic performance "
            "(tau-Voice customer service in airline / retail / telecom, English only), arena "
            "preference (people complete the same scenario with two hidden models) and task "
            "success rate. Only models with all four have an index, so all are closed models "
            "so far. Latency is time to first audio on Big Bench Audio; cost per hour of "
            "input audio on a fixed 40-question set.",
        },
        "metrics": [
            _m("index", "語音對語音指數", "Speech to Speech Index", "index", False),
            _m("ttfa", "第一段聲音延遲", "Time to first audio", "s", True),
            _m("price", "每小時輸入語音成本", "Cost per hour of input audio", "usd", True),
        ],
    },
    "image": {
        "page": "image/leaderboard/text-to-image",
        "parse": elo_rows("initialData", "pricePer1kImages"),
        "name": {"zh": "文字生圖", "en": "Text to image"},
        "about": {
            "zh": "AA-Image-T2I v2.1：同一段提示詞由兩個模型各生一張，AA 審核過的私人評審在不知道"
            "模型的情況下選比較好的，用 Bradley-Terry 算成 Elo（2026 年 10 月換新提示詞後重新"
            "計票）。提示詞依使用情境與能力平均抽樣。價格是開發商 API 以 1024×1024 生成 1,000"
            " 張的費用。",
            "en": "AA-Image-T2I v2.1: two models draw the same prompt; AA's vetted private "
            "evaluators pick the better one blind, fitted to Elo with Bradley-Terry (votes "
            "since the October 2026 prompt refresh). Prompts are sampled evenly across use "
            "cases and capabilities. Price per 1,000 images at 1024x1024 on the creator's API.",
        },
        "metrics": [ELO, _m("price", "每千張價格", "Price per 1,000 images", "usd", True)],
    },
    "video": {
        "page": "video/leaderboard/text-to-video",
        "parse": elo_rows("withAudioData", "pricePerMinute"),
        "name": {"zh": "文字生影片（有聲音）", "en": "Text to video (with audio)"},
        "about": {
            "zh": "AA-Video-T2V v2.1：同一段提示詞由兩個模型各生一段有聲音的影片，評審戴耳機連聲音"
            "一起評、盲選比較喜歡的，用 Bradley-Terry 算成 Elo。無聲影片另有一個榜，兩者不互相"
            "比。價格是開發商 API 在 AA 測試設定（有 1080p 就用 1080p）下生成 1 分鐘影片的費用。",
            "en": "AA-Video-T2V v2.1: two models make a video with sound from the same prompt; "
            "evaluators judge picture and sound (with headphones) blind and pick one, fitted "
            "to Elo with Bradley-Terry. Silent video has its own board; the two are not "
            "compared. Price per minute of video on the creator's API at AA's settings "
            "(1080p where offered).",
        },
        "metrics": [ELO, _m("price", "每分鐘影片價格", "Price per minute of video", "usd", True)],
    },
    "music": {
        "page": "music/leaderboard/vocals",
        "parse": elo_rows("initialData", None),
        "name": {"zh": "音樂生成（有人聲）", "en": "Music generation (with vocals)"},
        "about": {
            "zh": "AA-Music-Vocal v1.1：同一段提示詞由兩個模型各生成一首有人聲的曲子"
            "（先統一音量），評審聽完盲選比較喜歡的，用 Bradley-Terry 算成 Elo。"
            "純音樂另有一個榜。AA 這個榜沒有標示是否開源，也沒有價格。",
            "en": "AA-Music-Vocal v1.1: two models make a song with vocals from the same prompt "
            "(loudness normalized); evaluators listen and pick one blind, fitted to Elo with "
            "Bradley-Terry. Instrumental music has its own board. This board shows neither "
            "open-weights status nor prices.",
        },
        "metrics": [ELO],
    },
}


def category(cid: str, data: str, cache: Path | None = None) -> dict:
    """One board from its page data; raises when the page has none (AA changed the page).
    "rows" are the first TOP of the ranking, "open_rows" the first OPEN_TOP open-weights models
    (an empty list when the board has none or does not say); language models also get
    "small_rows" (see llm_lists)."""
    spec = CATEGORIES[cid]
    rows = spec["parse"](data)
    if not rows:
        raise ValueError("no ranked rows")
    out = {"name": spec["name"], "url": SITE + spec["page"], "about": spec["about"],
           "metrics": spec["metrics"], "total": len(rows)}  # fmt: skip
    if cid == "llm":
        out |= llm_lists(rows, cache)
    else:
        out["rows"] = rows[:TOP]
        out["open_rows"] = [r for r in rows if r["open_weights"]][:OPEN_TOP]
    return out


def fetch_all(cache: Path | None = None, progress=lambda done, total: None) -> dict:
    """Download every board and write OUT. A board that fails is named under "failed" and keeps
    its copy from the old file; when every board fails nothing is written (raises)."""
    today = datetime.date.today().isoformat()
    old = (load() or {}).get("categories", {})
    out, failed, errors = {}, [], []
    for n, cid in enumerate(CATEGORIES):
        try:
            out[cid] = category(cid, page_data(fetch(CATEGORIES[cid]["page"], cache)), cache)
            out[cid]["fetched"] = today
        except Exception as exc:  # noqa: BLE001  (network, or AA changed this page)
            failed.append(cid)
            errors.append(f"{cid}: {exc}")
            if cid in old:
                out[cid] = old[cid]
        progress(n + 1, len(CATEGORIES))
    if len(failed) == len(CATEGORIES):
        raise RuntimeError("every AA board failed; " + "; ".join(errors))
    data = {"fetched": today, "categories": out, "failed": failed, "errors": errors}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    return data


def load() -> dict | None:
    """The saved boards, or None when this server has not fetched them yet."""
    try:
        data = json.loads(OUT.read_text(encoding="utf-8"))
        return data if isinstance(data.get("categories"), dict) else None
    except (OSError, ValueError, AttributeError):
        return None


def main() -> None:
    cache = Path(sys.argv[sys.argv.index("--cache") + 1]) if "--cache" in sys.argv else None
    data = fetch_all(cache)
    for cid, cat in data["categories"].items():
        print(f"{cid}: {len(cat['rows'])} of {cat['total']} rows ({cat['fetched']})")
    for err in data["errors"]:
        print(f"failed: {err}", file=sys.stderr)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()

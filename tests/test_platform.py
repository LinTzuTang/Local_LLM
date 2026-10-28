import re

import pytest

from local_llm import catalog, ollama_admin
from local_llm.streaming import Backend


def test_catalog_entries_are_complete():
    data = catalog.load_catalog()
    assert catalog.default_model() == "qwen3:30b-a3b"
    assert [m["tag"] for m in data["models"] if m.get("default")] == ["qwen3:30b-a3b"]
    required = {"tag", "family", "description", "params", "context", "size_gb", "updated"}
    required |= {"caps", "tags", "license", "fit", "note", "note_en", "tasks"}
    for m in data["models"]:
        assert required <= m.keys(), (m["tag"], required - m.keys())
        assert set(m["tasks"]) <= data["tasks"].keys(), m["tag"]
        assert m["fit"] in catalog.FIT_LABELS
        assert set(m["caps"]) <= set(catalog.CAPABILITY_KEYS), m["tag"]
        assert set(m["tags"]) <= set(catalog.TAG_KEYS), m["tag"]
        assert re.fullmatch(r"\d{4}-\d{2}", m["updated"]), m["tag"]
        assert isinstance(m["context"], int) and m["context"] >= 512, m["tag"]
    tags = [m["tag"] for m in data["models"]]
    assert len(tags) == len(set(tags))
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", data["checked_date"])


def test_catalog_english_text_is_complete():
    data = catalog.load_catalog()
    assert data["tasks_en"].keys() == data["tasks"].keys()
    assert data["other_categories_en"].keys() == data["other_categories"].keys()
    for t in data["tiers"]:
        assert t["name_en"] and t["summary_en"] and len(t["suggested_en"]) == len(t["suggested"])
    han = re.compile(r"[\u4e00-\u9fff]")
    for m in data["models"]:
        assert m["note_en"] and not han.search(m["note_en"]), m["tag"]
        for field in ("license", "description"):
            if han.search(m[field]):
                assert m.get(f"{field}_en") or field == "description", (m["tag"], field)
            if m.get(f"{field}_en"):
                assert not han.search(m[f"{field}_en"]), (m["tag"], field)


def test_derived_tags_match_the_model_data():
    for m in catalog.models():
        tags, tag = set(m["tags"]), m["tag"]
        assert ("long_context" in tags) == (m["context"] >= catalog.LONG_CONTEXT), tag
        assert ("moe" in tags) == ("active" in m["params"]), tag
        for key in ("code", "translate", "ocr", "taiwan"):
            assert (key in tags) == (key in m["tasks"]), (tag, key)
        assert ("non_commercial" in tags) == ("非商用" in m["license"]), tag
        assert not {"commercial_ok", "non_commercial"} <= tags, tag


def test_catalog_helpers():
    m = catalog.find("qwen3:30b-a3b")
    caps = catalog.all_capabilities(m)
    assert caps[: len(m["caps"])] == m["caps"] and len(caps) == len(set(caps))
    assert {"tools", "thinking", "moe", "long_context"} <= set(caps)
    assert catalog.context_label({"context": 262144}) == "256K"
    assert catalog.context_label({"context": 1048576}) == "1M"
    assert catalog.context_label({"context": 512}) == "512"
    assert catalog.page_url(m) == "https://ollama.com/library/qwen3"
    assert catalog.page_url(catalog.find("command-r7b")).endswith("/command-r7b")
    assert catalog.pull_command(m) == "ollama pull qwen3:30b-a3b"
    assert catalog.families() == sorted(catalog.families(), key=str.casefold)
    # models the other pages refer to by name
    for tag in ("qwen3:1.7b", "bge-m3", "nimble:9b", "command-r7b", "aya-expanse:8b"):
        assert catalog.find(tag), tag


def test_tag_without_version_means_latest():
    assert catalog.same_model("bge-m3", "bge-m3:latest")
    assert not catalog.same_model("qwen3:8b", "qwen3:latest")
    assert catalog.find("bge-m3:latest")["tag"] == "bge-m3"


@pytest.mark.parametrize(
    ("show", "expected"),
    [
        ({"capabilities": ["completion"]}, "none"),
        ({"capabilities": ["thinking"], "template": "{{ if (not $.Think) }}"}, "optional"),
        ({"capabilities": ["thinking"], "template": "<|im_start|>assistant\n<think>\n"}, "always"),
    ],
)  # fmt: skip
def test_thinking_mode_from_template(monkeypatch, show, expected):
    ollama_admin.thinking_mode.cache_clear()
    monkeypatch.setattr(ollama_admin, "show", lambda model, base_url: show)
    assert ollama_admin.thinking_mode("m") == expected
    assert (
        ollama_admin.think_value("m", False)
        == {"none": None, "optional": False, "always": True}[expected]
    )
    ollama_admin.thinking_mode.cache_clear()


def test_openai_backends_need_v1():
    with pytest.raises(ValueError):
        Backend("vLLM", "vllm", "http://127.0.0.1:8000")
    with pytest.raises(ValueError):
        Backend("x", "openai", "http://127.0.0.1:8000/v1")
    Backend("vLLM", "vllm", "http://127.0.0.1:8000/v1")


def test_every_model_has_original_description_and_tier():
    for m in catalog.models():
        assert m["description"], m["tag"]
        assert catalog.tier_of(m)["name"]
    assert catalog.tier_of({"size_gb": 1.2})["key"] == "micro"
    assert catalog.tier_of({"size_gb": 18})["key"] == "large"
    assert catalog.tier_of({"size_gb": 65})["key"] == "xl"


def test_every_agent_tool_is_documented():
    from local_llm.agents import TOOL_INFO, all_tools

    # load_skill is created per agent from its skills, so it is documented but not listed
    assert set(all_tools()) | {"load_skill"} == set(TOOL_INFO)


@pytest.fixture
def company_tools():
    from local_llm import company_db

    if company_db.status():
        pytest.skip("company database not running (./scripts/start_db.sh)")
    return {role: {t.name: t for t in company_db.make_tools(role)} for role in company_db.ROLES}


def test_company_roles_are_enforced_by_postgres(company_tools):
    sales, hr = company_tools["agent_sales"], company_tools["agent_hr"]
    assert "Permission denied" in sales["db_query"].invoke("SELECT salary FROM hr.employees")
    described = sales["db_describe_table"].invoke("hr.employees")
    assert '"column_name": "salary"' not in described
    assert described.startswith("PERMISSION NOTICE")
    assert "Permission denied" in hr["db_query"].invoke("SELECT count(*) FROM sales.orders")
    assert '"n": 4200' in sales["db_query"].invoke("SELECT count(*) AS n FROM sales.orders")


def test_company_db_is_read_only(company_tools):
    query = company_tools["agent_sales"]["db_query"]
    query.invoke("WITH x AS (DELETE FROM sales.orders RETURNING 1) SELECT count(*) FROM x")
    assert query.invoke("DELETE FROM sales.orders") == "Only SELECT queries are allowed."
    assert '"n": 4200' in query.invoke("SELECT count(*) AS n FROM sales.orders")


def test_uploaded_table_is_private_until_granted(company_tools):
    import pandas as pd

    from local_llm import company_db

    df = pd.DataFrame({"專案": ["甲", "乙"], "預算": [100, 250]})
    company_db.import_dataframe(df, "uploads", "pytest_projects", "測試", replace=True)
    try:
        sales = {t.name: t for t in company_db.make_tools("agent_sales")}
        query = 'SELECT sum("預算") AS total FROM uploads.pytest_projects'
        assert "Permission denied" in sales["db_query"].invoke(query)
        assert sales["db_list_tables"].invoke({}).startswith("PERMISSION NOTICE")
        company_db.set_access("uploads.pytest_projects", "agent_sales", ["專案"])
        assert "Permission denied" in sales["db_query"].invoke(query)
        company_db.set_access("uploads.pytest_projects", "agent_sales", [])
        assert '"total": 350' in sales["db_query"].invoke(query)
        company_db.set_access("uploads.pytest_projects", "agent_sales", None)
        assert "Permission denied" in sales["db_query"].invoke(query)
    finally:
        company_db.drop_table("uploads.pytest_projects")


def test_upload_rejects_unsafe_table_names(company_tools):
    import pandas as pd

    from local_llm import company_db

    for name in ["Drop Table", "1abc", "x;drop", "訂單"]:
        with pytest.raises(ValueError):
            company_db.import_dataframe(pd.DataFrame({"a": [1]}), "uploads", name)
    with pytest.raises(ValueError):
        company_db.drop_table("sales.orders")


def test_hf_models_are_marked():
    assert catalog.is_huggingface("hf.co/audreyt/Gemma-3-TAIDE-12b-Chat-2602-GGUF:Q4_K_M")
    assert catalog.models("taiwan")


def test_other_runtime_models_are_complete():
    data = catalog.load_catalog()
    for m in catalog.other_models():
        assert {
            "name",
            "repo",
            "use",
            "params",
            "license",
            "runtime",
            "install",
            "note",
        } <= m.keys()
        assert m["category"] in data["other_categories"], m["name"]
        assert m["mac"] in data["mac_labels"], m["name"]
    assert {m["category"] for m in catalog.other_models()} == set(data["other_categories"])


SPEECH_CATEGORIES = {"asr", "stream", "tts", "voice_chat", "speech_translate", "vad", "diarize"}


def test_other_categories_cover_every_speech_task():
    data = catalog.load_catalog()
    assert SPEECH_CATEGORIES <= data["other_categories"].keys()
    for field in ("other_category_notes", "other_category_notes_en"):
        assert data[field].keys() == data["other_categories"].keys(), field
    assert data["lang_labels_en"].keys() == data["lang_labels"].keys()
    assert data["mac_labels_en"].keys() == data["mac_labels"].keys()


def test_catalog_only_entries_link_to_their_repo_and_cannot_be_downloaded():
    data = catalog.load_catalog()
    han = re.compile(r"[\u4e00-\u9fff]")
    names = [m["name"] for m in catalog.other_models()]
    assert len(names) == len(set(names))
    served = {m["name"] for m in catalog.gateway_models()}
    gateway_ids = {m["id"] for m in catalog.gateway_models()}
    for m in catalog.other_models():
        assert catalog.is_catalog_only(m) == (m.get("status") == "catalog"), m["name"]
        assert set(m.get("langs") or []) <= data["lang_labels"].keys(), m["name"]
        assert set(m.get("license_flags") or []) <= {"non_commercial", "gated", "custom"}
        if not catalog.is_catalog_only(m):
            assert m["gateway"]["id"] in gateway_ids
            continue
        assert m["name"] not in served, m["name"]
        if via := m.get("served_by"):  # a mode of a model the gateway already serves
            assert via in gateway_ids, m["name"]
        assert m["mac"] == "native" or catalog.other_url(m), m["name"]
        assert m.get("size_gb") is None or m["size_gb"] >= 0, m["name"]
        for field in ("name", "params", "license", "runtime", "install", "use", "note"):
            text = m.get(f"{field}_en") or m[field]
            assert not han.search(text), (m["name"], field)  # English text is complete
    for key in SPEECH_CATEGORIES - {"diarize"}:
        assert any(catalog.is_catalog_only(m) for m in catalog.other_models(key)), key


def test_cloud_models_are_separate_and_marked():
    data = catalog.load_catalog()
    han = re.compile(r"[\u4e00-\u9fff]")
    clouds = catalog.cloud_models()
    deployed = {m["name"] for m in clouds if m["deployed"]}
    assert deployed >= {"whisper", "gpt-4o-transcribe", "gpt-realtime"}
    assert {m["name"] for m in clouds if not m["deployed"]} >= {"gpt-4o-mini-tts"}
    assert all(m.get("provider") and m.get("paid") for m in clouds)  # every one is paid
    assert any(m.get("narration") for m in clouds if m["category"] == "tts")
    assert not {m["name"] for m in clouds} & {m["name"] for m in catalog.other_models()}
    for m in clouds:
        assert m["category"] in data["other_categories"], m["name"]
        assert isinstance(m["deployed"], bool) and m["url"].startswith("https://")
        assert m["use"] and m["use_en"] and not han.search(m["use_en"] + m["note_en"])
        assert "gateway" not in m and "size_gb" not in m  # nothing to download
    assert data["cloud_note"] and not han.search(data["cloud_note_en"])


def test_licence_keys():
    assert catalog.licence_keys({"license": "apache-2.0"}) == ["commercial_ok"]
    assert catalog.licence_keys({"license": "cc-by-nc-4.0（非商用）"}) == ["non_commercial"]
    gated = {"license": "mit", "license_flags": ["gated"]}
    assert catalog.licence_keys(gated) == ["commercial_ok", "gated"]
    custom = {"license": "Llama 3.2 Community License", "license_flags": ["custom"]}
    assert catalog.licence_keys(custom) == ["custom_license"]


def test_decision_and_embedding_models_are_not_offered_for_chat():
    assert catalog.is_chat("qwen3:30b-a3b")
    assert not catalog.is_chat("nimble:9b")
    assert not catalog.is_chat("bge-m3")
    for m in catalog.models():
        special = {"embedding", "decision"} & set(m["caps"])
        assert catalog.is_chat(m["tag"]) == (not special), m["tag"]


def test_thinking_mode_reads_declared_values(monkeypatch):
    from local_llm import ollama_admin

    shows = {
        "mlx": {"capabilities": ["thinking"], "thinking": {"values": [False, True]}},
        "always": {"capabilities": ["thinking"], "thinking": {"values": [True]}},
        "template": {"capabilities": ["thinking"], "template": "{{ if not $.Think }}"},
        "plain": {"capabilities": ["completion"]},
    }
    monkeypatch.setattr(ollama_admin, "show", lambda model, base_url=None: shows[model])
    assert ollama_admin.thinking_mode("mlx") == "optional"
    assert ollama_admin.thinking_mode("always") == "always"
    assert ollama_admin.thinking_mode("template") == "optional"
    assert ollama_admin.thinking_mode("plain") == "none"

"""Find every cloud language model this server's keys can actually call, and write the list the
platform offers behind the cloud switch (data/lab/cloud_llms.json, read by local_llm.cloud_llm).

    uv run python scripts/cloud_llm_probe.py          # probe and write the list
    uv run python scripts/cloud_llm_probe.py --dry    # probe and print only

A model is listed only after it answers one short prompt: a model the key can see but not call
(no access granted, not in this region, retired) stays off the list.
- AWS Bedrock (IAM keys): every text model in ListFoundationModels, called through a global or US
  inference profile when there is one, else on demand; Converse API.
- Azure OpenAI: every deployment (GET /openai/deployments), OpenAI-compatible /openai/v1.
- Vertex AI (service account): Google's Gemini models (publishers/google/models), the
  OpenAI-compatible endpoint.
Embedding, image, video, audio and rerank models are skipped.
"""

import argparse
import concurrent.futures as cf
import datetime as dt
import json
import re
import sys
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
OUT = ROOT / "data" / "lab" / "cloud_llms.json"
PROMPT = "用一句話說你好。"
SKIP = re.compile(
    r"embed|rerank|image|vision-only|video|pegasus|marengo|tts|whisper|transcri|audio|"
    r"realtime|dall|sora|canvas|reel|sonic|guard|moderation|titan-text-lite|banana|voxtral|palmyra-vision|mixtral",
    re.I,
)


def env() -> dict:
    from dotenv import dotenv_values

    return {k: v for k, v in dotenv_values(ROOT / ".env").items() if v}


def bedrock(e: dict) -> list[dict]:
    import boto3

    s = boto3.Session(
        aws_access_key_id=e["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=e["AWS_SECRET_ACCESS_KEY"],
        region_name=e.get("AWS_REGION") or "us-east-1",
    )
    b, rt = s.client("bedrock"), s.client("bedrock-runtime")
    models = b.list_foundation_models(byOutputModality="TEXT")["modelSummaries"]
    profiles = {
        p["inferenceProfileId"]
        for p in b.list_inference_profiles(maxResults=1000)["inferenceProfileSummaries"]
    }

    def call_id(m):
        for pre in ("global.", "us."):
            if pre + m["modelId"] in profiles:
                return pre + m["modelId"]
        return m["modelId"] if "ON_DEMAND" in m.get("inferenceTypesSupported", []) else None

    def probe(m):
        cid = call_id(m)
        if not cid or SKIP.search(m["modelId"]) or "TEXT" not in m.get("inputModalities", ["TEXT"]):
            return None
        try:
            r = rt.converse(
                modelId=cid,
                messages=[{"role": "user", "content": [{"text": PROMPT}]}],
                inferenceConfig={"maxTokens": 200},
            )
            text = "".join(c.get("text", "") for c in r["output"]["message"]["content"])
            if not text.strip():
                return None
        except Exception as exc:  # noqa: BLE001 - not callable with this key: left off the list
            print("  bedrock", cid, "->", str(exc)[:90])
            return None
        return {
            "id": f"cloud/bedrock/{cid}",
            "call": cid,
            "provider": "AWS",
            "vendor": m["providerName"],
            "label": m["modelName"],
            "api": "anthropic" if m["providerName"] == "Anthropic" else "converse",
        }

    with cf.ThreadPoolExecutor(8) as ex:
        return [r for r in ex.map(probe, models) if r]


def azure(e: dict) -> list[dict]:
    ep = e["AZURE_OPENAI_ENDPOINT"].rstrip("/").split("/openai")[0]
    h = {"api-key": e["AZURE_OPENAI_API_KEY"]}
    deps = httpx.get(
        f"{ep}/openai/deployments?api-version=2023-03-15-preview", headers=h, timeout=30
    ).json()
    names = sorted({d["id"] for d in deps.get("data", [])})

    def probe(name):
        if SKIP.search(name):
            return None
        r = httpx.post(
            f"{ep}/openai/v1/chat/completions",
            headers=h,
            timeout=90,
            json={
                "model": name,
                "messages": [{"role": "user", "content": PROMPT}],
                "max_completion_tokens": 2000,
            },
        )
        if (
            r.status_code != 200
            or not (r.json()["choices"][0]["message"].get("content") or "").strip()
        ):
            print("  azure", name, "->", r.status_code, r.text[:90])
            return None
        return {
            "id": f"cloud/azure/{name}",
            "call": name,
            "provider": "Azure",
            "vendor": "OpenAI",
            "label": name,
            "api": "openai",
        }

    with cf.ThreadPoolExecutor(6) as ex:
        return [r for r in ex.map(probe, names) if r]


def vertex() -> list[dict]:
    from local_llm import gcp_auth

    token, project = gcp_auth.token()
    h = {"Authorization": f"Bearer {token}"}
    r = httpx.get(
        "https://aiplatform.googleapis.com/v1beta1/publishers/google/models?pageSize=200",
        headers=h,
        timeout=30,
    )
    names = sorted(
        {
            m["name"].split("/")[-1]
            for m in r.json().get("publisherModels", [])
            if m["name"].split("/")[-1].startswith("gemini")
        }
    )
    url = f"https://aiplatform.googleapis.com/v1/projects/{project}/locations/global/endpoints/openapi/chat/completions"

    def probe(name):
        if SKIP.search(name) or "live" in name or "native-audio" in name:
            return None
        r = httpx.post(
            url,
            headers=h,
            timeout=90,
            json={
                "model": f"google/{name}",
                "max_tokens": 2000,
                "messages": [{"role": "user", "content": PROMPT}],
            },
        )
        if (
            r.status_code != 200
            or not (r.json()["choices"][0]["message"].get("content") or "").strip()
        ):
            print("  vertex", name, "->", r.status_code, r.text[:90])
            return None
        return {
            "id": f"cloud/vertex/{name}",
            "call": f"google/{name}",
            "provider": "Google",
            "vendor": "Google",
            "label": name,
            "api": "openai",
        }

    with cf.ThreadPoolExecutor(6) as ex:
        return [r for r in ex.map(probe, names) if r]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()
    e = env()
    found = []
    for name, fn in (("AWS", lambda: bedrock(e)), ("Azure", lambda: azure(e)), ("Google", vertex)):
        try:
            rows = fn()
        except Exception as exc:  # noqa: BLE001 - no key / no listing permission for this provider
            print(name, "skipped:", str(exc)[:120])
            continue
        print(name, len(rows), "callable")
        found += rows
    found.sort(key=lambda r: (r["provider"], r["vendor"], r["label"]))
    for r in found:
        print(f"  {r['provider']:6} {r['vendor']:12} {r['label']:40} {r['call']}")
    if not args.dry:
        OUT.write_text(
            json.dumps(
                {"checked": dt.date.today().isoformat(), "models": found},
                ensure_ascii=False,
                indent=1,
            )
            + "\n"
        )
        print("wrote", OUT.relative_to(ROOT), len(found))


if __name__ == "__main__":
    main()

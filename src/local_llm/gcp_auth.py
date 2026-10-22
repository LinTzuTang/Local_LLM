"""Google Cloud access tokens from a service account key, without the Google client libraries.

    from local_llm import gcp_auth
    token, project = gcp_auth.token()    # cached until a minute before it expires

The key file comes from GOOGLE_APPLICATION_CREDENTIALS (the environment or the project's .env);
it stays outside the repository (e.g. ~/.config/local_llm/gcp-service-account.json, mode 600).
A JWT signed with the key (RS256, scope cloud-platform) is exchanged at the key's token_uri
for an OAuth access token, as google-auth does. The key and the token are never logged.
"""

import base64
import json
import os
import threading
import time
from pathlib import Path

import httpx

PROJECT = Path(__file__).resolve().parents[2]
SCOPE = "https://www.googleapis.com/auth/cloud-platform"
_lock = threading.Lock()
_cached: dict = {}


def key_path() -> Path | None:
    path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    dotenv = PROJECT / ".env"
    if not path and dotenv.exists():
        for line in dotenv.read_text(encoding="utf-8").splitlines():
            if line.startswith("GOOGLE_APPLICATION_CREDENTIALS="):
                path = line.split("=", 1)[1].strip().strip("'\"")
    return Path(os.path.expanduser(path)) if path else None


def available() -> bool:
    path = key_path()
    return path is not None and path.exists()


def _b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def _assertion(key: dict, now: int) -> str:
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import padding

    header = {"alg": "RS256", "typ": "JWT", "kid": key.get("private_key_id")}
    claims = {"iss": key["client_email"], "scope": SCOPE, "aud": key["token_uri"],
              "iat": now, "exp": now + 3600}  # fmt: skip
    signing = f"{_b64(json.dumps(header).encode())}.{_b64(json.dumps(claims).encode())}"
    private = serialization.load_pem_private_key(key["private_key"].encode(), password=None)
    signature = private.sign(signing.encode(), padding.PKCS1v15(), hashes.SHA256())
    return f"{signing}.{_b64(signature)}"


def token() -> tuple[str, str]:
    """(access token, project id)."""
    with _lock:
        if _cached.get("expires", 0) > time.time() + 60:
            return _cached["token"], _cached["project"]
        path = key_path()
        if path is None or not path.exists():
            raise RuntimeError("GOOGLE_APPLICATION_CREDENTIALS is not set to a key file")
        key = json.loads(path.read_text(encoding="utf-8"))
        now = int(time.time())
        r = httpx.post(key["token_uri"], timeout=30, data={
            "grant_type": "urn:ietf:params:oauth:grant-type:jwt-bearer",
            "assertion": _assertion(key, now)})  # fmt: skip
        r.raise_for_status()
        body = r.json()
        _cached.update(token=body["access_token"], project=key["project_id"],
                       expires=now + int(body.get("expires_in", 3600)))  # fmt: skip
        return _cached["token"], _cached["project"]

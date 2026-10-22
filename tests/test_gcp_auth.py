"""Google Cloud tokens from a service account key (gcp_auth.py), with a throw-away key and the
token endpoint mocked."""

import base64
import json

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa

from local_llm import gcp_auth


def _unb64(text: str) -> bytes:
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


def test_signed_assertion_is_exchanged_once_and_cached(tmp_path, monkeypatch):
    private = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    pem = private.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                                serialization.NoEncryption()).decode()  # fmt: skip
    key = {"type": "service_account", "project_id": "demo", "private_key_id": "k1",
           "private_key": pem, "client_email": "svc@demo.iam.gserviceaccount.com",
           "token_uri": "https://oauth2.example/token"}  # fmt: skip
    path = tmp_path / "key.json"
    path.write_text(json.dumps(key))
    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", str(path))
    monkeypatch.setattr(gcp_auth, "_cached", {})
    posts = []

    class Reply:
        def raise_for_status(self):
            pass

        def json(self):
            return {"access_token": "tok-123", "expires_in": 3600}

    def post(url, timeout, data):
        posts.append((url, data))
        return Reply()

    monkeypatch.setattr(gcp_auth.httpx, "post", post)
    assert gcp_auth.available()
    assert gcp_auth.token() == ("tok-123", "demo")
    assert gcp_auth.token() == ("tok-123", "demo")  # cached
    assert len(posts) == 1 and posts[0][0] == key["token_uri"]
    head, claims, signature = posts[0][1]["assertion"].split(".")
    body = json.loads(_unb64(claims))
    assert body["iss"] == key["client_email"] and body["scope"] == gcp_auth.SCOPE
    private.public_key().verify(_unb64(signature), f"{head}.{claims}".encode(),
                                padding.PKCS1v15(), hashes.SHA256())  # fmt: skip

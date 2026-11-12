"""Accounts and the login service used when the platform is shared (auth.py, auth_server.py)."""

import pytest
from fastapi.testclient import TestClient

from local_llm import auth, auth_server
from local_llm.gateway import server as gateway_server

PASSWORD = "correct horse battery"


@pytest.fixture
def users(tmp_path):
    u = auth.Users(tmp_path)
    u.set_password("alice", PASSWORD, "admin")
    u.set_password("bob", PASSWORD, "user")
    return u


@pytest.fixture
def client(users, tmp_path):
    app = auth_server.create_app(users, auth.secret(tmp_path))
    return TestClient(app, base_url="https://lab.example.com", follow_redirects=False)


def login(client, name="bob", password=PASSWORD, next="/"):
    return client.post("/auth/login", data={"name": name, "password": password, "next": next})


def test_passwords_are_salted_scrypt_hashes(users):
    stored = users.get("alice")["hash"]
    assert stored.startswith("scrypt$") and PASSWORD not in stored
    assert stored != users.get("bob")["hash"]  # different salt, same password
    assert users.verify("alice", PASSWORD) and not users.verify("alice", "wrong password")
    assert users.verify("nobody", PASSWORD) is None
    assert not auth.check_password(PASSWORD, "garbage")


def test_account_rules(users):
    with pytest.raises(ValueError, match="at least"):
        users.set_password("carol", "short")
    with pytest.raises(ValueError, match="name"):
        users.set_password("Bad Name", PASSWORD)
    with pytest.raises(ValueError, match="role"):
        users.set_password("carol", PASSWORD, "root")
    assert oct(users.file.stat().st_mode & 0o777) == "0o600"


def test_tokens_expire_and_end_on_password_change(users, tmp_path):
    key = auth.secret(tmp_path)
    token = auth.make_token("bob", users.get("bob"), key, now=1000)
    assert auth.read_token(token, users, key, now=1001) == ("bob", "user")
    assert auth.read_token(token, users, key, now=1000 + auth.SESSION_SECONDS + 1) is None
    assert auth.read_token(token, users, b"another key" * 3, now=1001) is None
    body, sig = token.split(".")
    forged = auth._b64(b'{"u": "alice", "g": 1, "e": 99999999999}')
    assert auth.read_token(f"{forged}.{sig}", users, key, now=1001) is None
    users.set_password("bob", "a new password!")  # signs bob out everywhere
    assert auth.read_token(token, users, key, now=1001) is None
    users.remove("bob")
    fresh = auth.make_token("bob", {"gen": 2}, key, now=1000)
    assert auth.read_token(fresh, users, key, now=1001) is None  # account gone


def test_check_redirects_pages_and_refuses_the_rest_without_a_session(client):
    r = client.get("/auth/check", headers={"X-Forwarded-Uri": "/models?x=1"})
    assert r.status_code == 302 and r.headers["location"] == "/auth/login?next=/models%3Fx%3D1"
    for headers in (
        {"X-Forwarded-Uri": "/_stcore/stream"},
        {"X-Forwarded-Uri": "/gateway/v1/voice/agent/ws", "Upgrade": "websocket"},
        {"X-Forwarded-Uri": "/", "X-Forwarded-Method": "POST"},
    ):
        assert client.get("/auth/check", headers=headers).status_code == 401


def test_login_sets_a_cookie_that_check_turns_into_headers(client):
    r = login(client, next="/views/agent")
    assert r.status_code == 303 and r.headers["location"] == "/views/agent"
    cookie = r.headers["set-cookie"].lower()
    assert "httponly" in cookie and "samesite=lax" in cookie and "secure" in cookie
    r = client.get("/auth/check")
    assert r.status_code == 204
    assert (r.headers["x-lab-user"], r.headers["x-lab-role"]) == ("bob", "user")
    r = client.get("/auth/logout")
    assert r.status_code == 303 and 'lab_session=""' in r.headers["set-cookie"]


def test_wrong_passwords_lock_the_name_for_a_while(client):
    for _ in range(auth_server.MAX_FAILURES):
        assert login(client, password="wrong password").status_code == 401
    assert login(client).status_code == 429  # even the right password, until it expires
    assert login(client, name="alice").status_code == 429  # same address too


def test_login_page_escapes_and_keeps_redirects_on_site(client):
    page = client.get("/auth/login", params={"next": '/"><script>x</script>'}).text
    assert "<script>x" not in page
    for bad in ("//evil.example", "https://evil.example", "/\\evil.example", "evil"):
        assert auth_server.safe_next(bad) == "/"
        assert login(client, next=bad).headers["location"] == "/"


def test_users_command(tmp_path, monkeypatch, capsys):
    store = auth.Users(tmp_path)
    monkeypatch.setattr(auth.getpass, "getpass", lambda prompt="": PASSWORD)
    assert auth.main(["add", "dana", "--admin"], store) == 0
    assert auth.main(["add", "dana"], store) == 1  # exists
    assert auth.main(["role", "dana", "user"], store) == 0
    assert store.get("dana")["role"] == "user"
    assert auth.main(["list"], store) == 0
    assert "dana" in capsys.readouterr().out
    assert auth.main(["remove", "dana"], store) == 0 and store.get("dana") is None


def test_admin_passwords_are_temporary(tmp_path, monkeypatch):
    store = auth.Users(tmp_path)
    monkeypatch.setattr(auth.getpass, "getpass", lambda prompt="": PASSWORD)
    assert auth.main(["add", "erin"], store) == 0
    assert store.get("erin")["must_change"]
    store.change_own("erin", PASSWORD, "erin's own password")
    assert not store.get("erin")["must_change"]
    assert auth.main(["passwd", "erin"], store) == 0  # a reset is temporary again
    assert store.get("erin")["must_change"]


def test_first_sign_in_with_a_temporary_password_must_change_it(client, users):
    users.set_password("carol", PASSWORD, temporary=True)
    r = login(client, name="carol", next="/models")
    assert r.status_code == 303 and r.headers["location"] == "/auth/password?next=/models"
    # Until she does, every page sends her there and nothing else gets through.
    r = client.get("/auth/check", headers={"X-Forwarded-Uri": "/agent"})
    assert r.status_code == 302 and r.headers["location"] == "/auth/password?next=/agent"
    assert (
        client.get("/auth/check", headers={"X-Forwarded-Uri": "/_stcore/stream"}).status_code == 401
    )
    page = client.get("/auth/password", params={"next": "/models"}).text
    assert "臨時密碼" in page and "Cancel" not in page
    new = "carol's own secret"
    r = client.post("/auth/password", data={"old": PASSWORD, "new": new, "again": new,
                                            "next": "/models"})  # fmt: skip
    assert r.status_code == 303 and r.headers["location"] == "/models"
    assert client.get("/auth/check").headers["x-lab-user"] == "carol"  # fresh cookie works
    assert users.verify("carol", new) and not users.verify("carol", PASSWORD)
    assert new not in users.file.read_text()


def test_changing_your_password_checks_the_old_one_and_signs_out_elsewhere(client, users, tmp_path):
    login(client)
    other = TestClient(client.app, base_url="https://lab.example.com", follow_redirects=False)
    login(other)
    assert "Cancel" in client.get("/auth/password").text
    for data, why in (
        ({"old": "wrong pass", "new": "a fresh password", "again": "a fresh password"}, "目前的"),
        ({"old": PASSWORD, "new": "a fresh password", "again": "a different one!"}, "不一樣"),
        ({"old": PASSWORD, "new": "short", "again": "short"}, "至少"),
        ({"old": PASSWORD, "new": PASSWORD, "again": PASSWORD}, "一樣"),
    ):  # fmt: skip
        r = client.post("/auth/password", data=data)
        assert r.status_code == 400 and why in r.text
    r = client.post("/auth/password", data={"old": PASSWORD, "new": "a fresh password",
                                            "again": "a fresh password"})  # fmt: skip
    assert r.status_code == 303
    assert client.get("/auth/check").status_code == 204
    assert other.get("/auth/check", headers={"X-Forwarded-Uri": "/"}).status_code == 302


def test_password_page_needs_a_session(client):
    r = client.get("/auth/password")
    assert r.status_code == 303 and r.headers["location"].startswith("/auth/login?next=")
    r = client.post("/auth/password", data={"old": PASSWORD, "new": "x" * 12, "again": "x" * 12})
    assert r.status_code == 303 and r.headers["location"] == "/auth/login"


def test_gateway_accepts_the_public_origin_only_when_shared(monkeypatch):
    monkeypatch.setenv("LOCAL_LLM_PUBLIC_URL", "https://lab.example.com")
    monkeypatch.delenv("LOCAL_LLM_AUTH", raising=False)
    assert gateway_server.local_origin("http://127.0.0.1:8501")
    assert not gateway_server.local_origin("https://lab.example.com")
    monkeypatch.setenv("LOCAL_LLM_AUTH", "1")
    assert gateway_server.local_origin("https://lab.example.com")
    assert not gateway_server.local_origin("https://evil.example.com")

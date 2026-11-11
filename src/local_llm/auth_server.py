"""Login service behind the reverse proxy: uv run python -m local_llm.auth_server (127.0.0.1:11600).

- GET  /auth/check   Caddy's forward_auth asks this for every request. A valid session cookie
                     -> 204 with X-Lab-User / X-Lab-Role (Caddy copies them onto the request);
                     otherwise a page load is redirected to the login form and anything else
                     (WebSocket, Streamlit's own calls) gets 401.
- GET  /auth/login   the form; POST checks the password and sets the cookie.
- GET  /auth/logout  clears the cookie.

Five wrong passwords for one name or from one address within 10 minutes lock that name or
address for 10 minutes.
"""

import html
import os
import threading
import time
from collections import defaultdict, deque
from urllib.parse import quote

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse, Response

from . import auth

PORT = int(os.getenv("LOCAL_LLM_AUTH_PORT", "11600"))
MAX_FAILURES = 5
WINDOW_S = 600


class Throttle:
    """Recent failed logins per key (name or client address)."""

    def __init__(self, limit: int = MAX_FAILURES, window: float = WINDOW_S):
        self.limit, self.window = limit, window
        self.failures: dict[str, deque] = defaultdict(deque)
        self.lock = threading.Lock()

    def _recent(self, key: str, now: float) -> deque:
        q = self.failures[key]
        while q and q[0] < now - self.window:
            q.popleft()
        return q

    def blocked(self, *keys: str, now: float | None = None) -> bool:
        now = now or time.time()
        with self.lock:
            return any(len(self._recent(k, now)) >= self.limit for k in keys)

    def fail(self, *keys: str, now: float | None = None) -> None:
        now = now or time.time()
        with self.lock:
            for k in keys:
                self._recent(k, now).append(now)

    def clear(self, key: str) -> None:
        with self.lock:
            self.failures.pop(key, None)


def safe_next(target: str | None) -> str:
    """Only same-site paths: "/x" yes; "//evil.com", "https://...", "/\\evil" no."""
    if not target or not target.startswith("/") or target.startswith(("//", "/\\")):
        return "/"
    return target


def client_address(request: Request) -> str:
    # Caddy sets X-Forwarded-For to the client's address (it does not trust an incoming one).
    forwarded = request.headers.get("x-forwarded-for", "")
    return forwarded.split(",")[0].strip() or (request.client.host if request.client else "?")


PAGE = """<!doctype html>
<html lang="zh-Hant"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Local Agent Lab · 登入</title>
<style>
:root {{ color-scheme: light; --accent: #4F46E5; --text: #1F2937; --muted: #6B7280;
        --line: #E5E7EB; --bg: #F5F7FB; --error: #B91C1C; }}
* {{ box-sizing: border-box; }}
body {{ margin: 0; min-height: 100vh; display: grid; place-items: center; padding: 16px;
       background: var(--bg); color: var(--text);
       font: 15px/1.5 -apple-system, "Segoe UI", "Noto Sans TC", "PingFang TC", sans-serif; }}
form {{ width: 100%; max-width: 360px; background: #fff; border: 1px solid var(--line);
       border-radius: 12px; padding: 28px; }}
h1 {{ font-size: 20px; margin: 0 0 4px; }}
p {{ margin: 0 0 20px; color: var(--muted); font-size: 13px; }}
label {{ display: block; font-size: 13px; margin: 14px 0 6px; }}
input {{ width: 100%; padding: 10px 12px; border: 1px solid var(--line); border-radius: 8px;
        font: inherit; }}
input:focus {{ outline: 2px solid var(--accent); outline-offset: -1px; }}
button {{ width: 100%; margin-top: 22px; padding: 10px; border: 0; border-radius: 8px;
         background: var(--accent); color: #fff; font: inherit; font-weight: 600;
         cursor: pointer; }}
.error {{ color: var(--error); font-size: 13px; margin: 14px 0 0; }}
</style></head>
<body><form method="post" action="/auth/login">
<h1>Local Agent Lab</h1>
<p>請用管理者發給你的帳號登入 · Sign in with the account you were given</p>
<input type="hidden" name="next" value="{next}">
<label for="name">帳號 Account</label>
<input id="name" name="name" autocomplete="username" required autofocus value="{name}">
<label for="password">密碼 Password</label>
<input id="password" name="password" type="password" autocomplete="current-password" required>
{error}
<button type="submit">登入 Sign in</button>
</form></body></html>
"""


def login_page(next_path: str = "/", name: str = "", error: str = "", status: int = 200):
    body = PAGE.format(
        next=html.escape(next_path),
        name=html.escape(name),
        error=f'<div class="error" role="alert">{html.escape(error)}</div>' if error else "",
    )
    return HTMLResponse(body, status_code=status, headers={"Cache-Control": "no-store"})


def create_app(users: auth.Users | None = None, key: bytes | None = None) -> FastAPI:
    users = users or auth.Users()
    key = key or auth.secret()
    throttle = Throttle()
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)

    def secure(request: Request) -> bool:
        return request.headers.get("x-forwarded-proto", request.url.scheme) == "https"

    @app.get("/auth/check")
    def check(request: Request):
        who = auth.read_token(request.cookies.get(auth.COOKIE), users, key)
        if who:
            return Response(status_code=204, headers={"X-Lab-User": who[0], "X-Lab-Role": who[1]})
        uri = request.headers.get("x-forwarded-uri", "/")
        page_load = (
            request.headers.get("x-forwarded-method", "GET") == "GET"
            and "websocket" not in request.headers.get("upgrade", "").lower()
            and not uri.startswith(("/_stcore", "/gateway", "/static", "/component"))
        )
        if page_load:
            return RedirectResponse(f"/auth/login?next={quote(safe_next(uri))}", status_code=302)
        return Response("login required", status_code=401)

    @app.get("/auth/login")
    def form(request: Request, next: str = "/"):
        return login_page(safe_next(next))

    @app.post("/auth/login")
    def login(
        request: Request,
        name: str = Form(""),
        password: str = Form(""),
        next: str = Form("/"),
    ):
        name, target = name.strip().lower(), safe_next(next)
        address = client_address(request)
        keys = (f"name:{name}", f"addr:{address}")
        if throttle.blocked(*keys):
            return login_page(target, name, "嘗試太多次，請 10 分鐘後再試。Too many attempts; "
                              "try again in 10 minutes.", 429)  # fmt: skip
        user = users.verify(name, password)
        if user is None:
            throttle.fail(*keys)
            return login_page(target, name, "帳號或密碼錯誤。Wrong account or password.", 401)
        throttle.clear(f"name:{name}")
        response = RedirectResponse(target, status_code=303)
        response.set_cookie(
            auth.COOKIE, auth.make_token(name, user, key), max_age=auth.SESSION_SECONDS,
            httponly=True, secure=secure(request), samesite="lax", path="/",
        )  # fmt: skip
        return response

    @app.get("/auth/logout")
    def logout():
        response = RedirectResponse("/auth/login", status_code=303)
        response.delete_cookie(auth.COOKIE, path="/")
        return response

    return app


def main() -> None:
    import uvicorn

    if not auth.Users().load():
        print("No accounts yet: uv run local-agent users add <name> --admin", flush=True)
    uvicorn.run(create_app(), host="127.0.0.1", port=PORT, log_level="warning")


if __name__ == "__main__":
    main()

"""Login service behind the reverse proxy: uv run python -m local_llm.auth_server (127.0.0.1:11600).

- GET  /auth/check   Caddy's forward_auth asks this for every request. A valid session cookie
                     -> 204 with X-Lab-User / X-Lab-Role (Caddy copies them onto the request);
                     otherwise a page load is redirected to the login form and anything else
                     (WebSocket, Streamlit's own calls) gets 401.
- GET  /auth/login   the form; POST checks the password and sets the cookie.
- GET  /auth/password  change your own password (needs a session; old password asked again).
                     An account whose password the admin set (`must_change`) is sent here
                     by /auth/check until it picks its own.
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


STYLE = """<style>
:root { color-scheme: light; --accent: #4F46E5; --text: #1F2937; --muted: #6B7280;
        --line: #E5E7EB; --bg: #F5F7FB; --error: #B91C1C; }
* { box-sizing: border-box; }
body { margin: 0; min-height: 100vh; display: grid; place-items: center; padding: 16px;
       background: var(--bg); color: var(--text);
       font: 15px/1.5 -apple-system, "Segoe UI", "Noto Sans TC", "PingFang TC", sans-serif; }
form { width: 100%; max-width: 360px; background: #fff; border: 1px solid var(--line);
       border-radius: 12px; padding: 28px; }
h1 { font-size: 20px; margin: 0 0 4px; }
p { margin: 0 0 20px; color: var(--muted); font-size: 13px; }
label { display: block; font-size: 13px; margin: 14px 0 6px; }
input { width: 100%; padding: 10px 12px; border: 1px solid var(--line); border-radius: 8px;
        font: inherit; }
input:focus { outline: 2px solid var(--accent); outline-offset: -1px; }
button { width: 100%; margin-top: 22px; padding: 10px; border: 0; border-radius: 8px;
         background: var(--accent); color: #fff; font: inherit; font-weight: 600;
         cursor: pointer; }
.error { color: var(--error); font-size: 13px; margin: 14px 0 0; }
.hint { color: var(--muted); font-size: 12px; margin: 6px 0 0; }
.back { display: block; text-align: center; margin-top: 14px; font-size: 13px;
        color: var(--muted); }
</style>"""


def _page(title: str, form: str, status: int) -> HTMLResponse:
    body = (
        '<!doctype html>\n<html lang="zh-Hant"><head><meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f"<title>Local Agent Lab · {title}</title>\n{STYLE}</head>\n<body>{form}</body></html>\n"
    )
    return HTMLResponse(body, status_code=status, headers={"Cache-Control": "no-store"})


def _error(text: str) -> str:
    return f'<div class="error" role="alert">{html.escape(text)}</div>' if text else ""


def login_page(next_path: str = "/", name: str = "", error: str = "", status: int = 200):
    form = f"""<form method="post" action="/auth/login">
<h1>Local Agent Lab</h1>
<p>請用管理者發給你的帳號登入 · Sign in with the account you were given</p>
<input type="hidden" name="next" value="{html.escape(next_path)}">
<label for="name">帳號 Account</label>
<input id="name" name="name" autocomplete="username" required autofocus value="{html.escape(name)}">
<label for="password">密碼 Password</label>
<input id="password" name="password" type="password" autocomplete="current-password" required>
{_error(error)}
<button type="submit">登入 Sign in</button>
</form>"""
    return _page("登入", form, status)


def password_page(
    name: str, next_path: str = "/", first: bool = False, error: str = "", status: int = 200
):
    """Change your own password; `first` when the admin's temporary one has to go."""
    intro = (
        "這是管理者給的臨時密碼，請改成只有你知道的密碼才能繼續使用。管理者看不到你設定的密碼。"
        " · This is a temporary password from the admin: choose your own to continue. "
        "The admin cannot see the password you choose."
        if first
        else "改完後，其他瀏覽器上的登入會失效。 · Other signed-in browsers are signed out."
    )
    back = "" if first else f'<a class="back" href="{html.escape(next_path)}">取消 Cancel</a>'
    form = f"""<form method="post" action="/auth/password">
<h1>修改密碼 Change password</h1>
<p>{html.escape(name)} · {intro}</p>
<input type="hidden" name="next" value="{html.escape(next_path)}">
<input type="text" name="username" value="{html.escape(name)}" autocomplete="username" hidden>
<label for="old">目前的密碼 Current password</label>
<input id="old" name="old" type="password" autocomplete="current-password" required autofocus>
<label for="new">新密碼 New password</label>
<input id="new" name="new" type="password" autocomplete="new-password" required
       minlength="{auth.MIN_PASSWORD}">
<div class="hint">至少 {auth.MIN_PASSWORD} 個字元 · At least {auth.MIN_PASSWORD} characters</div>
<label for="again">再輸入一次新密碼 New password again</label>
<input id="again" name="again" type="password" autocomplete="new-password" required>
{_error(error)}
<button type="submit">儲存 Save</button>
{back}
</form>"""
    return _page("修改密碼", form, status)


PASSWORD_ERRORS = {
    "old": "目前的密碼不對。Wrong current password.",
    "same": "新密碼不能和目前的一樣。The new password must differ from the current one.",
    "short": f"新密碼至少要 {auth.MIN_PASSWORD} 個字元。"
    f"The new password needs at least {auth.MIN_PASSWORD} characters.",
    "again": "兩次輸入的新密碼不一樣。The two new passwords differ.",
}


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
        uri = request.headers.get("x-forwarded-uri", "/")
        page_load = (
            request.headers.get("x-forwarded-method", "GET") == "GET"
            and "websocket" not in request.headers.get("upgrade", "").lower()
            and not uri.startswith(("/_stcore", "/gateway", "/static", "/component"))
        )
        if who and not (users.get(who[0]) or {}).get("must_change"):
            return Response(status_code=204, headers={"X-Lab-User": who[0], "X-Lab-Role": who[1]})
        if who:  # signed in with the admin's temporary password: replace it first
            if page_load:
                return RedirectResponse(
                    f"/auth/password?next={quote(safe_next(uri))}", status_code=302
                )
            return Response("password change required", status_code=401)
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
        if user.get("must_change"):
            target = f"/auth/password?next={quote(target)}"
        return signed_in(RedirectResponse(target, status_code=303), request, name, user)

    def signed_in(response: Response, request: Request, name: str, user: dict) -> Response:
        response.set_cookie(
            auth.COOKIE, auth.make_token(name, user, key), max_age=auth.SESSION_SECONDS,
            httponly=True, secure=secure(request), samesite="lax", path="/",
        )  # fmt: skip
        return response

    def session(request: Request) -> tuple[str, dict] | None:
        who = auth.read_token(request.cookies.get(auth.COOKIE), users, key)
        user = users.get(who[0]) if who else None
        return (who[0], user) if user else None

    @app.get("/auth/password")
    def password_form(request: Request, next: str = "/"):
        current, target = session(request), safe_next(next)
        if current is None:
            return RedirectResponse(
                f"/auth/login?next={quote('/auth/password?next=' + target)}", status_code=303
            )
        name, user = current
        return password_page(name, target, first=bool(user.get("must_change")))

    @app.post("/auth/password")
    def change_password(
        request: Request,
        old: str = Form(""),
        new: str = Form(""),
        again: str = Form(""),
        next: str = Form("/"),
    ):
        current, target = session(request), safe_next(next)
        if current is None:
            return RedirectResponse("/auth/login", status_code=303)
        name, user = current
        first = bool(user.get("must_change"))
        keys = (f"name:{name}", f"addr:{client_address(request)}")
        if throttle.blocked(*keys):
            return password_page(name, target, first, "嘗試太多次，請 10 分鐘後再試。Too many "
                                 "attempts; try again in 10 minutes.", 429)  # fmt: skip
        try:
            if new != again:
                raise ValueError("again")
            users.change_own(name, old, new)
        except ValueError as exc:
            if str(exc) == "old":
                throttle.fail(*keys)
            reason = PASSWORD_ERRORS.get(str(exc), str(exc))
            return password_page(name, target, first, reason, 400)
        # The change ended every session of this account; this browser gets a fresh one.
        return signed_in(RedirectResponse(target, status_code=303), request, name, users.get(name))

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

"""Who is using the platform when it is shared behind the login proxy (local_llm.auth).

With LOCAL_LLM_AUTH off (the Mac) everyone is the admin and nothing changes. With it on, the
proxy (deploy/Caddyfile) puts the signed-in account in X-Lab-User / X-Lab-Role after removing
any such header the browser sent. A request without X-Forwarded-For did not come through the
proxy at all: it was opened on the machine itself or through an SSH tunnel, i.e. by someone
with shell access, so it counts as the admin.
"""

import streamlit as st

from local_llm import auth


def viewer() -> dict | None:
    """{"user": name or None, "role": "admin" | "user", "via": "local" | "proxy"}; None for a
    proxied request without an identity (a proxy that is not set up as in deploy/Caddyfile)."""
    if not auth.enabled():
        return {"user": None, "role": "admin", "via": "local"}
    headers = st.context.headers
    user, role = headers.get("X-Lab-User"), headers.get("X-Lab-Role")
    if user and role in auth.ROLES:
        return {"user": user, "role": role, "via": "proxy"}
    if not headers.get("X-Forwarded-For"):
        return {"user": None, "role": "admin", "via": "local"}
    return None


def is_admin() -> bool:
    who = viewer()
    return bool(who) and who["role"] == "admin"


def browser_gateway(url: str) -> str:
    """Where the browser's WebSockets reach the model gateway: through the proxy
    (wss://<host>/gateway, same login) when the page came that way, else the gateway itself
    (http://127.0.0.1:11500 -> ws://127.0.0.1:11500)."""
    who, headers = viewer(), st.context.headers
    host = headers.get("X-Forwarded-Host") or headers.get("Host")
    if who and who["via"] == "proxy" and host:
        scheme = "ws" if headers.get("X-Forwarded-Proto") == "http" else "wss"
        return f"{scheme}://{host}/gateway"
    base = url.rstrip("/")
    return "wss://" + base[8:] if base.startswith("https://") else "ws://" + base.split("://")[-1]

"""Which signed-in account the model requests of this page run are for.

The app sets it once per run (app/main.py); the platform's chat calls pass it to Ollama as the
X-Lab-User header, which the request log (ollama_proxy.py) records so real use can be told apart
from scripts on the same machine."""

from contextvars import ContextVar
from urllib.parse import quote

_user: ContextVar[str | None] = ContextVar("lab_user", default=None)


def set_user(name: str | None) -> None:
    _user.set(name)


def headers() -> dict[str, str]:
    """{"X-Lab-User": name} (percent-encoded: header values must be ASCII), or {} outside a
    signed-in page run."""
    name = _user.get()
    return {"X-Lab-User": quote(name, safe="")} if name else {}

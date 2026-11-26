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


_installed: set[tuple[str, int]] = set()


def install(*ollama_urls: str) -> None:
    """Tag every request this process sends to Ollama with the account, whichever library sends
    it (httpx directly, the ollama and openai clients, LangChain: all go through httpx). Only
    requests to Ollama's address get the header, never those to cloud services."""
    import httpx

    first = not _installed
    for url in ollama_urls:
        u = httpx.URL(url)
        loopback = {"127.0.0.1", "localhost"}
        for host in loopback if u.host in loopback else {u.host}:
            _installed.add((host, u.port or 80))
    if not first:
        return

    def tag(request: httpx.Request) -> None:
        if (request.url.host, request.url.port or 80) in _installed and (
            "x-lab-user" not in request.headers
        ):
            request.headers.update(headers())

    send, asend = httpx.Client.send, httpx.AsyncClient.send

    def client_send(self, request, *args, **kwargs):
        tag(request)
        return send(self, request, *args, **kwargs)

    async def async_send(self, request, *args, **kwargs):
        tag(request)
        return await asend(self, request, *args, **kwargs)

    httpx.Client.send, httpx.AsyncClient.send = client_send, async_send

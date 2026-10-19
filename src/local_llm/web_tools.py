"""External (network) tools. Every call here sends data off this machine.

They are only bound to an agent when network tools are allowed, and the platform asks a
human to approve each call before it runs (see agents.build).
"""

import ipaddress
import json
import socket
from urllib.parse import quote, urlparse

import httpx
import trafilatura
from langchain.tools import tool

from . import geo_tools

TIMEOUT = 15
MAX_CHARS = 4000
HEADERS = {"User-Agent": "Local_LLM-demo/0.1 (on-prem agent platform demo)"}
WEATHER_CODES = {
    0: "晴", 1: "大致晴朗", 2: "局部多雲", 3: "陰", 45: "霧", 48: "霧淞",
    51: "毛毛雨", 53: "毛毛雨", 55: "毛毛雨", 61: "小雨", 63: "中雨", 65: "大雨",
    71: "小雪", 73: "中雪", 75: "大雪", 80: "陣雨", 81: "陣雨", 82: "強陣雨",
    95: "雷雨", 96: "雷雨伴冰雹", 99: "強雷雨伴冰雹",
}  # fmt: skip


def _get(url: str, **params) -> httpx.Response:
    with httpx.Client(timeout=TIMEOUT, headers=HEADERS, follow_redirects=False) as client:
        response = client.get(url, params=params or None)
        response.raise_for_status()
        return response


@tool
def get_weather(city: str) -> str:
    """Get current weather and a 3-day forecast from Open-Meteo (external internet API).
    Use only when the user asks about real current or upcoming weather.
    `city` must be an English place name, e.g. 'Taipei', 'Hsinchu', 'Tokyo'."""
    try:
        places = _get(
            "https://geocoding-api.open-meteo.com/v1/search", name=city, count=1, language="zh"
        ).json()
        if not places.get("results"):
            return f"No location found for {city!r}; retry with an English city name."
        place = places["results"][0]
        data = _get(
            "https://api.open-meteo.com/v1/forecast",
            latitude=place["latitude"],
            longitude=place["longitude"],
            current="temperature_2m,relative_humidity_2m,weather_code,wind_speed_10m",
            daily="temperature_2m_max,temperature_2m_min,precipitation_probability_max,weather_code",
            timezone="auto",
            forecast_days=3,
        ).json()
    except httpx.HTTPError as exc:
        return f"Weather API error: {exc}"
    current, daily = data["current"], data["daily"]
    return json.dumps(
        {
            "location": f"{place.get('name')}, {place.get('country', '')}",
            "now": {
                "time": current["time"],
                "temperature_c": current["temperature_2m"],
                "humidity_pct": current["relative_humidity_2m"],
                "wind_kmh": current["wind_speed_10m"],
                "condition": WEATHER_CODES.get(current["weather_code"], current["weather_code"]),
            },
            "forecast": [
                {
                    "date": day,
                    "min_c": daily["temperature_2m_min"][i],
                    "max_c": daily["temperature_2m_max"][i],
                    "rain_chance_pct": daily["precipitation_probability_max"][i],
                    "condition": WEATHER_CODES.get(
                        daily["weather_code"][i], daily["weather_code"][i]
                    ),
                }
                for i, day in enumerate(daily["time"])
            ],
            "source": "open-meteo.com",
        },
        ensure_ascii=False,
    )


@tool
def search_places(location: str, query: str = "早午餐") -> str:
    """Find shops near a place from OpenStreetMap (external internet APIs: Open-Meteo
    geocoding, Nominatim and Overpass). `location` is a place name as the user said it, e.g.
    '內湖', '台北市信義區'; `query` what to look for, e.g. '早午餐', '咖啡', '拉麵'. Returns up to
    five places within about 1.5 km with distance, opening hours and address when OSM has
    them. Use only names from the result; never invent places."""
    try:
        return json.dumps(geo_tools.nearby(location, query), ensure_ascii=False)
    except (geo_tools.GeoError, httpx.HTTPError) as exc:
        return f"Place search error: {exc}"


@tool
def wikipedia_search(query: str) -> str:
    """Search Chinese Wikipedia (external internet API) and return summaries of the top
    articles in Traditional Chinese. Use for factual background knowledge."""
    try:
        hits = _get(
            "https://zh.wikipedia.org/w/api.php",
            action="query", list="search", srsearch=query, format="json", srlimit=3,
        ).json()["query"]["search"]  # fmt: skip
        results = []
        # Accept-Language zh-tw makes Wikipedia convert the text to Traditional Chinese.
        headers = {**HEADERS, "Accept-Language": "zh-tw"}
        for hit in hits:
            title = hit["title"]
            with httpx.Client(timeout=TIMEOUT, headers=headers) as client:
                summary = client.get(
                    "https://zh.wikipedia.org/api/rest_v1/page/summary/" + quote(title, safe="")
                ).json()
            results.append(
                {
                    "title": summary.get("title", title),
                    "summary": summary.get("extract", "")[:800],
                    "url": summary.get("content_urls", {}).get("desktop", {}).get("page"),
                }
            )
    except (httpx.HTTPError, KeyError) as exc:
        return f"Wikipedia API error: {exc}"
    return json.dumps(results, ensure_ascii=False) if results else "No Wikipedia results."


def check_public_url(url: str) -> str | None:
    """Return an error message unless `url` is http(s) to a public internet host.

    Blocks localhost and private networks so a prompt-injected URL cannot reach internal
    services (e.g. Ollama on 127.0.0.1 or company intranet systems).
    """
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        return "Only http(s) URLs are allowed."
    try:
        addresses = {info[4][0] for info in socket.getaddrinfo(parsed.hostname, None)}
    except socket.gaierror:
        return f"Cannot resolve host {parsed.hostname}."
    for address in addresses:
        ip = ipaddress.ip_address(address.split("%")[0])
        if not ip.is_global:
            return f"Blocked non-public address {ip} for {parsed.hostname}."
    return None


@tool
def fetch_webpage(url: str) -> str:
    """Download a public web page (external internet request) and return its main text.
    Use when the user gives a URL or a search result needs to be read in full."""
    if error := check_public_url(url):
        return error
    try:
        response = _get(url)
    except httpx.HTTPError as exc:
        return f"Fetch error: {exc}"
    if response.is_redirect:
        return f"Redirected to {response.headers.get('location')}; call again with that URL."
    text = trafilatura.extract(response.text, include_comments=False) or ""
    if not text:
        return "No readable text found on the page."
    suffix = f"\n…(truncated, {len(text)} chars total)" if len(text) > MAX_CHARS else ""
    return text[:MAX_CHARS] + suffix

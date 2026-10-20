"""Place-aware lookups for the live voice assistant: geocoding, weather and nearby places.

Every call here sends data off this machine (read-only requests, no keys):

- geocode(name): Open-Meteo's geocoding API (GeoNames data), which knows Chinese place names
  such as 「內湖」; results are cached for a day.
- weather(location): geocode + Open-Meteo forecast (now, today's and tomorrow's range and
  rain chance).
- nearby(location, query): geocode + one OpenStreetMap Overpass query for cafés, restaurants
  and fast food within RADIUS_M, ranked in Python: places whose name or tags say brunch
  first, then breakfast shops, and only when OSM has too few of those, the nearest cafés and
  restaurants, with `match` saying which tier the answer comes from so the assistant can say
  so honestly. Places come only from OSM; nothing is made up. Results are cached briefly.

Data: © OpenStreetMap contributors (ODbL), https://www.openstreetmap.org/copyright;
weather and geocoding by Open-Meteo (CC BY 4.0).
"""

import json
import math
import re
import threading
import time

import httpx

TIMEOUT = 20
HEADERS = {"User-Agent": "LocalAgentLab/0.1 (on-prem voice assistant demo; httpx)"}
GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
OVERPASS_URLS = ("https://overpass-api.de/api/interpreter",)
NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
NOMINATIM_INTERVAL_S = 1.1  # usage policy: at most one request per second
OVERPASS_TIMEOUT_S = 12
OVERPASS_WAIT_S = 4.0  # how long a lookup waits for Overpass in all
OVERPASS_GRACE_S = 0.5  # ... once Nominatim has already found enough places
RADIUS_M = 1500
MAX_PLACES = 5
OSM_ATTRIBUTION = "© OpenStreetMap contributors (ODbL)"
WEATHER_ATTRIBUTION = "Weather data by Open-Meteo.com (CC BY 4.0)"
GEOCODE_TTL_S = 24 * 3600
WEATHER_TTL_S = 10 * 60
PLACES_TTL_S = 15 * 60
WEATHER_CODES = {
    0: "晴", 1: "大致晴朗", 2: "局部多雲", 3: "陰", 45: "霧", 48: "霧淞",
    51: "毛毛雨", 53: "毛毛雨", 55: "毛毛雨", 61: "小雨", 63: "中雨", 65: "大雨",
    71: "小雪", 73: "中雪", 75: "大雪", 80: "陣雨", 81: "陣雨", 82: "強陣雨",
    95: "雷雨", 96: "雷雨伴冰雹", 99: "強雷雨伴冰雹",
}  # fmt: skip
BRUNCH = re.compile(r"早午餐|brunch", re.I)
BREAKFAST = re.compile(r"早餐|breakfast|美而美|早安", re.I)
CAFE = re.compile(r"咖啡|cafe|café|coffee", re.I)
KINDS = {"cafe": "咖啡店", "restaurant": "餐廳", "fast_food": "速食／早餐店"}


class GeoError(RuntimeError):
    """A lookup that cannot be answered (unknown place, API down); the message says why."""


# -- small TTL cache --------------------------------------------------------------------------

_cache: dict[tuple, tuple[float, object]] = {}
_cache_lock = threading.Lock()


def _cached(key: tuple, ttl: float, make):
    now = time.time()
    with _cache_lock:
        hit = _cache.get(key)
        if hit and now - hit[0] < ttl:
            return hit[1]
    value = make()
    with _cache_lock:
        _cache[key] = (now, value)
    return value


def clear_cache() -> None:
    with _cache_lock:
        _cache.clear()


TRANSPORT: httpx.BaseTransport | None = None  # tests put an httpx.MockTransport here


def _client(timeout: float = TIMEOUT) -> httpx.Client:
    return httpx.Client(
        timeout=timeout, headers=HEADERS, follow_redirects=False, transport=TRANSPORT
    )


# -- geocoding --------------------------------------------------------------------------------


def clean_place(name: str) -> str:
    """「台北市內湖區附近」-> 「台北市內湖區」: words around a place name that geocoders do not
    know. Keeps the name when nothing is left."""
    text = re.sub(r"[\s，,。！？?!]+", " ", name or "").strip()
    stripped = re.sub(r"(這|那)?(附近|一帶|周邊|周圍|這邊|那邊)$", "", text).strip()
    return stripped or text


def geocode(name: str) -> dict:
    """{"name", "label", "lat", "lon", "country_code", "timezone"} of the best match, or
    GeoError. Tries the spellings of `geocode_queries` in order and takes the first major
    place in Taiwan (an administrative area or a place with a population: 「板橋」 alone is a
    village in Yunlin, 「板橋區」 the New Taipei district). With only minor places in Taiwan,
    asks Nominatim (OpenStreetMap knows districts and landmarks GeoNames lacks: 信義區,
    西門町, 台北車站) and falls back to the first minor place, then to any result."""
    name = clean_place(name)
    if not name:
        raise GeoError("no place name given")
    minor = abroad = fallback = None
    for query in geocode_queries(name):
        rows = _cached(("geo", query), GEOCODE_TTL_S, lambda q=query: _geocode_rows(q))
        taiwan = [r for r in rows[:5] if r.get("country_code") == "TW"]
        if major := next((r for r in taiwan if _major(r)), None):
            return _place([major])
        minor = minor or (taiwan[0] if taiwan else None)
        abroad = abroad or next((r for r in rows[:5] if _major(r)), None)
        fallback = fallback or rows
    if abroad and not minor:  # "Tokyo": a city elsewhere, not a gap in Taiwan's data
        return _place([abroad])
    try:
        if found := _nominatim_place(name):
            return found
    except (httpx.HTTPError, json.JSONDecodeError, KeyError, ValueError):
        pass  # OpenStreetMap unavailable: keep what GeoNames had
    if minor or fallback:
        return _place([minor] if minor else fallback)
    raise GeoError(f"no location found for {name!r}")


def _major(row: dict) -> bool:
    code = str(row.get("feature_code") or "")
    return bool(row.get("population")) or code.startswith(("PPLA", "PPLC", "ADM"))


def _nominatim_place(name: str) -> dict | None:
    """A place in Taiwan by name from Nominatim, as geocode() returns it, or None."""

    def fetch() -> list[dict]:
        with _nominatim_lock:
            wait = _nominatim_last[0] + NOMINATIM_INTERVAL_S - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            try:
                with _client() as client:
                    response = client.get(NOMINATIM_URL, params={
                        "q": name, "format": "jsonv2", "limit": 1, "countrycodes": "tw",
                        "addressdetails": 1, "accept-language": "zh-TW",
                    })  # fmt: skip
                    response.raise_for_status()
                    return response.json()
            finally:
                _nominatim_last[0] = time.monotonic()

    rows = _cached(("nomgeo", name), GEOCODE_TTL_S, fetch)
    if not rows:
        return None
    row, address = rows[0], rows[0].get("address") or {}
    label = row.get("name") or name
    area = address.get("city") or address.get("county") or address.get("state")
    district = address.get("suburb") or address.get("city_district") or address.get("town")
    extra = "".join(p for p in (area, district) if p and p != label)
    return {
        "name": label,
        "label": f"{label}（{extra}）" if extra else label,
        "lat": float(row["lat"]),
        "lon": float(row["lon"]),
        "country_code": "TW",
        "timezone": "Asia/Taipei",
    }


def geocode_queries(name: str) -> list[str]:
    """The spellings to try for a place name. GeoNames (behind Open-Meteo) has gaps: it lists
    「內湖」 but not 「台北市內湖區」, 「台北市」 but not 「台北」 or 「臺北」, 「板橋區」 but
    「板橋」 only as a village, and 「台南」 only as places in China. So: the name, without a
    trailing 區/市/鄉/鎮, the part after 市/縣, a two-character name with 市 and 區 added, 台 and 臺
    swapped, and for "Xinyi Taipei" each word from the last (the city: "Xinyi" alone finds
    信宜 in China first)."""
    tries = [name]
    short = re.sub(r"(區|市|鄉|鎮|district|city)$", "", name, flags=re.I).strip()
    tries.append(short)
    if m := re.search(r"[市縣]\s*(.+?)[區鄉鎮]?$", name):
        tries.append(m[1])
    for t in list(tries):
        if re.fullmatch(r"[\u4e00-\u9fff]{2}", t):
            tries += [t + "市", t + "區"]
    for t in list(tries):
        if "台" in t or "臺" in t:
            tries.append(t.replace("臺", "台") if "臺" in t else t.replace("台", "臺"))
    words = name.split()
    if len(words) > 1:
        tries += reversed(words)
    return list(dict.fromkeys(t for t in tries if t))


def _geocode_rows(query: str) -> list[dict]:
    with _client() as client:
        response = client.get(GEOCODE_URL, params={"name": query, "count": 5, "language": "zh"})
        response.raise_for_status()
        return response.json().get("results") or []


def _place(rows: list[dict]) -> dict:
    """The most likely result: Open-Meteo orders by relevance and population; among the top
    results prefer a populated place in Taiwan (this is a Taiwan demo) when there is one."""
    taiwan = [r for r in rows[:5] if r.get("country_code") == "TW"]
    best = next((r for r in taiwan if r.get("population")), taiwan[0] if taiwan else rows[0])
    parts = [best.get("admin2"), best.get("admin3")]
    label = best.get("name", "")
    extra = "".join(p for p in parts if p and p != label)
    return {
        "name": label,
        "label": f"{label}（{extra}）" if extra else label,
        "lat": float(best["latitude"]),
        "lon": float(best["longitude"]),
        "country_code": best.get("country_code"),
        "timezone": best.get("timezone") or "auto",
    }


# -- weather ----------------------------------------------------------------------------------


def weather(location: str) -> dict:
    """Current weather and the outlook for today and tomorrow at a place name (any language
    Open-Meteo knows)."""
    place = geocode(location)
    key = ("wx", round(place["lat"], 3), round(place["lon"], 3))
    data = _cached(key, WEATHER_TTL_S, lambda: _forecast(place))
    return {"location": place["label"], **data, "source": WEATHER_ATTRIBUTION}


def _forecast(place: dict) -> dict:
    params = {
        "latitude": place["lat"],
        "longitude": place["lon"],
        "current": (
            "temperature_2m,apparent_temperature,relative_humidity_2m,weather_code,wind_speed_10m"
        ),
        "hourly": "precipitation_probability",
        "daily": (
            "temperature_2m_max,temperature_2m_min,precipitation_probability_max,weather_code"
        ),
        "timezone": place.get("timezone") or "auto",
        "forecast_days": 2,
    }
    with _client() as client:
        response = client.get(FORECAST_URL, params=params)
        response.raise_for_status()
        data = response.json()
    current, daily, hourly = data["current"], data["daily"], data.get("hourly") or {}
    now = current["time"][:13]  # "2026-10-19T17"
    times, probs = hourly.get("time") or [], hourly.get("precipitation_probability") or []
    upcoming = [p for t, p in zip(times, probs, strict=False) if t[:13] >= now][:6]
    upcoming = [p for p in upcoming if p is not None]

    def day(i: int) -> dict | None:
        if len(daily.get("temperature_2m_max") or []) <= i:
            return None
        code = daily["weather_code"][i]
        return {
            "min_c": daily["temperature_2m_min"][i],
            "max_c": daily["temperature_2m_max"][i],
            "rain_chance_pct": daily["precipitation_probability_max"][i],
            "condition": WEATHER_CODES.get(code, str(code)),
        }

    return {
        "time": current["time"],
        "temperature_c": current["temperature_2m"],
        "feels_like_c": current.get("apparent_temperature"),
        "humidity_pct": current.get("relative_humidity_2m"),
        "wind_kmh": current.get("wind_speed_10m"),
        "condition": WEATHER_CODES.get(current["weather_code"], str(current["weather_code"])),
        "today": day(0),
        "tomorrow": day(1),
        "rain_chance_next_6h_pct": max(upcoming) if upcoming else None,
    }


def weather_summary(result: dict) -> str:
    """One line for the UI's tool card."""
    today = result.get("today") or {}
    return (
        f"{result.get('location')}：{result.get('temperature_c')}°C {result.get('condition')}，"
        f"今天 {today.get('min_c')}–{today.get('max_c')}°C，"
        f"降雨機率 {today.get('rain_chance_pct')}%"
    )


# -- nearby places (OpenStreetMap) ------------------------------------------------------------


def distance_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in metres (haversine)."""
    r = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def wanted(query: str) -> str:
    """The kind of place a request asks for: "brunch", "cafe" or the query's own words."""
    q = (query or "").strip()
    if BRUNCH.search(q) or BREAKFAST.search(q):
        return "brunch"
    if CAFE.search(q):
        return "cafe"
    q = re.sub(r"(好吃的|推薦的|附近的|店家?|餐廳|的)$", "", q).strip()
    return q or "restaurant"


def _words(want: str) -> list[str]:
    return [w for w in re.split(r"[\s，,、]+", want) if w]


def overpass_query(lat: float, lon: float, want: str, radius: int = RADIUS_M) -> str:
    """Named cafés, restaurants and fast-food places around a point that may match `want`
    (a brunch / breakfast name or tag, a café, or the query's words in name or cuisine)."""
    around = f"nwr(around:{radius},{lat:.6f},{lon:.6f})"
    amenity = '["amenity"~"^(cafe|restaurant|fast_food)$"]'
    if want == "brunch":
        names = "早午餐|早餐|brunch|breakfast|美而美|早安"
        parts = [f'{around}{amenity}["name"~"{names}",i];',
                 f'{around}{amenity}["cuisine"~"breakfast|brunch",i];',
                 f'{around}{amenity}["breakfast"="yes"];']  # fmt: skip
    elif want == "cafe":
        parts = [f'{around}["amenity"="cafe"]["name"];']
    else:
        words = "|".join(re.escape(w).replace('"', "") for w in _words(want)) or "."
        parts = [f'{around}{amenity}["name"~"{words}",i];',
                 f'{around}{amenity}["cuisine"~"{words}",i];']  # fmt: skip
    return f"[out:json][timeout:{OVERPASS_TIMEOUT_S}];({''.join(parts)});out center tags 120;"


def _overpass(query: str) -> list[dict]:
    error: Exception | None = None
    with _client(OVERPASS_TIMEOUT_S + 5) as client:
        for url in OVERPASS_URLS:
            try:
                response = client.post(url, data={"data": query})
                response.raise_for_status()
                return response.json().get("elements") or []
            except (httpx.HTTPError, json.JSONDecodeError) as exc:  # try the next mirror
                error = exc
    raise GeoError(f"OpenStreetMap Overpass API error: {error}")


_nominatim_lock = threading.Lock()
_nominatim_last = [0.0]


def _nominatim(term: str, box: tuple[float, float, float, float]) -> list[dict]:
    """Nominatim free-text search inside a bounding box (OSM data, ~1 s), as Overpass-style
    elements. At most one request per second from this process (Nominatim's usage policy);
    results are cached."""

    def fetch() -> list[dict]:
        with _nominatim_lock:
            wait = _nominatim_last[0] + NOMINATIM_INTERVAL_S - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            try:
                with _client() as client:
                    response = client.get(NOMINATIM_URL, params={
                        "q": term, "format": "jsonv2", "limit": 15, "bounded": 1,
                        "viewbox": ",".join(f"{v:.5f}" for v in box), "addressdetails": 1,
                        "extratags": 1, "accept-language": "zh-TW",
                    })  # fmt: skip
                    response.raise_for_status()
                    rows = response.json()
            finally:
                _nominatim_last[0] = time.monotonic()
        return [_from_nominatim(r) for r in rows if r.get("name")]

    return _cached(("nom", term, tuple(round(v, 3) for v in box)), PLACES_TTL_S, fetch)


def _from_nominatim(row: dict) -> dict:
    extra, address = row.get("extratags") or {}, row.get("address") or {}
    tags = {"name": row["name"], "amenity": row.get("type") if row.get("category") == "amenity"
            else row.get("type")}  # fmt: skip
    for key in ("opening_hours", "cuisine", "breakfast"):
        if extra.get(key):
            tags[key] = extra[key]
    road, number = address.get("road"), address.get("house_number")
    district = address.get("suburb") or address.get("city_district")
    if road:
        tags["addr:full"] = f"{district or ''}{road}{f'{number}號' if number else ''}"
    return {"type": row.get("osm_type", "node"), "id": row.get("osm_id"),
            "lat": float(row["lat"]), "lon": float(row["lon"]), "tags": tags}  # fmt: skip


def viewbox(lat: float, lon: float, radius: int) -> tuple[float, float, float, float]:
    """(west, north, east, south) of the square around a point."""
    dlat = radius / 111_320
    dlon = radius / (111_320 * max(math.cos(math.radians(lat)), 0.01))
    return lon - dlon, lat + dlat, lon + dlon, lat - dlat


def tier(tags: dict, want: str) -> int:
    """How well a place matches the request (lower is better; 9 = only nearby).

    want "brunch": 1 brunch by name or cuisine, 2 breakfast shop; "cafe": 1 café;
    anything else: 1 when the words appear in the name or cuisine."""
    name = " ".join(str(tags.get(k) or "") for k in ("name", "name:zh", "name:en", "brand"))
    cuisine = str(tags.get("cuisine") or "")
    if want == "brunch":
        if BRUNCH.search(name) or "brunch" in cuisine:
            return 1
        if BREAKFAST.search(name) or "breakfast" in cuisine or tags.get("breakfast") == "yes":
            return 2
        return 9
    if want == "cafe":
        return 1 if tags.get("amenity") == "cafe" or CAFE.search(name) else 9
    words = _words(want)
    if words and any(w.lower() in (name + " " + cuisine).lower() for w in words):
        return 1
    return 9


def _address(tags: dict) -> str | None:
    if tags.get("addr:full"):
        return tags["addr:full"]
    parts = [tags.get(k) for k in ("addr:city", "addr:district", "addr:street")]
    number = tags.get("addr:housenumber")
    text = "".join(p for p in parts if p) + (f"{number}號" if number and parts[2] else "")
    return text or None


def rank_places(
    elements: list[dict],
    lat: float,
    lon: float,
    query: str,
    limit: int = MAX_PLACES,
    radius: int = RADIUS_M,
) -> tuple[str, list[dict]]:
    """(match, places): the best `limit` places for `query` within `radius` of (lat, lon).

    match: "brunch" (enough brunch places), "breakfast" (brunch topped up with breakfast
    shops), "match" (the query's words), or "nearby" (too few matches: the nearest cafés and
    restaurants fill the list). The same place from two sources is kept once, with the
    tags of both."""
    want = wanted(query)
    by_name: dict[str, dict] = {}
    for e in elements:
        tags = e.get("tags") or {}
        name = tags.get("name")
        p_lat = e.get("lat", (e.get("center") or {}).get("lat"))
        p_lon = e.get("lon", (e.get("center") or {}).get("lon"))
        if not name or p_lat is None or p_lon is None:
            continue
        dist = distance_m(lat, lon, p_lat, p_lon)
        if dist > radius * 1.15:
            continue
        old = by_name.get(name)
        if old and old["_dist"] <= dist + 50:  # same place (or a nearer branch) seen already
            old["_tags"] = {**tags, **old["_tags"]}
            continue
        by_name[name] = {"_tags": dict(tags) | (old["_tags"] if old else {}), "_dist": dist,
                         "osm": f"https://www.openstreetmap.org/{e.get('type', 'node')}/"
                                f"{e.get('id')}"}  # fmt: skip
    rows = []
    for name, r in by_name.items():
        tags = r["_tags"]
        rows.append({
            "name": name,
            "kind": KINDS.get(tags.get("amenity"), tags.get("amenity")),
            "distance_m": int(round(r["_dist"], -1)),
            "opening_hours": tags.get("opening_hours"),
            "address": _address(tags),
            "cuisine": tags.get("cuisine"),
            "osm": r["osm"],
            "tier": tier(tags, want),
        })  # fmt: skip
    rows.sort(key=lambda r: (r["tier"], r["distance_m"]))
    good = [r for r in rows if r["tier"] < 9]
    if want == "brunch":
        brunch = [r for r in good if r["tier"] == 1]
        match = "brunch" if len(brunch) >= 2 else "breakfast"
        picked = good[:limit]
    else:
        match, picked = "match", good[:limit]
    if len(picked) < 2:  # OSM knows too few: be honest and offer what is near
        match = "nearby"
        rest = sorted((r for r in rows if r not in picked), key=lambda r: r["distance_m"])
        picked = picked + rest[: limit - len(picked)]
    for r in picked:
        level = r.pop("tier")
        r["match"] = {1: "brunch" if want == "brunch" else "match", 2: "breakfast"}.get(
            level, "nearby"
        )
    return match, picked


NOTES = {
    "brunch": "OpenStreetMap 上附近有標成早午餐的店。",
    "breakfast": "OpenStreetMap 上附近明確標成早午餐的店不多，一起列出附近的早餐店。",
    "match": "OpenStreetMap 上附近符合的店家。",
    "nearby": "OpenStreetMap 上附近幾乎沒有符合條件的店，以下改列最近的咖啡店和餐廳。",
}


def nearby(location: str, query: str = "早午餐", radius: int = RADIUS_M) -> dict:
    """Places matching `query` within `radius` m of a place name, from OpenStreetMap.

    Two OSM sources run at once: an Overpass query (tags such as cuisine=breakfast; the
    public server is sometimes slow, so it gets OVERPASS_WAIT_S, or only OVERPASS_GRACE_S
    when Nominatim already found enough; a late answer is still cached for next time) and
    Nominatim name / category searches in the surrounding box (about a second each). Their
    places are merged and ranked."""
    place = geocode(location)
    lat, lon = place["lat"], place["lon"]
    want = wanted(query)
    started = time.monotonic()
    box = viewbox(lat, lon, radius)
    found: dict[str, object] = {}

    def run_overpass() -> None:
        key = ("osm", want, round(lat, 4), round(lon, 4), radius)
        try:
            query_text = overpass_query(lat, lon, want, radius)
            found["overpass"] = _cached(key, PLACES_TTL_S, lambda: _overpass(query_text))
        except GeoError as exc:
            found["overpass_error"] = str(exc)

    worker = threading.Thread(target=run_overpass, name="overpass", daemon=True)
    worker.start()
    terms = {"brunch": ["早午餐", "早餐"], "cafe": ["cafe"]}.get(want, [want])
    elements, sources, errors = [], [], []
    for term in terms:
        try:
            elements += _nominatim(term, box)
            sources.append("nominatim")
        except httpx.HTTPError as exc:
            errors.append(f"Nominatim: {exc}")
    # Enough good matches already: give Overpass only a moment more.
    _, early = rank_places(elements, lat, lon, query, radius=radius)
    enough = sum(p["match"] != "nearby" for p in early) >= 3
    budget = OVERPASS_GRACE_S if enough else OVERPASS_WAIT_S - (time.monotonic() - started)
    worker.join(timeout=max(budget, 0.0))
    if "overpass" in found:
        elements += found["overpass"]
        sources.append("overpass")
    elif "overpass_error" in found:
        errors.append(str(found["overpass_error"]))
    else:
        errors.append(f"Overpass: no answer within {OVERPASS_WAIT_S:.0f} s (still loading)")
    match, places = rank_places(elements, lat, lon, query, radius=radius)
    if match == "nearby":  # too few matches: the nearest cafés and restaurants instead
        for term in ("cafe", "restaurant"):
            try:
                elements += _nominatim(term, box)
            except httpx.HTTPError as exc:
                errors.append(f"Nominatim: {exc}")
        match, places = rank_places(elements, lat, lon, query, radius=radius)
    if not elements and errors:
        raise GeoError("; ".join(errors))
    return {
        "location": place["label"],
        "query": query,
        "radius_m": radius,
        "match": match,
        "note": NOTES[match],
        "places": places,
        "sources": sorted(set(sources)),
        "errors": errors,
        "attribution": OSM_ATTRIBUTION,
    }


def nearby_summary(result: dict) -> str:
    names = "、".join(p["name"] for p in result.get("places", [])[:3])
    return f"{result.get('location')}附近：{names or '沒有結果'}"

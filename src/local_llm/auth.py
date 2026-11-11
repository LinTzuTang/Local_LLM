"""Accounts for sharing the platform with colleagues (off unless LOCAL_LLM_AUTH=1).

The platform itself stays on 127.0.0.1. A reverse proxy (Caddy, deploy/Caddyfile) is the only
way in from outside: for every request it asks the login service (auth_server.py) whether the
browser's session cookie is valid, then forwards the request with X-Lab-User / X-Lab-Role set,
so Streamlit pages and the gateway's WebSockets sit behind the same login.

Accounts live in storage/auth/users.json (scrypt password hashes, role "admin" or "user"); the
session cookie is signed with storage/auth/secret. Manage accounts on the machine:

    uv run local-agent users add alice [--admin]
    uv run local-agent users passwd alice
    uv run local-agent users list
    uv run local-agent users remove alice
"""

import argparse
import base64
import getpass
import hashlib
import hmac
import json
import os
import re
import secrets
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

PROJECT_DIR = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_DIR / ".env")
AUTH_DIR = PROJECT_DIR / "storage" / "auth"
ROLES = ("admin", "user")
COOKIE = "lab_session"
SESSION_SECONDS = 7 * 24 * 3600
NAME_RE = re.compile(r"[a-z0-9][a-z0-9._-]{1,31}")
MIN_PASSWORD = 10
# scrypt cost: about 50 ms and 16 MB per check, so guessing offline is slow.
SCRYPT = {"n": 2**14, "r": 8, "p": 1}


def enabled() -> bool:
    return os.getenv("LOCAL_LLM_AUTH", "").strip().lower() in ("1", "true", "yes")


def public_url() -> str:
    """The HTTPS address colleagues open (e.g. https://lab.example.com), or ""."""
    return os.getenv("LOCAL_LLM_PUBLIC_URL", "").strip().rstrip("/")


# --- passwords ----------------------------------------------------------------------------


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.scrypt(password.encode(), salt=salt, **SCRYPT)
    return "scrypt${n}${r}${p}${salt}${hash}".format(**SCRYPT, salt=salt.hex(), hash=digest.hex())


def check_password(password: str, stored: str) -> bool:
    try:
        kind, n, r, p, salt, digest = stored.split("$")
        if kind != "scrypt":
            return False
        got = hashlib.scrypt(
            password.encode(), salt=bytes.fromhex(salt), n=int(n), r=int(r), p=int(p)
        )
    except (ValueError, TypeError):
        return False
    return hmac.compare_digest(got.hex(), digest)


# --- the account file ---------------------------------------------------------------------


class Users:
    """storage/auth/users.json: {"users": {name: {"hash", "role", "gen", "created"}}}.
    `gen` goes up on every password or role change, which ends that user's open sessions."""

    def __init__(self, folder: Path = AUTH_DIR):
        self.folder = folder
        self.file = folder / "users.json"

    def load(self) -> dict[str, dict]:
        try:
            return json.loads(self.file.read_text())["users"]
        except (OSError, ValueError, KeyError):
            return {}

    def save(self, users: dict[str, dict]) -> None:
        self.folder.mkdir(parents=True, exist_ok=True)
        tmp = self.file.with_suffix(".tmp")
        tmp.write_text(json.dumps({"users": users}, indent=2, ensure_ascii=False) + "\n")
        os.chmod(tmp, 0o600)
        tmp.replace(self.file)

    def get(self, name: str) -> dict | None:
        return self.load().get(name)

    def set_password(self, name: str, password: str, role: str | None = None) -> None:
        if not NAME_RE.fullmatch(name):
            raise ValueError("name: 2-32 characters, lowercase letters, digits, . _ -")
        if len(password) < MIN_PASSWORD:
            raise ValueError(f"password: at least {MIN_PASSWORD} characters")
        users = self.load()
        old = users.get(name, {})
        role = role or old.get("role", "user")
        if role not in ROLES:
            raise ValueError(f"role must be one of {ROLES}")
        users[name] = {
            "hash": hash_password(password),
            "role": role,
            "gen": old.get("gen", 0) + 1,
            "created": old.get("created", time.strftime("%Y-%m-%dT%H:%M:%S%z")),
        }
        self.save(users)

    def remove(self, name: str) -> bool:
        users = self.load()
        if users.pop(name, None) is None:
            return False
        self.save(users)
        return True

    def verify(self, name: str, password: str) -> dict | None:
        user = self.get(name)
        if user is None:
            check_password(password, DUMMY_HASH)  # same time whether or not the name exists
            return None
        return user if check_password(password, user["hash"]) else None


DUMMY_HASH = hash_password(secrets.token_hex(8))


# --- session cookies ----------------------------------------------------------------------


def secret(folder: Path = AUTH_DIR) -> bytes:
    """The cookie signing key, created on first use (only this machine knows it)."""
    path = folder / "secret"
    try:
        return bytes.fromhex(path.read_text().strip())
    except (OSError, ValueError):
        folder.mkdir(parents=True, exist_ok=True)
        key = secrets.token_bytes(32)
        path.write_text(key.hex())
        os.chmod(path, 0o600)
        return key


def _b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def _unb64(text: str) -> bytes:
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


def make_token(name: str, user: dict, key: bytes, now: float | None = None) -> str:
    exp = int((now or time.time()) + SESSION_SECONDS)
    body = _b64(json.dumps({"u": name, "g": user.get("gen", 0), "e": exp}).encode())
    sig = _b64(hmac.new(key, body.encode(), hashlib.sha256).digest())
    return f"{body}.{sig}"


def read_token(token: str | None, users: Users, key: bytes, now: float | None = None):
    """(name, role) for a valid, unexpired session of an existing account, else None."""
    if not token or token.count(".") != 1:
        return None
    body, sig = token.split(".")
    want = _b64(hmac.new(key, body.encode(), hashlib.sha256).digest())
    if not hmac.compare_digest(sig, want):
        return None
    try:
        data = json.loads(_unb64(body))
        name, gen, exp = data["u"], data["g"], data["e"]
    except (ValueError, KeyError, TypeError):
        return None
    user = users.get(name)
    if user is None or user.get("gen", 0) != gen or exp < (now or time.time()):
        return None
    return name, user["role"]


# --- command line: local-agent users ... ----------------------------------------------------


def _ask_password() -> str:
    first = getpass.getpass("New password: ")
    if first != getpass.getpass("Again: "):
        raise ValueError("the two passwords differ")
    return first


def main(argv: list[str] | None = None, users: Users | None = None) -> int:
    parser = argparse.ArgumentParser(prog="local-agent users", description="Platform accounts")
    sub = parser.add_subparsers(dest="cmd", required=True)
    add = sub.add_parser("add", help="create an account (asks for the password)")
    add.add_argument("name")
    add.add_argument("--admin", action="store_true", help="may use the coding agent page")
    passwd = sub.add_parser("passwd", help="change a password (signs the user out)")
    passwd.add_argument("name")
    role = sub.add_parser("role", help="change a role (signs the user out)")
    role.add_argument("name")
    role.add_argument("role", choices=ROLES)
    sub.add_parser("list", help="list accounts")
    rm = sub.add_parser("remove", help="delete an account")
    rm.add_argument("name")
    args = parser.parse_args(argv)
    users = users or Users()
    try:
        if args.cmd == "list":
            for name, user in sorted(users.load().items()):
                print(f"{name:24} {user['role']:6} created {user.get('created', '?')}")
        elif args.cmd == "add":
            if users.get(args.name):
                raise ValueError(f"{args.name} exists; use `passwd` or `role`")
            users.set_password(args.name, _ask_password(), "admin" if args.admin else "user")
            print(f"Created {args.name}")
        elif args.cmd == "passwd":
            if not users.get(args.name):
                raise ValueError(f"no account {args.name}")
            users.set_password(args.name, _ask_password())
            print(f"Changed the password of {args.name}")
        elif args.cmd == "role":
            user = users.get(args.name)
            if not user:
                raise ValueError(f"no account {args.name}")
            all_users = users.load()
            all_users[args.name] = {**user, "role": args.role, "gen": user.get("gen", 0) + 1}
            users.save(all_users)
            print(f"{args.name} is now {args.role}")
        elif args.cmd == "remove":
            if not users.remove(args.name):
                raise ValueError(f"no account {args.name}")
            print(f"Removed {args.name}")
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0

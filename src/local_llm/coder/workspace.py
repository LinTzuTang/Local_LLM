"""Project-scoped file access for the coding agent: safe paths, diffs, backups, undo, checks.

Everything is relative to one project root. Paths that leave the root (also through symlinks)
are refused, and so are secrets (`.env`), model weights (`models/`), the database files
(`storage/pgdata`), the virtualenv and git internals. Every write first stores the old and new
content under `storage/coder/backups/<id>/`, so any change can be shown as a diff and undone
(the project folder is not a git repository). Checks run only from a fixed allowlist.
"""

import difflib
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
PROCESS_START = time.time()

# Never read, listed or written.
DENIED_DIRS = ("models", "storage/pgdata", ".venv", ".git")
ALLOWED_ENV_TEMPLATES = {".env.example", ".env.sample", ".env.template"}
SECRET_SUFFIXES = (".pem", ".key", ".p12")
# Configuration with its own validated, versioned tools (see local_llm.lab.control).
CONFIG_FILES = ("data/lab/agents.json", "data/lab/scenarios.json")
# Readable, but never written by the agent (runtime data, caches).
READ_ONLY_TOP = ("storage", "runs")
CACHE_DIRS = {"__pycache__", ".pytest_cache", ".ruff_cache", ".mypy_cache", "node_modules"}
# Not listed or searched unless the glob names them explicitly (large or noisy).
QUIET_TOP = ("storage", "runs")

MAX_READ_LINES = 400
MAX_LINE_CHARS = 500
MAX_FILE_BYTES = 2_000_000
MAX_WRITE_BYTES = 300_000
MAX_LIST = 300
MAX_MATCHES = 80
CHECK_OUTPUT_CHARS = 6000


class PathError(ValueError):
    """The path is outside the project or not allowed."""


class EditError(ValueError):
    """The edit cannot be applied (text not found, not unique, file exists, ...)."""


def is_secret(name: str) -> bool:
    if name == ".env" or (name.startswith(".env.") and name not in ALLOWED_ENV_TEMPLATES):
        return True
    return name.endswith(SECRET_SUFFIXES)


def denied_reason(rel: str, write: bool = False) -> str | None:
    """Why `rel` (a project-relative POSIX path) may not be accessed, or None."""
    if rel in ("", "."):
        return "the project root itself cannot be written" if write else None
    for d in DENIED_DIRS:
        if rel == d or rel.startswith(d + "/"):
            return f"'{d}/' is off limits (model weights, database files, virtualenv, git)"
    parts = rel.split("/")
    if any(is_secret(p) for p in parts):
        return "secrets (.env, keys) are never read or written"
    if not write:
        return None
    if rel in CONFIG_FILES:
        return (
            "agents.json and scenarios.json are changed with the configuration tools "
            "(update_agent, create_agent, update_node, add_node, ...), which validate the "
            "change and keep every version"
        )
    if parts[0] in READ_ONLY_TOP:
        return f"'{parts[0]}/' holds runtime data and is not written by the coding agent"
    if CACHE_DIRS & set(parts):
        return "cache directories are not written"
    return None


def glob_regex(pattern: str) -> re.Pattern:
    """Translate a glob with `**`, `*`, `?` and `{a,b}` into a regex over POSIX paths."""
    out, i = [], 0
    while i < len(pattern):
        if pattern.startswith("**/", i):
            out.append("(?:.*/)?")
            i += 3
        elif pattern.startswith("**", i):
            out.append(".*")
            i += 2
        elif pattern[i] == "*":
            out.append("[^/]*")
            i += 1
        elif pattern[i] == "?":
            out.append("[^/]")
            i += 1
        elif pattern[i] == "{" and "}" in pattern[i:]:
            j = pattern.index("}", i)
            out.append("(?:" + "|".join(re.escape(x) for x in pattern[i + 1 : j].split(",")) + ")")
            i = j + 1
        else:
            out.append(re.escape(pattern[i]))
            i += 1
    return re.compile("".join(out) + r"\Z")


def unified_diff(before: str | None, after: str | None, path: str) -> str:
    a = (before or "").splitlines(keepends=True)
    b = (after or "").splitlines(keepends=True)
    lines = difflib.unified_diff(
        a,
        b,
        "/dev/null" if before is None else f"a/{path}",
        "/dev/null" if after is None else f"b/{path}",
        n=3,
    )
    return "".join(x if x.endswith("\n") else x + "\n" for x in lines).rstrip("\n")


def _read_text(path: Path) -> str:
    with open(path, encoding="utf-8", newline="") as f:  # keep line endings as they are
        return f.read()


@dataclass
class Change:
    """A write the agent wants to make: `before` is None when the file is new."""

    path: str
    before: str | None
    after: str

    @property
    def action(self) -> str:
        return "create" if self.before is None else "edit"

    def diff(self) -> str:
        return unified_diff(self.before, self.after, self.path)

    def counts(self) -> tuple[int, int]:
        added = removed = 0
        for line in self.diff().splitlines():
            if line.startswith("+") and not line.startswith("+++"):
                added += 1
            elif line.startswith("-") and not line.startswith("---"):
                removed += 1
        return added, removed


# --- checks the agent may run (nothing else is ever executed) ---------------------------------

_BIN = Path(sys.executable).parent
_RUFF = str(_BIN / "ruff") if (_BIN / "ruff").exists() else (shutil.which("ruff") or "ruff")
_VALIDATE_CONFIG = """
import sys
from local_llm.lab import store
bad = 0
for a in store.agents():
    c = store.validate_agent(a)
    bad += not c.ok
    print(("OK  " if c.ok else "ERR ") + "agent " + a["id"], *c.errors, sep="\\n    ")
for s in store.scenarios():
    c = store.validate_scenario(s)
    bad += not c.ok
    print(("OK  " if c.ok else "ERR ") + "scenario " + s["id"], *c.errors, sep="\\n    ")
sys.exit(1 if bad else 0)
"""
# name -> (shown as, command, default targets, timeout seconds)
CHECKS: dict[str, tuple[str, list[str], list[str], int]] = {
    "pytest": (
        "pytest -q",
        [sys.executable, "-m", "pytest", "-q", "--tb=short", "-p", "no:cacheprovider"],
        [],
        300,
    ),
    "ruff": ("ruff check", [_RUFF, "check", "--no-cache"], ["app", "src", "tests"], 60),
    "ruff_format": (
        "ruff format --check",
        [_RUFF, "format", "--check", "--no-cache"],
        ["app", "src", "tests"],
        60,
    ),
    "compileall": (
        "python -m compileall",
        [sys.executable, "-m", "compileall", "-q"],
        ["app", "src", "tests"],
        120,
    ),
    "config": (
        "validate agents.json + scenarios.json",
        [sys.executable, "-c", _VALIDATE_CONFIG],
        [],
        120,
    ),
}


def _truncate(text: str, limit: int = CHECK_OUTPUT_CHARS) -> str:
    if len(text) <= limit:
        return text
    head = limit // 4
    cut = len(text) - limit
    return f"{text[:head]}\n… [{cut} characters cut] …\n{text[-(limit - head) :]}"


# --- the workspace -----------------------------------------------------------------------------


class Workspace:
    """File tools bound to one project root, with backups of every change."""

    def __init__(self, root: Path | str = PROJECT_ROOT, backup_dir: Path | str | None = None):
        self.root = Path(root).resolve()
        self.backup_dir = (
            Path(backup_dir) if backup_dir else self.root / "storage" / "coder" / "backups"
        )
        self._lock = threading.Lock()

    # -- paths ---------------------------------------------------------------------------------

    def rel(self, path: Path) -> str:
        return path.relative_to(self.root).as_posix() or "."

    def resolve(self, path: str, write: bool = False) -> Path:
        """Absolute path inside the project, or PathError."""
        path = (path or "").strip()
        if not path or "\x00" in path:
            raise PathError("empty or invalid path")
        p = Path(path).expanduser()
        p = (p if p.is_absolute() else self.root / p).resolve()  # follows symlinks
        if not p.is_relative_to(self.root):
            raise PathError(f"{path}: outside the project ({self.root})")
        rel = self.rel(p)
        if reason := denied_reason("" if rel == "." else rel, write):
            raise PathError(f"{path}: {reason}")
        return p

    def _walk(self, pattern: str) -> Iterator[str]:
        """Allowed project files (relative) matching a glob; prunes denied and noisy folders."""
        pattern = pattern.strip().removeprefix("./") or "**/*"
        if pattern.endswith("/"):
            pattern += "**"
        elif not any(c in pattern for c in "*?{") and (self.root / pattern).is_dir():
            pattern += "/**"
        rx = glob_regex(pattern)
        by_name = "/" not in pattern
        explicit_top = pattern.split("/")[0]
        for dirpath, dirnames, filenames in os.walk(self.root):
            rel_dir = Path(dirpath).relative_to(self.root).as_posix()
            rel_dir = "" if rel_dir == "." else rel_dir + "/"
            keep = []
            for d in sorted(dirnames):
                rel = rel_dir + d
                if d in CACHE_DIRS or denied_reason(rel) or rel == "storage/coder":
                    continue
                if not rel_dir and d in QUIET_TOP and explicit_top != d:
                    continue
                if not rel_dir and d.startswith(".") and not explicit_top.startswith("."):
                    continue
                keep.append(d)
            dirnames[:] = keep
            for name in sorted(filenames):
                rel = rel_dir + name
                if name == ".DS_Store" or denied_reason(rel):
                    continue
                if rx.match(rel) or (by_name and rx.match(name)):
                    yield rel

    # -- reading -------------------------------------------------------------------------------

    def list_files(self, pattern: str = "**/*", limit: int = MAX_LIST) -> str:
        files = []
        for rel in self._walk(pattern):
            files.append(rel)
            if len(files) > limit * 4:
                break
        if not files:
            return f"No files match {pattern!r}."
        shown = files[:limit]
        more = f"\n… and {len(files) - limit}+ more; use a narrower glob" if files[limit:] else ""
        return f"{len(shown)} file(s) matching {pattern!r}:\n" + "\n".join(shown) + more

    def read_file(self, path: str, start: int = 1, end: int | None = None) -> str:
        p = self.resolve(path)
        rel = self.rel(p)
        if p.is_dir():
            return f"{rel} is a folder. Use list_files('{rel}/**') to see its files."
        if not p.exists():
            near = [f for f in self._walk("**/" + p.name)][:5]
            hint = f" Did you mean: {', '.join(near)}?" if near else ""
            raise PathError(f"{rel}: no such file.{hint}")
        if p.stat().st_size > MAX_FILE_BYTES:
            raise PathError(f"{rel}: file is larger than {MAX_FILE_BYTES // 1_000_000} MB")
        data = p.read_bytes()
        if b"\x00" in data[:8000]:
            raise PathError(f"{rel}: binary file")
        lines = data.decode("utf-8", errors="replace").splitlines()
        total = len(lines)
        if total == 0:
            return f"{rel} is empty."
        start = min(max(1, int(start or 1)), total)
        end = total if end is None else min(max(int(end), start), total)
        end = min(end, start + MAX_READ_LINES - 1)
        body = "\n".join(
            f"{n:>5}\t{line[:MAX_LINE_CHARS]}{' …' if len(line) > MAX_LINE_CHARS else ''}"
            for n, line in enumerate(lines[start - 1 : end], start)
        )
        more = f"\n(more: read_file('{rel}', start={end + 1}))" if end < total else ""
        return f"{rel} — lines {start}-{end} of {total}\n{body}{more}"

    def search(self, pattern: str, glob: str | None = None, limit: int = MAX_MATCHES) -> str:
        try:
            rx = re.compile(pattern)
        except re.error as exc:
            raise ValueError(f"invalid regex {pattern!r}: {exc}") from exc
        hits = self._search_rg(pattern, glob, limit) if shutil.which("rg") else None
        if hits is None:
            hits = []
            for rel in self._walk(glob or "**/*"):
                p = self.root / rel
                try:
                    if p.stat().st_size > MAX_FILE_BYTES:
                        continue
                    text = p.read_text(encoding="utf-8")
                except (UnicodeDecodeError, OSError):
                    continue
                for n, line in enumerate(text.splitlines(), 1):
                    if rx.search(line):
                        hits.append(f"{rel}:{n}: {line.strip()[:200]}")
                        if len(hits) > limit:
                            break
                if len(hits) > limit:
                    break
        if not hits:
            where = f" in {glob!r}" if glob else ""
            return f"No matches for {pattern!r}{where}."
        more = "\n… more matches; narrow the pattern or glob" if len(hits) > limit else ""
        return "\n".join(hits[:limit]) + more

    def _search_rg(self, pattern: str, glob: str | None, limit: int) -> list[str] | None:
        cmd = ["rg", "--line-number", "--no-heading", "--color=never", "--max-columns=200"]
        cmd += ["--hidden", "--no-ignore", "--max-count=20", "-e", pattern]
        if glob:
            cmd += ["-g", glob]
        top = (glob or "").split("/")[0]
        for d in (*DENIED_DIRS, *QUIET_TOP, *CACHE_DIRS, "storage/coder"):
            if d != top:
                cmd += ["-g", f"!{d}/**" if "/" in d or d in DENIED_DIRS else f"!**/{d}/**"]
        # .env files are dropped below (templates such as .env.example stay searchable)
        cmd += ["-g", "!**/.env", "-g", "!.DS_Store", "."]
        try:
            proc = subprocess.run(
                cmd, cwd=self.root, capture_output=True, text=True, timeout=30, check=False
            )
        except (OSError, subprocess.TimeoutExpired):
            return None
        if proc.returncode not in (0, 1):
            return None
        hits = []
        for line in proc.stdout.splitlines():
            rel = line.removeprefix("./").split(":", 1)[0]
            if not denied_reason(rel):
                hits.append(line.removeprefix("./"))
            if len(hits) > limit:
                break
        return hits

    # -- planning writes -----------------------------------------------------------------------

    def plan_edit(self, path: str, old: str, new: str) -> Change:
        p = self.resolve(path, write=True)
        rel = self.rel(p)
        if not p.is_file():
            raise EditError(f"{rel} does not exist; create it with write_file")
        if not old:
            raise EditError("`old` is empty; copy the exact text to replace from read_file")
        if old == new:
            raise EditError("`old` and `new` are identical; nothing to change")
        text = _read_text(p)
        count = text.count(old)
        if count == 0:
            raise EditError(f"`old` was not found in {rel}. {_near_miss(text, old)}")
        if count > 1:
            lines = [text[: m.start()].count("\n") + 1 for m in re.finditer(re.escape(old), text)]
            raise EditError(
                f"`old` occurs {count} times in {rel} (lines {lines[:10]}); include more "
                "surrounding lines so it matches exactly one place"
            )
        return Change(rel, text, text.replace(old, new, 1))

    def plan_write(self, path: str, content: str) -> Change:
        p = self.resolve(path, write=True)
        rel = self.rel(p)
        if p.exists():
            raise EditError(f"{rel} already exists; change it with edit_file instead")
        if len(content.encode()) > MAX_WRITE_BYTES:
            raise EditError(f"content is larger than {MAX_WRITE_BYTES // 1000} kB")
        if content and not content.endswith("\n"):
            content += "\n"
        return Change(rel, None, content)

    # -- applying writes with backups ----------------------------------------------------------

    def apply(self, change: Change, source: str = "coder") -> dict:
        """Write the change after storing a backup; returns the change-log entry."""
        with self._lock:
            p = self.resolve(change.path, write=True)
            current = _read_text(p) if p.exists() else None
            if current != change.before:
                raise EditError(f"{change.path} changed since the edit was planned; read it again")
            now = time.time()
            entry_id = time.strftime("%Y%m%d-%H%M%S", time.localtime(now))
            entry_id += f"-{int(now * 1e6) % 1_000_000:06d}-{uuid.uuid4().hex[:4]}"
            folder = self.backup_dir / entry_id
            folder.mkdir(parents=True, exist_ok=False)
            if change.before is not None:
                (folder / "before").write_text(change.before, encoding="utf-8")
            (folder / "after").write_text(change.after, encoding="utf-8")
            added, removed = change.counts()
            entry = {
                "id": entry_id,
                "path": change.path,
                "action": change.action,
                "source": source,
                "at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now)),
                "ts": now,
                "added": added,
                "removed": removed,
            }
            (folder / "meta.json").write_text(json.dumps(entry, ensure_ascii=False, indent=1))
            p.parent.mkdir(parents=True, exist_ok=True)
            tmp = p.with_name(f".{p.name}.coder-tmp")
            with open(tmp, "w", encoding="utf-8", newline="") as f:
                f.write(change.after)
            if p.exists():
                shutil.copymode(p, tmp)
            tmp.replace(p)
            return entry

    def changes(self, limit: int = 50) -> list[dict]:
        """Change-log entries, newest first."""
        if not self.backup_dir.exists():
            return []
        entries = []
        for meta in sorted(self.backup_dir.glob("*/meta.json"), reverse=True)[:limit]:
            try:
                entries.append(json.loads(meta.read_text()))
            except (OSError, json.JSONDecodeError):
                continue
        return entries

    def entry_change(self, entry_id: str) -> Change:
        folder = self._entry_folder(entry_id)
        meta = json.loads((folder / "meta.json").read_text())
        before = (folder / "before").read_text() if (folder / "before").exists() else None
        return Change(meta["path"], before, (folder / "after").read_text())

    def _entry_folder(self, entry_id: str) -> Path:
        folder = self.backup_dir / Path(entry_id).name
        if not (folder / "meta.json").exists():
            raise EditError(f"no change '{entry_id}' in the change log")
        return folder

    def undo(self, entry_id: str | None = None) -> dict:
        """Restore the file from before a change (default: the latest change not yet undone).

        Refused when the file was changed again afterwards: undo the later change first.
        """
        with self._lock:
            if entry_id is None:
                open_entries = [e for e in self.changes(1000) if not e.get("undone_at")]
                if not open_entries:
                    raise EditError("nothing to undo")
                entry_id = open_entries[0]["id"]
            folder = self._entry_folder(entry_id)
            meta = json.loads((folder / "meta.json").read_text())
            if meta.get("undone_at"):
                raise EditError(f"change {entry_id} was already undone")
            change = self.entry_change(entry_id)
            p = self.resolve(change.path, write=True)
            current = _read_text(p) if p.exists() else None
            if current != change.after:
                raise EditError(
                    f"{change.path} was changed after {entry_id}; undo the later change first"
                )
            if change.before is None:
                p.unlink()
            else:
                with open(p, "w", encoding="utf-8", newline="") as f:
                    f.write(change.before)
            meta["undone_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
            (folder / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1))
            return meta

    def code_changed_since(self, ts: float = PROCESS_START) -> bool:
        """Whether Python code was changed (and not undone) since `ts` (default: process start).

        The running process still has the old code imported, so configuration that depends on
        code (e.g. a new tool name) must then be validated in a fresh process.
        """
        return any(
            e["path"].endswith(".py") and e.get("ts", 0) >= ts and not e.get("undone_at")
            for e in self.changes(200)
        )

    # -- checks --------------------------------------------------------------------------------

    def run_check(self, name: str, path: str | None = None, timeout: int | None = None) -> str:
        if name not in CHECKS:
            raise ValueError(f"unknown check {name!r}; allowed: {', '.join(CHECKS)}")
        shown, cmd, targets, limit = CHECKS[name]
        if path:
            if name == "config":
                raise ValueError("the config check takes no path")
            targets = [self.rel(self.resolve(path))]
        else:
            targets = [t for t in targets if (self.root / t).exists()]
        env = {
            **os.environ,
            "PYTHONPATH": str(self.root / "src"),  # this project's code, also in a copy
            "PYTHONDONTWRITEBYTECODE": "1",
            "NO_COLOR": "1",
        }
        started = time.perf_counter()
        try:
            proc = subprocess.run(
                cmd + targets,
                cwd=self.root,
                env=env,
                capture_output=True,
                text=True,
                timeout=timeout or limit,
                stdin=subprocess.DEVNULL,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            out = (exc.stdout or b"").decode() if isinstance(exc.stdout, bytes) else exc.stdout
            return f"$ {shown}\nTIMED OUT after {timeout or limit}s\n{_truncate(out or '')}"
        seconds = time.perf_counter() - started
        status = "PASSED" if proc.returncode == 0 else f"FAILED (exit {proc.returncode})"
        output = (proc.stdout + ("\n" + proc.stderr if proc.stderr.strip() else "")).strip()
        label = f"$ {shown} {' '.join(targets)}".rstrip()
        return f"{label}\n{status} in {seconds:.1f}s\n{_truncate(output)}".rstrip()


def _near_miss(text: str, old: str) -> str:
    """A hint for why `old` did not match."""
    if re.search(r"(?m)^\s*\d+\t", old):
        return "It contains read_file line-number prefixes; remove them."
    squash = re.compile(r"\s+")
    if squash.sub(" ", old).strip() in squash.sub(" ", text):
        return "The text exists with different whitespace or indentation; copy it exactly."
    first = old.strip().splitlines()[0].strip() if old.strip() else ""
    if first and first in text:
        line = text[: text.index(first)].count("\n") + 1
        return f"Its first line occurs at line {line}, but the following lines differ."
    return "Read the file again and copy the exact lines (without line-number prefixes)."

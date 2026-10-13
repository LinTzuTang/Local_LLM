"""Agent skills: reusable instructions an agent loads only when a task needs them.

A skill is a folder `data/lab/skills/<name>/` with a `SKILL.md` (the open Agent Skills format,
https://agentskills.io: YAML front matter with `name` and `description`, then Markdown
instructions) and optional extra files (examples, references, a LICENSE). This platform also
reads `description_zh`, `source` and `license` from the front matter for the UI.

An agent that lists skills sees only their names and descriptions in its system prompt and
calls `load_skill(name)` to read the full instructions, then `load_skill(name, file)` for an
extra file the instructions point to (progressive disclosure). That keeps prompts short for
small local models. Skills are plain text: nothing in them is executed.
"""

import re
from functools import lru_cache
from pathlib import Path

import yaml
from langchain.tools import tool

SKILLS_DIR = Path(__file__).resolve().parents[2] / "data" / "lab" / "skills"
FRONT_MATTER = re.compile(r"^---\s*\n(.*?)\n---\s*\n?(.*)$", re.S)
TEXT_TYPES = {".md", ".txt", ".csv", ".json", ".yaml", ".yml", ".sql", ".html", ""}
MAX_FILE_CHARS = 12_000  # a skill file loaded into the conversation (about 8k tokens)


def _front_matter(raw: str) -> dict:
    """YAML front matter as {str: str}; falls back to `key: value` lines if YAML fails."""
    try:
        data = yaml.safe_load(raw)
    except yaml.YAMLError:
        data = None
    if not isinstance(data, dict):
        data = {}
        for line in raw.splitlines():
            if ":" in line and not line.startswith((" ", "\t")):
                key, value = line.split(":", 1)
                data[key.strip()] = value.strip().strip('"').strip("'")
    return {str(k): " ".join(str(v).split()) for k, v in data.items() if v is not None}


def _files(folder: Path) -> list[str]:
    """Extra files of a skill as relative paths (no SKILL.md, nothing hidden)."""
    return sorted(
        p.relative_to(folder).as_posix()
        for p in folder.rglob("*")
        if p.is_file()
        and p.name != "SKILL.md"
        and not any(part.startswith(".") for part in p.relative_to(folder).parts)
    )


def _parse(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    meta, body = {}, text
    if m := FRONT_MATTER.match(text):
        meta, body = _front_matter(m.group(1)), m.group(2)
    return {
        "name": meta.get("name") or path.parent.name,
        "description": meta.get("description", ""),
        "description_zh": meta.get("description_zh", ""),
        "source": meta.get("source", "local"),
        "license": meta.get("license", ""),
        "path": str(path.parent),
        "body": body.strip(),
        "files": _files(path.parent),
    }


@lru_cache(maxsize=1)
def _all(stamp: float) -> dict[str, dict]:
    if not SKILLS_DIR.exists():
        return {}
    return {s["name"]: s for s in (_parse(p) for p in sorted(SKILLS_DIR.glob("*/SKILL.md")))}


def _stamp() -> float:
    files = [p for p in SKILLS_DIR.rglob("*") if p.is_file()] if SKILLS_DIR.exists() else []
    return max((p.stat().st_mtime for p in files), default=0.0) + len(files)


def list_skills() -> list[dict]:
    """All installed skills: name, description(_zh), source, license, path, files (no body)."""
    return [{k: v for k, v in s.items() if k != "body"} for s in _all(_stamp()).values()]


def get_skill(name: str) -> dict:
    skills = _all(_stamp())
    if name not in skills:
        raise KeyError(f"unknown skill '{name}'; installed: {sorted(skills)}")
    return skills[name]


def read_skill_file(name: str, file: str) -> str:
    """One extra text file of a skill (e.g. 'examples/faq-answers.md'), cut to MAX_FILE_CHARS."""
    skill = get_skill(name)
    file = file.strip().lstrip("./")
    if file not in skill["files"]:
        listed = ", ".join(skill["files"]) or "none"
        raise FileNotFoundError(f"skill '{name}' has no file '{file}'; its files: {listed}")
    folder = Path(skill["path"]).resolve()
    path = (folder / file).resolve()
    if not path.is_relative_to(folder) or path.suffix.lower() not in TEXT_TYPES:
        raise PermissionError(f"'{file}' is not a readable text file of skill '{name}'")
    text = path.read_text(encoding="utf-8", errors="replace")
    if len(text) > MAX_FILE_CHARS:
        text = text[:MAX_FILE_CHARS] + f"\n[cut at {MAX_FILE_CHARS} characters]"
    return text


def skill_prompt(names: list[str]) -> str:
    """System-prompt section listing the skills an agent may load."""
    if not names:
        return ""
    lines = [f"- {n}: {get_skill(n)['description']}" for n in names]
    return (
        "\n\nSkills you can use (call load_skill with the name before doing a task that "
        "matches; follow the loaded instructions):\n" + "\n".join(lines)
    )


def skill_tool(names: list[str]):
    """A `load_skill` tool restricted to this agent's skills."""
    allowed = sorted(names)

    @tool
    def load_skill(name: str, file: str = "") -> str:
        """Load the full instructions of one of your skills by name. If the instructions
        mention an extra file of the skill (e.g. 'examples/faq-answers.md'), call again with
        that path as `file` to read it."""
        if name not in allowed:
            return f"Unknown skill '{name}'. Your skills: {', '.join(allowed)}"
        if file.strip():
            try:
                return read_skill_file(name, file)
            except (FileNotFoundError, PermissionError) as exc:
                return str(exc)
        skill = get_skill(name)
        extra = (
            "\n\nFiles in this skill (read one with load_skill(name, file)): "
            + ", ".join(skill["files"])
            if skill["files"]
            else ""
        )
        return skill["body"] + extra

    return load_skill

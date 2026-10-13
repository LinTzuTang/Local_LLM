import ast
import json
import operator
import re
from pathlib import Path

from langchain.tools import tool

CATALOG = Path(__file__).resolve().parents[2] / "data" / "models.json"
OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Mod: operator.mod,
}


def calculate(expression: str) -> float:
    if len(expression) > 200:
        raise ValueError("Expression is too long")
    tree = ast.parse(expression, mode="eval")
    if len(list(ast.walk(tree))) > 60:
        raise ValueError("Expression is too complex")

    def visit(node):
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            result = node.value
        elif isinstance(node, ast.BinOp) and type(node.op) in OPS:
            result = OPS[type(node.op)](visit(node.left), visit(node.right))
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            result = visit(node.operand) * (-1 if isinstance(node.op, ast.USub) else 1)
        else:
            raise ValueError("Only numeric arithmetic (+ - * / %) is supported")
        if not -1e12 <= result <= 1e12:
            raise ValueError("Result is out of range")
        return result

    return visit(tree.body)


@tool
def calculator(expression: str) -> str:
    """Calculate arithmetic exactly. Input examples: '37 * 19 + 8' or '128 / 4'."""
    try:
        return str(calculate(expression))
    except (ValueError, SyntaxError, ZeroDivisionError, OverflowError) as exc:
        return f"Invalid expression: {exc}"


NAME_FIELDS = ("id", "name")
TEXT_FIELDS = ("category", "license", "notes", "deployment", "hardware_plan")
# Words that describe what the user wants to know rather than which model to find.
STOPWORDS = {
    "a", "an", "and", "catalog", "find", "for", "in", "is", "its", "license", "licence",
    "model", "models", "of", "report", "search", "the", "what", "which",
}  # fmt: skip


def search_terms(query: str) -> list[str]:
    tokens = re.findall(r"[a-z0-9][a-z0-9._/:-]*|[\u4e00-\u9fff]+", query.lower())
    return [t for t in (tok.rstrip(".:") for tok in tokens) if t and t not in STOPWORDS]


def rank_models(query: str, rows: list[dict]) -> list[dict]:
    """Rank catalog rows by field values only; JSON keys never count as matches."""
    terms = search_terms(query)
    scored = []
    for index, row in enumerate(rows):
        names = " ".join(str(row.get(f) or "") for f in NAME_FIELDS).lower()
        text = " ".join(str(row.get(f) or "") for f in TEXT_FIELDS).lower()
        score = sum(3 if t in names else 1 if t in text else 0 for t in terms)
        if score:
            scored.append((-score, index, row))
    return [row for *_, row in sorted(scored, key=lambda item: item[:2])]


@tool
def search_models(query: str) -> str:
    """Search the verified local model catalog by model name or category. Use short keywords."""
    matches = rank_models(query, json.loads(CATALOG.read_text()))
    return json.dumps(matches[:3], ensure_ascii=False) if matches else "No catalog match."

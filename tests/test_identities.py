"""Database identities beyond the three departments (needs ./scripts/start_db.sh)."""

import shutil

import psycopg
import pytest

from local_llm import company_db

pytestmark = pytest.mark.skipif(
    company_db.status() is not None, reason="company database is not running"
)


def tools(role: str) -> dict:
    return {t.name: t for t in company_db.make_tools(role)}


def test_identities_list_builtin_and_custom_with_both_languages():
    company_db.ensure_seed_identities()
    idents = company_db.identities()
    roles = [i["role"] for i in idents]
    assert roles[:3] == ["agent_sales", "agent_hr", "agent_support"]
    for ident in idents:
        assert ident["label"]["zh"] and ident["label"]["en"]
        assert ident["description"]["zh"] and ident["description"]["en"]
        assert ident["builtin"] == (ident["role"] in company_db.ROLES)


@pytest.mark.parametrize(
    ("role", "allowed", "denied"),
    [
        ("agent_finance", "SELECT sum(quantity * unit_price) FROM sales.order_items",
         "SELECT count(*) FROM hr.employees"),
        ("agent_management", "SELECT count(*) FROM hr.employees WHERE status = '在職'",
         "SELECT max(salary) FROM hr.employees"),
        ("agent_intern", "SELECT count(*) FROM inventory.warehouses",
         "SELECT max(unit_cost) FROM sales.products"),
    ],
)  # fmt: skip
def test_seed_identities_are_enforced_by_postgres(role, allowed, denied):
    if role not in company_db.identity_roles():
        pytest.skip(f"{role} was deleted on this machine")
    query = tools(role)["db_query"]
    assert query.invoke({"query": allowed}).startswith("[")
    assert "Permission denied" in query.invoke({"query": denied})


def test_create_preview_grant_and_delete_an_identity(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    shutil.copy(company_db.ENV_FILE, env)
    monkeypatch.setattr(company_db, "ENV_FILE", env)
    monkeypatch.setattr(company_db, "IDENTITIES_FILE", tmp_path / "identities.json")
    role = "agent_pytest_tmp"
    company_db.create_identity(
        role,
        {"zh": "測試", "en": "Test"},
        {"zh": "暫時", "en": "temporary"},
        {"sales.products": ["id", "name"], "inventory.warehouses": []},
    )
    try:
        assert role in company_db.identity_roles()
        assert f"{company_db._env_key(role)}=" in env.read_text()
        cols, rows = company_db.preview(role, "sales.products", limit=3)
        assert cols == ["id", "name"] and len(rows) == 3
        with pytest.raises(psycopg.errors.InsufficientPrivilege):
            company_db.preview(role, "sales.products", ["id", "unit_cost"])
        assert "Permission denied" in tools(role)["db_query"].invoke(
            {"query": "SELECT 1 FROM hr.employees"}
        )
        assert "Only SELECT" in tools(role)["db_query"].invoke(
            {"query": "INSERT INTO inventory.warehouses VALUES (9, 'x', 'y')"}
        )
        matrix = company_db.access_matrix([role])[role]
        assert matrix["sales.products"][0] == ["id", "name"]
        assert matrix["inventory.warehouses"][0] == ["id", "name", "city"]

        company_db.set_access("sales.products", role, [])
        assert company_db.readable_columns(role)["sales.products"][0][-1] == "unit_cost"
        company_db.set_access("inventory.warehouses", role, None)
        saved = company_db._read_store()["identities"][0]["grants"]
        assert saved == {"sales.products": []}  # kept for re-init
    finally:
        company_db.delete_identity(role)
    assert role not in company_db.identity_roles()
    assert company_db._env_key(role) not in env.read_text()
    with company_db._admin() as db:
        assert not db.execute("SELECT 1 FROM pg_roles WHERE rolname = %s", [role]).fetchone()


def test_identity_rules():
    for bad in ["postgres", "agent_", "agent_X", "sales", "agent_a-b"]:
        with pytest.raises(ValueError):
            company_db.create_identity(bad, {"zh": "x", "en": "x"})
    with pytest.raises(ValueError):
        company_db.create_identity("agent_sales", {"zh": "x"})  # already exists
    with pytest.raises(ValueError):
        company_db.delete_identity("agent_hr")  # built in
    with pytest.raises(ValueError):
        company_db.set_access("sales.orders", "postgres", [])  # not an identity

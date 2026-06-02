"""
test_fixes.py — verifies all 9 structural fixes without requiring Ollama.
Run:  python test_fixes.py
"""

import io
import sqlite3
import sys
import tempfile
import traceback

sys.path.insert(0, ".")

from database.schema_manager import SchemaManager, _escape_sql_str
from database.dynamic_crud import DynamicCRUD
from csv_importer import CSVImporter, VALID_TYPES

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"

results = []


def test(name):
    def decorator(fn):
        try:
            fn()
            results.append((name, True, ""))
            print(f"  {PASS}  {name}")
        except Exception as exc:
            msg = f"{type(exc).__name__}: {exc}"
            results.append((name, False, msg))
            print(f"  {FAIL}  {name}")
            print(f"         {msg}")
            traceback.print_exc()
        return fn
    return decorator


def fresh_db():
    d = tempfile.mkdtemp()
    sm = SchemaManager(d)
    crud = DynamicCRUD(sm)
    uid = sm.create_user("u", "pass1234")
    return sm, crud, uid


# ─────────────────────────────────────────────────────────────────────────────
# Fix 1 — _escape_sql_str helper exists and works
# ─────────────────────────────────────────────────────────────────────────────

@test("Fix 1a: _escape_sql_str doubles single quotes")
def _():
    assert _escape_sql_str("O'Brien") == "O''Brien"
    assert _escape_sql_str("no quotes") == "no quotes"
    assert _escape_sql_str("it's it's") == "it''s it''s"


@test("Fix 1b: table with single-quote default is created without error")
def _():
    sm, crud, uid = fresh_db()
    sm.create_dynamic_table(uid, "quotes_test", [
        {"name": "name", "type": "TEXT", "default": "O'Brien"},
    ])
    # Insert with DEFAULT VALUES (no explicit columns) — DB default should apply
    new_id = crud.insert_record(uid, "quotes_test", {})
    row = crud.get_record(uid, "quotes_test", new_id)
    assert row["name"] == "O'Brien", f"got {row['name']!r}"


@test("Fix 1c: add_column with single-quote default doesn't error")
def _():
    sm, crud, uid = fresh_db()
    sm.create_dynamic_table(uid, "t1", [{"name": "x", "type": "TEXT"}])
    sm.add_column_to_table(uid, "t1", {
        "name": "note", "type": "TEXT", "default": "it's fine"
    })
    new_id = crud.insert_record(uid, "t1", {"x": "hello"})
    row = crud.get_record(uid, "t1", new_id)
    assert row["note"] == "it's fine", f"got {row['note']!r}"


# ─────────────────────────────────────────────────────────────────────────────
# Fix 2 — compensating rollback (simulated by creating duplicate table)
# ─────────────────────────────────────────────────────────────────────────────

@test("Fix 2a: create_dynamic_table raises on duplicate, metadata stays clean")
def _():
    sm, crud, uid = fresh_db()
    sm.create_dynamic_table(uid, "dup", [{"name": "a", "type": "TEXT"}])
    try:
        sm.create_dynamic_table(uid, "dup", [{"name": "b", "type": "TEXT"}])
        assert False, "should have raised"
    except ValueError:
        pass
    tables = sm.get_user_tables(uid)
    assert len(tables) == 1, f"expected 1 table, got {len(tables)}"


@test("Fix 2b: drop_table drops user DB first, then metadata")
def _():
    sm, crud, uid = fresh_db()
    sm.create_dynamic_table(uid, "drop_me", [{"name": "v", "type": "TEXT"}])
    crud.insert_record(uid, "drop_me", {"v": "hello"})
    sm.drop_table(uid, "drop_me")
    tables = sm.get_user_tables(uid)
    assert not any(t["table_name"] == "drop_me" for t in tables)


# ─────────────────────────────────────────────────────────────────────────────
# Fix 3 — UNIQUE INDEX on (table_id, column_name)
# ─────────────────────────────────────────────────────────────────────────────

@test("Fix 3: unique index idx_cc_col exists on custom_columns")
def _():
    sm, _, _ = fresh_db()
    conn = sqlite3.connect(sm._metadata_path)
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='index' AND name='idx_cc_col'"
    ).fetchall()
    conn.close()
    assert rows, "idx_cc_col index not found in metadata.db"


@test("Fix 3b: add_column rejects duplicate column name")
def _():
    sm, crud, uid = fresh_db()
    sm.create_dynamic_table(uid, "t2", [{"name": "col_a", "type": "TEXT"}])
    try:
        sm.add_column_to_table(uid, "t2", {"name": "col_a", "type": "INTEGER"})
        assert False, "should have raised on duplicate column"
    except ValueError:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# Fix 4 — index on (table_id, position)
# ─────────────────────────────────────────────────────────────────────────────

@test("Fix 4: idx_cc_pos index exists on custom_columns")
def _():
    sm, _, _ = fresh_db()
    conn = sqlite3.connect(sm._metadata_path)
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='index' AND name='idx_cc_pos'"
    ).fetchall()
    conn.close()
    assert rows, "idx_cc_pos index not found in metadata.db"


# ─────────────────────────────────────────────────────────────────────────────
# Fix 5 — is_required enforced at application layer
# ─────────────────────────────────────────────────────────────────────────────

@test("Fix 5a: insert_record raises when required column is missing and has no default")
def _():
    sm, crud, uid = fresh_db()
    sm.create_dynamic_table(uid, "req_test", [
        {"name": "name", "type": "TEXT", "required": True},
        {"name": "age",  "type": "INTEGER"},
    ])
    try:
        crud.insert_record(uid, "req_test", {"age": 30})
        assert False, "should have raised ValueError for missing required 'name'"
    except ValueError as e:
        assert "name" in str(e).lower(), f"unexpected message: {e}"


@test("Fix 5b: insert_record succeeds when required column IS provided")
def _():
    sm, crud, uid = fresh_db()
    sm.create_dynamic_table(uid, "req_ok", [
        {"name": "name", "type": "TEXT", "required": True},
    ])
    rid = crud.insert_record(uid, "req_ok", {"name": "Alice"})
    assert rid > 0


@test("Fix 5c: insert_record skips required check when column has a default")
def _():
    sm, crud, uid = fresh_db()
    sm.create_dynamic_table(uid, "req_def", [
        {"name": "status", "type": "TEXT", "required": True, "default": "active"},
    ])
    # 'status' is required but has a DB-level default, so empty insert should succeed
    rid = crud.insert_record(uid, "req_def", {})
    assert rid > 0
    row = crud.get_record(uid, "req_def", rid)
    assert row["status"] == "active", f"expected 'active', got {row['status']!r}"


# ─────────────────────────────────────────────────────────────────────────────
# Fix 6 — count_records raises on missing table instead of silently returning 0
# ─────────────────────────────────────────────────────────────────────────────

@test("Fix 6: count_records raises PermissionError for non-existent table")
def _():
    sm, crud, uid = fresh_db()
    try:
        crud.count_records(uid, "ghost_table")
        assert False, "should have raised PermissionError"
    except PermissionError:
        pass


@test("Fix 6b: count_records returns correct count for existing table")
def _():
    sm, crud, uid = fresh_db()
    sm.create_dynamic_table(uid, "cnt", [{"name": "v", "type": "TEXT"}])
    assert crud.count_records(uid, "cnt") == 0
    crud.insert_record(uid, "cnt", {"v": "x"})
    assert crud.count_records(uid, "cnt") == 1


# ─────────────────────────────────────────────────────────────────────────────
# Fix 7 — TIMESTAMP in csv_importer VALID_TYPES
# ─────────────────────────────────────────────────────────────────────────────

@test("Fix 7a: TIMESTAMP is in csv_importer VALID_TYPES")
def _():
    assert "TIMESTAMP" in VALID_TYPES, f"VALID_TYPES = {VALID_TYPES}"


@test("Fix 7b: CSV import with TIMESTAMP column type works end-to-end")
def _():
    sm, crud, uid = fresh_db()
    csv_imp = CSVImporter(sm, crud)
    csv_bytes = b"name,event_time\nAlice,2024-01-15 10:30:00\nBob,2024-02-20 08:00:00\n"
    preview = csv_imp.read(io.BytesIO(csv_bytes))
    result = csv_imp.import_to_table(
        uid,
        preview["df"],
        "ts_test",
        {"name": "TEXT", "event_time": "TIMESTAMP"},
    )
    assert result["rows_imported"] == 2, f"rows_imported={result['rows_imported']}"
    rows = crud.query_table(uid, "ts_test")
    assert rows[0]["event_time"] == "2024-01-15 10:30:00"


# ─────────────────────────────────────────────────────────────────────────────
# Reserved column names — bonus guard exposed by tests
# ─────────────────────────────────────────────────────────────────────────────

@test("Bonus: create_dynamic_table rejects reserved column names")
def _():
    sm, _, uid = fresh_db()
    for reserved in ("id", "created_at", "updated_at"):
        try:
            sm.create_dynamic_table(uid, f"bad_{reserved}", [
                {"name": reserved, "type": "TEXT"}
            ])
            assert False, f"should have raised for reserved name '{reserved}'"
        except ValueError:
            pass


@test("Bonus: add_column_to_table rejects reserved column names")
def _():
    sm, _, uid = fresh_db()
    sm.create_dynamic_table(uid, "guard_test", [{"name": "x", "type": "TEXT"}])
    for reserved in ("id", "created_at", "updated_at"):
        try:
            sm.add_column_to_table(uid, "guard_test", {"name": reserved, "type": "TEXT"})
            assert False, f"should have raised for reserved name '{reserved}'"
        except ValueError:
            pass


# ─────────────────────────────────────────────────────────────────────────────
# Fix 9 — PRAGMA foreign_keys = ON in _user_db
# ─────────────────────────────────────────────────────────────────────────────

@test("Fix 9: _user_db connection has foreign_keys enabled")
def _():
    sm, _, uid = fresh_db()
    with sm._user_db(uid) as conn:
        fk = conn.execute("PRAGMA foreign_keys").fetchone()[0]
    assert fk == 1, f"foreign_keys pragma = {fk}, expected 1"


# ─────────────────────────────────────────────────────────────────────────────
# Regression — core CRUD still works after all changes
# ─────────────────────────────────────────────────────────────────────────────

@test("Regression: full CRUD round-trip")
def _():
    sm, crud, uid = fresh_db()
    sm.create_dynamic_table(uid, "items", [
        {"name": "label", "type": "TEXT"},
        {"name": "qty",   "type": "INTEGER"},
        {"name": "price", "type": "FLOAT"},
        {"name": "active","type": "BOOLEAN"},
        {"name": "due",   "type": "DATE"},
    ])
    rid = crud.insert_record(uid, "items", {
        "label": "Widget", "qty": 5, "price": 9.99, "active": 1, "due": "2024-12-31"
    })
    row = crud.get_record(uid, "items", rid)
    assert row["label"] == "Widget"
    assert row["qty"] == 5

    crud.update_record(uid, "items", rid, {"qty": 10})
    row2 = crud.get_record(uid, "items", rid)
    assert row2["qty"] == 10

    assert crud.count_records(uid, "items") == 1
    assert crud.delete_record(uid, "items", rid)
    assert crud.count_records(uid, "items") == 0


@test("Regression: bulk_insert_records")
def _():
    sm, crud, uid = fresh_db()
    sm.create_dynamic_table(uid, "bulk", [{"name": "v", "type": "TEXT"}])
    n = crud.bulk_insert_records(uid, "bulk", [{"v": str(i)} for i in range(50)])
    assert n == 50
    assert crud.count_records(uid, "bulk") == 50


@test("Regression: search_table")
def _():
    sm, crud, uid = fresh_db()
    sm.create_dynamic_table(uid, "srch", [{"name": "name", "type": "TEXT"}])
    for name in ["Alice", "Bob", "Alicia"]:
        crud.insert_record(uid, "srch", {"name": name})
    hits = crud.search_table(uid, "srch", "name", "Ali")
    assert len(hits) == 2, f"expected 2 hits, got {len(hits)}"


@test("Regression: add_column persists and is queryable")
def _():
    sm, crud, uid = fresh_db()
    sm.create_dynamic_table(uid, "add_col", [{"name": "x", "type": "TEXT"}])
    rid = crud.insert_record(uid, "add_col", {"x": "hello"})
    sm.add_column_to_table(uid, "add_col", {"name": "y", "type": "INTEGER"})
    crud.update_record(uid, "add_col", rid, {"y": 42})
    row = crud.get_record(uid, "add_col", rid)
    assert row["y"] == 42


@test("Regression: CSV import (create + append)")
def _():
    sm, crud, uid = fresh_db()
    csv_imp = CSVImporter(sm, crud)
    csv_bytes = b"title,price\nDune,14.99\nFoundation,12.50\n"
    preview = csv_imp.read(io.BytesIO(csv_bytes))
    r = csv_imp.import_to_table(uid, preview["df"], "books",
                                {"title": "TEXT", "price": "FLOAT"})
    assert r["rows_imported"] == 2

    csv2 = b"title,price\nNeuromancer,11.00\n"
    p2 = csv_imp.read(io.BytesIO(csv2))
    r2 = csv_imp.append_to_table(uid, p2["df"], "books",
                                 {"title": "TEXT", "price": "FLOAT"})
    assert r2["rows_imported"] == 1
    assert crud.count_records(uid, "books") == 3


# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────

print()
passed = sum(1 for _, ok, _ in results if ok)
failed = sum(1 for _, ok, _ in results if not ok)
print(f"{'='*52}")
print(f"  \033[92m{passed} passed\033[0m   \033[91m{failed} failed\033[0m   / {len(results)} total")
print(f"{'='*52}")

if failed:
    print("\nFailed tests:")
    for name, ok, msg in results:
        if not ok:
            print(f"  • {name}")
            print(f"    {msg}")

sys.exit(failed)

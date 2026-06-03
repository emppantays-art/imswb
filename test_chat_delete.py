"""
test_chat_delete.py — chat delete tool + wrongful-deletion safety guard.

Requires Ollama (llama3.2:3b) for the end-to-end cases; the guard/executor unit
cases are deterministic.

Run:  uv run python test_chat_delete.py
"""

import sys, tempfile
sys.path.insert(0, ".")

from database.schema_manager import SchemaManager
from database.dynamic_crud import DynamicCRUD
from ai.query_parser import ChatEngine, _delete_target_ok
from ai.dynamic_tools import execute_tool

PASS = "\033[92mPASS\033[0m"; FAIL = "\033[91mFAIL\033[0m"
results = []

def check(name, cond, detail=""):
    results.append((name, bool(cond)))
    print(f"  {PASS if cond else FAIL}  {name}" + (f"  — {detail}" if detail and not cond else ""))

def fixture():
    sm = SchemaManager(tempfile.mkdtemp()); crud = DynamicCRUD(sm)
    uid = sm.create_user("u", "p1234")
    sm.create_dynamic_table(uid, "books", [
        {"name": "title", "type": "TEXT"}, {"name": "price", "type": "FLOAT"}])
    for t, pr in [("Dune", 14.99), ("Foundation", 12.5), ("Neuromancer", 11.0)]:
        crud.insert_record(uid, "books", {"title": t, "price": pr})
    return sm, crud, uid


# ── deterministic: the safety guard ───────────────────────────────────────────
_neu = {"id": 3, "title": "Neuromancer", "price": 11.0}
_fnd = {"id": 2, "title": "Foundation", "price": 12.5}
check("guard: named match allowed", _delete_target_ok(_fnd, "Delete the book Foundation"))
check("guard: named non-match blocked", not _delete_target_ok(_neu, "Delete the book Harry Potter"))
check("guard: contextual 'delete it' allowed", _delete_target_ok(_neu, "Delete it"))
check("guard: positional 'last one' allowed", _delete_target_ok(_neu, "Delete the last one"))

# ── deterministic: the delete_data executor ───────────────────────────────────
sm, crud, uid = fixture()
res = execute_tool("delete_data", {"table_name": "books", "record_id": 2}, uid, sm, crud)
check("executor: delete by id succeeds", res.success and res.payload.get("deleted_id") == 2)
check("executor: row actually removed", "Foundation" not in [r["title"] for r in crud.query_table(uid, "books")])

res = execute_tool("delete_data", {"table_name": "books", "record_id": 999}, uid, sm, crud)
check("executor: delete missing id fails", (not res.success) and "No record" in res.payload.get("error", ""))

res = execute_tool("delete_data", {"table_name": "books"}, uid, sm, crud)
check("executor: missing record_id is retryable", (not res.success) and res.retryable)

# ── end-to-end (Ollama) ───────────────────────────────────────────────────────
sm, crud, uid = fixture()
eng = ChatEngine(sm, crud, model="llama3.2:3b", active_table="books")
reply, tools = eng.chat(uid, "Delete the book Foundation", history=[])
titles = [r["title"] for r in crud.query_table(uid, "books")]
check("e2e: 'Delete Foundation' removes it", "Foundation" not in titles,
      f"remaining={titles}")
check("e2e: delete_data tool actually fired",
      any(t.name == "delete_data" and t.success for t in tools))

sm, crud, uid = fixture()
eng = ChatEngine(sm, crud, model="llama3.2:3b", active_table="books")
before = crud.count_records(uid, "books")
reply, tools = eng.chat(uid, "Delete the book Harry Potter", history=[])
check("e2e: deleting a non-existent item deletes NOTHING",
      crud.count_records(uid, "books") == before, f"count {before}→{crud.count_records(uid,'books')}")

print()
p = sum(1 for _, ok in results if ok)
print("=" * 56)
print(f"  \033[92m{p} passed\033[0m   \033[91m{len(results)-p} failed\033[0m   / {len(results)} total")
print("=" * 56)
sys.exit(len(results) - p)

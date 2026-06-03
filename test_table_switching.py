"""
test_table_switching.py — regression for the chat "tables not switching" bug.

When a table is pinned as the chat's Active table, the model used to anchor to a
table from earlier conversation history and query THAT instead of the pinned one.
The fix overrides the model's table choice to the pin when the user didn't name a
different table in the current message.

Requires Ollama running with llama3.2:3b.
Run:  uv run python test_table_switching.py
"""

import sys, tempfile
sys.path.insert(0, ".")

from database.schema_manager import SchemaManager
from database.dynamic_crud import DynamicCRUD
from ai.query_parser import ChatEngine

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
results = []


def fixture():
    sm = SchemaManager(tempfile.mkdtemp()); crud = DynamicCRUD(sm)
    uid = sm.create_user("u", "p1234")
    sm.create_dynamic_table(uid, "books", [
        {"name": "title", "type": "TEXT"}, {"name": "price", "type": "FLOAT"}])
    for t, pr in [("Dune", 14.99), ("Foundation", 12.5), ("Neuromancer", 11.0)]:
        crud.insert_record(uid, "books", {"title": t, "price": pr})
    sm.create_dynamic_table(uid, "shop", [
        {"name": "name", "type": "TEXT"}, {"name": "price", "type": "FLOAT"}])
    for n, pr in [("Apple", 0.99), ("Orange", 0.59)]:
        crud.insert_record(uid, "shop", {"name": n, "price": pr})
    return sm, crud, uid


def _qtables(tools):
    return [t.args.get("table_name") for t in tools if t.name == "query_data"]


def check(name, cond, detail=""):
    ok = bool(cond)
    results.append((name, ok))
    print(f"  {PASS if ok else FAIL}  {name}" + (f"  — {detail}" if detail and not ok else ""))


# 1. THE BUG: pinned table must win over a different table anchored in history
sm, crud, uid = fixture()
hist = [{"role": "user", "content": "show all records"},
        {"role": "assistant", "content": "There are 3 records: Dune, Foundation, Neuromancer."}]
eng = ChatEngine(sm, crud, model="llama3.2:3b", active_table="shop")
reply, tools = eng.chat(uid, "show all records", history=hist)
qt = _qtables(tools)
check("pinned 'shop' wins over 'books' in history (generic question)",
      qt and all(t == "shop" for t in qt), f"queried {qt}")

# 2. NO REGRESSION: an explicitly named table still overrides the pin
sm, crud, uid = fixture()
eng = ChatEngine(sm, crud, model="llama3.2:3b", active_table="books")
reply, tools = eng.chat(uid, "show all records in the shop table", history=[])
qt = _qtables(tools)
check("explicitly named 'shop' overrides 'books' pin",
      qt and "shop" in qt, f"queried {qt}")

# 3. BASELINE: pin respected with no misleading history
sm, crud, uid = fixture()
eng = ChatEngine(sm, crud, model="llama3.2:3b", active_table="shop")
reply, tools = eng.chat(uid, "how many records", history=[])
qt = _qtables(tools)
check("pinned 'shop' used for a generic question (no history)",
      qt and all(t == "shop" for t in qt), f"queried {qt}")

# 4. NO PIN: model still free to infer the table from the message
sm, crud, uid = fixture()
eng = ChatEngine(sm, crud, model="llama3.2:3b", active_table="")
reply, tools = eng.chat(uid, "show all records in books", history=[])
qt = _qtables(tools)
check("no pin → model infers 'books' from the message",
      qt and "books" in qt, f"queried {qt}")

print()
passed = sum(1 for _, ok in results if ok)
print("=" * 56)
print(f"  \033[92m{passed} passed\033[0m   \033[91m{len(results) - passed} failed\033[0m   / {len(results)} total")
print("=" * 56)
sys.exit(len(results) - passed)

"""
test_billing.py — Verify DynamicBillingSystem end-to-end.

Run:  uv run python test_billing.py
      (or whichever Python has the project deps installed)
"""

import sys, tempfile
sys.path.insert(0, ".")

from database.schema_manager import SchemaManager
from database.dynamic_crud import DynamicCRUD
from billing_dynamic import DynamicBillingSystem

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
            import traceback
            results.append((name, False, str(exc)))
            print(f"  {FAIL}  {name}")
            print(f"         {exc}")
            traceback.print_exc()
        return fn
    return decorator


def fresh():
    d   = tempfile.mkdtemp()
    sm  = SchemaManager(d)
    c   = DynamicCRUD(sm)
    uid = sm.create_user("tester", "pass1234")
    b   = DynamicBillingSystem(sm, c)
    return sm, c, uid, b


# ── 1. Column detection ───────────────────────────────────────────────────────

@test("detect_columns: price/name/stock found by keyword")
def _():
    sm, crud, uid, b = fresh()
    sm.create_dynamic_table(uid, "menu", [
        {"name": "dish_name", "type": "TEXT"},
        {"name": "rental_fee", "type": "FLOAT"},
        {"name": "copies", "type": "INTEGER"},
    ])
    cols = b.detect_columns(uid, "menu")
    assert cols["name"]  == "dish_name",  cols
    assert cols["price"] == "rental_fee", cols
    assert cols["stock"] == "copies",     cols


@test("detect_columns: returns None when column missing")
def _():
    sm, crud, uid, b = fresh()
    sm.create_dynamic_table(uid, "notes", [{"name": "body", "type": "TEXT"}])
    cols = b.detect_columns(uid, "notes")
    assert cols["price"] is None
    assert cols["name"]  is None


# ── 2. Invoice creation ───────────────────────────────────────────────────────

@test("create_invoice: happy path, correct totals, invoices table created")
def _():
    sm, crud, uid, b = fresh()
    sm.create_dynamic_table(uid, "products", [
        {"name": "name",  "type": "TEXT"},
        {"name": "price", "type": "FLOAT"},
    ])
    crud.insert_record(uid, "products", {"name": "Apple",  "price": 0.99})
    crud.insert_record(uid, "products", {"name": "Orange", "price": 0.59})

    inv = b.create_invoice(uid, "products",
                           [{"item_name": "Apple", "quantity": 2},
                            {"item_name": "Orange", "quantity": 1}],
                           customer_name="Maria")

    assert inv["invoice_id"].startswith("INV-")
    assert inv["customer_name"] == "Maria"
    assert abs(inv["total"] - 2.57) < 0.01, inv["total"]
    assert len(inv["items"]) == 2

    # invoices table must exist and contain our record
    invoices = crud.query_table(uid, "invoices")
    assert len(invoices) == 1
    assert invoices[0]["invoice_id"] == inv["invoice_id"]


@test("create_invoice: stock reduced correctly")
def _():
    sm, crud, uid, b = fresh()
    sm.create_dynamic_table(uid, "shop", [
        {"name": "name",  "type": "TEXT"},
        {"name": "price", "type": "FLOAT"},
        {"name": "stock", "type": "INTEGER"},
    ])
    rid = crud.insert_record(uid, "shop", {"name": "Widget", "price": 5.00, "stock": 10})

    b.create_invoice(uid, "shop", [{"item_name": "Widget", "quantity": 3}], "Bob")

    row = crud.get_record(uid, "shop", rid)
    assert row["stock"] == 7, f"Expected 7, got {row['stock']}"


@test("create_invoice: raises on insufficient stock")
def _():
    sm, crud, uid, b = fresh()
    sm.create_dynamic_table(uid, "store", [
        {"name": "name",  "type": "TEXT"},
        {"name": "price", "type": "FLOAT"},
        {"name": "stock", "type": "INTEGER"},
    ])
    crud.insert_record(uid, "store", {"name": "Gadget", "price": 20.00, "stock": 2})

    try:
        b.create_invoice(uid, "store", [{"item_name": "Gadget", "quantity": 5}], "Alice")
        assert False, "should have raised"
    except ValueError as e:
        assert "insufficient" in str(e).lower(), str(e)


@test("create_invoice: stock NOT modified if any item fails (atomicity)")
def _():
    sm, crud, uid, b = fresh()
    sm.create_dynamic_table(uid, "atomic_test", [
        {"name": "name",  "type": "TEXT"},
        {"name": "price", "type": "FLOAT"},
        {"name": "stock", "type": "INTEGER"},
    ])
    r1 = crud.insert_record(uid, "atomic_test", {"name": "A", "price": 1.00, "stock": 5})
    crud.insert_record(uid, "atomic_test",       {"name": "B", "price": 1.00, "stock": 1})

    try:
        b.create_invoice(uid, "atomic_test",
                         [{"item_name": "A", "quantity": 2},
                          {"item_name": "B", "quantity": 3}],   # B only has 1
                         "Test")
    except ValueError:
        pass

    row = crud.get_record(uid, "atomic_test", r1)
    assert row["stock"] == 5, f"Stock of A should be unchanged (5), got {row['stock']}"


@test("create_invoice: same item twice can't oversell stock (cumulative check)")
def _():
    sm, crud, uid, b = fresh()
    sm.create_dynamic_table(uid, "shop", [
        {"name": "name",  "type": "TEXT"},
        {"name": "price", "type": "FLOAT"},
        {"name": "stock", "type": "INTEGER"},
    ])
    rid = crud.insert_record(uid, "shop", {"name": "Apple", "price": 1.0, "stock": 5})
    # 3 + 3 = 6 > 5 must fail entirely, stock untouched
    try:
        b.create_invoice(uid, "shop",
                         [{"item_name": "Apple", "quantity": 3},
                          {"item_name": "Apple", "quantity": 3}], "Bob")
        assert False, "should have rejected oversell across duplicate line items"
    except ValueError as e:
        assert "insufficient" in str(e).lower(), str(e)
    assert crud.get_record(uid, "shop", rid)["stock"] == 5, "stock must be unchanged"
    # 2 + 2 = 4 <= 5 succeeds and reduces stock once to 1
    b.create_invoice(uid, "shop",
                     [{"item_name": "Apple", "quantity": 2},
                      {"item_name": "Apple", "quantity": 2}], "Bob")
    assert crud.get_record(uid, "shop", rid)["stock"] == 1


@test("create_invoice: exact name match wins over substring (no wrong-item billing)")
def _():
    sm, crud, uid, b = fresh()
    sm.create_dynamic_table(uid, "shop", [
        {"name": "name", "type": "TEXT"}, {"name": "price", "type": "FLOAT"}])
    crud.insert_record(uid, "shop", {"name": "Apple Pie", "price": 5.0})
    crud.insert_record(uid, "shop", {"name": "Apple",     "price": 1.0})
    inv = b.create_invoice(uid, "shop", [{"item_name": "Apple", "quantity": 1}], "X")
    assert inv["items"][0]["unit_price"] == 1.0, "must bill exact 'Apple', not 'Apple Pie'"
    assert abs(inv["total"] - 1.0) < 0.001


@test("create_invoice: non-numeric stock is treated as untracked, not a crash")
def _():
    sm, crud, uid, b = fresh()
    sm.create_dynamic_table(uid, "shop", [
        {"name": "name", "type": "TEXT"}, {"name": "price", "type": "FLOAT"},
        {"name": "stock", "type": "TEXT"}])
    crud.insert_record(uid, "shop", {"name": "Apple", "price": 1.0, "stock": "plenty"})
    inv = b.create_invoice(uid, "shop", [{"item_name": "Apple", "quantity": 3}], "X")
    assert abs(inv["total"] - 3.0) < 0.001


@test("create_invoice: non-numeric price coerces to 0, not a crash")
def _():
    sm, crud, uid, b = fresh()
    sm.create_dynamic_table(uid, "shop", [
        {"name": "name", "type": "TEXT"}, {"name": "cost", "type": "TEXT"}])
    crud.insert_record(uid, "shop", {"name": "Apple", "cost": "free"})
    inv = b.create_invoice(uid, "shop", [{"item_name": "Apple", "quantity": 2}], "X")
    assert inv["total"] == 0.0


@test("create_invoice: non-integer quantity raises a clear error")
def _():
    sm, crud, uid, b = fresh()
    sm.create_dynamic_table(uid, "shop", [
        {"name": "name", "type": "TEXT"}, {"name": "price", "type": "FLOAT"}])
    crud.insert_record(uid, "shop", {"name": "Apple", "price": 1.0})
    try:
        b.create_invoice(uid, "shop", [{"item_name": "Apple", "quantity": "two"}], "X")
        assert False, "should reject non-integer quantity"
    except ValueError as e:
        assert "quantity" in str(e).lower(), str(e)


@test("create_invoice: raises on unknown item")
def _():
    sm, crud, uid, b = fresh()
    sm.create_dynamic_table(uid, "books", [
        {"name": "title", "type": "TEXT"},
        {"name": "price", "type": "FLOAT"},
    ])
    crud.insert_record(uid, "books", {"title": "Dune", "price": 14.99})

    try:
        b.create_invoice(uid, "books", [{"item_name": "Nonexistent", "quantity": 1}], "X")
        assert False, "should have raised"
    except ValueError as e:
        assert "not found" in str(e).lower(), str(e)


@test("create_invoice: raises when no price column")
def _():
    sm, crud, uid, b = fresh()
    sm.create_dynamic_table(uid, "notes", [{"name": "body", "type": "TEXT"}])
    crud.insert_record(uid, "notes", {"body": "hello"})
    try:
        b.create_invoice(uid, "notes", [{"item_name": "hello", "quantity": 1}], "X")
        assert False, "should have raised"
    except ValueError as e:
        assert "price" in str(e).lower(), str(e)


# ── 3. Daily sales ────────────────────────────────────────────────────────────

@test("get_daily_sales: correct aggregation")
def _():
    sm, crud, uid, b = fresh()
    sm.create_dynamic_table(uid, "items", [
        {"name": "name",  "type": "TEXT"},
        {"name": "price", "type": "FLOAT"},
    ])
    crud.insert_record(uid, "items", {"name": "X", "price": 10.00})
    crud.insert_record(uid, "items", {"name": "Y", "price":  5.00})

    b.create_invoice(uid, "items", [{"item_name": "X", "quantity": 2}], "C1")
    b.create_invoice(uid, "items", [{"item_name": "Y", "quantity": 1}], "C2")

    from datetime import datetime
    today   = datetime.now().strftime("%Y-%m-%d")
    summary = b.get_daily_sales(uid, today)

    assert summary["invoice_count"]  == 2,     summary
    assert abs(summary["total_revenue"] - 25.0) < 0.01, summary
    assert abs(summary["average_ticket"] - 12.5) < 0.01, summary


@test("get_daily_sales: zero on a day with no sales")
def _():
    sm, crud, uid, b = fresh()
    summary = b.get_daily_sales(uid, "2000-01-01")
    assert summary["invoice_count"] == 0
    assert summary["total_revenue"] == 0.0


# ── 4. Invoice history ────────────────────────────────────────────────────────

@test("get_invoices: returns newest-first")
def _():
    sm, crud, uid, b = fresh()
    sm.create_dynamic_table(uid, "prod", [
        {"name": "name",  "type": "TEXT"},
        {"name": "price", "type": "FLOAT"},
    ])
    crud.insert_record(uid, "prod", {"name": "A", "price": 1.00})

    b.create_invoice(uid, "prod", [{"item_name": "A", "quantity": 1}], "First")
    b.create_invoice(uid, "prod", [{"item_name": "A", "quantity": 1}], "Second")

    rows = b.get_invoices(uid, limit=2)
    assert len(rows) == 2
    assert rows[0]["customer_name"] == "Second"   # newest first


# ── 5. Receipt HTML ───────────────────────────────────────────────────────────

@test("generate_receipt_html: contains invoice ID and total")
def _():
    sm, crud, uid, b = fresh()
    sm.create_dynamic_table(uid, "p", [
        {"name": "name",  "type": "TEXT"},
        {"name": "price", "type": "FLOAT"},
    ])
    crud.insert_record(uid, "p", {"name": "Dune", "price": 14.99})

    inv  = b.create_invoice(uid, "p", [{"item_name": "Dune", "quantity": 1}], "Reader")
    html = b.generate_receipt_html(inv)

    assert inv["invoice_id"] in html
    assert "14.99" in html
    assert "Reader" in html
    assert "<table" in html.lower()


# ── 6. billable_tables ────────────────────────────────────────────────────────

@test("billable_tables: excludes tables without name+price, excludes invoices table")
def _():
    sm, crud, uid, b = fresh()
    sm.create_dynamic_table(uid, "goods",  [
        {"name": "name",  "type": "TEXT"},
        {"name": "price", "type": "FLOAT"},
    ])
    sm.create_dynamic_table(uid, "notes",  [{"name": "body", "type": "TEXT"}])

    # trigger invoices table creation
    crud.insert_record(uid, "goods", {"name": "X", "price": 1.0})
    b.create_invoice(uid, "goods", [{"item_name": "X", "quantity": 1}], "Y")

    billable = b.billable_tables(uid)
    assert "goods"   in billable
    assert "notes"   not in billable
    assert "invoices" not in billable


# ── Summary ───────────────────────────────────────────────────────────────────

print()
passed = sum(1 for _, ok, _ in results if ok)
failed = len(results) - passed
print("=" * 54)
print(f"  \033[92m{passed} passed\033[0m   \033[91m{failed} failed\033[0m   / {len(results)} total")
print("=" * 54)

if failed:
    print("\nFailed tests:")
    for name, ok, msg in results:
        if not ok:
            print(f"  • {name}\n    {msg}")

sys.exit(failed)

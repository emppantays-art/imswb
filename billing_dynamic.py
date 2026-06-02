"""
billing_dynamic.py — Auto-adapting billing / POS system for any dynamic table.

Works by keyword-matching column names to detect price / name / stock columns —
no hardcoded table or column names required.
"""

import json
import random
import string
from datetime import datetime
from typing import Any, Dict, List, Optional

from database.schema_manager import SchemaManager
from database.dynamic_crud import DynamicCRUD


def to_number(val: Any) -> float:
    """Parse a price-like value to float; 0.0 if blank/non-numeric (never raises)."""
    try:
        return float(val)
    except (ValueError, TypeError):
        return 0.0


def to_stock(val: Any) -> Optional[int]:
    """
    Parse a stock-like value to int. Returns None when the value isn't a
    trackable number (blank, text like 'plenty', etc.) so callers can treat
    that row as 'stock not tracked' instead of crashing.
    """
    if val is None:
        return None
    try:
        return int(float(val))
    except (ValueError, TypeError):
        return None


class DynamicBillingSystem:
    """
    POS / invoicing layer that sits on top of SchemaManager + DynamicCRUD.

    Column auto-detection uses keyword matching so it works with tables named
    anything — products, menu, books, rentals, etc.
    """

    PRICE_KEYWORDS  = {"price", "cost", "fee", "rate", "amount", "rental_fee", "unit_price", "sale_price"}
    NAME_KEYWORDS   = {"name", "title", "item", "product", "dish", "label", "description", "sku"}
    STOCK_KEYWORDS  = {"stock", "quantity", "inventory", "available", "copies", "qty", "units", "count"}
    INVOICES_TABLE  = "invoices"

    def __init__(self, sm: SchemaManager, crud: DynamicCRUD):
        self.sm   = sm
        self.crud = crud

    # ── column detection ─────────────────────────────────────────────────────

    def detect_columns(self, user_id: int, table_name: str) -> Dict[str, Optional[str]]:
        """
        Return {"price": col|None, "name": col|None, "stock": col|None}
        by matching column names against keyword sets.
        """
        schema = self.sm.get_table_schema(user_id, table_name) or []
        result: Dict[str, Optional[str]] = {"price": None, "name": None, "stock": None}
        for col in schema:
            low = col["column_name"].lower()
            if result["price"] is None and any(kw in low for kw in self.PRICE_KEYWORDS):
                result["price"] = col["column_name"]
            if result["name"]  is None and any(kw in low for kw in self.NAME_KEYWORDS):
                result["name"]  = col["column_name"]
            if result["stock"] is None and any(kw in low for kw in self.STOCK_KEYWORDS):
                result["stock"] = col["column_name"]
        return result

    def billable_tables(self, user_id: int) -> List[str]:
        """Return tables that have at least a name + price column (excludes invoices table)."""
        out = []
        for t in self.sm.get_user_tables(user_id):
            name = t["table_name"]
            if name == self.INVOICES_TABLE:
                continue
            cols = self.detect_columns(user_id, name)
            if cols["name"] and cols["price"]:
                out.append(name)
        return out

    # ── invoices table ───────────────────────────────────────────────────────

    def _ensure_invoices_table(self, user_id: int) -> None:
        """Create the shared invoices table if it doesn't already exist."""
        existing = {t["table_name"] for t in self.sm.get_user_tables(user_id)}
        if self.INVOICES_TABLE not in existing:
            self.sm.create_dynamic_table(user_id, self.INVOICES_TABLE, [
                {"name": "invoice_id",    "type": "TEXT",  "required": True},
                {"name": "customer_name", "type": "TEXT",  "required": True},
                {"name": "source_table",  "type": "TEXT",  "required": True},
                {"name": "items_json",    "type": "TEXT",  "required": True},
                {"name": "total",         "type": "FLOAT", "required": True},
                {"name": "date",          "type": "TEXT"},
                {"name": "status",        "type": "TEXT",  "default": "paid"},
            ])

    # ── item lookup ──────────────────────────────────────────────────────────

    def _find_item(self, user_id: int, table_name: str,
                   name_col: str, item_name: str) -> Optional[Dict]:
        """
        Resolve an item name to a single row. Prefers an exact (case-insensitive)
        name match over a substring match, so asking for 'Apple' never silently
        bills 'Apple Pie'. Returns None if nothing matches at all.
        """
        rows = self.crud.search_table(user_id, table_name, name_col, item_name, limit=25)
        if not rows:
            return None
        target = item_name.strip().lower()
        for r in rows:
            if str(r.get(name_col, "")).strip().lower() == target:
                return r
        return rows[0]   # no exact match — fall back to the first substring hit

    # ── create invoice ───────────────────────────────────────────────────────

    def create_invoice(
        self,
        user_id: int,
        table_name: str,
        items: List[Dict[str, Any]],
        customer_name: str,
    ) -> Dict:
        """
        Create an invoice / POS sale from any product table.

        Parameters
        ----------
        user_id       : owner
        table_name    : source table (products, menu, books …)
        items         : [{"item_name": str, "quantity": int}, …]
        customer_name : buyer name for the receipt

        Returns the completed invoice dict.
        Raises ValueError on missing items, bad stock, or undetectable columns.
        The stock reduction is applied atomically — if any item fails the whole
        transaction is aborted before any DB write.
        """
        cols = self.detect_columns(user_id, table_name)
        if not cols["price"]:
            raise ValueError(
                f"No price column found in '{table_name}'. "
                "Name a column with one of: price, cost, fee, rate, amount, rental_fee."
            )
        if not cols["name"]:
            raise ValueError(
                f"No name column found in '{table_name}'. "
                "Name a column with one of: name, title, item, product, dish."
            )

        line_items: List[Dict] = []
        total = 0.0
        # Track per-row state so the same item appearing in several line items is
        # checked against ONE cumulative quantity (prevents overselling), and so
        # stock is reduced once per row by the total quantity sold.
        rows_by_id: Dict[int, Dict] = {}
        qty_by_id:  Dict[int, int]  = {}

        for entry in items:
            item_name = str(entry.get("item_name", "")).strip()
            try:
                qty = int(entry.get("quantity"))
            except (ValueError, TypeError):
                raise ValueError(f"Quantity for '{item_name}' must be a positive integer.")
            if qty <= 0:
                raise ValueError(f"Quantity for '{item_name}' must be a positive integer.")

            row = self._find_item(user_id, table_name, cols["name"], item_name)
            if row is None:
                raise ValueError(f"Item '{item_name}' not found in '{table_name}'.")
            rid = row["id"]
            rows_by_id[rid] = row
            qty_by_id[rid]  = qty_by_id.get(rid, 0) + qty

            unit_price = to_number(row.get(cols["price"]))
            subtotal   = round(unit_price * qty, 2)

            # Stock check against the CUMULATIVE quantity requested for this row.
            if cols["stock"]:
                avail = to_stock(row.get(cols["stock"]))
                if avail is not None and qty_by_id[rid] > avail:
                    raise ValueError(
                        f"Insufficient stock for '{item_name}': "
                        f"requested {qty_by_id[rid]}, available {avail}."
                    )

            line_items.append({
                "item_name":  item_name,
                "quantity":   qty,
                "unit_price": unit_price,
                "subtotal":   subtotal,
            })
            total += subtotal

        # All validation passed — reduce stock once per row by total sold.
        if cols["stock"]:
            for rid, sold in qty_by_id.items():
                avail = to_stock(rows_by_id[rid].get(cols["stock"]))
                if avail is not None:
                    self.crud.update_record(
                        user_id, table_name, rid, {cols["stock"]: avail - sold}
                    )

        # Generate invoice ID: INV-YYYYMMDD-XXXX
        now        = datetime.now()
        rand_part  = "".join(random.choices(string.digits, k=4))
        invoice_id = f"INV-{now.strftime('%Y%m%d')}-{rand_part}"
        today      = now.strftime("%Y-%m-%d")

        self._ensure_invoices_table(user_id)
        self.crud.insert_record(user_id, self.INVOICES_TABLE, {
            "invoice_id":    invoice_id,
            "customer_name": customer_name,
            "source_table":  table_name,
            "items_json":    json.dumps(line_items),
            "total":         round(total, 2),
            "date":          today,
            "status":        "paid",
        })

        return {
            "invoice_id":    invoice_id,
            "customer_name": customer_name,
            "source_table":  table_name,
            "items":         line_items,
            "total":         round(total, 2),
            "date":          today,
        }

    # ── reporting ────────────────────────────────────────────────────────────

    def get_daily_sales(self, user_id: int, date: Optional[str] = None) -> Dict:
        """Return sales summary for a given date (defaults to today)."""
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")
        self._ensure_invoices_table(user_id)
        rows    = self.crud.query_table(
            user_id, self.INVOICES_TABLE, filters={"date": date}, limit=1000
        )
        count   = len(rows)
        revenue = sum(to_number(r.get("total")) for r in rows)
        avg     = round(revenue / count, 2) if count else 0.0
        return {
            "date":           date,
            "invoice_count":  count,
            "total_revenue":  round(revenue, 2),
            "average_ticket": avg,
        }

    def get_invoices(self, user_id: int, limit: int = 20) -> List[Dict]:
        """Return the most recent invoices, newest first."""
        self._ensure_invoices_table(user_id)
        return self.crud.query_table(
            user_id, self.INVOICES_TABLE,
            limit=limit, order_by="id", descending=True,
        )

    # ── receipt HTML ─────────────────────────────────────────────────────────

    def generate_receipt_html(self, invoice: Dict) -> str:
        """Render a printable HTML receipt from an invoice dict."""
        items = invoice.get("items", [])
        if isinstance(items, str):          # stored as JSON string in DB
            items = json.loads(items)

        rows_html = "".join(
            f"<tr>"
            f"<td>{it['item_name']}</td>"
            f"<td style='text-align:center'>{it['quantity']}</td>"
            f"<td style='text-align:right'>${float(it['unit_price']):.2f}</td>"
            f"<td style='text-align:right'>${float(it['subtotal']):.2f}</td>"
            f"</tr>"
            for it in items
        )
        return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<style>
  body  {{ font-family: 'Courier New', monospace; max-width: 440px;
           margin: 20px auto; background:#fff; color:#111; }}
  h2    {{ text-align:center; letter-spacing:3px;
           border-bottom:2px solid #111; padding-bottom:8px; }}
  .meta {{ font-size:.85em; margin-bottom:12px; line-height:1.6; }}
  table {{ width:100%; border-collapse:collapse; font-size:.9em; }}
  th    {{ background:#111; color:#fff; padding:5px 7px; text-align:left; }}
  td    {{ padding:4px 7px; border-bottom:1px dotted #ccc; }}
  .tot  {{ font-weight:bold; font-size:1.15em; border-top:2px solid #111; }}
  .foot {{ text-align:center; margin-top:16px; font-size:.78em; color:#666; }}
</style>
</head>
<body>
  <h2>🧾 RECEIPT</h2>
  <div class="meta">
    <strong>Invoice :</strong> {invoice['invoice_id']}<br>
    <strong>Customer:</strong> {invoice['customer_name']}<br>
    <strong>Date    :</strong> {invoice['date']}
  </div>
  <table>
    <tr><th>Item</th><th>Qty</th><th>Unit</th><th>Total</th></tr>
    {rows_html}
    <tr class="tot">
      <td colspan="3">TOTAL</td>
      <td style="text-align:right">${float(invoice['total']):.2f}</td>
    </tr>
  </table>
  <p class="foot">Thank you for your purchase!</p>
</body>
</html>"""

"""
csv_importer.py

Reads an uploaded CSV, auto-detects column types, and creates or
appends to a dynamic table.

Used by app_db.py — never raises on individual bad rows; collects
errors and returns them in the result dict so the UI can show them.
"""

import io
import re
from typing import Any, Dict, List

import pandas as pd

from database.schema_manager import SchemaManager, _safe_name
from database.dynamic_crud import DynamicCRUD

VALID_TYPES = ["TEXT", "INTEGER", "FLOAT", "BOOLEAN", "DATE", "TIMESTAMP"]

_DATE_RE = re.compile(
    r"^\d{4}[-/]\d{2}[-/]\d{2}$"          # YYYY-MM-DD  or  YYYY/MM/DD
    r"|^\d{2}[-/]\d{2}[-/]\d{4}$"          # DD-MM-YYYY  or  MM/DD/YYYY
)


# ── type detection ─────────────────────────────────────────────────────────────

def detect_type(series: pd.Series) -> str:
    """
    Infer the best column type from a pandas Series of raw strings.
    Order: BOOLEAN → INTEGER → FLOAT → DATE → TEXT
    """
    sample = series.dropna().astype(str).str.strip()
    sample = sample[sample != ""]
    if len(sample) == 0:
        return "TEXT"

    lower = sample.str.lower()

    # BOOLEAN: only true/false/yes/no/0/1
    if set(lower.unique()).issubset({"true", "false", "yes", "no", "1", "0"}):
        return "BOOLEAN"

    # INTEGER: every non-empty value parses as a whole number
    try:
        as_float = pd.to_numeric(sample, errors="raise")
        if (as_float == as_float.astype("int64")).all():
            return "INTEGER"
        return "FLOAT"
    except (ValueError, TypeError):
        pass

    # DATE: every non-empty value matches a date pattern
    if sample.str.match(_DATE_RE).all():
        return "DATE"

    return "TEXT"


def _coerce(value: Any, col_type: str) -> Any:
    """Convert a raw string to the target Python type. Returns None on failure."""
    if value is None:
        return None
    s = str(value).strip()
    if not s or s.lower() in ("nan", "none", "null", ""):
        return None
    try:
        if col_type == "INTEGER":
            return int(float(s))
        if col_type == "FLOAT":
            return float(s)
        if col_type == "BOOLEAN":
            return 1 if s.lower() in ("true", "yes", "1") else 0
    except (ValueError, TypeError):
        pass
    return s


# ── CSVImporter class ──────────────────────────────────────────────────────────

class CSVImporter:
    def __init__(self, sm: SchemaManager, crud: DynamicCRUD):
        self.sm   = sm
        self.crud = crud

    # ── parsing ───────────────────────────────────────────────────────────────

    def read(self, file_obj, max_preview: int = 5) -> Dict:
        """
        Parse a CSV file-like object and return a preview dict:

        {
          "headers":         [str, ...],
          "suggested_types": {col: type},
          "preview":         list of dicts  (first max_preview rows),
          "total_rows":      int,
          "df":              pd.DataFrame   (full data, strings only),
        }
        """
        df = self._read_df(file_obj)
        df.columns = [str(c).strip() for c in df.columns]
        df = df.dropna(how="all").reset_index(drop=True)

        # Sanitise then deduplicate column names.
        # _safe_name converts pandas-mangled "price.1" → "price_1" first;
        # the dedup loop then handles any remaining collisions.
        seen: Dict[str, int] = {}
        new_cols = []
        for idx, col in enumerate(df.columns):
            try:
                safe = _safe_name(col)
            except ValueError:
                safe = f"col_{idx}"
            if safe in seen:
                seen[safe] += 1
                new_cols.append(f"{safe}_{seen[safe]}")
            else:
                seen[safe] = 0
                new_cols.append(safe)
        df.columns = new_cols

        suggested = {col: detect_type(df[col]) for col in df.columns}
        preview   = df.head(max_preview).fillna("").to_dict(orient="records")

        return {
            "headers":         list(df.columns),
            "suggested_types": suggested,
            "preview":         preview,
            "total_rows":      len(df),
            "df":              df,
        }

    @staticmethod
    def _read_df(file_obj) -> pd.DataFrame:
        """Try multiple encodings; detect delimiter automatically."""
        raw = file_obj.read() if hasattr(file_obj, "read") else open(file_obj, "rb").read()
        for enc in ("utf-8-sig", "utf-8", "latin-1", "cp1252"):
            try:
                text = raw.decode(enc)
                # Sniff delimiter (comma or semicolon or tab)
                first_line = text.split("\n")[0]
                sep = ";"  if first_line.count(";") > first_line.count(",") else (
                      "\t" if first_line.count("\t") > first_line.count(",") else ","
                )
                return pd.read_csv(
                    io.StringIO(text), sep=sep, dtype=str,
                    on_bad_lines="skip",   # silently drop rows with wrong field count
                )
            except (UnicodeDecodeError, pd.errors.ParserError):
                continue
        raise ValueError("Could not decode the CSV file. Try saving it as UTF-8.")

    # ── helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _normalize(df: pd.DataFrame, column_types: Dict[str, str]):
        """
        Rename df columns to their _safe_name equivalents and update
        column_types keys to match. Call this before any DB operation so
        column names in the insert data match what SchemaManager stored.

        Returns (renamed_df, updated_column_types).
        """
        col_map: Dict[str, str] = {}
        for col in df.columns:
            try:
                col_map[col] = _safe_name(col)
            except ValueError:
                col_map[col] = f"col_{list(df.columns).index(col)}"

        df = df.copy()
        df.columns = [col_map[c] for c in df.columns]
        safe_types = {col_map.get(k, k): v for k, v in column_types.items()}
        return df, safe_types

    # ── import ────────────────────────────────────────────────────────────────

    def import_to_table(
        self,
        user_id: int,
        df: pd.DataFrame,
        table_name: str,
        column_types: Dict[str, str],
        description: str = "",
    ) -> Dict:
        """
        Create a new table from df and import all rows.
        Returns {"rows_imported": int, "rows_failed": int, "errors": [str]}.
        """
        # Align df column names with what SchemaManager will store in the DB.
        # Without this, "Product Name" in df never matches "Product_Name" in DB
        # and every row silently gets dropped.
        df, column_types = self._normalize(df, column_types)
        # SchemaManager.create_dynamic_table sanitizes the table name internally;
        # use the same safe name so _bulk_insert can find the table afterwards.
        table_name = _safe_name(table_name)

        columns_schema = [
            {
                "name":     col,
                "type":     column_types.get(col, "TEXT"),
                "required": False,
                "default":  None,
            }
            for col in df.columns
            if col in column_types
        ]
        self.sm.create_dynamic_table(user_id, table_name, columns_schema)
        return self._bulk_insert(user_id, df, table_name, column_types)

    def append_to_table(
        self,
        user_id: int,
        df: pd.DataFrame,
        table_name: str,
        column_types: Dict[str, str],
    ) -> Dict:
        """
        Append rows to an existing table.
        Columns in df that don't exist in the table are silently skipped.
        """
        table_name = _safe_name(table_name)
        if not self.sm.verify_user_owns_table(user_id, table_name):
            raise ValueError(f"Table '{table_name}' not found")
        df, column_types = self._normalize(df, column_types)
        return self._bulk_insert(user_id, df, table_name, column_types)

    def _bulk_insert(
        self,
        user_id: int,
        df: pd.DataFrame,
        table_name: str,
        column_types: Dict[str, str],
    ) -> Dict:
        rows_imported = 0
        rows_failed   = 0
        errors: List[str] = []

        records = []
        for _, row in df.iterrows():
            data: Dict[str, Any] = {}
            for col in df.columns:
                if col not in column_types:
                    continue
                raw = row.get(col)
                val = None if (not isinstance(raw, str) and pd.isna(raw)) else raw
                coerced = _coerce(val, column_types[col])
                if coerced is not None:
                    data[col] = coerced
            records.append(data)

        # Use bulk insert for performance (single DB connection for all rows)
        try:
            rows_imported = self.crud.bulk_insert_records(user_id, table_name, records)
        except Exception as exc:
            # Fall back to row-by-row so partial imports still work
            for data in records:
                try:
                    self.crud.insert_record(user_id, table_name, data)
                    rows_imported += 1
                except Exception as e:
                    rows_failed += 1
                    if len(errors) < 10:
                        errors.append(str(e))

        return {
            "rows_imported": rows_imported,
            "rows_failed":   rows_failed,
            "errors":        errors,
        }

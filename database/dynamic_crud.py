"""
dynamic_crud.py — CRUD operations against user-owned dynamic tables.

All methods verify ownership before touching the user's DB, so callers
cannot reach another user's data by guessing a table name.
"""

import sqlite3
from typing import Any, Dict, List, Optional

from .schema_manager import SchemaManager


class DynamicCRUD:
    def __init__(self, schema_manager: SchemaManager):
        self.sm = schema_manager

    # ── write ────────────────────────────────────────────────────────────────

    def insert_record(
        self, user_id: int, table_name: str, data: Dict[str, Any]
    ) -> int:
        """Insert a row. Returns the new row's id."""
        self._assert_access(user_id, table_name)
        schema = self.sm.get_table_schema(user_id, table_name) or []
        valid = {c["column_name"] for c in schema}
        row = {k: v for k, v in data.items() if k in valid}
        for col in schema:
            if col["is_required"] and col.get("default_value") is None and col["column_name"] not in row:
                raise ValueError(f"Column '{col['column_name']}' is required")

        conn = self.sm._open_user_db(user_id)
        try:
            if row:
                cols = list(row)
                quoted = ", ".join(f'"{c}"' for c in cols)
                placeholders = ", ".join("?" * len(cols))
                sql = f'INSERT INTO "{table_name}" ({quoted}) VALUES ({placeholders})'
                cur = conn.execute(sql, [row[c] for c in cols])
            else:
                # All columns have DB-level defaults; use DEFAULT VALUES.
                cur = conn.execute(f'INSERT INTO "{table_name}" DEFAULT VALUES')
            conn.commit()
            return cur.lastrowid
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def update_record(
        self,
        user_id: int,
        table_name: str,
        record_id: int,
        updates: Dict[str, Any],
    ) -> bool:
        """Update columns of a row by id. Returns True if a row was changed."""
        self._assert_access(user_id, table_name)
        valid = self._valid_cols(user_id, table_name)
        row = {k: v for k, v in updates.items() if k in valid}
        if not row:
            raise ValueError("No valid columns to update")

        sets = [f'"{k}" = ?' for k in row]
        sets.append('"updated_at" = datetime(\'now\')')
        sql = f'UPDATE "{table_name}" SET {", ".join(sets)} WHERE id = ?'

        conn = self.sm._open_user_db(user_id)
        try:
            cur = conn.execute(sql, [*row.values(), record_id])
            conn.commit()
            return cur.rowcount > 0
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def delete_record(self, user_id: int, table_name: str, record_id: int) -> bool:
        """Delete a row by id. Returns True if a row was removed."""
        self._assert_access(user_id, table_name)
        conn = self.sm._open_user_db(user_id)
        try:
            cur = conn.execute(
                f'DELETE FROM "{table_name}" WHERE id = ?', (record_id,)
            )
            conn.commit()
            return cur.rowcount > 0
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    # ── read ─────────────────────────────────────────────────────────────────

    def query_table(
        self,
        user_id: int,
        table_name: str,
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 500,
        offset: int = 0,
        order_by: str = "id",
        descending: bool = False,
    ) -> List[Dict]:
        """
        Return rows as a list of dicts.

        filters: { column_name: exact_value } — all conditions ANDed together.
        """
        self._assert_access(user_id, table_name)
        system_cols = {"id", "created_at", "updated_at"}
        valid = self._valid_cols(user_id, table_name) | system_cols

        params: List[Any] = []
        where = ""
        if filters:
            clauses = []
            for k, v in filters.items():
                # Skip non-primitive values (dicts, lists) — only exact-match scalars work
                if k in valid and isinstance(v, (str, int, float, bool, type(None))):
                    clauses.append(f'"{k}" = ?')
                    params.append(v)
            if clauses:
                where = " WHERE " + " AND ".join(clauses)

        sort_col = order_by if order_by in valid else "id"
        direction = "DESC" if descending else "ASC"
        sql = (
            f'SELECT * FROM "{table_name}"{where} '
            f'ORDER BY "{sort_col}" {direction} '
            f'LIMIT ? OFFSET ?'
        )
        params += [limit, offset]

        conn = self.sm._open_user_db(user_id)
        try:
            conn.row_factory = sqlite3.Row
            return [dict(r) for r in conn.execute(sql, params).fetchall()]
        finally:
            conn.close()

    def get_record(
        self, user_id: int, table_name: str, record_id: int
    ) -> Optional[Dict]:
        """Fetch a single row by id."""
        self._assert_access(user_id, table_name)
        conn = self.sm._open_user_db(user_id)
        try:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                f'SELECT * FROM "{table_name}" WHERE id = ?', (record_id,)
            ).fetchone()
            return dict(row) if row else None
        finally:
            conn.close()

    def count_records(self, user_id: int, table_name: str) -> int:
        self._assert_access(user_id, table_name)
        conn = self.sm._open_user_db(user_id)
        try:
            return conn.execute(
                f'SELECT COUNT(*) FROM "{table_name}"'
            ).fetchone()[0]
        finally:
            conn.close()

    def bulk_insert_records(
        self,
        user_id: int,
        table_name: str,
        records: List[Dict[str, Any]],
    ) -> int:
        """
        Insert many rows in a single DB transaction. Returns count inserted.
        Much faster than calling insert_record() in a loop for large CSV imports.
        """
        self._assert_access(user_id, table_name)
        valid = self._valid_cols(user_id, table_name)
        conn = self.sm._open_user_db(user_id)
        count = 0
        try:
            for data in records:
                row = {k: v for k, v in data.items() if k in valid}
                if not row:
                    continue
                cols = list(row)
                quoted = ", ".join(f'"{c}"' for c in cols)
                placeholders = ", ".join("?" * len(cols))
                conn.execute(
                    f'INSERT INTO "{table_name}" ({quoted}) VALUES ({placeholders})',
                    [row[c] for c in cols],
                )
                count += 1
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
        return count

    def search_table(
        self,
        user_id: int,
        table_name: str,
        column: str,
        value: str,
        limit: int = 500,
    ) -> List[Dict]:
        """LIKE-based substring search on a single text column."""
        self._assert_access(user_id, table_name)
        valid = self._valid_cols(user_id, table_name)
        if column not in valid:
            raise ValueError(f"Column '{column}' not found")

        sql = f'SELECT * FROM "{table_name}" WHERE "{column}" LIKE ? LIMIT ?'
        conn = self.sm._open_user_db(user_id)
        try:
            conn.row_factory = sqlite3.Row
            return [
                dict(r)
                for r in conn.execute(sql, [f"%{value}%", limit]).fetchall()
            ]
        finally:
            conn.close()

    # ── internal ─────────────────────────────────────────────────────────────

    def _assert_access(self, user_id: int, table_name: str):
        if not self.sm.verify_user_owns_table(user_id, table_name):
            raise PermissionError(
                f"Table '{table_name}' does not exist for this user"
            )

    def _valid_cols(self, user_id: int, table_name: str):
        schema = self.sm.get_table_schema(user_id, table_name) or []
        return {c["column_name"] for c in schema}

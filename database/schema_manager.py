"""
schema_manager.py — user creation, metadata tables, schema DDL
"""

import hashlib
import re
import secrets
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List, Optional

VALID_COLUMN_TYPES = ["TEXT", "INTEGER", "FLOAT", "BOOLEAN", "DATE", "TIMESTAMP"]

_SQL_TYPE = {
    "TEXT": "TEXT",
    "INTEGER": "INTEGER",
    "FLOAT": "REAL",
    "BOOLEAN": "INTEGER",  # SQLite has no native bool
    "DATE": "TEXT",
    "TIMESTAMP": "TEXT",
}

_METADATA_DDL = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS users (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    username      TEXT    UNIQUE NOT NULL,
    password_hash TEXT    NOT NULL,
    salt          TEXT    NOT NULL,
    created_at    TEXT    DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS custom_tables (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id    INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    table_name TEXT    NOT NULL,
    created_at TEXT    DEFAULT (datetime('now')),
    UNIQUE(user_id, table_name)
);

CREATE TABLE IF NOT EXISTS custom_columns (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    table_id      INTEGER NOT NULL REFERENCES custom_tables(id) ON DELETE CASCADE,
    column_name   TEXT    NOT NULL,
    column_type   TEXT    NOT NULL,
    is_required   INTEGER DEFAULT 0,
    default_value TEXT,
    position      INTEGER NOT NULL
);
"""


def _safe_name(name: str) -> str:
    """Strip everything except word chars; prepend 't_' if starts with digit."""
    s = re.sub(r"[^\w]", "_", name.strip())
    if not s:
        raise ValueError("Name cannot be empty")
    if s[0].isdigit():
        s = "t_" + s
    return s[:64]


class SchemaManager:
    def __init__(self, base_dir: str = "data"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        (self.base_dir / "users").mkdir(exist_ok=True)
        self._metadata_path = str(self.base_dir / "metadata.db")
        self._bootstrap()

    # ── connection helpers ───────────────────────────────────────────────────

    @contextmanager
    def _meta_db(self):
        conn = sqlite3.connect(self._metadata_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    @contextmanager
    def _user_db(self, user_id: int):
        path = str(self.base_dir / "users" / f"{user_id}.db")
        conn = sqlite3.connect(path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    # kept as a plain method so DynamicCRUD can call it too
    def _open_user_db(self, user_id: int) -> sqlite3.Connection:
        path = str(self.base_dir / "users" / f"{user_id}.db")
        conn = sqlite3.connect(path)
        conn.row_factory = sqlite3.Row
        return conn

    # ── bootstrap ───────────────────────────────────────────────────────────

    def _bootstrap(self):
        with self._meta_db() as conn:
            conn.executescript(_METADATA_DDL)

    # ── passwords ───────────────────────────────────────────────────────────

    @staticmethod
    def _hash(password: str, salt: str = None):
        if salt is None:
            salt = secrets.token_hex(16)
        digest = hashlib.pbkdf2_hmac(
            "sha256", password.encode(), salt.encode(), 100_000
        ).hex()
        return digest, salt

    # ── user API ────────────────────────────────────────────────────────────

    def create_user(self, username: str, password: str) -> int:
        """Create a new user. Returns the new user_id."""
        if not username.strip() or not password:
            raise ValueError("Username and password are required")
        ph, salt = self._hash(password)
        try:
            with self._meta_db() as conn:
                cur = conn.execute(
                    "INSERT INTO users (username, password_hash, salt) VALUES (?,?,?)",
                    (username.strip(), ph, salt),
                )
                user_id = cur.lastrowid
            # create the per-user DB file
            with self._user_db(user_id):
                pass
            return user_id
        except sqlite3.IntegrityError:
            raise ValueError(f"Username '{username}' is already taken")

    def authenticate_user(self, username: str, password: str) -> Optional[Dict]:
        """Return user dict on success, None on failure."""
        with self._meta_db() as conn:
            row = conn.execute(
                "SELECT * FROM users WHERE username = ?", (username,)
            ).fetchone()
        if not row:
            return None
        digest, _ = self._hash(password, row["salt"])
        if digest == row["password_hash"]:
            return {"id": row["id"], "username": row["username"]}
        return None

    def get_all_users(self) -> List[Dict]:
        with self._meta_db() as conn:
            return [
                dict(r)
                for r in conn.execute(
                    "SELECT id, username, created_at FROM users ORDER BY username"
                ).fetchall()
            ]

    # ── table schema API ────────────────────────────────────────────────────

    def create_dynamic_table(
        self, user_id: int, table_name: str, columns_schema: List[Dict]
    ) -> int:
        """
        Create a table in the user's DB and record its schema in metadata.

        columns_schema is a list of dicts:
          { name: str, type: TEXT|INTEGER|FLOAT|BOOLEAN|DATE|TIMESTAMP,
            required: bool (opt), default: str (opt) }

        Returns the new table_id.
        """
        table_name = _safe_name(table_name)
        if not columns_schema:
            raise ValueError("At least one column is required")

        sanitized = []
        for col in columns_schema:
            ctype = col["type"].upper()
            if ctype not in VALID_COLUMN_TYPES:
                raise ValueError(f"Invalid column type: {col['type']}")
            sanitized.append({
                "name": _safe_name(col["name"]),
                "type": ctype,
                "required": bool(col.get("required")),
                "default": col.get("default") or None,
            })

        # Build DDL
        col_defs = ["id INTEGER PRIMARY KEY AUTOINCREMENT"]
        for col in sanitized:
            sql_t = _SQL_TYPE[col["type"]]
            nn = " NOT NULL" if col["required"] else ""
            df = f" DEFAULT '{col['default']}'" if col["default"] is not None else ""
            col_defs.append(f'"{col["name"]}" {sql_t}{nn}{df}')
        col_defs += [
            "created_at TEXT DEFAULT (datetime('now'))",
            "updated_at TEXT DEFAULT (datetime('now'))",
        ]
        ddl = f'CREATE TABLE IF NOT EXISTS "{table_name}" ({", ".join(col_defs)})'

        try:
            with self._meta_db() as conn:
                cur = conn.execute(
                    "INSERT INTO custom_tables (user_id, table_name) VALUES (?,?)",
                    (user_id, table_name),
                )
                table_id = cur.lastrowid
                conn.executemany(
                    """INSERT INTO custom_columns
                       (table_id, column_name, column_type, is_required, default_value, position)
                       VALUES (?,?,?,?,?,?)""",
                    [
                        (table_id, c["name"], c["type"], int(c["required"]),
                         c["default"], i)
                        for i, c in enumerate(sanitized)
                    ],
                )
        except sqlite3.IntegrityError:
            raise ValueError(f"Table '{table_name}' already exists for this user")

        with self._user_db(user_id) as conn:
            conn.execute(ddl)

        return table_id

    def add_column_to_table(
        self, user_id: int, table_name: str, column_def: Dict
    ) -> None:
        """
        Add a column to an existing table.

        column_def: { name, type, required (opt), default (opt) }

        SQLite only supports ADD COLUMN (no NOT NULL without a default
        for existing rows), so is_required is stored in metadata only.
        """
        table_name = _safe_name(table_name)
        col_name = _safe_name(column_def["name"])
        col_type = column_def["type"].upper()
        if col_type not in VALID_COLUMN_TYPES:
            raise ValueError(f"Invalid column type: {col_type}")

        with self._meta_db() as conn:
            t_row = conn.execute(
                "SELECT id FROM custom_tables WHERE user_id=? AND table_name=?",
                (user_id, table_name),
            ).fetchone()
            if not t_row:
                raise ValueError(f"Table '{table_name}' not found")
            table_id = t_row["id"]

            if conn.execute(
                "SELECT 1 FROM custom_columns WHERE table_id=? AND column_name=?",
                (table_id, col_name),
            ).fetchone():
                raise ValueError(f"Column '{col_name}' already exists")

            max_pos = conn.execute(
                "SELECT COALESCE(MAX(position), -1) FROM custom_columns WHERE table_id=?",
                (table_id,),
            ).fetchone()[0]

            conn.execute(
                """INSERT INTO custom_columns
                   (table_id, column_name, column_type, is_required, default_value, position)
                   VALUES (?,?,?,?,?,?)""",
                (
                    table_id, col_name, col_type,
                    int(bool(column_def.get("required"))),
                    column_def.get("default") or None,
                    max_pos + 1,
                ),
            )

        default = column_def.get("default")
        df_clause = f" DEFAULT '{default}'" if default else ""
        with self._user_db(user_id) as conn:
            conn.execute(
                f'ALTER TABLE "{table_name}" ADD COLUMN "{col_name}" '
                f'{_SQL_TYPE[col_type]}{df_clause}'
            )

    def drop_table(self, user_id: int, table_name: str) -> None:
        table_name = _safe_name(table_name)
        with self._meta_db() as conn:
            t_row = conn.execute(
                "SELECT id FROM custom_tables WHERE user_id=? AND table_name=?",
                (user_id, table_name),
            ).fetchone()
            if not t_row:
                raise ValueError(f"Table '{table_name}' not found")
            # cascade deletes custom_columns too via FK
            conn.execute("DELETE FROM custom_tables WHERE id=?", (t_row["id"],))
        with self._user_db(user_id) as conn:
            conn.execute(f'DROP TABLE IF EXISTS "{table_name}"')

    # ── read-only schema queries ─────────────────────────────────────────────

    def get_user_tables(self, user_id: int) -> List[Dict]:
        with self._meta_db() as conn:
            return [
                dict(r)
                for r in conn.execute(
                    "SELECT id, table_name, created_at FROM custom_tables "
                    "WHERE user_id=? ORDER BY created_at DESC",
                    (user_id,),
                ).fetchall()
            ]

    def get_table_schema(self, user_id: int, table_name: str) -> Optional[List[Dict]]:
        """Returns ordered list of column dicts, or None if table doesn't exist."""
        with self._meta_db() as conn:
            t_row = conn.execute(
                "SELECT id FROM custom_tables WHERE user_id=? AND table_name=?",
                (user_id, table_name),
            ).fetchone()
            if not t_row:
                return None
            return [
                dict(r)
                for r in conn.execute(
                    "SELECT column_name, column_type, is_required, default_value, position "
                    "FROM custom_columns WHERE table_id=? ORDER BY position",
                    (t_row["id"],),
                ).fetchall()
            ]

    def verify_user_owns_table(self, user_id: int, table_name: str) -> bool:
        with self._meta_db() as conn:
            return bool(
                conn.execute(
                    "SELECT 1 FROM custom_tables WHERE user_id=? AND table_name=?",
                    (user_id, table_name),
                ).fetchone()
            )

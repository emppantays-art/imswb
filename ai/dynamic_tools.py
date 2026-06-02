"""
ai/dynamic_tools.py

Builds Ollama-compatible tool definitions dynamically from the live schema,
then executes whichever tool the LLM decides to call.

Tool descriptions embed real table/column names so the model always knows
what exists without needing a separate schema-lookup tool.
"""

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from database.schema_manager import SchemaManager, VALID_COLUMN_TYPES
from database.dynamic_crud import DynamicCRUD

# Maximum rows returned by query_data (keeps LLM context manageable)
QUERY_LIMIT = 50


# ── schema helpers ────────────────────────────────────────────────────────────

def schema_summary(user_id: int, sm: SchemaManager) -> str:
    """
    One-liner per table for the system prompt:
      books  (title:TEXT*, author:TEXT*, price:FLOAT, genre:TEXT)
    """
    tables = sm.get_user_tables(user_id)
    if not tables:
        return "  (no tables yet)"
    lines = []
    for t in tables:
        cols = sm.get_table_schema(user_id, t["table_name"]) or []
        col_parts = ", ".join(
            f"{c['column_name']}:{c['column_type']}"
            + ("*" if c["is_required"] else "")
            for c in cols
        )
        lines.append(f"  {t['table_name']}  ({col_parts})")
    return "\n".join(lines)


def _table_descriptions(user_id: int, sm: SchemaManager) -> str:
    """
    Inline description used inside tool parameter descriptions:
      'books' (columns: title, author, price); 'employees' (columns: name, role)
    """
    tables = sm.get_user_tables(user_id)
    if not tables:
        return "none yet"
    parts = []
    for t in tables:
        cols = sm.get_table_schema(user_id, t["table_name"]) or []
        col_str = ", ".join(c["column_name"] for c in cols)
        parts.append(f"'{t['table_name']}' (columns: {col_str})")
    return ";  ".join(parts)


# ── tool registry ─────────────────────────────────────────────────────────────

def build_tools(user_id: int, sm: SchemaManager) -> List[Dict]:
    """
    Return a list of Ollama tool dicts whose descriptions reflect the user's
    current schema. Call this again after create_table / add_column so the
    LLM sees the updated schema in subsequent rounds.
    """
    table_desc = _table_descriptions(user_id, sm)
    types_enum = VALID_COLUMN_TYPES          # used in enum fields
    types_str  = ", ".join(VALID_COLUMN_TYPES)

    return [
        # ── query_data ──────────────────────────────────────────────────────
        {
            "type": "function",
            "function": {
                "name": "query_data",
                "description": (
                    f"Fetch rows from a table (max {QUERY_LIMIT} rows). "
                    f"Available tables: {table_desc}. "
                    "Infer table_name from context: if the user mentions a column "
                    "name (e.g. 'price', 'author'), use the table that has that column. "
                    "Pass 'filters' as column:value pairs for exact equality. "
                    "Omit 'filters' to return all rows. "
                    "For comparisons or ranges (e.g. price < 20), fetch ALL rows "
                    "then filter in your reply."
                ),
                "parameters": {
                    "type": "object",
                    "required": ["table_name"],
                    "properties": {
                        "table_name": {
                            "type": "string",
                            "description": "Name of the table to query",
                        },
                        "filters": {
                            "type": "object",
                            "description": (
                                "Optional exact-match filters, e.g. "
                                '{"author": "Frank Herbert"} or {"active": 1}. '
                                "Omit to get all rows."
                            ),
                        },
                    },
                },
            },
        },

        # ── add_data ─────────────────────────────────────────────────────────
        {
            "type": "function",
            "function": {
                "name": "add_data",
                "description": (
                    "Insert a new record into an existing table. "
                    f"Available tables: {table_desc}. "
                    "Provide only columns that belong to the table."
                ),
                "parameters": {
                    "type": "object",
                    "required": ["table_name", "data"],
                    "properties": {
                        "table_name": {
                            "type": "string",
                            "description": "Name of the target table",
                        },
                        "data": {
                            "type": "object",
                            "description": (
                                "Column:value pairs matching the table's columns. "
                                'e.g. {"title": "Dune", "author": "Frank Herbert", '
                                '"price": 14.99}'
                            ),
                        },
                    },
                },
            },
        },

        # ── update_data ───────────────────────────────────────────────────────
        {
            "type": "function",
            "function": {
                "name": "update_data",
                "description": (
                    "Update columns of an existing record identified by its integer id. "
                    f"Available tables: {table_desc}. "
                    "ALWAYS call query_data first to find the record and get its id. "
                    "NEVER guess or assume an id — only use an id returned by a "
                    "prior query_data call in this conversation."
                ),
                "parameters": {
                    "type": "object",
                    "required": ["table_name", "record_id", "updates"],
                    "properties": {
                        "table_name": {
                            "type": "string",
                            "description": "Name of the table",
                        },
                        "record_id": {
                            "type": "integer",
                            "description": "Integer primary key (id column) of the row to update",
                        },
                        "updates": {
                            "type": "object",
                            "description": (
                                "Column:value pairs to change, e.g. "
                                '{"price": 18.99} or {"stock": 0, "active": 0}'
                            ),
                        },
                    },
                },
            },
        },

        # ── create_table ──────────────────────────────────────────────────────
        {
            "type": "function",
            "function": {
                "name": "create_table",
                "description": (
                    "Create a brand-new table with custom columns. "
                    "The id, created_at, and updated_at columns are added automatically. "
                    f"Allowed column types: {types_str}."
                ),
                "parameters": {
                    "type": "object",
                    "required": ["table_name", "columns"],
                    "properties": {
                        "table_name": {
                            "type": "string",
                            "description": "Name for the new table (letters, digits, underscores)",
                        },
                        "columns": {
                            "type": "array",
                            "description": "List of column definitions",
                            "items": {
                                "type": "object",
                                "required": ["name", "type"],
                                "properties": {
                                    "name": {
                                        "type": "string",
                                        "description": "Column name",
                                    },
                                    "type": {
                                        "type": "string",
                                        "enum": types_enum,
                                        "description": f"One of: {types_str}",
                                    },
                                    "required": {
                                        "type": "boolean",
                                        "description": "Whether this field is mandatory",
                                    },
                                    "default": {
                                        "type": "string",
                                        "description": "Optional default value",
                                    },
                                },
                            },
                        },
                    },
                },
            },
        },

        # ── add_column ────────────────────────────────────────────────────────
        {
            "type": "function",
            "function": {
                "name": "add_column",
                "description": (
                    "Add a new column to an existing table without losing data. "
                    f"Available tables: {table_desc}. "
                    f"Allowed column types: {types_str}."
                ),
                "parameters": {
                    "type": "object",
                    "required": ["table_name", "column"],
                    "properties": {
                        "table_name": {
                            "type": "string",
                            "description": "Name of the existing table",
                        },
                        "column": {
                            "type": "object",
                            "required": ["name", "type"],
                            "description": "Column definition",
                            "properties": {
                                "name": {"type": "string"},
                                "type": {
                                    "type": "string",
                                    "enum": types_enum,
                                },
                                "required": {"type": "boolean"},
                                "default": {"type": "string"},
                            },
                        },
                    },
                },
            },
        },
    ]


# ── executor ──────────────────────────────────────────────────────────────────

@dataclass
class ToolResult:
    name: str
    args: Dict[str, Any]
    success: bool
    payload: Any        # JSON-serialisable
    retryable: bool = False  # True → model should fix args and retry; False → report to user

    def content_str(self) -> str:
        """What the LLM sees in the tool-role message."""
        if not self.success:
            err = self.payload.get("error", "unknown error")

            if self.retryable:
                return (
                    f"ARGUMENT ERROR: {err}\n"
                    "Do NOT tell the user. Fix this by calling the correct tool NOW."
                )

            # Table-not-found: give the model a scripted reply to copy verbatim
            table = self.args.get("table_name", "?")
            if "does not exist for this user" in err or "not found" in err.lower():
                scripted = (
                    f"STOP → The '{table}' table doesn't exist yet. "
                    "Would you like me to create it?"
                )
                return scripted

            # Record-not-found on update: scripted to avoid "No records found" conflation
            if "No record with id=" in err:
                scripted = f"STOP → Error: {err}. The update was not applied."
                return scripted

            # Generic failure
            return (
                f"TOOL FAILED: {err}\n"
                "STOP calling tools. Reply: \"Error: " + err + "\""
            )

        # Successful query_data — lean payload only (timestamps excluded)
        if self.name == "query_data":
            return json.dumps({
                "count": self.payload["count"],
                "data":  self.payload["data"],
            }, default=str)

        # Successful update_data — include what changed so the model can confirm it
        if self.name == "update_data":
            updates = self.args.get("updates", {})
            changes = ", ".join(f"{k}={v}" for k, v in updates.items())
            return json.dumps({
                "result": "success",
                "updated_id": self.payload["updated_id"],
                "changes_applied": changes,
            }, default=str)

        return json.dumps(self.payload, default=str)


def execute_tool(
    name: str,
    args: Dict[str, Any],
    user_id: int,
    sm: SchemaManager,
    crud: DynamicCRUD,
    default_table: str = "",
) -> ToolResult:
    """
    Dispatch a single tool call and return a ToolResult.
    Never raises — errors are captured in the result payload.
    """
    # Normalize table_name — small models sometimes capitalize (e.g. "Books" vs "books")
    if isinstance(args, dict) and isinstance(args.get("table_name"), str):
        args = dict(args)
        args["table_name"] = args["table_name"].strip().lower()
    try:
        if name == "query_data":
            table = args.get("table_name") or ""
            if not table:
                if default_table:
                    table = default_table
                    args = dict(args)
                    args["table_name"] = table
                else:
                    _all_tables = sm.get_user_tables(user_id)
                    if len(_all_tables) == 1:
                        table = _all_tables[0]["table_name"]
                        args = dict(args)
                        args["table_name"] = table
                    else:
                        available = ", ".join(t["table_name"] for t in _all_tables) or "none"
                        first = available.split(",")[0].strip() if available != "none" else "table"
                        return ToolResult(name, args, False, {"error": (
                            f"table_name is required as a top-level parameter. "
                            f"Example: {{\"table_name\": \"{first}\", \"filters\": {{}}}}. "
                            f"Available tables: {available}."
                        )}, retryable=True)

            filters = args.get("filters") or None
            # Guard: filters must be a dict, not a bare string
            if filters is not None and not isinstance(filters, dict):
                filters = None
            rows    = crud.query_table(user_id, table,
                                       filters=filters, limit=QUERY_LIMIT)
            # Strip timestamp noise so the model gets a smaller, cleaner payload
            _SKIP = {"created_at", "updated_at"}
            lean  = [{k: v for k, v in r.items() if k not in _SKIP} for r in rows]
            # payload["rows"] = full rows for the UI; payload["data"] = lean for the LLM
            payload = {"count": len(rows), "data": lean, "rows": rows}
            return ToolResult(name, args, True, payload)

        if name == "add_data":
            table  = args.get("table_name") or ""
            if not table:
                if default_table:
                    table = default_table
                    args = dict(args)
                    args["table_name"] = table
                else:
                    _all_tables = sm.get_user_tables(user_id)
                    if len(_all_tables) == 1:
                        table = _all_tables[0]["table_name"]
                        args = dict(args)
                        args["table_name"] = table
            data   = args.get("data", {})
            if not isinstance(data, dict):
                cols = [c["column_name"]
                        for c in (sm.get_table_schema(user_id, table) or [])]
                example = {c: "..." for c in cols[:3]}
                return ToolResult(name, args, False, {"error": (
                    f"'data' must be a JSON object of column:value pairs, "
                    f"e.g. {json.dumps(example)}. "
                    f"Available columns: {', '.join(cols)}."
                )}, retryable=True)
            new_id = crud.insert_record(user_id, table, data)
            return ToolResult(name, args, True,
                              {"inserted_id": new_id,
                               "message": f"Inserted record id={new_id}"})

        if name == "update_data":
            table  = args.get("table_name") or ""
            if not table:
                if default_table:
                    table = default_table
                    args = dict(args)
                    args["table_name"] = table
                else:
                    _all_tables = sm.get_user_tables(user_id)
                    if len(_all_tables) == 1:
                        table = _all_tables[0]["table_name"]
                        args = dict(args)
                        args["table_name"] = table
            raw_id = args.get("record_id")
            if raw_id is None or str(raw_id).strip() == "":
                return ToolResult(name, args, False, {"error": (
                    "record_id is missing or empty. You MUST call query_data first "
                    "to locate the record and obtain its integer id. Then call "
                    "update_data with that id."
                )}, retryable=True)
            try:
                record_id = int(raw_id)
            except (ValueError, TypeError):
                return ToolResult(name, args, False, {"error": (
                    f"record_id must be an integer, got {raw_id!r}. "
                    "Search for the record first using query_data, then pass its numeric id here."
                )}, retryable=True)
            updates   = args.get("updates", {})
            if not isinstance(updates, dict):
                cols = [c["column_name"]
                        for c in (sm.get_table_schema(user_id, table) or [])]
                return ToolResult(name, args, False, {"error": (
                    f"'updates' must be a JSON object of column:value pairs, "
                    f"e.g. {{\"quantity\": 5}}. "
                    f"Available columns: {', '.join(cols)}."
                )}, retryable=True)
            try:
                changed = crud.update_record(user_id, table, record_id, updates)
            except ValueError as exc:
                err_str = str(exc)
                if "No valid columns" in err_str:
                    valid_cols = ", ".join(sorted(
                        c["column_name"]
                        for c in (sm.get_table_schema(user_id, table) or [])
                    ))
                    return ToolResult(name, args, False, {"error": (
                        f"{err_str}. Valid column names for '{table}': {valid_cols}. "
                        "Use one of those exact names in your updates dict."
                    )}, retryable=True)
                raise
            if changed:
                return ToolResult(name, args, True,
                                  {"updated_id": record_id,
                                   "message": f"Record {record_id} updated"})
            return ToolResult(name, args, False,
                              {"error": f"No record with id={record_id}"})

        if name == "create_table":
            table_name = args["table_name"]
            raw_cols   = args["columns"]
            cols = [
                {
                    "name":     c["name"],
                    "type":     c.get("type", "TEXT").upper(),
                    "required": bool(c.get("required", False)),
                    "default":  c.get("default"),
                }
                for c in raw_cols
            ]
            sm.create_dynamic_table(user_id, table_name, cols)
            col_summary = ", ".join(f"{c['name']} ({c['type']})" for c in cols)
            return ToolResult(name, args, True,
                              {"message": f"Table '{table_name}' created",
                               "columns": col_summary})

        if name == "add_column":
            table_name = args["table_name"]
            col = args["column"]
            col_def = {
                "name":     col["name"],
                "type":     col.get("type", "TEXT").upper(),
                "required": bool(col.get("required", False)),
                "default":  col.get("default"),
            }
            sm.add_column_to_table(user_id, table_name, col_def)
            return ToolResult(name, args, True,
                              {"message": (
                                  f"Column '{col['name']}' ({col_def['type']}) "
                                  f"added to '{table_name}'"
                              )})

        return ToolResult(name, args, False, {"error": f"Unknown tool: {name}"})

    except Exception as exc:
        return ToolResult(name, args, False, {"error": str(exc)})

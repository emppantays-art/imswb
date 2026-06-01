"""
ai/dynamic_tools.py

Builds Ollama-compatible tool definitions dynamically from the live schema,
then executes whichever tool the LLM decides to call.

Tool descriptions embed real table/column names so the model always knows
what exists without needing a separate schema-lookup tool.
"""

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

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
                    "Pass 'filters' as column:value pairs for exact equality. "
                    "Omit 'filters' to return all rows. "
                    "For comparisons or ranges (e.g. price < 20), fetch all rows "
                    "and note which ones satisfy the condition in your reply."
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
                    "IMPORTANT: if you do not know the record's id, call query_data "
                    "first to retrieve it, then call update_data."
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
    payload: Any   # JSON-serialisable

    def content_str(self) -> str:
        """What the LLM sees in the tool-role message."""
        if not self.success:
            err = self.payload.get("error", "unknown error")
            # Plain English — small models misread JSON-formatted errors as query results
            return (
                f"TOOL FAILED: {err}\n"
                "Tell the user about this error word-for-word. "
                "Do NOT say 'no records found'. Do NOT pretend the operation succeeded."
            )
        # For query_data, send only the lean payload (no timestamps)
        if self.name == "query_data":
            return json.dumps({
                "count": self.payload["count"],
                "data":  self.payload["data"],
            }, default=str)
        return json.dumps(self.payload, default=str)


def execute_tool(
    name: str,
    args: Dict[str, Any],
    user_id: int,
    sm: SchemaManager,
    crud: DynamicCRUD,
) -> ToolResult:
    """
    Dispatch a single tool call and return a ToolResult.
    Never raises — errors are captured in the result payload.
    """
    try:
        if name == "query_data":
            table   = args["table_name"]
            filters = args.get("filters") or None
            rows    = crud.query_table(user_id, table,
                                       filters=filters, limit=QUERY_LIMIT)
            # Strip timestamp noise so the model gets a smaller, cleaner payload
            _SKIP = {"created_at", "updated_at"}
            lean  = [{k: v for k, v in r.items() if k not in _SKIP} for r in rows]
            # payload["rows"] = full rows for the UI; payload["data"] = lean for the LLM
            payload = {"count": len(rows), "data": lean, "rows": rows}
            return ToolResult(name, args, True, payload)

        if name == "add_data":
            table  = args["table_name"]
            data   = args["data"]
            new_id = crud.insert_record(user_id, table, data)
            return ToolResult(name, args, True,
                              {"inserted_id": new_id,
                               "message": f"Inserted record id={new_id}"})

        if name == "update_data":
            table     = args["table_name"]
            record_id = int(args["record_id"])
            updates   = args["updates"]
            changed   = crud.update_record(user_id, table, record_id, updates)
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

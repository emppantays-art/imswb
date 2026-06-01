"""
ai/query_parser.py

ChatEngine: stateless class that wraps the Ollama tool-calling loop.

Flow per turn:
  1. Build system prompt with live schema
  2. Build tools from live schema
  3. Send to Ollama; if no tool_calls → return text
  4. Execute each tool call, append tool-role messages
  5. Rebuild tools (schema may have changed), repeat
  6. After MAX_ROUNDS, force a plain-text summary
"""

import json
import re
from typing import Dict, List, Optional, Tuple

import ollama

from database.schema_manager import SchemaManager
from database.dynamic_crud import DynamicCRUD
from .dynamic_tools import build_tools, execute_tool, schema_summary, ToolResult

DEFAULT_MODEL = "llama3.2:3b"
MAX_TOOL_ROUNDS = 8   # caps multi-step chains (e.g. query-then-update)
MEMORY_TURNS = 5      # user+assistant pairs kept in context


def _system_prompt(user_id: int, sm: SchemaManager) -> str:
    schema = schema_summary(user_id, sm)
    table_names = [t["table_name"] for t in sm.get_user_tables(user_id)]
    table_list  = ", ".join(table_names) if table_names else "none"
    return f"""\
You are a database assistant. Answer questions about the user's data using tools.

Current schema (* = required column):
{schema}

Tables: {table_list}

Rules:
1. Use a tool for every database read or write. Never describe without doing.
2. UPDATE: if you don't know the record id, call query_data first, then update_data.
3. After tool calls, reply in 1-2 sentences. Show rows as a numbered list.
4. MISSING TABLE: if the user asks about a table that does not exist, say \
"That table doesn't exist yet. Would you like me to create it?"
5. TOOL ERROR: if a tool result starts with "TOOL FAILED:", tell the user \
the exact error message. NEVER say "No records found" for a failed insert/update. \
NEVER pretend the operation succeeded.
6. EMPTY RESULT: if query_data returns count 0, say "No records found." \
Do NOT make up example rows. Only say this after a successful query_data call.
7. UNKNOWN QUESTION: if the question has no connection to any table or column \
in the schema, reply: "I can only query your database tables: {table_list}."

Example — if schema has table PC with columns (component, brand, quantity):
  User: "how many keyboards do I have?"
  → call query_data(table_name="PC", filters={{"component": "Keyboard"}})
  → reply: "You have 2 keyboards (id=1, brand=Logitech, quantity=2)."
Do NOT treat column values like "Keyboard" or "RAM" as hardware questions."""


class ChatEngine:
    def __init__(
        self,
        sm: SchemaManager,
        crud: DynamicCRUD,
        model: str = DEFAULT_MODEL,
    ):
        self.sm    = sm
        self.crud  = crud
        self.model = model

    # ── health ────────────────────────────────────────────────────────────────

    def check_ollama(self) -> Tuple[bool, str]:
        """Return (ok, message). Safe to call without crashing on failure."""
        try:
            result = ollama.list()
            names  = [m.model or "" for m in result.models]
            base   = self.model.split(":")[0]
            if not any(base in (n or "") for n in names):
                avail = ", ".join(filter(None, names)) or "none"
                return False, (
                    f"Model '{self.model}' not pulled yet. "
                    f"Available: {avail}.\n"
                    f"Fix: `ollama pull {self.model}`"
                )
            return True, f"Ollama connected  ·  model: {self.model}"
        except Exception as exc:
            return False, (
                f"Ollama unreachable ({exc}).\n"
                "Start it with: `ollama serve`"
            )

    # ── main entry ────────────────────────────────────────────────────────────

    def chat(
        self,
        user_id: int,
        user_message: str,
        history: List[Dict],
    ) -> Tuple[str, List[ToolResult]]:
        """
        Process one user turn.

        history: trimmed list of {"role": "user"|"assistant", "content": str}
                 from the last MEMORY_TURNS turns (built by the caller).

        Returns (assistant_reply_text, list_of_ToolResult).
        """
        system    = _system_prompt(user_id, self.sm)
        tools     = build_tools(user_id, self.sm)
        all_tools: List[ToolResult] = []

        # Full message list for this request
        messages: List[Dict] = [{"role": "system", "content": system}]
        messages.extend(history)
        messages.append({"role": "user", "content": user_message})

        for _round in range(MAX_TOOL_ROUNDS):
            response = ollama.chat(
                model=self.model,
                messages=messages,
                tools=tools,
            )
            msg = response.message

            # ── no tool calls → final answer ──────────────────────────────
            if not msg.tool_calls:
                return _clean_reply(msg.content or "(no response)"), all_tools

            # ── append the assistant's tool-requesting message ────────────
            messages.append({
                "role": "assistant",
                "content": msg.content or "",
                "tool_calls": msg.tool_calls,
            })

            # ── execute each requested tool ───────────────────────────────
            for tc in msg.tool_calls:
                name = tc.function.name
                args = tc.function.arguments
                # Guard: some builds return JSON string instead of dict
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except (json.JSONDecodeError, TypeError):
                        args = {}

                result = execute_tool(name, args, user_id, self.sm, self.crud)
                all_tools.append(result)

                messages.append({
                    "role": "tool",
                    "content": result.content_str(),
                })

            # Rebuild: schema may have changed if create_table / add_column ran
            tools = build_tools(user_id, self.sm)

        # ── exceeded max rounds ───────────────────────────────────────────────
        messages.append({
            "role": "user",
            "content": "Please give a brief summary of what was just done.",
        })
        response = ollama.chat(model=self.model, messages=messages)
        return _clean_reply(response.message.content or ""), all_tools


def _clean_reply(text: str) -> str:
    """Strip leaked tool-call syntax that small models sometimes append."""
    text = re.sub(r'\{function\s+\w+[^}]*\}',  '', text, flags=re.DOTALL)
    text = re.sub(r'\[TOOL_CALL\]\s*\{.*?\}',   '', text, flags=re.DOTALL)
    text = re.sub(r'<tool_call>.*?</tool_call>', '', text, flags=re.DOTALL)
    return text.strip()


def trim_history(
    history: List[Dict],
    max_turns: int = MEMORY_TURNS,
) -> List[Dict]:
    """
    Keep only the last `max_turns` user+assistant pairs.
    Preserves the alternating user/assistant structure.
    """
    # Filter to only user and assistant messages (no tool/system noise)
    clean = [m for m in history if m["role"] in ("user", "assistant")]
    # Each turn = 2 messages; keep the last max_turns turns
    cutoff = max_turns * 2
    return clean[-cutoff:] if len(clean) > cutoff else clean

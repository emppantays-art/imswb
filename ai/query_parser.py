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
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

import ollama

from database.schema_manager import SchemaManager
from database.dynamic_crud import DynamicCRUD
from .dynamic_tools import build_tools, execute_tool, schema_summary, ToolResult

if TYPE_CHECKING:
    from .rag_engine import RAGEngine

DEFAULT_MODEL = "llama3.2:3b"
MAX_TOOL_ROUNDS = 8   # caps multi-step chains (e.g. query-then-update)
MEMORY_TURNS = 5      # user+assistant pairs kept in context
KEEP_ALIVE = "30m"    # keep the model resident in Ollama between turns (avoids
                      # multi-second cold reloads when the user keeps chatting)

# Generation options for the tool-calling loop.
#   num_ctx     — the system prompt + tool defs alone are ~2200 tokens, which
#                 overflows Ollama's 2048 default and silently truncates the
#                 rules. 8192 gives ample headroom (the model supports 131k).
#   temperature — low for reliable, repeatable tool selection (this is a
#                 deterministic DB assistant, not creative writing).
#   num_predict — cap runaway generations (a stuck model can otherwise emit
#                 hundreds of filler tokens); normal replies are well under this.
TOOL_LOOP_OPTIONS = {"num_ctx": 8192, "temperature": 0.1, "num_predict": 1024}


def _system_prompt(user_id: int, sm: SchemaManager, rag_context: str = "",
                   active_table: str = "", billing=None) -> str:
    schema      = schema_summary(user_id, sm)
    table_names = [t["table_name"] for t in sm.get_user_tables(user_id)]
    table_list  = ", ".join(table_names) if table_names else "none"

    # Column→table map so the model can infer table_name from field names in the query
    col_hints = []
    for t in sm.get_user_tables(user_id):
        cols = sm.get_table_schema(user_id, t["table_name"]) or []
        col_hints.append(
            f"  {t['table_name']}: " + ", ".join(c["column_name"] for c in cols)
        )
    col_map = "\n".join(col_hints) if col_hints else "  (none)"

    rag_section = ""
    if rag_context:
        rag_section = (
            "\nRelevant rows (semantic search — may be partial; "
            "verify with query_data before any write):\n"
            f"{rag_context}\n"
        )

    active_section = (
        f"\nACTIVE TABLE: '{active_table}' — always use table_name='{active_table}' "
        "unless the user explicitly names a different table.\n"
        if active_table else ""
    )

    billing_section = ""
    if billing is not None:
        billable = billing.billable_tables(user_id)
        if billable:
            billing_section = (
                "\nBILLING TOOLS AVAILABLE: create_invoice, get_daily_sales, view_invoices.\n"
                f"Billable tables (name + price columns detected): {', '.join(billable)}.\n"
                "Use create_invoice to sell items. Use get_daily_sales for revenue summary. "
                "Use view_invoices to list past sales.\n"
            )

    return f"""\
You are a database assistant. You may ONLY use data from tool results. Never invent values.
{active_section}
Schema (* = required):
{schema}

Columns by table (use to choose table_name):
{col_map}

Valid tables: {table_list}
{rag_section}{billing_section}
Rules:
1. TABLE CHECK — run before calling any tool:
   Valid table NAMES (not values) are: {table_list}
   If the user requests a table whose NAME is not in that list, tell them it doesn't
   exist and offer to create it. Do NOT call any tool.
   Do NOT use a different existing table as a substitute.
   NOTE: values inside rows (book titles, item names, product names, author names, etc.)
   are NOT table names. E.g. "Oak Lumber Planks" is an Item_Name filter value, not a table.
   Use filters={{"Item_Name": "Oak Lumber Planks"}} against the correct table.
   CRITICAL: If query_data already returned rows from a table this turn, that table
   EXISTS — never claim it is missing after a successful query.
2. READS: Use query_data for every question about records or values.
   Infer table_name from the column names above. When uncertain, call query_data.
   Always report actual field values from results, not descriptions of table structure.
3. RANGES: For comparisons or ranges (e.g., price < 20, rating > 4), call query_data
   with NO filter to get all rows, then list only the rows that match the condition.
   NEVER say "No records found" when query_data returned rows — examine them.
4. EMPTY: Say EXACTLY "No records found." ONLY when query_data returns count=0.
   If count > 0: examine the rows, count ONLY matches present in those results
   (not from training knowledge), and state how many match (possibly 0).
   NEVER say "No records found" followed by any qualifier or condition.
   NEVER say "No records found" when count > 0. Never make up rows.
5. UPDATE: Call query_data first to get the record id, then call update_data with that id.
   Skip the query only when the user explicitly gives an integer id (e.g. "id=5").
5b. DELETE: To remove a record, call query_data first to get its integer id, then call
   delete_data with that id. NEVER claim a record was deleted without calling delete_data —
   there is a real delete_data tool, so use it. Do not invent a deletion.
6. STOP → [text]: copy [text] word-for-word. Do not call any more tools.
7. TOOL FAILED: [msg]: reply "Error: [msg]". Do not call any more tools.
   Never say "No records found" for a failed insert or update.
8. OFF-TOPIC: Only if the question is clearly unrelated to any database table
   (pure geography, math, weather, etc.) → briefly explain you are a database
   assistant and mention the available tables. Do NOT apply this rule to questions
   about data values, items, or records — always call query_data when in doubt."""


# Command/filler words that aren't the name of a record the user wants to delete.
_DELETE_STOPWORDS = {
    "delete", "remove", "drop", "erase", "the", "a", "an", "book", "record",
    "row", "item", "entry", "data", "from", "table", "please", "this", "that",
    "it", "its", "one", "ones", "last", "first", "newest", "oldest", "my", "of",
    "in", "with", "and", "id", "number", "all", "for", "to", "by",
}


def _delete_target_ok(record: Dict, user_message: str) -> bool:
    """
    Safety check before deleting `record`: return True only if it's a sound
    target for this request. Small models, when they can't find the named item,
    will happily delete an arbitrary row — so we verify the user's named entity
    actually appears in the record's values. Purely contextual/positional
    requests ("delete it", "the last one", "id 5") carry no entity tokens and
    are allowed through.
    """
    toks = [w for w in re.findall(r"[a-z0-9]+", user_message.lower())
            if len(w) >= 3 and w not in _DELETE_STOPWORDS]
    if not toks:
        return True
    values = " ".join(
        str(v).lower() for k, v in record.items()
        if k not in ("id", "created_at", "updated_at") and v is not None
    )
    return any(t in values for t in toks)


class ChatEngine:
    def __init__(
        self,
        sm: SchemaManager,
        crud: DynamicCRUD,
        model: str = DEFAULT_MODEL,
        rag: Optional["RAGEngine"] = None,
        active_table: str = "",
        billing=None,
    ):
        self.sm           = sm
        self.crud         = crud
        self.model        = model
        self.rag          = rag
        self.active_table = active_table
        self.billing      = billing

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
        # Retrieve RAG context before the first LLM call
        rag_context = ""
        if self.rag:
            snippets = self.rag.retrieve(user_id, user_message)
            if snippets:
                rag_context = "\n".join(f"  • {s}" for s in snippets)

        active_table = self.active_table
        system    = _system_prompt(user_id, self.sm, rag_context=rag_context,
                                   active_table=active_table, billing=self.billing)
        tools     = build_tools(user_id, self.sm, billing=self.billing)
        all_tools: List[ToolResult] = []
        _update_nudge_sent = False
        _in_update_flow = False   # True after retryable update/delete failure → suppress post-query nudge
        _pending_op = "update"    # which write the query-first flow is leading to: "update" or "delete"
        _schema_dirty = False     # set when create_table/add_column runs → rebuild tools only then

        # Pre-flight: detect explicit nonexistent-table mention in the user message
        # e.g. "records in the employees table" or "Ghost to the readers table"
        user_tables = {t["table_name"] for t in self.sm.get_user_tables(user_id)}
        _msg_lower = user_message.lower()

        # Greeting shortcut — avoid tool calls for social messages
        _GREETINGS = {"hi", "hello", "hey", "howdy", "greetings", "good morning",
                      "good afternoon", "good evening"}
        if user_message.strip().lower().rstrip("!.,?") in _GREETINGS:
            _tlist = ", ".join(sorted(user_tables)) or "none yet"
            return (
                f"Hi! I can help you query and manage your database. "
                f"Your tables: {_tlist}. What would you like to know?"
            ), []

        # Off-topic patterns that can be intercepted before the LLM sees them.
        # These are clear geography/trivia questions with no possible database relevance.
        _OFFTOPIC_TRIGGERS = [
            "capital of", "population of", "president of",
            "what country", "what continent", "history of the world",
        ]
        if any(t in _msg_lower for t in _OFFTOPIC_TRIGGERS):
            _table_list = ", ".join(sorted(user_tables))
            return _clean_reply(
                f"I can only help with your database tables: {_table_list}."
            ), []

        # Explicit nonexistent-table mention: "records in the X table" / "to the Y table"
        _tbl_match = re.search(
            r'\b(?:in|from|to|into)\s+(?:the\s+)?(\w+)\s+table\b',
            user_message, re.IGNORECASE
        )
        if _tbl_match:
            _mentioned = _tbl_match.group(1).lower()
            if _mentioned not in user_tables and _mentioned not in {
                "my", "a", "an", "this", "that", "your", "our"
            }:
                _reply = (
                    f"The '{_mentioned}' table doesn't exist yet. "
                    "Would you like me to create it?"
                )
                return _clean_reply(_reply), []

        # "Add/create/insert a [new] X" where X implies a nonexistent domain entity
        _GENERIC_NOUNS = {
            "record", "row", "entry", "item", "data", "value",
            "table", "column", "field", "new",
        }
        _add_match = re.search(
            r'\b(?:add|create|insert)\s+(?:a|an)\s+(?:new\s+)?(\w+)\b',
            user_message, re.IGNORECASE
        )
        if _add_match:
            _noun = _add_match.group(1).lower()
            if _noun not in _GENERIC_NOUNS:
                _candidates = {_noun, _noun + "s", _noun + "es"}
                if not (_candidates & user_tables):
                    _reply = (
                        f"The '{_noun}' table doesn't exist yet. "
                        "Would you like me to create it?"
                    )
                    return _clean_reply(_reply), []

        # Context-dependent follow-up: "What X is it/that/this?"
        # If the last assistant message already contains field:value, answer directly.
        if history:
            _last_asst = next(
                (m["content"] for m in reversed(history) if m["role"] == "assistant"),
                ""
            )
            if _last_asst:
                _fu_match = re.search(
                    r'\bwhat\s+(\w+)\s+is\s+(?:it|that|this)\b',
                    user_message.strip(), re.IGNORECASE
                )
                if _fu_match:
                    _field = _fu_match.group(1).lower()
                    _fp = re.search(
                        rf'\b{re.escape(_field)}\s*[:\-–]\s*([^\s,;.]+)',
                        _last_asst, re.IGNORECASE
                    )
                    if _fp:
                        _val = _fp.group(1).rstrip(".,;)")
                        return _clean_reply(f"The {_field} is {_val}."), []

        # Full message list for this request
        messages: List[Dict] = [{"role": "system", "content": system}]
        messages.extend(history)
        messages.append({"role": "user", "content": user_message})

        for _round in range(MAX_TOOL_ROUNDS):
            response = ollama.chat(
                model=self.model,
                messages=messages,
                tools=tools,
                keep_alive=KEEP_ALIVE,
                options=TOOL_LOOP_OPTIONS,
            )
            msg = response.message

            # ── no tool calls → final answer ──────────────────────────────
            if not msg.tool_calls:
                content = _clean_reply(msg.content or "")
                if content:
                    return content, all_tools
                # Empty response — nudge the model and let the loop retry
                messages.append({"role": "user", "content": "Please answer the question or look up the data needed."})
                continue

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

                # Safety guard: never delete a record that doesn't match what the
                # user named. Small models pick an arbitrary id when they can't
                # find the target — this prevents wrongful deletions.
                if name == "delete_data" and isinstance(args, dict) and args.get("record_id") is not None:
                    _tbl = (args.get("table_name") or active_table or "").strip().lower()
                    try:
                        _target = self.crud.get_record(user_id, _tbl, int(args["record_id"]))
                    except Exception:
                        _target = None
                    if _target and not _delete_target_ok(_target, user_message):
                        return ("I couldn't find a record matching that request, "
                                "so nothing was deleted."), all_tools

                result = execute_tool(name, args, user_id, self.sm, self.crud,
                                     default_table=active_table, billing=self.billing)
                all_tools.append(result)
                if name in ("create_table", "add_column") and result.success:
                    _schema_dirty = True

                messages.append({
                    "role": "tool",
                    "content": result.content_str(),
                })

                # Non-retryable update failure (e.g. record not found) — stop the
                # loop immediately so the model cannot substitute a different record.
                if name in ("update_data", "delete_data") and not result.success and not result.retryable:
                    _err = result.payload.get("error", "Operation failed")
                    return f"Error: {_err}", all_tools

                # Return immediately after a successful delete — a real tool ran,
                # so confirm deterministically instead of letting the model
                # hallucinate (it used to claim deletions with no delete tool).
                if name == "delete_data" and result.success:
                    _rec = result.payload.get("deleted_id", "")
                    _tbl = result.args.get("table_name", "the table")
                    return f"Deleted record {_rec} from {_tbl}.", all_tools

                # Return immediately after successful update_data — prevents
                # post-update query loops even when model batches tool calls.
                if name == "update_data" and result.success and not _update_nudge_sent:
                    _update_nudge_sent = True
                    _upd = result.args.get("updates", {})
                    _changes = ", ".join(f"{k} → {v}" for k, v in _upd.items())
                    _rec = result.payload.get("updated_id", "")
                    _tbl = result.args.get("table_name", "record")
                    _simple_upd = [
                        {"role": "system", "content": "You are a database assistant. Reply in one sentence."},
                        {"role": "user", "content": (
                            f"A database record was updated. "
                            f"Table: {_tbl}, Record ID: {_rec}, Changes: {_changes}. "
                            "Confirm to the user what was changed."
                        )},
                    ]
                    _confirm = ollama.chat(model=self.model, messages=_simple_upd,
                                           keep_alive=KEEP_ALIVE)
                    _text = _clean_reply(_confirm.message.content or "")
                    if not _text:
                        _text = f"Updated {_tbl} record {_rec}: {_changes}."
                    return _text, all_tools

            last = all_tools[-1] if all_tools else None

            # When the model batches multiple queries in one turn (e.g. one with a
            # wrong equality filter returning 0 + one unfiltered returning rows),
            # prefer the result that actually has data so the wrong empty result
            # doesn't trigger the zero-result handler incorrectly.
            if not _in_update_flow:
                for _r in reversed(all_tools):
                    if _r.name == "query_data" and _r.success and _r.payload.get("count", 0) > 0:
                        last = _r
                        break

            # Auto-unfilter: if every query this turn used filters and still returned
            # 0 rows, the model likely applied an equality filter to a range question
            # (e.g. price=13 instead of fetching all and filtering in text).
            # Retry once without filters so the secondary LLM gets full data.
            if (
                not _in_update_flow
                and last
                and last.name == "query_data"
                and last.success
                and last.payload.get("count", 0) == 0
                and last.args.get("filters")
            ):
                _tbl = last.args.get("table_name", "")
                if _tbl:
                    _retry = execute_tool(
                        "query_data", {"table_name": _tbl},
                        user_id, self.sm, self.crud, default_table=active_table,
                    )
                    all_tools.append(_retry)
                    messages.append({"role": "tool", "content": _retry.content_str()})
                    if _retry.success and _retry.payload.get("count", 0) > 0:
                        last = _retry

            # Nudge when delete_data fails retryably (usually a missing id).
            if last and last.name == "delete_data" and last.retryable:
                table = last.args.get("table_name", "the table")
                _in_update_flow = True
                _pending_op = "delete"
                messages.append({
                    "role": "user",
                    "content": (
                        f"You need to find the record first. "
                        f"Call query_data on '{table}' to get the row and its integer id, "
                        "then call delete_data with that id."
                    ),
                })

            # Nudge when update_data fails retryably
            elif last and last.name == "update_data" and last.retryable:
                table = last.args.get("table_name", "the table")
                if _round == 0:
                    _in_update_flow = True
                    _pending_op = "update"
                    _cols0 = sorted(
                        c["column_name"]
                        for c in (self.sm.get_table_schema(user_id, table) or [])
                    )
                    messages.append({
                        "role": "user",
                        "content": (
                            f"You need to find the record first. "
                            f"Call query_data on '{table}' to get the row and its id. "
                            f"Valid column names for 'updates': {', '.join(_cols0)}. "
                            "Then call update_data with the integer id and a JSON object "
                            f"using exactly those names, e.g. {{\"price\": 15.99}}."
                        ),
                    })
                else:
                    # Subsequent retryable failure: correct column names
                    _valid_cols = sorted(
                        c["column_name"]
                        for c in (self.sm.get_table_schema(user_id, table) or [])
                    )
                    messages.append({
                        "role": "user",
                        "content": (
                            f"The update failed. Valid column names for '{table}': "
                            f"{', '.join(_valid_cols)}. "
                            "Use EXACTLY those names in 'updates', with the integer id "
                            "from your earlier query_data result."
                        ),
                    })

            # In update flow: after query_data succeeds, push model to call update_data
            if (
                last
                and last.name == "query_data"
                and last.success
                and last.payload.get("count", 0) > 0
                and _in_update_flow
            ):
                _d = last.payload.get("data", [])
                if _d:
                    # If the user message contains an explicit numeric id (e.g. "id=999")
                    # and that id is not among the query results, return an error immediately
                    # instead of substituting a different record.
                    _explicit = re.search(r'\bid[=:\s]+(\d+)', user_message, re.IGNORECASE)
                    if _explicit:
                        _asked_id = int(_explicit.group(1))
                        _found_ids = {r.get("id") for r in _d}
                        if _asked_id not in _found_ids:
                            return (
                                f"No record with id={_asked_id} found in '{last.args.get('table_name', 'the table')}'.",
                                all_tools,
                            )
                    # Show all available ids so the model picks the right one
                    _rows_desc = "; ".join(
                        f"id={r['id']}: " + ", ".join(
                            f"{k}={v}" for k, v in r.items()
                            if k not in ("id", "created_at", "updated_at")
                        )
                        for r in _d[:10]
                    )
                    _tool = "delete_data" if _pending_op == "delete" else "update_data"
                    messages.append({
                        "role": "user",
                        "content": (
                            f"Found {len(_d)} record(s). Based on the user's request, "
                            f"identify the correct one and call {_tool} with its integer id.\n"
                            f"Available records: {_rows_desc}"
                        ),
                    })

            # Post-query force-return: on the first successful query with data,
            # immediately force a text answer using all returned rows.
            elif (
                last
                and last.name == "query_data"
                and last.success
                and last.payload.get("count", 0) > 0
                and not _in_update_flow
            ):
                _d = last.payload.get("data", [])
                _n = last.payload.get("count", len(_d))
                _rows_text = "\n".join(
                    "  • " + ", ".join(f"{_k}={_v}" for _k, _v in _r.items() if _k != "id")
                    for _r in _d
                )
                _simple2 = [
                    {"role": "system", "content": (
                        "You are a database assistant writing the FINAL answer to the user. "
                        "You have NO tools — never emit JSON, tool calls, function names, or "
                        "phrases like 'I will call'. Answer directly in plain English using ONLY "
                        "the rows provided. Never invent data.\n"
                        "For max/min/most/least/highest/lowest: compare the numeric values and "
                        "name the correct row. Read each value; do not guess."
                    )},
                    {"role": "user", "content": (
                        f"Database rows ({_n} total):\n{_rows_text}\n\n"
                        f"Question: {user_message}\n\n"
                        "Write the final answer in one or two plain sentences using only the data above. "
                        "If counting total rows: give ONE total number — do not break down by category. "
                        "If filtering by a condition: count only matching rows and state the number "
                        "(say 'no records' / '0' when none match). "
                        "If finding a max or min value: scan every row and name the one with the "
                        "highest/lowest number.\nAnswer:"
                    )},
                ]
                _confirm2 = ollama.chat(
                    model=self.model, messages=_simple2,
                    options={"temperature": 0},
                    keep_alive=KEEP_ALIVE,
                )
                _raw2  = _confirm2.message.content or ""
                _text2 = _clean_reply(_raw2)
                # Detect tool-call leakage the cleaner may not fully remove, or a reply
                # that became a non-answer after stripping.
                _leaked = bool(re.search(
                    r'"name"\s*:\s*"|"parameters"\s*:|I will call \w|\bquery_data\s*[\(\{]',
                    _raw2, re.IGNORECASE,
                ))
                if not _text2 or _leaked:
                    # Model failed to produce a clean answer — list the rows directly
                    _row_summaries = "; ".join(
                        ", ".join(f"{_k}={_v}" for _k, _v in _r.items() if _k != "id")
                        for _r in _d[:5]
                    )
                    _text2 = f"Found {_n} record(s): {_row_summaries}" + ("…" if _n > 5 else ".")
                return _text2, all_tools

            # Zero-result force-return: immediately answer when query returns nothing.
            # Skip a second LLM call — the model only adds non-deterministic phrasing
            # variance; a direct reply is more reliable.
            elif (
                last
                and last.name == "query_data"
                and last.success
                and last.payload.get("count", 0) == 0
                and not _in_update_flow
            ):
                _tbl = last.args.get("table_name", "the table")
                _flt = last.args.get("filters")
                if not isinstance(_flt, dict):
                    _flt = {}
                if _flt:
                    _flt_desc = ", ".join(f"{k}={v}" for k, v in _flt.items())
                    return f"No records found matching {_flt_desc}.", all_tools
                return f"No records found in '{_tbl}'.", all_tools

            # Rebuild tools ONLY when the schema actually changed this round.
            # Otherwise the tool definitions are identical, so re-querying the full
            # schema every round (≈8 schema reads per table) is pure overhead.
            if _schema_dirty:
                tools = build_tools(user_id, self.sm, billing=self.billing)
                _schema_dirty = False

        # ── exceeded max rounds ───────────────────────────────────────────────
        messages.append({
            "role": "user",
            "content": "Please give a brief summary of what was just done.",
        })
        response = ollama.chat(model=self.model, messages=messages,
                               keep_alive=KEEP_ALIVE, options=TOOL_LOOP_OPTIONS)
        return _clean_reply(response.message.content or ""), all_tools


def _clean_reply(text: str) -> str:
    """Strip leaked tool-call syntax and model self-commentary."""
    # JSON-style tool calls
    text = re.sub(r'\{function\s+\w+[^}]*\}',  '', text, flags=re.DOTALL)
    text = re.sub(r'\[TOOL_CALL\]\s*\{.*?\}',   '', text, flags=re.DOTALL)
    text = re.sub(r'<tool_call>.*?</tool_call>', '', text, flags=re.DOTALL)
    # {"name": "tool_name", "parameters": {...}} leakage
    text = re.sub(r'\{"name"\s*:\s*"[^"]+"\s*,\s*"parameters"[^}]*\}[^"]*\}?',
                  '', text, flags=re.DOTALL)
    # Python function-call style: query_data(...) / add_data(...)
    text = re.sub(r'\b(query_data|add_data|update_data|delete_data|create_table|add_column)\s*\([^)]*\)',
                  '', text, flags=re.DOTALL)
    # Generic call_tool("name", {...}) leakage from some model builds
    text = re.sub(r'\bcall_tool\s*\([^)]*\)', '', text, flags=re.DOTALL)
    # "call query_data with {...}" and "call tool X with {...}" instruction-echo leakage
    text = re.sub(
        r'\bcall\s+(?:tool\s+)?\w+\s+with\b.*',
        '', text, flags=re.DOTALL | re.IGNORECASE,
    )
    # Retryable / argument-error messages leaking as text output
    text = re.sub(r'(?i)ARGUMENT ERROR:.*', '', text)
    text = re.sub(r'(?i)Error:\s*Required argument[^\n]*', '', text)
    # Trailing JSON key-value fragments: `", "table_name": "books"}}` or `"}}` at end
    text = re.sub(r'[\s",]+\"[\w_]+\"\s*:\s*\"[^\"]*\"\s*\}*\s*$', '', text)
    text = re.sub(r'[\s"]*\}+\s*$', '', text)
    # Model self-commentary: "(Note: ...)", "(I've reformatted ...)", "I've reformatted ..."
    text = re.sub(r'\(Note:[^)]*\)', '', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'\(I\'ve [^)]*\)', '', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"I've reformatted.*", '', text, flags=re.DOTALL | re.IGNORECASE)
    # Instruction-echo thinking fragments at the start of lines
    text = re.sub(r'^\(I will [^)]*\)\s*', '', text, flags=re.MULTILINE | re.IGNORECASE)
    # "I'll call the correct tool now" / "I will now call..." / "Let me call the tool"
    text = re.sub(r"I['']ll\s+call\s+(?:the\s+)?(?:correct\s+)?tool.*?[\.\!\n]?",
                  '', text, flags=re.IGNORECASE)
    text = re.sub(r"I\s+will\s+(?:now\s+)?call\s+.*?tool.*?[\.\!\n]?",
                  '', text, flags=re.IGNORECASE)
    # "I will call query_data to get more information" (no 'tool' keyword, uses tool name directly)
    text = re.sub(r"\bI\s+will\s+call\s+\w+\b.*?[\.\!\n]", '', text, flags=re.IGNORECASE)
    text = re.sub(r"Let\s+me\s+(?:call|use|run|check)\s+(?:the\s+)?(?:correct\s+)?tool.*?[\.\!\n]?",
                  '', text, flags=re.IGNORECASE)
    # "Fix this by calling..." leaked from ARGUMENT ERROR messages
    text = re.sub(r"Fix\s+this\s+by\s+calling.*?[\.\!\n]?", '', text, flags=re.IGNORECASE)
    # "Do NOT tell the user" leaked from retryable error content
    text = re.sub(r"Do\s+NOT\s+tell\s+the\s+user.*?[\.\!\n]?", '', text, flags=re.IGNORECASE)
    # Bare JSON fragment without leading {: "name": "query_data", "parameters": ...
    text = re.sub(r'"name"\s*:\s*"[^"]+"\s*,.*', '', text, flags=re.DOTALL)
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

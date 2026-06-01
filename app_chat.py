"""
app_chat.py — AI-powered chat interface for the dynamic multi-tenant database.

Run:  streamlit run app_chat.py
      (Ollama must be running: `ollama serve` and `ollama pull llama3.2`)
"""

import json
import streamlit as st
import pandas as pd

from database.schema_manager import SchemaManager
from database.dynamic_crud import DynamicCRUD
from ai.query_parser import ChatEngine, trim_history, DEFAULT_MODEL, MEMORY_TURNS
from ai.dynamic_tools import schema_summary, ToolResult

# ─── page config ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="DB Chat",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container { padding-top: 1.5rem; }
    .stChatMessage p { margin-bottom: 0.3rem; }
    .tool-badge {
        display: inline-block;
        background: #1e3a5f;
        color: #7eb8f7;
        border-radius: 4px;
        padding: 1px 7px;
        font-size: 0.78rem;
        font-family: monospace;
        margin-right: 4px;
    }
    .tool-ok   { background: #1a3d2b; color: #6ddb9b; }
    .tool-fail { background: #3d1a1a; color: #db6d6d; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─── singletons ──────────────────────────────────────────────────────────────

@st.cache_resource
def _init_services():
    sm   = SchemaManager(base_dir="data")
    crud = DynamicCRUD(sm)
    return sm, crud

sm, crud = _init_services()


def _engine() -> ChatEngine:
    return ChatEngine(sm, crud, model=st.session_state.get("model", DEFAULT_MODEL))


# ─── session defaults ─────────────────────────────────────────────────────────

def _ss(key, default):
    if key not in st.session_state:
        st.session_state[key] = default

_ss("user",           None)
_ss("messages",       [])   # display list: {role, content, tool_results?}
_ss("llm_history",    [])   # trimmed context for the LLM: {role, content}
_ss("model",          DEFAULT_MODEL)
_ss("pending_prompt", None)


# ═════════════════════════════════════════════════════════════════════════════
#  AUTH
# ═════════════════════════════════════════════════════════════════════════════

def auth_page():
    _, col, _ = st.columns([1, 1.4, 1])
    with col:
        st.markdown("## DB Chat")
        st.caption("Talk to your database in plain English")
        st.divider()
        tab_login, tab_reg = st.tabs(["Login", "Create Account"])

        with tab_login:
            with st.form("f_login"):
                u = st.text_input("Username")
                p = st.text_input("Password", type="password")
                if st.form_submit_button("Login", use_container_width=True,
                                         type="primary"):
                    user = sm.authenticate_user(u, p)
                    if user:
                        st.session_state.user = user
                        st.rerun()
                    else:
                        st.error("Invalid credentials")

        with tab_reg:
            with st.form("f_reg"):
                u  = st.text_input("Username")
                p  = st.text_input("Password", type="password")
                p2 = st.text_input("Confirm Password", type="password")
                if st.form_submit_button("Create Account", use_container_width=True,
                                         type="primary"):
                    if p != p2:
                        st.error("Passwords do not match")
                    elif len(p) < 4:
                        st.error("Password must be at least 4 characters")
                    else:
                        try:
                            sm.create_user(u.strip(), p)
                            st.success("Account created — you can now log in")
                        except ValueError as e:
                            st.error(str(e))


# ═════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ═════════════════════════════════════════════════════════════════════════════

# Cache the Ollama health check for 20 s so it doesn't fire on every keypress
@st.cache_data(ttl=20)
def _cached_health(model: str) -> tuple:
    return ChatEngine(sm, crud, model=model).check_ollama()


def sidebar():
    user = st.session_state.user
    with st.sidebar:
        # ── user / logout ─────────────────────────────────────────────────
        st.markdown(f"**{user['username']}**")
        if st.button("Logout", use_container_width=True):
            st.session_state.update(
                user=None, messages=[], llm_history=[], pending_prompt=None
            )
            st.rerun()

        # ── model picker ──────────────────────────────────────────────────
        st.divider()
        model_choices = ["llama3.2:3b", "llama3.1:8b", "hermes3:8b",
                         "mistral:7b", "mistral-nemo:12b", "phi4-mini:3.8b",
                         "qwen2.5-coder:7b"]
        cur = st.session_state.model
        if cur not in model_choices:
            model_choices.insert(0, cur)
        chosen = st.selectbox(
            "Model",
            model_choices,
            index=model_choices.index(cur),
            key="model_sel",
        )
        if chosen != st.session_state.model:
            st.session_state.model = chosen
            _cached_health.clear()
            st.rerun()

        # ── ollama status ─────────────────────────────────────────────────
        ok, msg = _cached_health(st.session_state.model)
        if ok:
            st.success(msg, icon="✅")
        else:
            st.error(msg, icon="❌")

        # ── tables overview ───────────────────────────────────────────────
        st.divider()
        st.caption("YOUR TABLES")
        tables = sm.get_user_tables(user["id"])
        if not tables:
            st.caption("None yet — ask the chat to create one")
        else:
            for t in tables:
                n    = crud.count_records(user["id"], t["table_name"])
                cols = sm.get_table_schema(user["id"], t["table_name"]) or []
                with st.expander(f"{t['table_name']}  ({n} rows)", expanded=False):
                    if cols:
                        df = pd.DataFrame([{
                            "column": c["column_name"],
                            "type":   c["column_type"],
                        } for c in cols])
                        st.dataframe(df, use_container_width=True,
                                     hide_index=True, height=130)
                    else:
                        st.caption("No columns")

        # ── example prompts ───────────────────────────────────────────────
        st.divider()
        st.caption("EXAMPLE PROMPTS  (click to send)")

        examples = _example_prompts(user["id"])
        for label, prompt in examples:
            if st.button(label, use_container_width=True, key=f"ex_{label[:30]}"):
                st.session_state.pending_prompt = prompt
                st.rerun()

        # ── clear history ────────────────────────────────────────────────
        st.divider()
        if st.button("Clear chat history", use_container_width=True):
            st.session_state.messages    = []
            st.session_state.llm_history = []
            st.rerun()


def _example_prompts(user_id: int):
    tables = sm.get_user_tables(user_id)
    if not tables:
        return [
            ("📚 Create a books table",
             "Create a table called books with columns: title (TEXT), author (TEXT), "
             "price (FLOAT), and genre (TEXT)"),
            ("👥 Create an employees table",
             "Create a table for employees with columns: name, role, salary (FLOAT), "
             "and department"),
            ("✅ Create a tasks table",
             "Create a tasks table with: title (TEXT), description (TEXT), "
             "done (BOOLEAN), due_date (DATE)"),
        ]

    first = tables[0]["table_name"]
    prompts = [
        (f"📋 Show all {first}",
         f"Show me all records in the {first} table"),
        (f"➕ Add a record to {first}",
         f"Add a new record to {first}"),
        (f"🔍 Query {first}",
         f"How many records are in {first}?"),
    ]
    if len(tables) == 1:
        prompts.append((
            "📚 Add books table",
            "Create a books table with title, author, price, and genre columns",
        ))
    return prompts


# ═════════════════════════════════════════════════════════════════════════════
#  TOOL CALL RENDERER
# ═════════════════════════════════════════════════════════════════════════════

_TOOL_ICONS = {
    "query_data":   "🔍",
    "add_data":     "➕",
    "update_data":  "✏️",
    "create_table": "🗄",
    "add_column":   "➕🏛",
}


def render_tool_results(tool_results: list):
    """Render tool call details inside the assistant bubble."""
    if not tool_results:
        return
    for tr in tool_results:
        icon  = _TOOL_ICONS.get(tr.name, "🔧")
        badge = "tool-ok" if tr.success else "tool-fail"
        label = f"{icon} {tr.name}"
        with st.expander(label, expanded=False):
            # Args
            st.markdown("**Arguments**")
            st.code(json.dumps(tr.args, indent=2, default=str), language="json")
            st.markdown("**Result**")
            # Special rendering for query results
            if tr.success and tr.name == "query_data":
                rows = tr.payload.get("rows", [])
                count = tr.payload.get("count", 0)
                st.caption(f"{count} row(s) returned")
                if rows:
                    # Drop internal columns from display
                    df = pd.DataFrame(rows)
                    st.dataframe(df, use_container_width=True,
                                 hide_index=True, height=min(200, 45 + 35 * len(rows)))
                else:
                    st.info("No rows returned")
            else:
                color = "#1a3d2b" if tr.success else "#3d1a1a"
                st.code(
                    json.dumps(tr.payload, indent=2, default=str),
                    language="json",
                )


def render_message(msg: dict):
    """Render one stored message (user or assistant)."""
    role = msg["role"]
    with st.chat_message(role):
        if role == "assistant" and msg.get("tool_results"):
            render_tool_results(msg["tool_results"])
        st.markdown(msg["content"])


# ═════════════════════════════════════════════════════════════════════════════
#  MAIN CHAT VIEW
# ═════════════════════════════════════════════════════════════════════════════

def chat_view():
    user    = st.session_state.user
    user_id = user["id"]

    st.title("DB Chat")
    st.caption(
        f"Talking to your database as **{user['username']}** · "
        f"model: `{st.session_state.model}` · "
        f"memory: last {MEMORY_TURNS} turns"
    )

    # ── render existing conversation ──────────────────────────────────────────
    for msg in st.session_state.messages:
        render_message(msg)

    # ── collect new input (chat widget OR pending button click) ───────────────
    pending = st.session_state.pop("pending_prompt", None)
    typed   = st.chat_input("Ask about your data…  e.g. 'Add Dune by Frank Herbert, price $14.99'")
    active  = pending or typed

    if not active:
        return

    # ── show user message immediately ─────────────────────────────────────────
    with st.chat_message("user"):
        st.markdown(active)
    st.session_state.messages.append({"role": "user", "content": active})

    # ── check Ollama before calling ───────────────────────────────────────────
    ok, health_msg = _cached_health(st.session_state.model)
    if not ok:
        err = f"Cannot reach Ollama: {health_msg}"
        with st.chat_message("assistant"):
            st.error(err)
        st.session_state.messages.append({"role": "assistant", "content": err,
                                           "tool_results": []})
        return

    # ── call the engine (with spinner) ────────────────────────────────────────
    with st.chat_message("assistant"):
        tool_placeholder = st.empty()
        with st.spinner("Thinking…"):
            try:
                reply, tool_results = _engine().chat(
                    user_id=user_id,
                    user_message=active,
                    history=st.session_state.llm_history,
                )
            except Exception as exc:
                reply        = f"Error communicating with Ollama: {exc}"
                tool_results = []

        # Render tool calls inside the assistant bubble
        tool_placeholder.empty()
        if tool_results:
            render_tool_results(tool_results)
        st.markdown(reply)

    # ── persist ───────────────────────────────────────────────────────────────
    st.session_state.messages.append({
        "role":         "assistant",
        "content":      reply,
        "tool_results": [tr.__dict__ for tr in tool_results],  # serialisable
    })

    # Update trimmed LLM history
    st.session_state.llm_history.append({"role": "user",      "content": active})
    st.session_state.llm_history.append({"role": "assistant", "content": reply})
    st.session_state.llm_history = trim_history(
        st.session_state.llm_history, max_turns=MEMORY_TURNS
    )


# ═════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═════════════════════════════════════════════════════════════════════════════

def main():
    if not st.session_state.user:
        auth_page()
        return
    sidebar()
    chat_view()


if __name__ == "__main__":
    main()

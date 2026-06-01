"""
app_db.py — Streamlit UI for the dynamic multi-tenant SQLite system.

Run:  streamlit run app_db.py
"""

import json
import pandas as pd
import streamlit as st
from datetime import date, datetime

from database.schema_manager import VALID_COLUMN_TYPES, SchemaManager
from database.dynamic_crud import DynamicCRUD
from ai.query_parser import ChatEngine, trim_history, DEFAULT_MODEL, MEMORY_TURNS
from ai.dynamic_tools import schema_summary, ToolResult

# ─── page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Dynamic DB Studio",
    page_icon="🗄",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container { padding-top: 1.5rem; }
    [data-testid="stSidebarContent"] button { justify-content: flex-start; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─── singletons ──────────────────────────────────────────────────────────────

@st.cache_resource
def _init():
    sm = SchemaManager(base_dir="data")
    return sm, DynamicCRUD(sm)

sm, crud = _init()

# ─── session state helpers ────────────────────────────────────────────────────

def _ss(key, default):
    if key not in st.session_state:
        st.session_state[key] = default

_ss("user", None)
_ss("view", "dashboard")
_ss("active_table", None)
_ss("num_new_cols", 1)
_ss("_toast", None)        # survives rerun; flushed once at top of each render
_ss("chat_messages", [])   # display history for the chat view
_ss("llm_history", [])     # trimmed LLM context
_ss("model", DEFAULT_MODEL)


def _queue_toast(msg: str, icon: str = "✅"):
    """Schedule a toast that will show after the next st.rerun()."""
    st.session_state["_toast"] = (msg, icon)


def _flush_toast():
    """Call once at the start of every page render to show any queued toast."""
    item = st.session_state.pop("_toast", None)
    if item:
        st.toast(item[0], icon=item[1])


def _nav(view, table=None):
    st.session_state.view = view
    st.session_state.active_table = table

# ═════════════════════════════════════════════════════════════════════════════
#  AUTH PAGE
# ═════════════════════════════════════════════════════════════════════════════

def auth_page():
    _, col, _ = st.columns([1, 1.4, 1])
    with col:
        st.markdown("## Dynamic DB Studio")
        st.caption("Multi-tenant SQLite · dynamic schemas · per-user isolation")
        st.divider()

        tab_login, tab_register = st.tabs(["Login", "Create Account"])

        with tab_login:
            with st.form("form_login"):
                username = st.text_input("Username")
                password = st.text_input("Password", type="password")
                if st.form_submit_button("Login", use_container_width=True, type="primary"):
                    user = sm.authenticate_user(username, password)
                    if user:
                        st.session_state.user = user
                        st.rerun()
                    else:
                        st.error("Invalid username or password")

        with tab_register:
            with st.form("form_register"):
                new_user = st.text_input("Username")
                new_pass = st.text_input("Password", type="password")
                confirm = st.text_input("Confirm Password", type="password")
                if st.form_submit_button("Create Account", use_container_width=True, type="primary"):
                    if new_pass != confirm:
                        st.error("Passwords do not match")
                    elif len(new_pass) < 4:
                        st.error("Password must be at least 4 characters")
                    else:
                        try:
                            sm.create_user(new_user.strip(), new_pass)
                            st.success("Account created — you can now log in")
                        except ValueError as e:
                            st.error(str(e))

# ═════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ═════════════════════════════════════════════════════════════════════════════

def sidebar():
    user = st.session_state.user
    with st.sidebar:
        st.markdown(f"Signed in as **{user['username']}**")
        if st.button("Logout", use_container_width=True):
            st.session_state.update(user=None, view="dashboard", active_table=None)
            st.rerun()

        st.divider()

        col_dash, col_chat = st.columns(2)
        if col_dash.button("Dashboard", use_container_width=True):
            _nav("dashboard")
            st.rerun()
        if col_chat.button("💬 Chat", use_container_width=True,
                           type="primary" if st.session_state.view == "chat" else "secondary"):
            _nav("chat")
            st.rerun()

        if st.button("＋ New Table", use_container_width=True, type="primary"):
            # clear any leftover column-builder keys
            for i in range(st.session_state.num_new_cols):
                for p in ("cn_", "ct_", "cr_", "cd_"):
                    st.session_state.pop(f"{p}{i}", None)
            st.session_state.num_new_cols = 1
            _nav("create_table")
            st.rerun()

        st.divider()
        st.caption("MY TABLES")

        tables = sm.get_user_tables(user["id"])
        if not tables:
            st.caption("None yet")
        for t in tables:
            n = crud.count_records(user["id"], t["table_name"])
            label = f"{t['table_name']}  ({n})"
            active = st.session_state.active_table == t["table_name"]
            if st.button(label, key=f"nav_{t['table_name']}",
                         use_container_width=True,
                         type="primary" if active else "secondary"):
                _nav("table", t["table_name"])
                st.rerun()

# ═════════════════════════════════════════════════════════════════════════════
#  DASHBOARD
# ═════════════════════════════════════════════════════════════════════════════

def dashboard_view():
    user_id = st.session_state.user["id"]
    st.title("Dashboard")
    tables = sm.get_user_tables(user_id)

    if not tables:
        st.info("No tables yet. Click **＋ New Table** in the sidebar to create one.")
        return

    cols = st.columns(min(3, len(tables)))
    for i, t in enumerate(tables):
        n = crud.count_records(user_id, t["table_name"])
        schema = sm.get_table_schema(user_id, t["table_name"]) or []
        with cols[i % 3]:
            with st.container(border=True):
                st.markdown(f"#### {t['table_name']}")
                m1, m2 = st.columns(2)
                m1.metric("Rows", n)
                m2.metric("Columns", len(schema))
                st.caption(f"Created {t['created_at'][:10]}")
                if st.button("Open →", key=f"open_{t['table_name']}",
                             use_container_width=True):
                    _nav("table", t["table_name"])
                    st.rerun()

# ═════════════════════════════════════════════════════════════════════════════
#  CREATE TABLE
# ═════════════════════════════════════════════════════════════════════════════

def create_table_view():
    st.title("Create New Table")
    user_id = st.session_state.user["id"]
    num = st.session_state.num_new_cols

    table_name = st.text_input("Table Name", placeholder="e.g. products, tasks, customers")
    st.caption("`id`, `created_at`, and `updated_at` columns are added automatically.")

    # Column builder header
    h = st.columns([3, 2, 1, 2, 0.5])
    h[0].markdown("**Column Name**")
    h[1].markdown("**Type**")
    h[2].markdown("**Req**")
    h[3].markdown("**Default**")

    for i in range(num):
        c = st.columns([3, 2, 1, 2, 0.5])
        c[0].text_input("Name", key=f"cn_{i}", label_visibility="collapsed",
                        placeholder=f"column_{i + 1}")
        c[1].selectbox("Type", VALID_COLUMN_TYPES, key=f"ct_{i}",
                       label_visibility="collapsed")
        c[2].checkbox("Req", key=f"cr_{i}", label_visibility="collapsed")
        c[3].text_input("Default", key=f"cd_{i}", label_visibility="collapsed",
                        placeholder="optional")
        # Remove-row button (only show when there is more than one row)
        if num > 1 and c[4].button("✕", key=f"rm_{i}"):
            _remove_col_row(i, num)
            st.rerun()

    btn1, btn2 = st.columns([1, 3])
    if btn1.button("＋ Add Column"):
        st.session_state.num_new_cols += 1
        st.rerun()

    if btn2.button("Create Table", type="primary", use_container_width=True):
        _do_create_table(user_id, table_name.strip(), num)


def _remove_col_row(index: int, total: int):
    """Shift widget keys down to close the gap left by removing row `index`."""
    for j in range(index, total - 1):
        st.session_state[f"cn_{j}"] = st.session_state.get(f"cn_{j+1}", "")
        st.session_state[f"ct_{j}"] = st.session_state.get(f"ct_{j+1}", "TEXT")
        st.session_state[f"cr_{j}"] = st.session_state.get(f"cr_{j+1}", False)
        st.session_state[f"cd_{j}"] = st.session_state.get(f"cd_{j+1}", "")
    for p in ("cn_", "ct_", "cr_", "cd_"):
        st.session_state.pop(f"{p}{total - 1}", None)
    st.session_state.num_new_cols -= 1


def _do_create_table(user_id: int, table_name: str, num: int):
    if not table_name:
        st.error("Table name is required")
        return

    columns_schema = []
    for i in range(num):
        name = st.session_state.get(f"cn_{i}", "").strip()
        if name:
            columns_schema.append({
                "name": name,
                "type": st.session_state.get(f"ct_{i}", "TEXT"),
                "required": st.session_state.get(f"cr_{i}", False),
                "default": st.session_state.get(f"cd_{i}", "") or None,
            })

    if not columns_schema:
        st.error("At least one named column is required")
        return

    try:
        sm.create_dynamic_table(user_id, table_name, columns_schema)
        # clean widget keys
        for i in range(num):
            for p in ("cn_", "ct_", "cr_", "cd_"):
                st.session_state.pop(f"{p}{i}", None)
        st.session_state.num_new_cols = 1
        _nav("table", table_name)
        st.rerun()
    except ValueError as e:
        st.error(str(e))

# ═════════════════════════════════════════════════════════════════════════════
#  TABLE VIEW  (data / insert / filter / add-column / danger zone)
# ═════════════════════════════════════════════════════════════════════════════

def table_view():
    user_id = st.session_state.user["id"]
    table_name = st.session_state.active_table
    schema = sm.get_table_schema(user_id, table_name)

    if schema is None:
        st.error("Table not found")
        return

    st.title(table_name)

    tabs = st.tabs(["Data", "Insert", "Filter & Search", "Add Column", "Danger Zone"])
    with tabs[0]: _tab_data(user_id, table_name, schema)
    with tabs[1]: _tab_insert(user_id, table_name, schema)
    with tabs[2]: _tab_filter(user_id, table_name, schema)
    with tabs[3]: _tab_add_col(user_id, table_name, schema)
    with tabs[4]: _tab_danger(user_id, table_name)


# ── Data tab ──────────────────────────────────────────────────────────────────

def _tab_data(user_id, table_name, schema):
    records = crud.query_table(user_id, table_name)

    hdr, btn = st.columns([4, 1])
    hdr.caption(f"{len(records)} records")
    if btn.button("↻ Refresh", key="btn_refresh"):
        st.rerun()

    if not records:
        st.info("No records yet — use the **Insert** tab to add data.")
        return

    df = pd.DataFrame(records)
    st.dataframe(df, use_container_width=True, hide_index=True)

    st.divider()
    st.markdown("#### Edit / Delete a Record")

    id_list = [r["id"] for r in records]
    chosen_id = st.selectbox("Select record by ID", options=id_list,
                              key="sel_edit_id")
    record = next((r for r in records if r["id"] == chosen_id), None)

    if record:
        # Encode the chosen_id into the field keys so switching records
        # forces fresh widgets with the correct pre-filled values.
        prefix = f"ed_{chosen_id}"
        with st.form("form_edit"):
            edits = {
                c["column_name"]: _make_field(
                    c["column_name"], c["column_type"],
                    record.get(c["column_name"]), prefix
                )
                for c in schema
            }
            c1, c2 = st.columns(2)
            do_save = c1.form_submit_button("Save Changes", type="primary",
                                            use_container_width=True)
            do_del = c2.form_submit_button("Delete Record",
                                           use_container_width=True)

        if do_save:
            try:
                crud.update_record(user_id, table_name, chosen_id, edits)
                _queue_toast(f"Record {chosen_id} updated")
                st.rerun()
            except Exception as e:
                st.error(str(e))

        if do_del:
            crud.delete_record(user_id, table_name, chosen_id)
            _queue_toast(f"Record {chosen_id} deleted", icon="🗑️")
            st.rerun()


# ── Insert tab ────────────────────────────────────────────────────────────────

def _tab_insert(user_id, table_name, schema):
    st.markdown("#### Insert New Record")
    with st.form("form_insert"):
        data = {
            c["column_name"]: _make_field(
                c["column_name"], c["column_type"], None, "ins"
            )
            for c in schema
        }
        if st.form_submit_button("Insert Record", type="primary",
                                 use_container_width=True):
            clean = {k: v for k, v in data.items()
                     if v is not None and v != ""}
            try:
                new_id = crud.insert_record(user_id, table_name, clean)
                _queue_toast(f"Record inserted  (id = {new_id})", icon="➕")
                st.rerun()
            except Exception as e:
                st.error(str(e))


# ── Filter & Search tab ───────────────────────────────────────────────────────

def _tab_filter(user_id, table_name, schema):
    st.markdown("#### Exact-match Filter")
    col_names = [c["column_name"] for c in schema]

    fc1, fc2, fc3 = st.columns([2, 2, 1])
    filter_col = fc1.selectbox("Column", col_names, key="flt_col")
    filter_val = fc2.text_input("Value equals", key="flt_val")
    fc3.markdown("<br>", unsafe_allow_html=True)
    run_filter = fc3.button("Apply", type="primary", key="btn_filter")

    if run_filter:
        filters = {filter_col: filter_val} if filter_val.strip() else None
        results = crud.query_table(user_id, table_name, filters=filters)
        st.caption(f"{len(results)} matching records")
        if results:
            st.dataframe(pd.DataFrame(results), use_container_width=True,
                         hide_index=True)
        else:
            st.info("No records match")

    st.divider()
    st.markdown("#### Substring Search (TEXT columns)")

    text_cols = [c["column_name"] for c in schema if c["column_type"] == "TEXT"]
    if not text_cols:
        st.caption("No TEXT columns in this table")
        return

    sc1, sc2, sc3 = st.columns([2, 2, 1])
    search_col = sc1.selectbox("Search in", text_cols, key="srch_col")
    search_val = sc2.text_input("Contains", key="srch_val")
    sc3.markdown("<br>", unsafe_allow_html=True)
    run_search = sc3.button("Search", key="btn_search")

    if run_search and search_val.strip():
        results = crud.search_table(user_id, table_name, search_col, search_val)
        st.caption(f"{len(results)} results")
        if results:
            st.dataframe(pd.DataFrame(results), use_container_width=True,
                         hide_index=True)
        else:
            st.info("No results")


# ── Add Column tab ────────────────────────────────────────────────────────────

def _tab_add_col(user_id, table_name, schema):
    st.markdown("#### Current Schema")
    st.dataframe(
        pd.DataFrame([{
            "Column": c["column_name"],
            "Type": c["column_type"],
            "Required": bool(c["is_required"]),
            "Default": c["default_value"] or "",
        } for c in schema]),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("#### Add New Column")
    with st.form("form_add_col"):
        ac1, ac2, ac3, ac4 = st.columns([3, 2, 1, 2])
        new_name = ac1.text_input("Column Name")
        new_type = ac2.selectbox("Type", VALID_COLUMN_TYPES)
        new_req = ac3.checkbox("Req", help="Required (stored in metadata; SQLite "
                               "cannot enforce NOT NULL on ALTER TABLE without a default)")
        new_default = ac4.text_input("Default (optional)")
        if st.form_submit_button("Add Column", type="primary"):
            if not new_name.strip():
                st.error("Column name is required")
            else:
                try:
                    sm.add_column_to_table(
                        user_id, table_name,
                        {
                            "name": new_name.strip(),
                            "type": new_type,
                            "required": new_req,
                            "default": new_default.strip() or None,
                        },
                    )
                    _queue_toast(f"Column '{new_name}' added")
                    st.rerun()
                except ValueError as e:
                    st.error(str(e))


# ── Danger Zone tab ───────────────────────────────────────────────────────────

def _tab_danger(user_id, table_name):
    st.warning("Dropping a table permanently deletes all rows and schema metadata.")
    confirm = st.text_input(f"Type **{table_name}** to confirm")
    if st.button("Drop Table", type="primary", key="btn_drop"):
        if confirm.strip() == table_name:
            sm.drop_table(user_id, table_name)
            _queue_toast(f"Table '{table_name}' dropped", icon="🗑️")
            st.session_state.update(view="dashboard", active_table=None)
            st.rerun()
        else:
            st.error("Name does not match — table was NOT dropped")


# ═════════════════════════════════════════════════════════════════════════════
#  FIELD FACTORY  — renders the right Streamlit widget for each column type
# ═════════════════════════════════════════════════════════════════════════════

def _make_field(col_name: str, col_type: str, current=None, prefix: str = ""):
    """
    Render a Streamlit input widget appropriate for col_type.
    Returns a Python value suitable for storing in SQLite.
    """
    key = f"{prefix}_{col_name}"
    label = col_name.replace("_", " ").title()

    if col_type == "INTEGER":
        val = int(current) if current is not None else 0
        return st.number_input(label, value=val, step=1, key=key)

    if col_type == "FLOAT":
        val = float(current) if current is not None else 0.0
        return st.number_input(label, value=val, step=0.01,
                               format="%.4f", key=key)

    if col_type == "BOOLEAN":
        # SQLite stores booleans as 0/1
        val = bool(int(current)) if current is not None else False
        return int(st.checkbox(label, value=val, key=key))

    if col_type == "DATE":
        try:
            val = date.fromisoformat(str(current)) if current else date.today()
        except (ValueError, TypeError):
            val = date.today()
        return str(st.date_input(label, value=val, key=key))

    if col_type == "TIMESTAMP":
        try:
            val = datetime.fromisoformat(str(current)).date() if current else date.today()
        except (ValueError, TypeError):
            val = date.today()
        return str(st.date_input(label, value=val, key=key))

    # TEXT (default)
    return st.text_input(label, value=str(current) if current is not None else "",
                         key=key)


# ═════════════════════════════════════════════════════════════════════════════
#  CHAT VIEW
# ═════════════════════════════════════════════════════════════════════════════

_TOOL_ICONS = {
    "query_data": "🔍", "add_data": "➕", "update_data": "✏️",
    "create_table": "🗄", "add_column": "➕🏛",
}


@st.cache_data(ttl=20)
def _check_ollama(model: str):
    return ChatEngine(sm, crud, model=model).check_ollama()


def _render_tool_results(tool_results: list):
    for tr in tool_results:
        icon = _TOOL_ICONS.get(tr["name"], "🔧")
        ok   = tr["success"]
        with st.expander(f"{'✅' if ok else '❌'} {icon} `{tr['name']}`", expanded=False):
            st.markdown("**Arguments**")
            st.code(json.dumps(tr["args"], indent=2, default=str), language="json")
            st.markdown("**Result**")
            if ok and tr["name"] == "query_data":
                rows = tr["payload"].get("rows", [])
                st.caption(f"{tr['payload'].get('count', 0)} row(s)")
                if rows:
                    st.dataframe(pd.DataFrame(rows), use_container_width=True,
                                 hide_index=True,
                                 height=min(200, 45 + 35 * len(rows)))
                else:
                    st.info("No rows returned")
            else:
                st.code(json.dumps(tr["payload"], indent=2, default=str), language="json")


def chat_view():
    user_id = st.session_state.user["id"]

    st.title("💬 DB Chat")
    st.caption(
        f"Talk to your database · model `{st.session_state.model}` · "
        f"memory: last {MEMORY_TURNS} turns"
    )

    # ── model picker + status inside main area ────────────────────────────────
    with st.expander("⚙️ Model settings", expanded=False):
        model_choices = ["llama3.2:3b", "llama3.1:8b", "hermes3:8b",
                         "mistral:7b", "mistral-nemo:12b", "phi4-mini:3.8b",
                         "qwen2.5-coder:7b"]
        cur = st.session_state.model
        if cur not in model_choices:
            model_choices.insert(0, cur)
        chosen = st.selectbox("Ollama model", model_choices,
                              index=model_choices.index(cur), key="chat_model_sel")
        if chosen != st.session_state.model:
            st.session_state.model = chosen
            _check_ollama.clear()
            st.rerun()

        ok, health_msg = _check_ollama(st.session_state.model)
        if ok:
            st.success(health_msg, icon="✅")
        else:
            st.error(health_msg, icon="❌")

    # ── example prompt buttons ────────────────────────────────────────────────
    tables = sm.get_user_tables(user_id)
    if tables:
        first = tables[0]["table_name"]
        examples = [
            f"Show all records in {first}",
            f"How many records are in {first}?",
        ]
    else:
        examples = [
            "Create a books table with title, author, price, and genre columns",
            "Create a tasks table with title, description, done (BOOLEAN), and due_date",
        ]

    cols = st.columns(len(examples))
    for i, ex in enumerate(examples):
        if cols[i].button(ex, use_container_width=True, key=f"chat_ex_{i}"):
            st.session_state["_pending_chat"] = ex
            st.rerun()

    st.divider()

    # ── render existing messages ──────────────────────────────────────────────
    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            if msg["role"] == "assistant" and msg.get("tool_results"):
                _render_tool_results(msg["tool_results"])
            st.markdown(msg["content"])

    # ── collect input ─────────────────────────────────────────────────────────
    pending = st.session_state.pop("_pending_chat", None)
    typed   = st.chat_input("Ask about your data…  e.g. 'Add Dune by Frank Herbert, $14.99'")
    active  = pending or typed

    if not active:
        return

    # show user bubble immediately
    with st.chat_message("user"):
        st.markdown(active)
    st.session_state.chat_messages.append({"role": "user", "content": active})

    # check Ollama
    ok, health_msg = _check_ollama(st.session_state.model)
    if not ok:
        err = f"Ollama is not running. {health_msg}"
        with st.chat_message("assistant"):
            st.error(err)
        st.session_state.chat_messages.append(
            {"role": "assistant", "content": err, "tool_results": []}
        )
        return

    # call LLM
    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            try:
                engine = ChatEngine(sm, crud, model=st.session_state.model)
                reply, tool_results = engine.chat(
                    user_id=user_id,
                    user_message=active,
                    history=st.session_state.llm_history,
                )
            except Exception as exc:
                reply        = f"Error: {exc}"
                tool_results = []

        if tool_results:
            _render_tool_results([tr.__dict__ for tr in tool_results])
        st.markdown(reply)

    # persist
    st.session_state.chat_messages.append({
        "role":         "assistant",
        "content":      reply,
        "tool_results": [tr.__dict__ for tr in tool_results],
    })
    st.session_state.llm_history.append({"role": "user",      "content": active})
    st.session_state.llm_history.append({"role": "assistant", "content": reply})
    st.session_state.llm_history = trim_history(
        st.session_state.llm_history, max_turns=MEMORY_TURNS
    )

    if st.button("🗑 Clear chat", key="clear_chat"):
        st.session_state.chat_messages = []
        st.session_state.llm_history   = []
        st.rerun()


# ═════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main():
    _flush_toast()
    if not st.session_state.user:
        auth_page()
        return

    sidebar()

    view = st.session_state.view
    if view == "chat":
        chat_view()
    elif view == "create_table":
        create_table_view()
    elif view == "table" and st.session_state.active_table:
        table_view()
    else:
        dashboard_view()


if __name__ == "__main__":
    main()

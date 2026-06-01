# Dynamic DB Studio

A multi-tenant, AI-powered database workbench. Create tables, insert data, and query everything in plain English — no SQL required.

Built with **Streamlit**, **SQLite**, and **Ollama** (local LLM, fully offline).

---

## Features

**Database**
- Multi-tenant isolation — every user gets their own SQLite database file
- Create tables with custom columns and types at runtime (no migrations)
- Add columns to existing tables without losing data
- Full CRUD — insert, edit, delete, filter, search
- Drop tables with confirmation guard

**AI Chat**
- Talk to your database in plain English: *"How many keyboards do I have?"*
- LLM calls tools (query, insert, update, create table, add column) automatically
- Tool-call transparency — every database action is shown in an expandable panel
- Multi-turn memory (last 5 conversation turns kept in context)
- Swap models live — any Ollama model that supports tool calling works
- Anti-hallucination guardrails — errors are reported verbatim, not invented

---

## Architecture

```
┌─────────────────────────────────────────────────┐
│  Streamlit UI  (app_db.py / app_chat.py)        │
│  ┌──────────────┐   ┌──────────────────────┐    │
│  │  DB Studio   │   │     AI Chat view     │    │
│  │  (CRUD UI)   │   │  (plain-English DB)  │    │
│  └──────┬───────┘   └──────────┬───────────┘    │
│         │                      │                │
│  ┌──────▼──────────────────────▼───────────┐    │
│  │         database/                        │    │
│  │  SchemaManager   DynamicCRUD            │    │
│  │  metadata.db     data/users/{id}.db     │    │
│  └──────────────────────┬──────────────────┘    │
│                         │                       │
│  ┌──────────────────────▼──────────────────┐    │
│  │              ai/                         │    │
│  │  ChatEngine (query_parser.py)            │    │
│  │  Dynamic tool builder (dynamic_tools.py) │    │
│  │         │                               │    │
│  │    Ollama (local LLM)                   │    │
│  └─────────────────────────────────────────┘    │
└─────────────────────────────────────────────────┘
```

### Key design decisions

| Decision | Reason |
|---|---|
| Per-user SQLite files | True tenant isolation; no cross-user query possible |
| Central metadata.db | Single source of truth for table/column definitions |
| Dynamic tool descriptions | Tools embed live table/column names so the LLM always knows the current schema |
| ToolResult anti-hallucination | Failed tool calls return plain-English error text, not JSON the model misreads |
| `trim_history()` | Keeps last 5 user+assistant pairs to prevent context overflow |

---

## Prerequisites

| Tool | Version | Notes |
|---|---|---|
| Python | 3.10+ | Tested on 3.13 |
| [Ollama](https://ollama.com) | any | Must be running locally |
| llama3.2:3b | — | Default model; others work too |

---

## Quick Start

### 1. Clone the repo

```bash
git clone https://github.com/your-username/imswb.git
cd imswb
```

### 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 3. Install and start Ollama

Download from [ollama.com](https://ollama.com), then:

```bash
ollama serve          # start the server (skip if already running)
ollama pull llama3.2:3b   # download the default model (~2 GB)
```

### 4. Run the app

```bash
streamlit run app_db.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

> **Standalone chat-only UI** — also available at port 8502:
> ```bash
> streamlit run app_chat.py --server.port 8502
> ```

---

## Usage

### Creating your first table

1. Log in or create an account (any username/password ≥ 4 chars)
2. Click **＋ New Table** in the sidebar
3. Name the table (e.g. `inventory`) and add columns
4. Click **Create Table**

Or just ask the chat: *"Create a table called inventory with columns: item (TEXT), quantity (INTEGER), price (FLOAT)"*

### Using the AI chat

Click **💬 Chat** in the sidebar. Example prompts:

```
Show all records in inventory
Add 10 keyboards at $29.99 each
How many items cost more than $50?
Update the keyboard price to $24.99
Create a sales table with product, amount, and date columns
```

The assistant will call the correct database tool automatically and show you exactly what it ran.

### Switching models

Open **⚙️ Model settings** in the chat view and pick any model you have pulled in Ollama. Models that support tool calling (function calling) work best:

- `llama3.2:3b` — fast, good enough for most queries (default)
- `llama3.1:8b` — better reasoning, slower
- `mistral:7b` — solid alternative
- `qwen2.5-coder:7b` — strong at structured data tasks

---

## Project Structure

```
imswb/
├── app_db.py               # Main Streamlit app (DB Studio + Chat)
├── app_chat.py             # Standalone chat-only app
├── requirements.txt
│
├── database/
│   ├── schema_manager.py   # User auth, DDL, metadata (metadata.db)
│   └── dynamic_crud.py     # CRUD against per-user SQLite files
│
├── ai/
│   ├── query_parser.py     # ChatEngine — Ollama tool-calling loop
│   └── dynamic_tools.py    # Builds tool defs from live schema; executes tools
│
└── data/                   # Created at runtime, gitignored
    ├── metadata.db         # Central user/table/column registry
    └── users/
        └── {user_id}.db    # One SQLite file per user
```

---

## Column Types

| Type | SQLite storage | Notes |
|---|---|---|
| TEXT | TEXT | Default for names, descriptions |
| INTEGER | INTEGER | Whole numbers |
| FLOAT | REAL | Prices, measurements |
| BOOLEAN | INTEGER (0/1) | Rendered as checkbox in UI |
| DATE | TEXT (ISO 8601) | Rendered as date picker |
| TIMESTAMP | TEXT (ISO 8601) | Rendered as date picker |

---

## How the AI Tool Loop Works

```
User message
     │
     ▼
Build system prompt  ← live schema embedded here
Build tool definitions  ← table/column names in descriptions
     │
     ▼
ollama.chat(model, messages, tools)
     │
     ├─ no tool_calls → return text to user
     │
     └─ tool_calls → execute each tool
                          │
                          ├─ query_data   → DynamicCRUD.query_table()
                          ├─ add_data     → DynamicCRUD.insert_record()
                          ├─ update_data  → DynamicCRUD.update_record()
                          ├─ create_table → SchemaManager.create_dynamic_table()
                          └─ add_column   → SchemaManager.add_column_to_table()
                          │
                          └─ append tool results, rebuild tools, repeat
                                                    (max 8 rounds)
```

---

## Security Notes

- Passwords are hashed with **PBKDF2-HMAC-SHA256** (100 000 iterations, random salt per user)
- All CRUD methods verify table ownership before executing — users cannot access each other's data
- SQL identifiers are sanitised with `_safe_name()` before any DDL
- The `data/` directory is `.gitignore`d — never commit user databases

---

## Contributing

Pull requests are welcome. For large changes, open an issue first to discuss the approach.

```bash
# Run a quick smoke test after changes
python -c "
from database.schema_manager import SchemaManager
from database.dynamic_crud import DynamicCRUD
sm = SchemaManager('data_test')
uid = sm.create_user('test', 'test1234')
sm.create_dynamic_table(uid, 'items', [{'name': 'name', 'type': 'TEXT', 'required': True}])
crud = DynamicCRUD(sm)
rid = crud.insert_record(uid, 'items', {'name': 'hello'})
rows = crud.query_table(uid, 'items')
assert rows[0]['name'] == 'hello'
sm.drop_table(uid, 'items')
print('All checks passed')
"
```

---

## License

MIT

"""
rag_diagnosis.py

Run with:  python rag_diagnosis.py
           (or the same python that runs the app — e.g. uv run python rag_diagnosis.py)

Checks every layer of a potential RAG stack and prints a structured report.
"""

import importlib
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).parent.resolve()

# ── 1. packages ───────────────────────────────────────────────────────────────

RAG_PACKAGES = {
    "chromadb":              "Vector DB (ChromaDB)",
    "langchain":             "LangChain orchestration",
    "langchain_community":   "LangChain community integrations",
    "sentence_transformers": "Local sentence embeddings",
    "faiss":                 "FAISS vector index",
    "faiss_cpu":             "FAISS (CPU wheel)",
    "numpy":                 "Numeric arrays (embedding ops)",
    "ollama":                "Ollama LLM / embedding client",
}


def check_packages() -> dict:
    installed, missing = {}, []
    for pkg, label in RAG_PACKAGES.items():
        mod_name = pkg.replace("-", "_")
        try:
            m = importlib.import_module(mod_name)
            ver = getattr(m, "__version__", "installed")
            installed[pkg] = ver
        except ImportError:
            missing.append(f"{pkg}  ({label})")
    return {"installed": installed, "missing": missing}


# ── 2. vector DB files ────────────────────────────────────────────────────────

VECTOR_DIR_NAMES = {
    "chroma_db", "chromadb", "vector_store", "vectorstore",
    "embeddings", "faiss_index", "data/chroma",
}
VECTOR_FILE_EXTS = {".parquet", ".sqlite3", ".index", ".bin", ".pkl"}


def check_vector_files() -> dict:
    found_dirs, found_files, collections = [], [], []

    for candidate in VECTOR_DIR_NAMES:
        p = ROOT / candidate
        if p.exists():
            found_dirs.append(str(p.relative_to(ROOT)))

    for ext in VECTOR_FILE_EXTS:
        for f in ROOT.rglob(f"*{ext}"):
            if ".git" not in f.parts and "__pycache__" not in f.parts:
                found_files.append(str(f.relative_to(ROOT)))

    # Try to list ChromaDB collections if the package is available
    try:
        import chromadb
        for search_path in [ROOT / "chroma_db", ROOT / "data" / "chroma",
                             ROOT / "vector_store"]:
            if search_path.exists():
                client = chromadb.PersistentClient(path=str(search_path))
                cols = client.list_collections()
                collections += [
                    {"name": c.name, "count": c.count()} for c in cols
                ]
    except Exception:
        pass

    return {
        "vector_dirs":  found_dirs,
        "vector_files": found_files,
        "collections":  collections,
        "exists":       bool(found_dirs or found_files or collections),
    }


# ── 3. RAG imports in source files ────────────────────────────────────────────

RAG_IMPORT_PATTERNS = [
    r"\bchromadb\b",
    r"\blangchain[._]vectorstores\b",
    r"\blangchain[._]embeddings\b",
    r"\bsentence_transformers\b",
    r"\bfaiss\b",
    r"\bPGVector\b",
    r"from\s+ollama\s+import.*embed",
    r"ollama\.embed",
    r"\bembed_documents\b",
    r"\bembed_query\b",
]


def check_imports() -> dict:
    hits = {}
    py_files = [
        f for f in ROOT.rglob("*.py")
        if ".git" not in f.parts and "__pycache__" not in f.parts
    ]
    for path in py_files:
        try:
            text = path.read_text(errors="ignore")
        except OSError:
            continue
        matched = [p for p in RAG_IMPORT_PATTERNS if re.search(p, text)]
        if matched:
            hits[str(path.relative_to(ROOT))] = matched
    return hits


# ── 4. RAG function definitions ───────────────────────────────────────────────

RAG_FUNC_PATTERNS = [
    r"def\s+\w*retrieve\w*",
    r"def\s+\w*embed\w*",
    r"def\s+\w*vector\w*",
    r"def\s+\w*similarity\w*",
    r"def\s+\w*index\w*",
    r"\.similarity_search\b",
    r"\.add_texts\b",
    r"\.add_documents\b",
    r"query.*collection",
    r"collection\.query",
    r"\.upsert\(",
]


def check_rag_functions() -> dict:
    hits = {}
    py_files = [
        f for f in ROOT.rglob("*.py")
        if ".git" not in f.parts and "__pycache__" not in f.parts
    ]
    for path in py_files:
        try:
            text = path.read_text(errors="ignore")
        except OSError:
            continue
        for i, line in enumerate(text.splitlines(), 1):
            for pat in RAG_FUNC_PATTERNS:
                if re.search(pat, line):
                    key = str(path.relative_to(ROOT))
                    hits.setdefault(key, []).append(f"line {i}: {line.strip()}")
    return hits


# ── 5. requirements.txt ───────────────────────────────────────────────────────

RAG_REQUIREMENT_KEYWORDS = {
    "chromadb", "langchain", "sentence-transformers",
    "faiss-cpu", "faiss-gpu", "pgvector", "openai",
    "tiktoken", "hnswlib", "annoy",
}


def check_requirements() -> dict:
    found, req_files = {}, []
    for name in ("requirements.txt", "requirements_db.txt",
                  "pyproject.toml", "Pipfile"):
        p = ROOT / name
        if p.exists():
            req_files.append(name)
            text = p.read_text(errors="ignore").lower()
            for kw in RAG_REQUIREMENT_KEYWORDS:
                if kw in text:
                    found[kw] = name
    return {"files_checked": req_files, "rag_packages_listed": found}


# ── 6. Ollama embedding models ────────────────────────────────────────────────

EMBED_MODEL_FAMILIES = {
    "nomic-embed-text", "mxbai-embed-large", "all-minilm",
    "bge-m3", "snowflake-arctic-embed",
}


def check_ollama_models() -> dict:
    chat_models, embed_models = [], []
    try:
        import ollama
        resp = ollama.list()
        models = getattr(resp, "models", None) or resp.get("models", [])
        for m in models:
            name = getattr(m, "model", None) or m.get("name", "?")
            is_embed = any(fam in name for fam in EMBED_MODEL_FAMILIES)
            (embed_models if is_embed else chat_models).append(name)
    except Exception as exc:
        return {"error": str(exc)}
    return {"chat_models": chat_models, "embedding_models": embed_models}


# ── 7. integration points ─────────────────────────────────────────────────────

INTEGRATION_FILES = [
    "ai/query_parser.py",
    "ai/dynamic_tools.py",
    "app_db.py",
    "app_chat.py",
    "llm_agent.py",
    "rag.py",
    "rag_engine.py",
]
INTEGRATION_PATTERNS = [
    r"\brag\b",
    r"\bretriev",
    r"\bvector\b",
    r"\bembed\b",
    r"\bchroma\b",
    r"\bsimilarity\b",
    r"context.*retriev",
]


def check_integration_points() -> dict:
    points = {}
    for rel in INTEGRATION_FILES:
        p = ROOT / rel
        if not p.exists():
            continue
        text = p.read_text(errors="ignore")
        matched_lines = []
        for i, line in enumerate(text.splitlines(), 1):
            low = line.lower()
            if any(re.search(pat, low) for pat in INTEGRATION_PATTERNS):
                matched_lines.append(f"  line {i}: {line.rstrip()}")
        if matched_lines:
            points[rel] = matched_lines
    return points


# ── 8. assemble report ────────────────────────────────────────────────────────

def build_report() -> dict:
    print("Running RAG diagnosis …\n")

    pkgs     = check_packages()
    vdb      = check_vector_files()
    imports  = check_imports()
    funcs    = check_rag_functions()
    reqs     = check_requirements()
    models   = check_ollama_models()
    integr   = check_integration_points()

    # ── derived booleans ──────────────────────────────────────────────────────
    rag_installed   = "chromadb" in pkgs["installed"]
    vector_db_exists = vdb["exists"]
    rag_functions   = [
        f"{f}: {ln}"
        for f, lines in funcs.items()
        for ln in lines
    ]
    embedding_model = (
        models.get("embedding_models", ["(none found)"])[0]
        if models.get("embedding_models") else "(none found)"
    )
    collections_count  = len(vdb["collections"])
    indexed_documents  = sum(c.get("count", 0) for c in vdb["collections"])
    integration_points = list(integr.keys())

    # ── gap analysis ─────────────────────────────────────────────────────────
    missing, recommendations = [], []

    if not rag_installed:
        missing.append("chromadb not installed")
        recommendations.append("pip install chromadb")

    if not vector_db_exists:
        missing.append("No vector DB directory found (chroma_db/, vector_store/, …)")
        recommendations.append(
            "Create a RAG engine that initialises a ChromaDB PersistentClient"
        )

    if not imports:
        missing.append("No RAG imports found in any Python file")
        recommendations.append("Create ai/rag_engine.py with ChromaDB + Ollama embeddings")

    if not funcs:
        missing.append("No retrieve/embed/vector/similarity functions found")
        recommendations.append(
            "Implement retrieve(query, user_id, top_k) and index_table(user_id, table_name)"
        )

    if not reqs["rag_packages_listed"]:
        missing.append("requirements.txt does not list chromadb")
        recommendations.append("Add chromadb to requirements.txt")

    embed_models = models.get("embedding_models", [])
    if not embed_models:
        missing.append("No embedding model pulled in Ollama")
        recommendations.append("ollama pull nomic-embed-text")
    else:
        recommendations.append(
            f"Embedding model ready: {embed_models[0]} — use it in OllamaEmbeddingFunction"
        )

    if not integr:
        missing.append("RAG is not wired into ai/query_parser.py or app_db.py")
        recommendations.append(
            "Inject RAG context into ChatEngine._system_prompt() before LLM call"
        )

    if rag_installed and not vector_db_exists:
        recommendations.append(
            "ChromaDB is installed — create ai/rag_engine.py and call index_all_tables() on login"
        )

    return {
        "rag_installed":       rag_installed,
        "vector_db_exists":    vector_db_exists,
        "rag_functions_found": rag_functions,
        "embedding_model":     embedding_model,
        "collections_count":   collections_count,
        "indexed_documents":   indexed_documents,
        "integration_points":  integration_points,
        "missing_components":  missing,
        "recommendations":     recommendations,
        # ── detail sections (not in the summary dict but useful) ──────────────
        "_detail": {
            "packages":          pkgs,
            "vector_db":         vdb,
            "rag_imports":       imports,
            "rag_functions":     funcs,
            "requirements":      reqs,
            "ollama_models":     models,
            "integration_points_detail": integr,
        },
    }


# ── pretty-print ──────────────────────────────────────────────────────────────

def _yn(v: bool) -> str:
    return "✅  YES" if v else "❌  NO"


def print_report(r: dict):
    d = r.pop("_detail")

    # ── summary table ─────────────────────────────────────────────────────────
    print("=" * 60)
    print("  RAG DIAGNOSIS REPORT")
    print("=" * 60)
    print(f"  chromadb installed       {_yn(r['rag_installed'])}")
    print(f"  Vector DB files on disk  {_yn(r['vector_db_exists'])}")
    print(f"  RAG functions in code    {_yn(bool(r['rag_functions_found']))}")
    print(f"  Embedding model          {r['embedding_model']}")
    print(f"  Collections in DB        {r['collections_count']}")
    print(f"  Documents indexed        {r['indexed_documents']}")
    print(f"  Wired into chat/app      {_yn(bool(r['integration_points']))}")
    print()

    # ── packages ──────────────────────────────────────────────────────────────
    print("── Installed packages ──────────────────────────────────")
    for pkg, ver in d["packages"]["installed"].items():
        print(f"   {pkg}=={ver}")
    if d["packages"]["missing"]:
        print("  Missing:")
        for m in d["packages"]["missing"]:
            print(f"   ✗  {m}")
    print()

    # ── Ollama models ─────────────────────────────────────────────────────────
    print("── Ollama models ───────────────────────────────────────")
    for m in d["ollama_models"].get("chat_models", []):
        print(f"   chat:  {m}")
    for m in d["ollama_models"].get("embedding_models", []):
        print(f"   embed: {m}  ← ready for RAG")
    print()

    # ── RAG imports found ─────────────────────────────────────────────────────
    if d["rag_imports"]:
        print("── RAG-related imports found ────────────────────────────")
        for f, pats in d["rag_imports"].items():
            print(f"   {f}")
            for p in pats:
                print(f"      {p}")
        print()

    # ── RAG functions found ───────────────────────────────────────────────────
    if d["rag_functions"]:
        print("── RAG-related function hits ────────────────────────────")
        for f, lines in d["rag_functions"].items():
            print(f"   {f}")
            for ln in lines[:5]:
                print(f"      {ln}")
        print()

    # ── integration points ────────────────────────────────────────────────────
    if d["integration_points_detail"]:
        print("── Integration points ──────────────────────────────────")
        for f, lines in d["integration_points_detail"].items():
            print(f"   {f}")
            for ln in lines[:4]:
                print(f"  {ln}")
        print()

    # ── gaps & recommendations ────────────────────────────────────────────────
    print("── Missing components ──────────────────────────────────")
    for m in r["missing_components"]:
        print(f"   ✗  {m}")
    print()

    print("── Recommendations ─────────────────────────────────────")
    for i, rec in enumerate(r["recommendations"], 1):
        print(f"   {i}. {rec}")
    print()

    print("── Full JSON result ─────────────────────────────────────")
    print(json.dumps(r, indent=2))


if __name__ == "__main__":
    report = build_report()
    print_report(report)

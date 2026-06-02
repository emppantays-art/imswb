"""
ai/rag_engine.py

RAG layer: embeds table rows with nomic-embed-text via Ollama,
stores them in a per-user ChromaDB collection, and retrieves
relevant context for the chat loop.

Design:
  • One ChromaDB collection per user  →  "user_{user_id}"
  • One document per row, ID "table__rowid"
  • Documents: plain-text  "tablename:  col=val  col=val …"
  • Embeddings: Ollama nomic-embed-text (already pulled)
  • Retrieval is optional/graceful — returns [] on any failure
    so the chat loop degrades cleanly when Ollama is down.
"""

from pathlib import Path
from typing import Dict, List, Optional

import chromadb
import ollama

from database.schema_manager import SchemaManager
from database.dynamic_crud import DynamicCRUD

EMBED_MODEL   = "nomic-embed-text"
CHROMA_DIR    = "data/chroma"
TOP_K_DEFAULT = 5
INDEX_ROW_CAP = 2000       # max rows indexed per table
_SKIP_COLS    = {"id", "created_at", "updated_at"}


# ── embedding helper ──────────────────────────────────────────────────────────

def _embed(texts: List[str]) -> List[List[float]]:
    """
    Embed a list of strings via Ollama.  Handles both SDK generations:
      ollama >= 0.4  → ollama.embed(model, input=list)  → .embeddings
      ollama <  0.4  → ollama.embeddings(model, prompt=str) → ["embedding"]
    Falls back to per-item calls if batch mode is unsupported.
    """
    try:
        # Batch API (ollama >= 0.4)
        resp = ollama.embed(model=EMBED_MODEL, input=texts)
        return list(resp.embeddings)
    except (AttributeError, TypeError, Exception):
        pass

    # Per-item fallback
    result = []
    for t in texts:
        try:
            resp = ollama.embed(model=EMBED_MODEL, input=t)
            result.append(list(resp.embeddings[0]))
        except Exception:
            resp = ollama.embeddings(model=EMBED_MODEL, prompt=t)
            vec  = resp.get("embedding") if isinstance(resp, dict) else resp.embedding
            result.append(list(vec))
    return result


# ── document representation ───────────────────────────────────────────────────

def _row_to_text(table_name: str, row: Dict) -> str:
    """
    Convert one DB row to a searchable text document.
    e.g. "books:  title=Dune  author=Frank Herbert  price=14.99"
    """
    parts = [f"{table_name}: "]
    for k, v in row.items():
        if k in _SKIP_COLS or v is None or str(v).strip() == "":
            continue
        parts.append(f"{k}={v}")
    return "  ".join(parts)


# ── RAGEngine ─────────────────────────────────────────────────────────────────

class RAGEngine:
    """
    Shared singleton (via st.cache_resource).  Thread-safe for reads;
    indexing is serialised inside the Streamlit event loop.
    """

    def __init__(
        self,
        sm: SchemaManager,
        crud: DynamicCRUD,
        chroma_dir: str = CHROMA_DIR,
    ):
        self.sm   = sm
        self.crud = crud
        Path(chroma_dir).mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=chroma_dir)

    # ── internal helpers ──────────────────────────────────────────────────────

    def _col(self, user_id: int):
        """Get-or-create the per-user ChromaDB collection (cosine similarity)."""
        return self._client.get_or_create_collection(
            name=f"user_{user_id}",
            metadata={"hnsw:space": "cosine"},
        )

    def _delete_table_docs(self, col, table_name: str) -> None:
        """Remove all previously indexed docs for one table."""
        try:
            existing = col.get(where={"table": {"$eq": table_name}})
            if existing.get("ids"):
                col.delete(ids=existing["ids"])
        except Exception:
            pass

    # ── indexing API ──────────────────────────────────────────────────────────

    def index_table(self, user_id: int, table_name: str) -> int:
        """
        (Re-)index all rows of one table.
        Old docs for the table are replaced so edits/deletes stay current.
        Returns the number of documents indexed.
        """
        rows = self.crud.query_table(user_id, table_name, limit=INDEX_ROW_CAP)
        col  = self._col(user_id)
        self._delete_table_docs(col, table_name)

        if not rows:
            return 0

        ids       = [f"{table_name}__{r['id']}" for r in rows]
        texts     = [_row_to_text(table_name, r) for r in rows]
        metadatas = [{"table": table_name, "row_id": str(r["id"])} for r in rows]
        vectors   = _embed(texts)

        col.upsert(ids=ids, documents=texts, embeddings=vectors, metadatas=metadatas)
        return len(rows)

    def index_all_tables(self, user_id: int) -> Dict[str, object]:
        """Index every table the user owns. Returns {table_name: row_count | error_str}."""
        results: Dict[str, object] = {}
        for t in self.sm.get_user_tables(user_id):
            name = t["table_name"]
            try:
                results[name] = self.index_table(user_id, name)
            except Exception as exc:
                results[name] = f"ERROR: {exc}"
        return results

    def remove_table(self, user_id: int, table_name: str) -> None:
        """Call after drop_table so stale embeddings are removed."""
        try:
            self._delete_table_docs(self._col(user_id), table_name)
        except Exception:
            pass

    # ── retrieval API ─────────────────────────────────────────────────────────

    def retrieve(
        self,
        user_id: int,
        query: str,
        top_k: int = TOP_K_DEFAULT,
        table_filter: Optional[str] = None,
    ) -> List[str]:
        """
        Return the top_k most semantically similar document strings.
        Returns [] gracefully on any error (Ollama down, empty index, etc.).
        """
        try:
            col = self._col(user_id)
            if col.count() == 0:
                return []

            q_vec = _embed([query])[0]
            n     = min(top_k, col.count())

            kwargs: Dict = dict(query_embeddings=[q_vec], n_results=n)
            if table_filter:
                kwargs["where"] = {"table": {"$eq": table_filter}}

            results = col.query(**kwargs)
            docs    = results.get("documents", [[]])[0]
            return [d for d in docs if d]
        except Exception:
            return []

    def doc_count(self, user_id: int) -> int:
        """Total indexed documents for this user."""
        try:
            return self._col(user_id).count()
        except Exception:
            return 0

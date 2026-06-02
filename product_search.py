"""
product_search.py — fast, typo-tolerant product search for the billing/POS UI.

Algorithm: a trigram (3-gram) inverted index with ranked scoring — the same
technique behind PostgreSQL's pg_trgm and many search engines.

Why this one:
  • Typo-tolerant: "aple" still finds "Apple" because they share trigrams,
    unlike a plain substring/LIKE match.
  • Scalable: the inverted index maps each trigram → the items containing it,
    so a query only scores the handful of items that share a trigram with it,
    not the whole catalog. Build is O(total chars); query is roughly
    O(query_trigrams × avg_postings) instead of O(N) per search.
  • Well-ranked: exact > prefix > substring > fuzzy(trigram-Jaccard), so the
    obvious match always sorts first.

Usage:
    idx = TrigramIndex(products, key="name")
    hits = idx.search("aple", limit=10)        # → [{"name": "Apple", ...}, ...]

or one-shot:
    hits = fuzzy_search("aple", products, key="name")
"""

from collections import defaultdict
from typing import Any, Dict, List, Sequence


def _trigrams(text: str) -> set:
    """
    Padded character trigrams. Padding with leading spaces makes prefixes
    produce distinctive trigrams (pg_trgm style), so "app" scores high against
    "apple". Short strings (<3 chars after padding) fall back to the whole token.
    """
    s = f"  {text.strip().lower()} "
    if len(s) < 3:
        return {s.strip()} if s.strip() else set()
    return {s[i:i + 3] for i in range(len(s) - 2)}


class TrigramIndex:
    """Reusable trigram inverted index over a list of item dicts."""

    def __init__(self, items: Sequence[Dict[str, Any]], key: str):
        self._items: List[Dict[str, Any]] = list(items)
        self._key = key
        self._names: List[str] = []
        self._item_trigrams: List[set] = []
        self._index: Dict[str, set] = defaultdict(set)   # trigram -> {item idx}

        for i, it in enumerate(self._items):
            name = str(it.get(key, "")).strip().lower()
            self._names.append(name)
            tg = _trigrams(name)
            self._item_trigrams.append(tg)
            for t in tg:
                self._index[t].add(i)

    def search(
        self,
        query: str,
        limit: int = 10,
        threshold: float = 0.3,
    ) -> List[Dict[str, Any]]:
        """
        Return up to `limit` items ranked by relevance to `query`.
        `threshold` is the minimum fuzzy score (0–1) for non-substring matches;
        exact/prefix/substring matches always qualify.
        """
        q = query.strip().lower()
        if not q:
            return []

        q_tg = _trigrams(q)

        # Candidate set: only items sharing at least one trigram with the query.
        # This is the speed win — we never touch unrelated products.
        candidates = set()
        for t in q_tg:
            candidates |= self._index.get(t, set())
        # Substrings shorter than a trigram (e.g. 1–2 char queries) won't share a
        # padded trigram with longer names, so also allow a direct substring sweep
        # for very short queries.
        if len(q) < 3:
            candidates |= {i for i, n in enumerate(self._names) if q in n}

        scored = []
        for i in candidates:
            s = self._score(q, q_tg, self._names[i], self._item_trigrams[i])
            if s >= threshold:
                scored.append((s, len(self._names[i]), i))

        # Highest score first; break ties by shorter name (more specific match).
        scored.sort(key=lambda x: (-x[0], x[1]))
        return [self._items[i] for _, _, i in scored[:limit]]

    @staticmethod
    def _score(q: str, q_tg: set, name: str, name_tg: set) -> float:
        if not name:
            return 0.0
        if name == q:
            return 1.0
        if name.startswith(q):
            return 0.95
        if q in name:
            return 0.85
        # Trigram Jaccard similarity for fuzzy / typo matches.
        if not q_tg or not name_tg:
            return 0.0
        inter = len(q_tg & name_tg)
        union = len(q_tg | name_tg)
        return inter / union if union else 0.0


def fuzzy_search(
    query: str,
    items: Sequence[Dict[str, Any]],
    key: str = "name",
    limit: int = 10,
    threshold: float = 0.3,
) -> List[Dict[str, Any]]:
    """One-shot fuzzy search (builds a throwaway index). Fine for a few hundred
    items per call; for repeated searches over a large catalog, build a
    TrigramIndex once and reuse it."""
    return TrigramIndex(items, key).search(query, limit=limit, threshold=threshold)

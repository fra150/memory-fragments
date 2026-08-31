"""Indexer implementations — BM25 (pure-Python) and embedding-based."""

from __future__ import annotations

import math
import re
import warnings
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from memory_fragments.config import default_config
from memory_fragments.models import Fragment

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_WORD_RE = re.compile(r"[A-Za-z0-9]+(?:['\u2019][A-Za-z0-9]+)?")


def _tokenize(text: str) -> List[str]:
    """Lowercase tokenisation using a simple regex."""
    return [t.lower() for t in _WORD_RE.findall(text)]


# ---------------------------------------------------------------------------
# BM25Indexer
# ---------------------------------------------------------------------------


class BM25Indexer:
    """In-memory BM25 indexer with incremental add/remove support.

    Uses pure-Python TF-IDF‑like scoring with the standard BM25 formula so no
    external downloads are needed.
    """

    def __init__(self) -> None:
        cfg = default_config.retriever
        self._k1: float = cfg.bm25_k1
        self._b: float = cfg.bm25_b

        # term -> set of fragment_ids containing the term
        self._inverted_index: Dict[str, Set[str]] = {}
        # fragment_id -> {term -> count}
        self._term_freqs: Dict[str, Dict[str, int]] = {}
        # fragment_id -> Fragment
        self._fragments: Dict[str, Fragment] = {}
        # total document length (in tokens) across all indexed fragments
        self._total_doc_len: int = 0

    # ---- public API -------------------------------------------------------

    def index_fragments(self, fragments: List[Fragment]) -> None:
        """Build (or rebuild) the BM25 index from a full list of fragments."""
        self._inverted_index.clear()
        self._term_freqs.clear()
        self._fragments.clear()
        self._total_doc_len = 0

        for frag in fragments:
            self._add_fragment_internal(frag)

    def search(self, query: str, top_k: int = 5) -> List[Tuple[Fragment, float]]:
        """Return up to *top_k* (Fragment, BM25_score) pairs sorted by score descending."""
        query_terms = _tokenize(query)
        if not query_terms or not self._fragments:
            return []

        num_docs = len(self._fragments)
        avg_doc_len = self._total_doc_len / num_docs if num_docs else 1.0

        # Pre‑compute IDF for each query term
        idf: Dict[str, float] = {}
        for term in set(query_terms):
            df = len(self._inverted_index.get(term, set()))
            # BM25 IDF variant (smooth)
            idf[term] = math.log((num_docs - df + 0.5) / (df + 0.5) + 1.0)

        scores: Dict[str, float] = {}
        for term in query_terms:
            term_idf = idf.get(term, 0.0)
            if term_idf == 0.0:
                continue
            for fid in self._inverted_index.get(term, set()):
                tf = self._term_freqs.get(fid, {}).get(term, 0)
                if tf == 0:
                    continue
                doc_len = sum(self._term_freqs[fid].values())
                denom = tf + self._k1 * (
                    1.0 - self._b + self._b * doc_len / avg_doc_len
                )
                term_score = term_idf * (tf * (self._k1 + 1.0)) / denom
                scores[fid] = scores.get(fid, 0.0) + term_score

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [(self._fragments[fid], score) for fid, score in ranked[:top_k]]

    def add_fragment(self, fragment: Fragment) -> None:
        """Incrementally add a single fragment to the index."""
        self._add_fragment_internal(fragment)

    def remove_fragment(self, fragment_id: str) -> None:
        """Remove a fragment from the index by its ID."""
        frag = self._fragments.pop(fragment_id, None)
        if frag is None:
            return
        term_counts = self._term_freqs.pop(fragment_id, {})
        self._total_doc_len -= sum(term_counts.values())
        for term in term_counts:
            doc_set = self._inverted_index.get(term)
            if doc_set:
                doc_set.discard(fragment_id)
                if not doc_set:
                    del self._inverted_index[term]

    # ---- internals --------------------------------------------------------

    def _add_fragment_internal(self, fragment: Fragment) -> None:
        """Core logic for indexing one fragment (no dedup guard)."""
        tokens = _tokenize(fragment.content)
        if not tokens:
            return

        tf: Dict[str, int] = {}
        for t in tokens:
            tf[t] = tf.get(t, 0) + 1

        fid = fragment.fragment_id
        # Remove existing entry if it already exists
        if fid in self._fragments:
            self.remove_fragment(fid)

        self._fragments[fid] = fragment
        self._term_freqs[fid] = tf
        self._total_doc_len += len(tokens)

        for term in tf:
            if term not in self._inverted_index:
                self._inverted_index[term] = set()
            self._inverted_index[term].add(fid)

    @property
    def fragment_count(self) -> int:
        """Number of fragments currently indexed."""
        return len(self._fragments)


# ---------------------------------------------------------------------------
# EmbeddingIndexer
# ---------------------------------------------------------------------------

_FALLBACK_EMBED_DIM = 384


def _simple_embed(text: str, dim: int = _FALLBACK_EMBED_DIM) -> np.ndarray:
    """Deterministic fallback embedding based on character hashes.

    Used when sentence-transformers is not available.  The result is a unit
    vector so cosine similarity remains meaningful (though coarse).
    """
    rng = np.random.RandomState(sum(ord(c) * (i + 1) for i, c in enumerate(text)) % (2**31 - 1))
    vec = rng.randn(dim)
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


class EmbeddingIndexer:
    """Embedding-based semantic indexer using sentence-transformers.

    The underlying model is lazily loaded on first use.  If sentence-transformers
    is not installed the indexer falls back to a deterministic random projection
    so that callers do not crash — but results will be of low quality.
    """

    def __init__(self, model_name: Optional[str] = None) -> None:
        self._model_name = model_name or default_config.retriever.embedding_model

        # Lazy-loaded model / flag
        self._model: Optional["SentenceTransformer"] = None  # noqa: F821
        self._fallback: bool = False

        # fragment_id -> embedding vector
        self._embeddings: Dict[str, np.ndarray] = {}
        # fragment_id -> Fragment
        self._fragments: Dict[str, Fragment] = {}

    # ---- public API -------------------------------------------------------

    def index_fragments(self, fragments: List[Fragment]) -> None:
        """Compute and store embeddings for a full list of fragments."""
        self._embeddings.clear()
        self._fragments.clear()

        texts: List[str] = []
        ids: List[str] = []
        for frag in fragments:
            texts.append(frag.content)
            ids.append(frag.fragment_id)
            self._fragments[frag.fragment_id] = frag

        if not texts:
            return

        embeddings = self._embed_batch(texts)
        for fid, vec in zip(ids, embeddings):
            self._embeddings[fid] = vec

    def search(self, query: str, top_k: int = 5) -> List[Tuple[Fragment, float]]:
        """Return up to *top_k* (Fragment, cosine_similarity) pairs."""
        if not self._embeddings:
            return []

        query_vec = self.embed_text(query)
        if query_vec is None:
            return []

        candidates: List[Tuple[str, float]] = []
        for fid, emb in self._embeddings.items():
            sim = float(np.dot(query_vec, emb))
            candidates.append((fid, sim))

        candidates.sort(key=lambda x: x[1], reverse=True)
        return [(self._fragments[fid], score) for fid, score in candidates[:top_k]]

    def embed_text(self, text: str) -> Optional[np.ndarray]:
        """Embed a single text string into a normalised vector."""
        self._ensure_model()
        try:
            if self._model is not None:
                vec = self._model.encode(text, normalize_embeddings=True)
                return np.asarray(vec, dtype=np.float64)
            return _simple_embed(text)
        except Exception:
            warnings.warn("Embedding failed, falling back to simple embed", stacklevel=2)
            return _simple_embed(text)

    def add_fragment(self, fragment: Fragment) -> None:
        """Incrementally add a single fragment to the embedding index."""
        self._ensure_model()
        vec = self.embed_text(fragment.content)
        if vec is not None:
            self._fragments[fragment.fragment_id] = fragment
            self._embeddings[fragment.fragment_id] = vec

    def remove_fragment(self, fragment_id: str) -> None:
        """Remove a fragment from the embedding index by its ID."""
        self._fragments.pop(fragment_id, None)
        self._embeddings.pop(fragment_id, None)

    # ---- internals --------------------------------------------------------

    def _ensure_model(self) -> None:
        if self._model is not None or self._fallback:
            return
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore[import-untyped]
            self._model = SentenceTransformer(self._model_name)
        except ImportError:
            warnings.warn(
                "sentence-transformers not installed — using fallback embeddings. "
                "Install with `pip install sentence-transformers` for real embeddings.",
                stacklevel=2,
            )
            self._fallback = True
        except Exception as exc:
            warnings.warn(
                f"Failed to load SentenceTransformer model '{self._model_name}': {exc} — "
                f"using fallback embeddings.",
                stacklevel=2,
            )
            self._fallback = True

    def _embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        """Embed a batch of texts."""
        self._ensure_model()
        try:
            if self._model is not None:
                embeddings = self._model.encode(texts, normalize_embeddings=True)
                return [np.asarray(v, dtype=np.float64) for v in embeddings]
            return [_simple_embed(t) for t in texts]
        except Exception:
            warnings.warn("Batch embedding failed, falling back per-item", stacklevel=2)
            return [_simple_embed(t) for t in texts]

    @property
    def fragment_count(self) -> int:
        return len(self._fragments)

    @property
    def embedding_dimension(self) -> int:
        if self._embeddings:
            return next(iter(self._embeddings.values())).shape[0]
        return _FALLBACK_EMBED_DIM

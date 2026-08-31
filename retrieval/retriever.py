"""Hybrid retriever — fuses BM25 keyword scores with embedding-based semantic scores."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

from memory_fragments.config import RetrieverConfig, default_config
from memory_fragments.models import Fragment
from memory_fragments.retrieval.indexer import BM25Indexer, EmbeddingIndexer

if TYPE_CHECKING:
    from memory_fragments.engine.conflict import ConflictReport


# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------


def _min_max_normalise(
    scored: List[Tuple[Fragment, float]],
) -> List[Tuple[Fragment, float]]:
    """In-place min‑max normalisation to [0, 1]."""
    if not scored:
        return scored
    scores = [s for _, s in scored]
    lo, hi = min(scores), max(scores)
    span = hi - lo
    if span < 1e-12:
        return [(f, 1.0) for f, _ in scored]
    return [(f, (s - lo) / span) for f, s in scored]


# ---------------------------------------------------------------------------
# HybridRetriever
# ---------------------------------------------------------------------------


class HybridRetriever:
    """Hybrid search combining keyword (BM25) and semantic (embedding) retrieval.

    The two result lists are individually min‑max normalised and then fused via a
    weighted average configured through *RetrieverConfig*.
    """

    def __init__(self, config: Optional[RetrieverConfig] = None) -> None:
        self._config = config or default_config.retriever

        self.bm25 = BM25Indexer()
        self.embedding = EmbeddingIndexer()

    # ---- public API -------------------------------------------------------

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
    ) -> List[Tuple[Fragment, float]]:
        """Hybrid search: fuse BM25 + embedding results.

        Results below the configured *semantic_threshold* are filtered out after
        fusion.
        """
        k = top_k or self._config.top_k
        cutoff = max(k * 2, 10)  # fetch extra candidates for fusion

        bm25_results = self.bm25.search(query, cutoff)
        embed_results = self.embedding.search(query, cutoff)

        return self._fuse(bm25_results, embed_results, k)

    def retrieve_with_conflicts(
        self,
        query: str,
        archive_fragments: Optional[List[Fragment]] = None,
        top_k: Optional[int] = None,
        conflict_threshold: Optional[float] = None,
    ) -> Tuple[List[Tuple[Fragment, float]], List[ConflictReport]]:
        """Hybrid search with conflict detection among results.

        Performs a standard hybrid retrieval, then scans every returned
        fragment against the full *archive_fragments* list for embedding-based
        conflicts.  This is an O(n × m) operation where n = returned results
        and m = archive size.

        Args:
            query: Search query.
            archive_fragments: Full list of archive fragments (for conflict scan).
                When *None*, conflict detection is skipped.
            top_k: Number of results to return.
            conflict_threshold: Similarity threshold for conflict detection
                (default: from ``EvaluatorConfig.conflict_similarity_threshold``).

        Returns:
            (results, conflict_reports)
        """
        from memory_fragments.engine.conflict import ConflictDetector

        results = self.retrieve(query, top_k)
        fragments = [f for f, _ in results]

        if archive_fragments is None:
            return results, []

        detector = ConflictDetector()
        reports: List[ConflictReport] = []
        for frag in fragments:
            report = detector.scan_fragment(frag, archive_fragments, conflict_threshold)
            if report.has_conflicts:
                reports.append(report)

        return results, reports

    def retrieve_semantic(
        self,
        query: str,
        top_k: Optional[int] = None,
    ) -> List[Tuple[Fragment, float]]:
        """Semantic‑only retrieval (embedding cosine similarity)."""
        k = top_k or self._config.top_k
        return self.embedding.search(query, k)

    def retrieve_keyword(
        self,
        query: str,
        top_k: Optional[int] = None,
    ) -> List[Tuple[Fragment, float]]:
        """Keyword‑only retrieval (BM25)."""
        k = top_k or self._config.top_k
        return self.bm25.search(query, k)

    # ---- fragment management ----------------------------------------------

    def add_fragment(self, fragment: Fragment) -> None:
        """Add a single fragment to both indexers."""
        self.bm25.add_fragment(fragment)
        self.embedding.add_fragment(fragment)

    def add_fragments(self, fragments: List[Fragment]) -> None:
        """Add multiple fragments to both indexers."""
        for frag in fragments:
            self.add_fragment(frag)

    def remove_fragment(self, fragment_id: str) -> None:
        """Remove a fragment from both indexers by ID."""
        self.bm25.remove_fragment(fragment_id)
        self.embedding.remove_fragment(fragment_id)

    def rebuild(self, fragments: List[Fragment]) -> None:
        """Full reindex — replaces all data in both indexers."""
        self.bm25.index_fragments(fragments)
        self.embedding.index_fragments(fragments)

    # ---- internals --------------------------------------------------------

    def _fuse(
        self,
        bm25_results: List[Tuple[Fragment, float]],
        embed_results: List[Tuple[Fragment, float]],
        top_k: int,
    ) -> List[Tuple[Fragment, float]]:
        """Min‑max normalise both result sets, then weighted‑average fuse."""
        bm25_norm = _min_max_normalise(bm25_results)
        embed_norm = _min_max_normalise(embed_results)

        w_bm25 = self._config.bm25_weight
        w_embed = self._config.embedding_weight

        # Build a lookup of fragment_id -> normalised BM25 score
        bm25_lookup: Dict[str, float] = {f.fragment_id: s for f, s in bm25_norm}
        embed_lookup: Dict[str, float] = {f.fragment_id: s for f, s in embed_norm}

        # Collect all unique fragment IDs from both result sets
        all_ids: Dict[str, Fragment] = {}
        for f, _ in bm25_results:
            all_ids[f.fragment_id] = f
        for f, _ in embed_results:
            all_ids[f.fragment_id] = f

        fused: List[Tuple[Fragment, float]] = []
        for fid, frag in all_ids.items():
            b_score = bm25_lookup.get(fid, 0.0)
            e_score = embed_lookup.get(fid, 0.0)
            combined = w_bm25 * b_score + w_embed * e_score
            fused.append((frag, combined))

        # Sort descending and apply semantic threshold
        fused.sort(key=lambda x: x[1], reverse=True)

        threshold = self._config.semantic_threshold
        if threshold > 0.0:
            # Keep results that either have significant embedding contribution
            # or are in the top_k even without it
            filtered = []
            for frag, score in fused:
                e_score = embed_lookup.get(frag.fragment_id, 0.0)
                if e_score >= threshold or score > 0.0:
                    filtered.append((frag, score))
            fused = filtered

        return fused[:top_k]

    @property
    def config(self) -> RetrieverConfig:
        """Return the active retriever configuration."""
        return self._config

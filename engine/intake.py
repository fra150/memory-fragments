"""Intake Verifier — lightweight pre-filter for the Dispatcher.

Checks if certified fragments exist that can handle a query before
the Dispatcher decides a path. Uses only embedding similarity + BM25 —
NO LLM calls. Latency target: < 5ms.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from memory_fragments.models import Fragment
from memory_fragments.models.quality import QualitySource, QualityProvenance
from memory_fragments.library.guardian import FragmentGuardian, QUALITY_THRESHOLD
from memory_fragments.retrieval.indexer import EmbeddingIndexer, BM25Indexer


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MIN_ASPECT_COUNT: int = 2
"""Minimum number of query aspects to extract (fallback for short queries)."""

DEFAULT_CANDIDATE_THRESHOLD: float = 0.35
"""Minimum similarity for a fragment to be considered a candidate."""


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------


@dataclass
class IntakeResult:
    """Result of an Intake Verifier scan."""

    should_proceed: bool
    """True if there are certified candidates worth passing to the Dispatcher."""

    candidates: List[Fragment] = field(default_factory=list)
    """Certified fragments that match the query aspects."""

    aspects: List[str] = field(default_factory=list)
    """Query aspects extracted from the query."""

    max_similarity: float = 0.0
    """Maximum similarity score among candidates."""

    matched_aspects: List[str] = field(default_factory=list)
    """Query aspects that have coverage in the archive."""

    missing_aspects: List[str] = field(default_factory=list)
    """Query aspects with NO coverage in the archive."""

    scan_time_ms: float = 0.0
    """Time taken for the scan."""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "should_proceed": self.should_proceed,
            "candidates": [f.fragment_id for f in self.candidates],
            "candidate_count": len(self.candidates),
            "aspects": self.aspects,
            "max_similarity": round(self.max_similarity, 4),
            "matched_aspects": self.matched_aspects,
            "missing_aspects": self.missing_aspects,
            "scan_time_ms": round(self.scan_time_ms, 2),
        }

    def __repr__(self) -> str:
        return (
            f"IntakeResult(proceed={self.should_proceed}, "
            f"candidates={len(self.candidates)}, "
            f"aspects={len(self.aspects)}, "
            f"matched={len(self.matched_aspects)}, "
            f"missing={len(self.missing_aspects)})"
        )


# ---------------------------------------------------------------------------
# Intake Verifier
# ---------------------------------------------------------------------------


class IntakeVerifier:
    """Lightweight pre-filter at the entrance of the Dispatch pipeline.

    No LLM calls — only embedding similarity + BM25. Communicates with
    the Guardian to check that candidates are certified before passing
    them forward.

    Usage::

        verifier = IntakeVerifier(guardian, archive)
        result = verifier.scan(query)
        if result.should_proceed:
            dispatcher.dispatch(result.candidates, query)
        else:
            generate_from_scratch(query)
    """

    def __init__(
        self,
        guardian: FragmentGuardian,
        archive: "StaticArchive",
        threshold: float = DEFAULT_CANDIDATE_THRESHOLD,
        min_aspects: int = MIN_ASPECT_COUNT,
    ) -> None:
        """
        Args:
            guardian: Guardian instance for certification checks.
            archive: StaticArchive to scan for fragments.
            threshold: Minimum similarity for candidates.
            min_aspects: Minimum query aspects to extract.
        """
        self._guardian = guardian
        self._archive = archive
        self._threshold = threshold
        self._min_aspects = min_aspects

        # Lazy-loaded embedder (same pattern as Evaluator/ConflictDetector)
        self._embedder: Optional[EmbeddingIndexer] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def scan(self, query: str, top_k: int = 10) -> IntakeResult:
        """Scan the archive for certified fragments matching the query.

        Args:
            query: The input query to scan for.
            top_k: Maximum candidates to return.

        Returns:
            IntakeResult with candidates and aspect coverage info.
        """
        start = time.perf_counter()

        # 1. Extract query aspects
        aspects = self._extract_aspects(query)

        # 2. Find candidate fragments via embedding + BM25
        candidates = self._find_candidates(query, aspects, top_k)

        # 3. Filter by Guardian certification
        certified = self._filter_certified(candidates)

        # 4. Determine aspect coverage
        matched_aspects = self._compute_aspect_coverage(aspects, certified)
        missing_aspects = [a for a in aspects if a not in matched_aspects]

        # 5. Decision
        should_proceed = len(certified) > 0 and (
            len(matched_aspects) >= len(aspects) * 0.4  # At least 40% aspect coverage
        )

        elapsed = (time.perf_counter() - start) * 1000

        return IntakeResult(
            should_proceed=should_proceed,
            candidates=certified[:top_k],
            aspects=aspects,
            max_similarity=max(
                (self._compute_similarity(query, f.content) for f in certified),
                default=0.0,
            ),
            matched_aspects=matched_aspects,
            missing_aspects=missing_aspects,
            scan_time_ms=elapsed,
        )

    def has_certified_for(self, query: str) -> bool:
        """Quick check: does the archive have certified fragments for this query?

        Faster than full scan — stops at first match.
        """
        aspects = self._extract_aspects(query)
        all_fragments = self._archive.list_all()
        for aspect in aspects:
            for frag in all_fragments:
                if self._is_certified(frag) and self._aspect_matches(aspect, frag):
                    return True
        return False

    def certified_count(self, query: str) -> int:
        """Count certified fragments matching the query."""
        result = self.scan(query, top_k=100)
        return len(result.candidates)

    # ------------------------------------------------------------------
    # Aspect extraction
    # ------------------------------------------------------------------

    def _extract_aspects(self, query: str) -> List[str]:
        """Extract query aspects via embedding + semantic chunking.

        For short queries (< 5 tokens), forces minimum aspect count
        by splitting on semantic boundaries.
        """
        tokens = query.lower().split()

        # For very short queries, use n-gram aspects
        if len(tokens) < 5:
            aspects = []
            # Single tokens as aspects
            for t in tokens:
                if len(t) > 2:  # Skip very short tokens
                    aspects.append(t)
            # Bigrams for context
            for i in range(len(tokens) - 1):
                aspects.append(f"{tokens[i]} {tokens[i+1]}")
            # Force minimum count
            while len(aspects) < self._min_aspects and len(query) > 2:
                aspects.append(query[: len(query) // 2])
                aspects.append(query[len(query) // 2 :])
                break
            # Order-preserving dedup
            seen: Set[str] = set()
            deduped: List[str] = []
            for a in aspects:
                if a not in seen:
                    seen.add(a)
                    deduped.append(a)
            return deduped[:5]

        # For longer queries, extract key phrases
        # Use simple heuristics: nouns, capitalized terms, quoted phrases
        aspects = []

        # Quoted phrases
        quotes = re.findall(r'"([^"]+)"', query)
        aspects.extend(quotes)

        # Capitalized terms (potential named entities)
        caps = re.findall(r'\b[A-Z][a-z]+\b', query)
        aspects.extend(caps)

        # Key terms (remove stopwords, take remaining)
        stopwords = {
            "the", "a", "an", "is", "are", "was", "were", "in", "on",
            "at", "to", "for", "of", "with", "by", "from", "as",
            "and", "or", "but", "not", "be", "been", "being",
            "have", "has", "had", "do", "does", "did", "will",
            "would", "could", "should", "may", "might", "can",
            "shall", "about", "into", "through", "during", "before",
            "after", "above", "below", "between", "out", "off",
            "over", "under", "again", "further", "then", "once",
            "here", "there", "when", "where", "why", "how",
            "all", "each", "every", "both", "few", "more",
            "most", "other", "some", "such", "no", "nor", "not",
            "only", "own", "same", "so", "than", "too", "very",
            "just", "because", "as", "until", "while", "if",
        }

        key_terms = [t for t in tokens if t not in stopwords and len(t) > 2]
        aspects.extend(key_terms[:6])  # Max 6 key terms

        # Order-preserving dedup
        seen: Set[str] = set()
        deduped: List[str] = []
        for a in aspects:
            if a not in seen:
                seen.add(a)
                deduped.append(a)
        return deduped[:10]

    # ------------------------------------------------------------------
    # Candidate finding
    # ------------------------------------------------------------------

    def _find_candidates(
        self, query: str, aspects: List[str], top_k: int
    ) -> List[Fragment]:
        """Find candidate fragments via embedding similarity."""
        all_fragments = self._archive.list_all()
        if not all_fragments:
            return []

        scored: List[Tuple[Fragment, float]] = []
        for frag in all_fragments:
            sim = self._compute_similarity(query, frag.content)
            if sim >= self._threshold:
                scored.append((frag, sim))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [f for f, _ in scored[:top_k]]

    # ------------------------------------------------------------------
    # Certification filter
    # ------------------------------------------------------------------

    def _filter_certified(self, fragments: List[Fragment]) -> List[Fragment]:
        """Filter to only fragments that pass the Guardian."""
        certified = []
        for frag in fragments:
            threshold = self._guardian.get_threshold_for(frag)
            if frag.metadata.quality >= threshold:
                certified.append(frag)
        return certified

    def _is_certified(self, fragment: Fragment) -> bool:
        """Check if a single fragment passes the Guardian."""
        threshold = self._guardian.get_threshold_for(fragment)
        return fragment.metadata.quality >= threshold

    # ------------------------------------------------------------------
    # Similarity & coverage
    # ------------------------------------------------------------------

    def _compute_similarity(self, text_a: str, text_b: str) -> float:
        """Compute query→document similarity using asymmetric word coverage.

        For query-document matching, uses **query coverage** (fraction of
        query words found in the document) as the primary metric. This is
        asymmetric: a short query against a long document scores well when
        most query terms match, unlike symmetric Jaccard which penalizes
        length differences.

        When a real embedding model is available (positive cosine similarity
        for related texts), boosts with the embedding score. When only fallback
        random projections are available (near-zero dot products), relies on
        query coverage alone.
        """
        # Handle empty inputs
        if not text_a or not text_b:
            return 0.0

        # Tokenize
        words_a = set(text_a.lower().split())
        words_b = set(text_b.lower().split())
        if not words_a or not words_b:
            return 0.0

        # Identify which text is the query (shorter = query)
        # Compute asymmetric query coverage
        if len(words_a) <= len(words_b):
            # text_a is the query
            query_words = words_a
            doc_words = words_b
        else:
            # text_b is the query
            query_words = words_b
            doc_words = words_a

        # Query coverage: fraction of query words found in the document
        intersection = query_words & doc_words
        query_coverage = len(intersection) / len(query_words)

        # Embedding-based semantic similarity (when available).
        # With real sentence-transformers, related texts have positive cosine
        # similarity. With the deterministic fallback (_simple_embed), dot
        # products are near-zero regardless of semantic relatedness, so we
        # only trust the embedding score when it clearly exceeds the query
        # coverage.
        try:
            vec_a = self._get_embedding(text_a)
            vec_b = self._get_embedding(text_b)
            if vec_a is not None and vec_b is not None:
                emb_sim = float(np.dot(vec_a, vec_b))
                # Real embeddings give positive meaningful similarity;
                # fallback random projections hover near zero.
                if emb_sim > query_coverage:
                    return emb_sim
        except Exception:
            # Swallow embedding errors silently — fallback to word coverage
            logger.debug("Embedding similarity computation failed, using word coverage fallback")
            pass

        return query_coverage

    def _get_embedding(self, text: str) -> Optional[np.ndarray]:
        """Get embedding vector, lazy-loading the embedder."""
        if self._embedder is None:
            self._embedder = EmbeddingIndexer()
        try:
            return self._embedder.embed_text(text)
        except Exception:
            return None

    def _compute_aspect_coverage(
        self, aspects: List[str], fragments: List[Fragment]
    ) -> List[str]:
        """Determine which query aspects are covered by certified fragments."""
        matched = []
        for aspect in aspects:
            for frag in fragments:
                if self._aspect_matches(aspect, frag):
                    matched.append(aspect)
                    break
        return matched

    @staticmethod
    def _aspect_matches(aspect: str, fragment: Fragment) -> bool:
        """Check if a fragment covers a query aspect."""
        aspect_lower = aspect.lower()
        content_lower = fragment.content.lower()

        # Direct substring match
        if aspect_lower in content_lower:
            return True

        # Topic match
        if aspect_lower in fragment.metadata.topic.lower():
            return True

        # Tag match
        for tag in fragment.metadata.tags:
            if aspect_lower in tag.lower():
                return True

        return False

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def threshold(self) -> float:
        return self._threshold

    def __repr__(self) -> str:
        return f"IntakeVerifier(threshold={self._threshold})"

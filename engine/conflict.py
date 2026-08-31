"""Cross-fragment conflict detection — embedding similarity scan + NLI contradiction check.

Detects when two fragments in the archive contain contradictory or inconsistent
information. Uses embedding cosine similarity as a fast pre-filter, then optionally
runs NLI (Natural Language Inference) for deeper analysis.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from memory_fragments.config import EvaluatorConfig, default_config
from memory_fragments.models import Fragment


@dataclass
class ConflictReport:
    """Result of a conflict scan between fragments."""

    fragment_id: str
    """The fragment being checked."""

    conflicts: List[ConflictEntry] = field(default_factory=list)
    """List of conflicts found with other fragments."""

    scan_timestamp: str = ""
    """ISO-8601 timestamp of the scan."""

    total_checked: int = 0
    """How many fragments were compared."""

    @property
    def has_conflicts(self) -> bool:
        return len(self.conflicts) > 0

    @property
    def max_contradiction_score(self) -> float:
        if not self.conflicts:
            return 0.0
        return max(c.contradiction_score for c in self.conflicts)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "fragment_id": self.fragment_id,
            "conflicts": [c.to_dict() for c in self.conflicts],
            "scan_timestamp": self.scan_timestamp,
            "total_checked": self.total_checked,
            "has_conflicts": self.has_conflicts,
            "max_contradiction_score": self.max_contradiction_score,
        }

    def __repr__(self) -> str:
        return (
            f"ConflictReport(fragment={self.fragment_id}, "
            f"conflicts={len(self.conflicts)}, "
            f"checked={self.total_checked})"
        )


@dataclass
class ConflictEntry:
    """A single conflict between two fragments."""

    other_fragment_id: str
    """The fragment this fragment conflicts with."""

    similarity: float
    """Embedding cosine similarity between the two fragments (0.0-1.0)."""

    contradiction_score: float = 0.0
    """NLI-based contradiction likelihood (0.0-1.0, 0.0 = not checked)."""

    overlap_excerpt: str = ""
    """Short overlapping text excerpt for human review."""

    detection_method: str = "embedding_similarity"
    """How the conflict was detected: 'embedding_similarity', 'nli_contradiction'."""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "other_fragment_id": self.other_fragment_id,
            "similarity": round(self.similarity, 4),
            "contradiction_score": round(self.contradiction_score, 4),
            "overlap_excerpt": self.overlap_excerpt[:200],
            "detection_method": self.detection_method,
        }


class ConflictDetector:
    """Detects conflicts between fragments using embedding similarity + optional NLI.

    The detector works in three stages:
    1. Embed all fragments (if not already indexed)
    2. For each fragment, find neighbors with similarity > threshold
    3. Optionally run NLI on high-similarity pairs to detect contradictions
    """

    def __init__(
        self,
        config: Optional[EvaluatorConfig] = None,
    ) -> None:
        self._config = config or default_config.evaluator
        self._embedding_model = None  # Lazy-loaded SentenceTransformer
        self._embedding_fallback = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def scan_fragment(
        self,
        fragment: Fragment,
        candidates: List[Fragment],
        threshold: Optional[float] = None,
    ) -> ConflictReport:
        """Scan a single fragment against a list of candidates for conflicts.

        Args:
            fragment: The fragment to check.
            candidates: Other fragments to compare against.
            threshold: Similarity threshold (default: from config).

        Returns:
            ConflictReport with any conflicts found.
        """
        from datetime import datetime, timezone

        effective_threshold = threshold or self._config.conflict_similarity_threshold
        fragment_vec = self._embed(fragment.content)
        report = ConflictReport(
            fragment_id=fragment.fragment_id,
            scan_timestamp=datetime.now(timezone.utc).isoformat(),
            total_checked=len(candidates),
        )

        for candidate in candidates:
            if candidate.fragment_id == fragment.fragment_id:
                continue

            candidate_vec = self._embed(candidate.content)
            if candidate_vec is None or fragment_vec is None:
                continue

            similarity = float(np.dot(fragment_vec, candidate_vec))

            if similarity >= effective_threshold:
                entry = ConflictEntry(
                    other_fragment_id=candidate.fragment_id,
                    similarity=round(similarity, 4),
                    overlap_excerpt=self._extract_overlap(fragment.content, candidate.content),
                    detection_method="embedding_similarity",
                )

                # Optionally run NLI contradiction check
                if (self._config.nli_contradiction_threshold > 0
                    and similarity >= self._config.nli_contradiction_threshold):
                    contradiction = self._detect_contradiction_nli(
                        fragment.content, candidate.content
                    )
                    entry.contradiction_score = contradiction
                    if contradiction > 0.5:
                        entry.detection_method = "nli_contradiction"

                report.conflicts.append(entry)

        return report

    def scan_archive(
        self,
        archive_fragments: List[Fragment],
        threshold: Optional[float] = None,
    ) -> List[ConflictReport]:
        """Scan all fragments in an archive for pairwise conflicts.

        Args:
            archive_fragments: All fragments in the archive.
            threshold: Similarity threshold (default: from config).

        Returns:
            List of ConflictReports (one per fragment with conflicts).
            Fragments with no conflicts are omitted.
        """
        reports: List[ConflictReport] = []
        for fragment in archive_fragments:
            candidates = [f for f in archive_fragments if f.fragment_id != fragment.fragment_id]
            report = self.scan_fragment(fragment, candidates, threshold)
            if report.has_conflicts:
                reports.append(report)
        return reports

    def has_conflict_with(
        self,
        fragment: Fragment,
        archive_fragments: List[Fragment],
        threshold: Optional[float] = None,
    ) -> bool:
        """Quick check: does *fragment* conflict with any archive fragment?"""
        report = self.scan_fragment(fragment, archive_fragments, threshold)
        return report.has_conflicts

    # ------------------------------------------------------------------
    # Embedding helpers
    # ------------------------------------------------------------------

    def _embed(self, text: str):
        """Get unit-normalized embedding vector for text."""
        self._ensure_model()
        try:
            if self._embedding_model is not None:
                vec = self._embedding_model.encode(text, normalize_embeddings=True)
                return np.asarray(vec, dtype=np.float64)
            return self._simple_embed(text)
        except Exception:
            return self._simple_embed(text)

    def _ensure_model(self) -> None:
        """Lazy-load sentence-transformers model."""
        if self._embedding_model is not None or self._embedding_fallback:
            return
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore[import-untyped]
            model_name = self._config.embedding_model
            self._embedding_model = SentenceTransformer(model_name)
        except ImportError:
            import warnings
            warnings.warn(
                "sentence-transformers not installed — using fallback embeddings. "
                "Install with `pip install sentence-transformers` for real embeddings.",
                stacklevel=2,
            )
            self._embedding_fallback = True
        except Exception as exc:
            import warnings
            warnings.warn(
                f"Failed to load model '{self._config.embedding_model}': {exc} — "
                f"using fallback embeddings.",
                stacklevel=2,
            )
            self._embedding_fallback = True

    @staticmethod
    def _simple_embed(text: str, dim: int = 384) -> np.ndarray:
        """Deterministic fallback embedding (same as EmbeddingIndexer)."""
        import hashlib
        # Use MD5 hash for deterministic pseudo-random seed
        hash_bytes = hashlib.md5(text.encode("utf-8")).digest()
        seed = int.from_bytes(hash_bytes[:4], "big")
        rng = np.random.RandomState(seed)
        vec = rng.randn(dim)
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    # ------------------------------------------------------------------
    # NLI contradiction detection (lazy, heuristic placeholder)
    # ------------------------------------------------------------------
    #
    # TODO: Replace with real ONNX-based NLI (BART-large-MNLI) in a future card.
    # The current heuristic checks for contradictory keyword pairs (e.g., "increase"/"decrease",
    # "buy"/"sell", "accept"/"reject"). It is retained for immediate use but is known to be
    # less precise than a proper NLI model.
    #
    # For production use, integrate: sentence-transformers + ONNX Runtime with
    # facebook/bart-large-mnli model, or use an OpenAI/Anthropic API call for NLI.
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_contradiction_nli(text_a: str, text_b: str) -> float:
        """Detect contradiction between two texts using NLI.

        Currently a heuristic-based placeholder. Checks for contradictory keyword pairs
        (e.g., "increase"/"decrease", "buy"/"sell", "accept"/"reject"). Real ONNX-based
        NLI (BART-large-MNLI) will be integrated here in a future card.

        Args:
            text_a: First text
            text_b: Second text

        Returns:
            Contradiction likelihood (0.0-1.0).
        """
        # Simple heuristic: check for contradictory keyword pairs
        contradiction_pairs = [
            ("increase", "decrease"), ("buy", "sell"), ("start", "stop"),
            ("true", "false"), ("yes", "no"), ("on", "off"),
            ("positive", "negative"), ("success", "failure"),
            ("win", "lose"), ("gain", "loss"), ("up", "down"),
            ("high", "low"), ("hot", "cold"), ("fast", "slow"),
            ("always", "never"), ("all", "none"), ("every", "no"),
            ("include", "exclude"), ("add", "remove"), ("create", "delete"),
            ("enter", "exit"), ("open", "close"), ("begin", "end"),
            ("accept", "reject"), ("approve", "deny"),
        ]

        words_a = set(text_a.lower().split())
        words_b = set(text_b.lower().split())

        contradictions = 0
        for w1, w2 in contradiction_pairs:
            if (w1 in words_a and w2 in words_b) or (w2 in words_a and w1 in words_b):
                contradictions += 1

        return min(contradictions * 0.25, 0.8)  # Max 0.8 from heuristic

    @staticmethod
    def _extract_overlap(text_a: str, text_b: str, max_chars: int = 100) -> str:
        """Extract a short overlapping excerpt between two texts."""
        a_words = text_a.split()
        b_words = text_b.split()
        overlap = []
        for w in a_words:
            if w in b_words:
                overlap.append(w)
                if len(" ".join(overlap)) > max_chars:
                    break

        excerpt = " ".join(overlap)
        if len(excerpt) > max_chars:
            excerpt = excerpt[:max_chars] + "..."
        return excerpt if len(excerpt) > 10 else ""

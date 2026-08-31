"""Cassetto — a domain-specific library shelf with quality guardian.

Each Cassetto wraps an Archive + Retriever + AppealTrialSpace and enforces
the FragmentGuardian at its entrance.  Only fragments with quality ≥ 0.80
(or improved to that level) are stored.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from memory_fragments.archive.static import StaticArchive
from memory_fragments.archive.appeal_space import AppealTrialSpace
from memory_fragments.retrieval.retriever import HybridRetriever
from memory_fragments.retrieval.indexer import BM25Indexer, EmbeddingIndexer
from memory_fragments.models import (
    Appeal,
    AppealMetrics,
    AppealOperation,
    AppealStatus,
    Fragment,
    FragmentConditions,
    FragmentMetadata,
    FragmentStatus,
    GenealogyGraph,
)
from memory_fragments.governance.api import GovernanceAPI, GovernanceReport
from memory_fragments.engine.evaluator import Evaluator
from memory_fragments.engine.diff_explain import DiffExplainEngine
from memory_fragments.engine.composer import Composer
from memory_fragments.config import (
    default_config,
    GovernanceConfig,
    RetrieverConfig,
)

from memory_fragments.library.guardian import FragmentGuardian, QUALITY_THRESHOLD
from memory_fragments.models.quality import QualitySource


@dataclass
class CassettoConfig:
    """Configuration for a single Cassetto (library shelf)."""

    name: str
    """Unique shelf name (e.g. ``"medical"``, ``"legal"``)."""

    topic: str = ""
    """Default topic assigned to fragments in this shelf."""

    guardian_threshold: float = QUALITY_THRESHOLD
    """Minimum quality for fragments entering this shelf."""

    guardian_improver: Any = None
    """LLM improvver callable (optional)."""

    guardian_source_overrides: Optional[Dict[str, float]] = None
    """QualitySource-specific threshold overrides (key = QualitySource value name).

    Example: ``{"user_claimed": 0.98, "independently_verified": 0.65}``
    Falls back to ``QUALITY_THRESHOLD_OVERRIDES`` per-source defaults when ``None``.
    """

    governance_config: Optional[GovernanceConfig] = None
    """Per-shelf governance rules (falls back to global default)."""

    retriever_config: Optional[RetrieverConfig] = None
    """Per-shelf retriever tuning (falls back to global default)."""


class Cassetto:
    """A domain-specific knowledge shelf.

    Each Cassetto is self-contained::
        - Own ``StaticArchive``, ``HybridRetriever``, ``AppealTrialSpace``
        - FragmentGuardian at the entrance (quality ≥ 80 %)
        - Optional per-shelf ``GovernanceAPI``

    Fragments that fail the guardian are tracked in a rejection log.
    """

    def __init__(self, config: CassettoConfig) -> None:
        self.config = config

        # Guardian at the door — with optional QualitySource overrides
        source_overrides = self._parse_source_overrides(config.guardian_source_overrides)
        self.guardian = FragmentGuardian(
            threshold=config.guardian_threshold,
            improver=config.guardian_improver,
            source_overrides=source_overrides,
        )

        # Core components
        self.archive = StaticArchive()
        self.retriever = HybridRetriever(config.retriever_config)
        self.appeal_space = AppealTrialSpace()
        self._genealogy = GenealogyGraph()

        # Engine components (lazy — created on first governance use)
        self._evaluator: Optional[Evaluator] = None
        self._diff_explain: Optional[DiffExplainEngine] = None
        self._composer: Optional[Composer] = None
        self._governance: Optional[GovernanceAPI] = None

        # Rejection log
        self._rejected: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Fragment management (guardian-enforced)
    # ------------------------------------------------------------------

    def add(self, fragment: Fragment) -> bool:
        """Add a fragment guarded by the quality threshold.

        Returns ``True`` if the fragment was accepted into the shelf,
        ``False`` if it was rejected (even after LLM improvement).
        """
        # Tag with shelf topic if empty
        if not fragment.metadata.topic and self.config.topic:
            fragment.metadata.topic = self.config.topic

        accepted, result = self.guardian.guard(fragment)
        if accepted and result is not None:
            self.archive.add(result)
            self.retriever.add_fragment(result)
            return True

        self._log_rejection(fragment, "quality_below_threshold")
        return False

    def add_many(self, fragments: List[Fragment]) -> Tuple[int, int]:
        """Batch add. Returns (accepted_count, rejected_count)."""
        ok = 0
        nope = 0
        for f in fragments:
            if self.add(f):
                ok += 1
            else:
                nope += 1
        return ok, nope

    def get(self, fragment_id: str) -> Optional[Fragment]:
        return self.archive.get(fragment_id)

    def search(self, query: str, top_k: int = 5) -> List[Tuple[Fragment, float]]:
        """Search within this shelf only (domain-scoped)."""
        return self.retriever.retrieve(query, top_k)

    def count(self) -> int:
        return self.archive.count()

    def rejected_count(self) -> int:
        return len(self._rejected)

    def rejected_log(self) -> List[Dict[str, Any]]:
        """Return the rejection history (for monitoring / reporting)."""
        return list(self._rejected)

    # ------------------------------------------------------------------
    # Appeal & Governance
    # ------------------------------------------------------------------

    def _ensure_engine(self) -> None:
        if self._evaluator is None:
            self._evaluator = Evaluator()
        if self._diff_explain is None:
            self._diff_explain = DiffExplainEngine()
        if self._composer is None:
            self._composer = Composer()

    def governance(self) -> GovernanceAPI:
        """Lazily initialise and return the per-shelf GovernanceAPI."""
        self._ensure_engine()
        if self._governance is None:
            gc = self.config.governance_config or default_config.governance
            self._governance = GovernanceAPI(
                archive=self.archive,
                appeal_space=self.appeal_space,
                evaluator=self._evaluator,
                diff_explain=self._diff_explain,
                graph=self._genealogy,
                config=gc,
            )
        return self._governance

    def create_appeal(
        self,
        appeal_id: str,
        sources: List[str],
        ops: List[AppealOperation],
    ) -> Appeal:
        """Create an Appeal scoped to this shelf's fragment IDs."""
        return self.appeal_space.create_appeal(appeal_id, sources, ops)

    def compose(self, query: str, top_k: int = 5) -> str:
        """Compose a response from this shelf's fragments for *query*."""
        self._ensure_engine()
        results = self.search(query, top_k)
        fragments = [f for f, _ in results]
        return self._composer.compose(fragments, query)

    # ------------------------------------------------------------------
    # Rejection log helpers
    # ------------------------------------------------------------------

    def _log_rejection(self, fragment: Fragment, reason: str) -> None:
        self._rejected.append({
            "fragment_id": fragment.fragment_id,
            "quality": fragment.metadata.quality,
            "topic": fragment.metadata.topic,
            "reason": reason,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

    # ------------------------------------------------------------------
    # Source override parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_source_overrides(
        raw: Optional[Dict[str, float]],
    ) -> Optional[Dict[QualitySource, float]]:
        """Convert string-keyed override dict to QualitySource-keyed dict.

        Silently skips keys that don't match any ``QualitySource`` value.
        Returns ``None`` when *raw* is ``None`` (so the guardian uses its
        built-in defaults).
        """
        if raw is None:
            return None
        overrides: Dict[QualitySource, float] = {}
        for key, val in raw.items():
            try:
                overrides[QualitySource(key)] = val
            except ValueError:
                pass  # Skip invalid source names
        return overrides if overrides else None

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return {
            "config": {
                "name": self.config.name,
                "topic": self.config.topic,
                "guardian_threshold": self.config.guardian_threshold,
            },
            "archive": self.archive.to_dict(),
            "appeal_space": self.appeal_space.to_dict(),
            "genealogy": self._genealogy.to_dict(),
            "rejected_count": len(self._rejected),
        }

    def __repr__(self) -> str:
        return (
            f"Cassetto(name={self.config.name!r}, "
            f"fragments={self.count()}, "
            f"rejected={self.rejected_count()})"
        )

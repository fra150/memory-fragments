"""Fragment Guardian — quality gatekeeper for the Cassetto system.

Only fragments with quality ≥ 0.80 pass through the guardian.
Below-threshold fragments are sent to an LLM improver; if the
improved version still falls short the fragment is discarded.

The guardian supports QualitySource-aware thresholds: different
QualitySource values trigger different minimum quality requirements.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from memory_fragments.models import Fragment, FragmentMetadata
from memory_fragments.models.quality import QualityProvenance, QualitySource

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

QUALITY_THRESHOLD: float = 0.80
"""Default minimum quality — used when no QualitySource override applies."""

# QualitySource-specific threshold overrides.
# When a fragment's QualitySource matches a key, that threshold is used instead
# of the default. The idea: untrusted sources need higher scores to pass.
QUALITY_THRESHOLD_OVERRIDES: Dict[QualitySource, float] = {
    QualitySource.USER_CLAIMED: 0.95,       # Unverified -> needs very high score
    QualitySource.LLM_SELF_REPORTED: 0.85,  # LLM assigned its own score -> extra scrutiny
    QualitySource.HEURISTIC_BOOSTED: 0.85,  # Simple rule boost -> still needs check
    QualitySource.INDEPENDENTLY_VERIFIED: 0.70,  # Third-party verified -> can be lower
    QualitySource.MAJORITY_VOTE: 0.75,           # Consensus -> trusted
    QualitySource.RASTRELLO_DISCOVERED: 0.80,    # Pattern scanner -> moderate trust
    QualitySource.EVALUATOR_COMPUTED: 0.75,      # Automated metrics -> moderately trusted
    QualitySource.INHERITED: 0.85,          # Inherited from sources -> need extra care
}

# ---------------------------------------------------------------------------
# LLM Improver interface
# ---------------------------------------------------------------------------

FragmentImprover = Callable[[Fragment], Optional[Fragment]]
"""Signature for an LLM-based (or heuristic) fragment improver."""


def _default_improver(fragment: Fragment) -> Optional[Fragment]:
    """Heuristic fallback improver.

    Boosts quality by adding structured context. If the boost pushes
    quality past the threshold the fragment is accepted; otherwise
    a real LLM improver (``AnthropicFragmentImprover``) is needed.
    """
    from memory_fragments.library.improver import HeuristicImprover
    return HeuristicImprover(boost=0.15)(fragment)


# ---------------------------------------------------------------------------
# Guardian
# ---------------------------------------------------------------------------


class FragmentGuardian:
    """Quality gatekeeper at the entrance of every Cassetto.

    The guardian enforces a two-stage filter::

        Fragment ──► quality ≥ threshold?
          ├── YES ──► accepted (pass-through)
          └── NO  ──► LLM improver
                        ├── success (≥ threshold) ──► accepted
                        └── failure (< threshold) ──► discarded

    The threshold used depends on the fragment's **QualitySource**:
    - Fragments with QualitySource.USER_CLAIMED need ≥ 0.95
    - Fragments with QualitySource.INDEPENDENTLY_VERIFIED need ≥ 0.70
    - Etc.
    """

    def __init__(
        self,
        threshold: float = QUALITY_THRESHOLD,
        improver: Optional[FragmentImprover] = None,
        source_overrides: Optional[Dict[QualitySource, float]] = None,
    ) -> None:
        self._default_threshold = threshold
        self._improver = improver or _default_improver
        self._source_overrides = source_overrides or dict(QUALITY_THRESHOLD_OVERRIDES)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def guard(self, fragment: Fragment) -> Tuple[bool, Optional[Fragment]]:
        """Run the quality gate with QualitySource-aware thresholds.

        Uses provenance information from the fragment (if available)
        to determine the effective threshold.

        Returns
        -------
        (accepted, result)
            *accepted* is ``True`` when the fragment (or its improved
            version) meets the effective quality threshold.
            *result* is the fragment to store (maybe improved) on
            success, or ``None`` on final rejection.
        """
        effective_threshold = self._get_effective_threshold(fragment)

        if fragment.metadata.quality >= effective_threshold:
            return True, copy.deepcopy(fragment)

        # Attempt improvement
        improved = self._try_improve(fragment)
        if improved is not None:
            improved_threshold = self._get_effective_threshold(improved)
            if improved.metadata.quality >= improved_threshold:
                return True, improved

        return False, None

    def guard_many(
        self, fragments: List[Fragment]
    ) -> Tuple[List[Fragment], List[Fragment]]:
        """Batch guard — returns (accepted, rejected)."""
        accepted: List[Fragment] = []
        rejected: List[Fragment] = []
        for f in fragments:
            ok, result = self.guard(f)
            if ok and result is not None:
                accepted.append(result)
            else:
                rejected.append(f)
        return accepted, rejected

    # ------------------------------------------------------------------
    # Source-aware threshold helpers
    # ------------------------------------------------------------------

    def _get_effective_threshold(self, fragment: Fragment) -> float:
        """Determine the effective quality threshold for a fragment.

        Checks provenance first, then falls back to the default threshold.
        """
        source = self._extract_source(fragment)
        if source is not None and source in self._source_overrides:
            return self._source_overrides[source]
        return self._default_threshold

    @staticmethod
    def _extract_source(fragment: Fragment) -> Optional[QualitySource]:
        """Extract the QualitySource from a fragment's provenance, if available."""
        if fragment.metadata.provenance is not None:
            return fragment.metadata.provenance.final_source
        return None

    def get_threshold_for(self, fragment: Fragment) -> float:
        """Public helper to query the effective threshold for a fragment."""
        return self._get_effective_threshold(fragment)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def threshold(self) -> float:
        return self._default_threshold

    @threshold.setter
    def threshold(self, value: float) -> None:
        self._default_threshold = value

    @property
    def source_overrides(self) -> Dict[QualitySource, float]:
        return dict(self._source_overrides)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _try_improve(self, fragment: Fragment) -> Optional[Fragment]:
        """Call the improver; return ``None`` on failure."""
        try:
            return self._improver(fragment)
        except Exception:
            return None

    def __repr__(self) -> str:
        overrides_str = ", ".join(
            f"{s.value}={t}" for s, t in sorted(
                self._source_overrides.items(), key=lambda x: x[1], reverse=True
            )
        )
        return (
            f"FragmentGuardian(default={self._default_threshold}, "
            f"overrides={{{overrides_str}}})"
        )

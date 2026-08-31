"""Quality provenance tracking — traces how quality scores are assigned and verified."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional


class QualitySource(str, Enum):
    """Identifies how a quality score was obtained."""

    USER_CLAIMED = "user_claimed"  # User-provided, unverified
    LLM_SELF_REPORTED = "llm_self_reported"  # LLM assigned its own score
    HEURISTIC_BOOSTED = "heuristic_boosted"  # Simple rule-based boost
    INDEPENDENTLY_VERIFIED = "independently_verified"  # Verified by independent evaluation
    MAJORITY_VOTE = "majority_vote"  # 3-agent consensus
    RASTRELLO_DISCOVERED = "rastrello_discovered"  # Discovered by the Rastrello (Rake) pattern scanner
    EVALUATOR_COMPUTED = "evaluator_computed"  # Computed by Evaluator metrics
    INHERITED = "inherited"  # Averaged from source fragments


@dataclass
class QualityEvaluation:
    """A single quality evaluation at one point in the pipeline."""

    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    model_id: str = "unknown"  # e.g. "phi-3-mini-4k", "gemma-2b", "gpt-4o", "heuristic", "user"
    model_version: str = ""  # e.g. "1.0" or git hash
    score: float = 0.5  # quality score assigned in this evaluation
    source: QualitySource = QualitySource.USER_CLAIMED
    metadata: Dict[str, Any] = field(
        default_factory=dict
    )  # extra: temperature, prompt, confidence, etc.

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "model_id": self.model_id,
            "model_version": self.model_version,
            "score": round(self.score, 4),
            "source": self.source.value,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> QualityEvaluation:
        ts = data.get("timestamp")
        if isinstance(ts, str):
            ts = datetime.fromisoformat(ts)
        return cls(
            timestamp=ts or datetime.now(timezone.utc),
            model_id=data.get("model_id", "unknown"),
            model_version=data.get("model_version", ""),
            score=float(data.get("score", 0.5)),
            source=QualitySource(data.get("source", "user_claimed")),
            metadata=data.get("metadata", {}),
        )


@dataclass
class QualityProvenance:
    """Full provenance trail for a fragment's quality score.

    Tracks every evaluation that contributed to the final quality,
    plus the aggregated consensus result when majority voting is used.
    """

    evaluations: List[QualityEvaluation] = field(default_factory=list)
    final_quality: float = 0.5
    final_source: QualitySource = QualitySource.USER_CLAIMED
    consensus_score: Optional[float] = None
    consensus_method: Optional[str] = None  # e.g. "majority_vote_3", "average", "max"

    def add_evaluation(self, evaluation: QualityEvaluation) -> None:
        """Append an evaluation and update final_quality/final_source."""
        self.evaluations.append(evaluation)
        self.final_quality = evaluation.score
        self.final_source = evaluation.source

    def set_consensus(self, score: float, method: str) -> None:
        """Record the consensus result from majority voting."""
        self.consensus_score = score
        self.consensus_method = method
        self.final_quality = score
        self.final_source = QualitySource.MAJORITY_VOTE

    def get_latest_evaluation(self) -> Optional[QualityEvaluation]:
        return self.evaluations[-1] if self.evaluations else None

    def get_evaluations_by_source(self, source: QualitySource) -> List[QualityEvaluation]:
        return [e for e in self.evaluations if e.source == source]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "evaluations": [e.to_dict() for e in self.evaluations],
            "final_quality": round(self.final_quality, 4),
            "final_source": self.final_source.value,
            "consensus_score": round(self.consensus_score, 4) if self.consensus_score is not None else None,
            "consensus_method": self.consensus_method,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> QualityProvenance:
        return cls(
            evaluations=[QualityEvaluation.from_dict(e) for e in data.get("evaluations", [])],
            final_quality=float(data.get("final_quality", 0.5)),
            final_source=QualitySource(data.get("final_source", "user_claimed")),
            consensus_score=float(data["consensus_score"]) if data.get("consensus_score") is not None else None,
            consensus_method=data.get("consensus_method"),
        )

    def __repr__(self) -> str:
        return (
            f"QualityProvenance(evaluations={len(self.evaluations)}, "
            f"final={self.final_quality:.2f}/{self.final_source.value})"
        )

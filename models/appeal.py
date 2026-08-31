"""Appeal data model — a temporary sandbox proposal to modify fragments."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional


class OperationType(str, Enum):
    MERGE = "merge"
    REORDER = "reorder"
    PRUNE = "prune"
    ANNOTATE = "annotate"
    SPLIT = "split"
    REWRITE = "rewrite"
    CONDENSE = "condense"


class AppealStatus(str, Enum):
    DRAFT = "draft"
    PENDING_USER_APPROVAL = "pending_user_approval"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXPIRED = "expired"


@dataclass
class AppealOperation:
    """A single transformation operation within an Appeal."""

    op_type: OperationType
    params: Dict[str, Any] = field(default_factory=dict)
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "op_type": self.op_type.value,
            "params": self.params,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> AppealOperation:
        return cls(
            op_type=OperationType(data["op_type"]),
            params=data.get("params", {}),
            description=data.get("description", ""),
        )


@dataclass
class AppealMetrics:
    """Metrics computed by the Automatic Evaluator for an Appeal."""

    delta_token: int = 0
    coverage: float = 0.0
    risk: float = 0.0
    aggregate_score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "delta_token": self.delta_token,
            "coverage": round(self.coverage, 4),
            "risk": round(self.risk, 4),
            "aggregate_score": round(self.aggregate_score, 4),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> AppealMetrics:
        return cls(
            delta_token=data.get("delta_token", 0),
            coverage=data.get("coverage", 0.0),
            risk=data.get("risk", 0.0),
            aggregate_score=data.get("aggregate_score", 0.0),
        )


@dataclass
class AppealDiff:
    """Structured diff between an Appeal and its source fragments."""

    added: List[str] = field(default_factory=list)
    removed: List[str] = field(default_factory=list)
    modified: List[Dict[str, Any]] = field(default_factory=list)
    reordered: List[int] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "added": self.added,
            "removed": self.removed,
            "modified": self.modified,
            "reordered": self.reordered,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> AppealDiff:
        return cls(
            added=data.get("added", []),
            removed=data.get("removed", []),
            modified=data.get("modified", []),
            reordered=data.get("reordered", []),
        )


@dataclass
class Appeal:
    """
    A temporary sandbox proposal in the Appeal Trial Space.

    An Appeal represents a user-proposed modification to the Static Archive.
    It carries source fragment references, a sequence of operations, a diff,
    computed metrics, and a status that transitions through the governance workflow.
    """

    appeal_id: str
    sources: List[str] = field(default_factory=list)
    ops: List[AppealOperation] = field(default_factory=list)
    diff: AppealDiff = field(default_factory=AppealDiff)
    metrics: AppealMetrics = field(default_factory=AppealMetrics)
    status: AppealStatus = AppealStatus.DRAFT
    proposed_content: str = ""
    explanation: str = ""
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    resolved_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.appeal_id,
            "sources": self.sources,
            "ops": [op.to_dict() for op in self.ops],
            "diff": self.diff.to_dict(),
            "metrics": self.metrics.to_dict(),
            "status": self.status.value,
            "proposed_content": self.proposed_content,
            "explanation": self.explanation,
            "created_at": self.created_at.isoformat(),
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Appeal:
        created_at = datetime.fromisoformat(data["created_at"]) if isinstance(data.get("created_at"), str) else data.get("created_at", datetime.now(timezone.utc))
        resolved_at = None
        if data.get("resolved_at"):
            resolved_at = datetime.fromisoformat(data["resolved_at"]) if isinstance(data["resolved_at"], str) else data["resolved_at"]
        return cls(
            appeal_id=data["id"],
            sources=data.get("sources", []),
            ops=[AppealOperation.from_dict(op) for op in data.get("ops", [])],
            diff=AppealDiff.from_dict(data.get("diff", {})),
            metrics=AppealMetrics.from_dict(data.get("metrics", {})),
            status=AppealStatus(data.get("status", "draft")),
            proposed_content=data.get("proposed_content", ""),
            explanation=data.get("explanation", ""),
            created_at=created_at,
            resolved_at=resolved_at,
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, default=str)

    def __repr__(self) -> str:
        return f"Appeal(id={self.appeal_id}, sources={self.sources}, status={self.status.value})"

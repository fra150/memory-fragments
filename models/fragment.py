"""Fragment data model — a discrete, conditionally-activated knowledge unit."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from memory_fragments.models.quality import QualityProvenance


class FragmentStatus(str, Enum):
    ACTIVE = "active"
    ARCHIVED = "archived"
    DEPRECATED = "deprecated"
    PENDING_REVIEW = "pending_review"


@dataclass
class FragmentConditions:
    """Activation conditions for a fragment."""

    keywords: List[str] = field(default_factory=list)
    semantic_threshold: float = 0.72
    booleans: Dict[str, bool] = field(default_factory=dict)
    regex_patterns: List[str] = field(default_factory=list)
    require_all_keywords: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "keywords": self.keywords,
            "semantic_threshold": self.semantic_threshold,
            "booleans": self.booleans,
            "regex_patterns": self.regex_patterns,
            "require_all_keywords": self.require_all_keywords,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> FragmentConditions:
        return cls(
            keywords=data.get("keywords", []),
            semantic_threshold=data.get("semantic_threshold", 0.72),
            booleans=data.get("booleans", {}),
            regex_patterns=data.get("regex_patterns", []),
            require_all_keywords=data.get("require_all_keywords", False),
        )


@dataclass
class FragmentMetadata:
    """Metadata attached to a fragment."""

    topic: str = ""
    source: str = ""
    date: Optional[datetime] = None
    quality: float = 0.5
    provenance: Optional[QualityProvenance] = None
    author: str = ""
    tags: List[str] = field(default_factory=list)
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "topic": self.topic,
            "source": self.source,
            "date": self.date.isoformat() if self.date else None,
            "quality": self.quality,
            "provenance": self.provenance.to_dict() if self.provenance else None,
            "author": self.author,
            "tags": self.tags,
            **self.extra,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> FragmentMetadata:
        date_str = data.get("date")
        date = None
        if date_str:
            try:
                date = datetime.fromisoformat(date_str)
            except (ValueError, TypeError):
                date = None
        prov_data = data.get("provenance")
        provenance = QualityProvenance.from_dict(prov_data) if prov_data else None
        extra = {k: v for k, v in data.items() if k not in ("topic", "source", "date", "quality", "provenance", "author", "tags")}
        return cls(
            topic=data.get("topic", ""),
            source=data.get("source", ""),
            date=date,
            quality=data.get("quality", 0.5),
            provenance=provenance,
            author=data.get("author", ""),
            tags=data.get("tags", []),
            extra=extra,
        )


@dataclass
class Fragment:
    """
    A discrete, immutable knowledge unit in the Static Archive.

    Each fragment carries content, metadata, activation conditions,
    and an optional list of parent fragment IDs for genealogy tracking.
    """

    fragment_id: str
    content: str
    metadata: FragmentMetadata = field(default_factory=FragmentMetadata)
    conditions: FragmentConditions = field(default_factory=FragmentConditions)
    parents: List[str] = field(default_factory=list)
    status: FragmentStatus = FragmentStatus.ACTIVE
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    checksum: str = ""

    def __post_init__(self) -> None:
        if not self.checksum:
            self.checksum = self._compute_checksum()
        if not (0.0 <= self.metadata.quality <= 1.0):
            self.metadata.quality = max(0.0, min(1.0, self.metadata.quality))

    def _compute_checksum(self) -> str:
        raw = f"{self.fragment_id}::{self.content}::{self.created_at.isoformat()}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.fragment_id,
            "content": self.content,
            "metadata": self.metadata.to_dict(),
            "conditions": self.conditions.to_dict(),
            "parents": self.parents,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "checksum": self.checksum,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Fragment:
        created_at = datetime.fromisoformat(data["created_at"]) if isinstance(data.get("created_at"), str) else data.get("created_at", datetime.now(timezone.utc))
        return cls(
            fragment_id=data["id"],
            content=data["content"],
            metadata=FragmentMetadata.from_dict(data.get("metadata", {})),
            conditions=FragmentConditions.from_dict(data.get("conditions", {})),
            parents=data.get("parents", []),
            status=FragmentStatus(data.get("status", "active")),
            created_at=created_at,
            checksum=data.get("checksum", ""),
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, default=str)

    def __repr__(self) -> str:
        return f"Fragment(id={self.fragment_id}, topic={self.metadata.topic}, quality={self.metadata.quality})"

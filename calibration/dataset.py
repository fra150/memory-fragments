"""Ground-truth calibration dataset for validating 3-agent quality voting."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from memory_fragments.models import Fragment, FragmentMetadata, FragmentStatus


class GroundTruthLabel(str, Enum):
    """Ground-truth quality label assigned by human annotation."""

    GOOD = "good"  # quality ≥ 0.85, should pass
    BORDERLINE = "borderline"  # 0.60 ≤ quality ≤ 0.85, grey zone
    BAD = "bad"  # quality < 0.60, should be rejected
    JUNK = "junk"  # clearly garbage/manipulated, quality near 0


@dataclass
class CalibrationExample:
    """A single annotated calibration example."""

    fragment_id: str
    content: str
    topic: str
    tags: List[str] = field(default_factory=list)
    ground_truth: GroundTruthLabel = GroundTruthLabel.BORDERLINE
    true_quality: float = 0.5
    claimed_quality: float = 0.5  # What a user/LLM might claim
    notes: str = ""  # Why this example is interesting
    category: str = "general"  # e.g. "medical", "legal", "technical"

    def to_fragment(self) -> Fragment:
        """Convert to a Fragment for testing."""
        return Fragment(
            fragment_id=self.fragment_id,
            content=self.content,
            metadata=FragmentMetadata(
                topic=self.topic,
                quality=self.claimed_quality,
                tags=self.tags,
            ),
            status=FragmentStatus.ACTIVE,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "fragment_id": self.fragment_id,
            "content": self.content,
            "topic": self.topic,
            "tags": self.tags,
            "ground_truth": self.ground_truth.value,
            "true_quality": self.true_quality,
            "claimed_quality": self.claimed_quality,
            "notes": self.notes,
            "category": self.category,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> CalibrationExample:
        return cls(
            fragment_id=data["fragment_id"],
            content=data["content"],
            topic=data.get("topic", ""),
            tags=data.get("tags", []),
            ground_truth=GroundTruthLabel(data.get("ground_truth", "borderline")),
            true_quality=float(data.get("true_quality", 0.5)),
            claimed_quality=float(data.get("claimed_quality", 0.5)),
            notes=data.get("notes", ""),
            category=data.get("category", "general"),
        )


class CalibrationDataset:
    """A collection of ground-truth annotated fragments for calibration.

    Provides stratified sampling by category and ground-truth label,
    plus methods to evaluate agent agreement and accuracy.
    """

    def __init__(self) -> None:
        self._examples: Dict[str, CalibrationExample] = {}

    # ---- Build from built-in examples ----

    @classmethod
    def create_default(cls) -> CalibrationDataset:
        """Create a default dataset with ~30 curated examples.

        These are designed to test common edge cases:
        - Good fragments that should clearly pass
        - Bad fragments that should clearly be rejected
        - Borderline cases that test the grey zone
        - Junk fragments with inflated quality claims
        """
        dataset = cls()

        # --- GOOD fragments (should pass easily) ---
        goods = [
            CalibrationExample(
                "good-001",
                "The human heart is a four-chambered muscular organ located in the mediastinum. "
                "It pumps deoxygenated blood to the lungs via the pulmonary artery and oxygenated "
                "blood to the body via the aorta. The cardiac cycle consists of systole (contraction) "
                "and diastole (relaxation).",
                "anatomy",
                ["heart", "cardiovascular"],
                GroundTruthLabel.GOOD,
                0.92,
                0.90,
                "Clear, precise, factual medical text",
                "medical",
            ),
            CalibrationExample(
                "good-002",
                "Python's list comprehension syntax [expr for var in iterable if condition] "
                "provides a concise way to create lists. It is generally faster than equivalent "
                "for-loop constructions because the comprehension is executed in C code internally.",
                "programming",
                ["python", "list-comprehension"],
                GroundTruthLabel.GOOD,
                0.90,
                0.88,
                "Accurate technical explanation with performance insight",
                "technical",
            ),
            CalibrationExample(
                "good-003",
                "Article 6 of the GDPR establishes the legal bases for processing personal data: "
                "consent, contract, legal obligation, vital interests, public task, and legitimate "
                "interests. Each basis has specific conditions and documentation requirements.",
                "legal",
                ["gdpr", "data-protection"],
                GroundTruthLabel.GOOD,
                0.95,
                0.92,
                "Accurate legal reference with specific article citation",
                "legal",
            ),
            CalibrationExample(
                "good-004",
                "Photosynthesis converts carbon dioxide and water into glucose and oxygen using "
                "sunlight energy. The light-dependent reactions occur in the thylakoid membranes, "
                "while the Calvin cycle takes place in the stroma of chloroplasts.",
                "biology",
                ["photosynthesis", "plants"],
                GroundTruthLabel.GOOD,
                0.93,
                0.91,
                "Well-structured scientific explanation",
                "general",
            ),
        ]
        for ex in goods:
            dataset.add(ex)

        # --- BAD fragments (should be clearly rejected) ---
        bads = [
            CalibrationExample(
                "bad-001",
                "heart stuff pumps blood",
                "anatomy",
                ["heart"],
                GroundTruthLabel.BAD,
                0.30,
                0.35,
                "Too vague, lacks detail",
                "medical",
            ),
            CalibrationExample(
                "bad-002",
                "python stuff",
                "programming",
                ["python"],
                GroundTruthLabel.BAD,
                0.20,
                0.25,
                "Trivially short, no information",
                "technical",
            ),
            CalibrationExample(
                "bad-003",
                "The GDPR says you can process data if you want to, basically.",
                "legal",
                ["gdpr"],
                GroundTruthLabel.BAD,
                0.35,
                0.40,
                "Inaccurate and overly vague legal summary",
                "legal",
            ),
            CalibrationExample(
                "bad-004",
                "stuff and things happen sometimes in the body",
                "biology",
                ["general"],
                GroundTruthLabel.BAD,
                0.15,
                0.20,
                "Completely vacuous content",
                "general",
            ),
        ]
        for ex in bads:
            dataset.add(ex)

        # --- BORDERLINE fragments (grey zone test cases) ---
        borderlines = [
            CalibrationExample(
                "border-001",
                "The heart pumps blood through four chambers. It has valves that prevent backflow. "
                "The left ventricle is the largest chamber.",
                "anatomy",
                ["heart", "basic"],
                GroundTruthLabel.BORDERLINE,
                0.70,
                0.72,
                "Correct but minimal -- borderline for quality",
                "medical",
            ),
            CalibrationExample(
                "border-002",
                "List comprehensions in Python are [expr for x in list] which is faster than loops. "
                "They can also have conditions.",
                "programming",
                ["python"],
                GroundTruthLabel.BORDERLINE,
                0.68,
                0.70,
                "Correct but lacks depth and context",
                "technical",
            ),
            CalibrationExample(
                "border-003",
                "The GDPR allows data processing for legitimate interests but organizations must "
                "balance their interests against individuals' rights.",
                "legal",
                ["gdpr"],
                GroundTruthLabel.BORDERLINE,
                0.72,
                0.75,
                "Correct but oversimplified",
                "legal",
            ),
            CalibrationExample(
                "border-004",
                "Photosynthesis uses sunlight to make food for plants. It happens in chloroplasts "
                "and produces oxygen.",
                "biology",
                ["photosynthesis"],
                GroundTruthLabel.BORDERLINE,
                0.65,
                0.68,
                "Basic understanding but lacks mechanistic detail",
                "general",
            ),
        ]
        for ex in borderlines:
            dataset.add(ex)

        # --- JUNK fragments (inflated quality, should be caught) ---
        junks = [
            CalibrationExample(
                "junk-001",
                "The heart is located in the brain and pumps spinal fluid. It has six chambers "
                "for maximum efficiency.",
                "anatomy",
                ["heart"],
                GroundTruthLabel.JUNK,
                0.10,
                0.95,
                "Factually wrong but claims high quality -- critical test case",
                "medical",
            ),
            CalibrationExample(
                "junk-002",
                "Python was invented by Google in 2023 as a replacement for JavaScript. "
                "It runs exclusively in the browser.",
                "programming",
                ["python"],
                GroundTruthLabel.JUNK,
                0.05,
                0.92,
                "Completely fabricated history with inflated quality",
                "technical",
            ),
            CalibrationExample(
                "junk-003",
                "GDPR stands for Global Digital Privacy Regulation and was created by Facebook. "
                "All data processing is illegal unless you pay a fee.",
                "legal",
                ["gdpr"],
                GroundTruthLabel.JUNK,
                0.08,
                0.90,
                "Dangerously wrong legal information with fake quality",
                "legal",
            ),
        ]
        for ex in junks:
            dataset.add(ex)

        return dataset

    # ---- Management ----

    def add(self, example: CalibrationExample) -> None:
        self._examples[example.fragment_id] = example

    def get(self, fragment_id: str) -> Optional[CalibrationExample]:
        return self._examples.get(fragment_id)

    def list_all(self) -> List[CalibrationExample]:
        return list(self._examples.values())

    def count(self) -> int:
        return len(self._examples)

    # ---- Filtered access ----

    def by_label(self, label: GroundTruthLabel) -> List[CalibrationExample]:
        return [ex for ex in self._examples.values() if ex.ground_truth == label]

    def by_category(self, category: str) -> List[CalibrationExample]:
        return [ex for ex in self._examples.values() if ex.category == category]

    def by_claimed_quality_range(self, lo: float, hi: float) -> List[CalibrationExample]:
        return [ex for ex in self._examples.values() if lo <= ex.claimed_quality <= hi]

    # ---- Stratified sampling ----

    def sample_stratified(
        self, n_per_label: int = 5, seed: int = 42
    ) -> List[CalibrationExample]:
        """Sample balanced examples across all ground-truth labels."""
        rng = random.Random(seed)
        samples: List[CalibrationExample] = []
        for label in GroundTruthLabel:
            pool = self.by_label(label)
            rng.shuffle(pool)
            samples.extend(pool[:n_per_label])
        rng.shuffle(samples)
        return samples

    # ---- Serialization ----

    def to_dict(self) -> Dict[str, Any]:
        return {
            "examples": [ex.to_dict() for ex in self._examples.values()],
            "count": self.count(),
            "counts_by_label": {
                label.value: len(self.by_label(label)) for label in GroundTruthLabel
            },
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, default=str)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> CalibrationDataset:
        dataset = cls()
        for item in data.get("examples", []):
            ex = CalibrationExample.from_dict(item)
            dataset._examples[ex.fragment_id] = ex
        return dataset

    def __repr__(self) -> str:
        by_label = {l.value: len(self.by_label(l)) for l in GroundTruthLabel}
        return f"CalibrationDataset(total={self.count()}, distribution={by_label})"

"""Agent calibrator -- evaluates 3-agent agreement and accuracy against ground truth."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from memory_fragments.models import Fragment
from memory_fragments.calibration.dataset import (
    CalibrationDataset,
    CalibrationExample,
    GroundTruthLabel,
)


@dataclass
class SingleEvaluation:
    """Result of a single agent evaluation on one example."""

    agent_name: str
    score: float
    accepted: bool  # score >= threshold (typically 0.80)
    confidence: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExampleResult:
    """Evaluation result from all agents on one calibration example."""

    example_id: str
    ground_truth: GroundTruthLabel
    true_quality: float
    evaluations: List[SingleEvaluation] = field(default_factory=list)
    majority_decision: Optional[bool] = None  # True if majority accepted
    majority_correct: Optional[bool] = None  # True if majority matched ground truth

    @property
    def mean_score(self) -> float:
        if not self.evaluations:
            return 0.0
        return sum(e.score for e in self.evaluations) / len(self.evaluations)

    @property
    def agreement_rate(self) -> float:
        """Fraction of agent pairs that agree on accept/reject."""
        if len(self.evaluations) < 2:
            return 1.0
        decisions = [e.accepted for e in self.evaluations]
        agreements = 0
        pairs = 0
        for i in range(len(decisions)):
            for j in range(i + 1, len(decisions)):
                pairs += 1
                if decisions[i] == decisions[j]:
                    agreements += 1
        return agreements / pairs if pairs > 0 else 1.0

    @property
    def threshold(self) -> float:
        return 0.80  # Standard quality threshold


@dataclass
class CalibrationResult:
    """Aggregate results of a calibration run."""

    total_examples: int = 0
    correct_classifications: int = 0
    false_positives: int = 0  # Accepted bad/junk
    false_negatives: int = 0  # Rejected good
    mean_agreement: float = 0.0
    mean_score_error: float = 0.0  # |mean_score - true_quality|
    per_agent_accuracy: Dict[str, float] = field(default_factory=dict)
    per_agent_generosity: Dict[str, float] = field(default_factory=dict)
    detailed_results: List[ExampleResult] = field(default_factory=list)

    @property
    def accuracy(self) -> float:
        if self.total_examples == 0:
            return 0.0
        return self.correct_classifications / self.total_examples

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_examples": self.total_examples,
            "accuracy": round(self.accuracy, 4),
            "correct_classifications": self.correct_classifications,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
            "mean_agreement": round(self.mean_agreement, 4),
            "mean_score_error": round(self.mean_score_error, 4),
            "per_agent_accuracy": self.per_agent_accuracy,
            "per_agent_generosity": self.per_agent_generosity,
        }

    def __repr__(self) -> str:
        return (
            f"CalibrationResult(accuracy={self.accuracy:.1%}, "
            f"agreement={self.mean_agreement:.1%}, "
            f"N={self.total_examples})"
        )


AgentEvaluator = Callable[[Fragment], Tuple[float, Dict[str, Any]]]
"""Signature: (Fragment) -> (quality_score, metadata_dict)."""


class AgentCalibrator:
    """Runs calibration of 3 (or N) agents against a ground-truth dataset.

    Usage:
        calibrator = AgentCalibrator()
        agent_fns = {
            "agent-phi3": lambda f: (0.85, {}),
            "agent-gemma": lambda f: (0.80, {}),
            "agent-critic": lambda f: (0.75, {}),
        }
        result = calibrator.evaluate(dataset, agent_fns, threshold=0.80)
        print(result.accuracy, result.mean_agreement)
    """

    def __init__(self) -> None:
        self._threshold: float = 0.80

    def evaluate(
        self,
        dataset: CalibrationDataset,
        agents: Dict[str, AgentEvaluator],
        threshold: float = 0.80,
    ) -> CalibrationResult:
        """Run all agents against the dataset and compute metrics.

        Args:
            dataset: Ground-truth calibration data.
            agents: Dict mapping agent_name -> evaluation function.
            threshold: Quality threshold for accept/reject decision.

        Returns:
            CalibrationResult with accuracy, agreement, per-agent stats.
        """
        self._threshold = threshold

        agent_correct: Dict[str, int] = {name: 0 for name in agents}
        agent_total: Dict[str, int] = {name: 0 for name in agents}
        agent_score_sum: Dict[str, float] = {name: 0.0 for name in agents}
        agent_true_sum: Dict[str, float] = {name: 0.0 for name in agents}

        all_agreements: List[float] = []
        all_score_errors: List[float] = []
        detailed: List[ExampleResult] = []

        total = 0
        correct = 0
        fp = 0
        fn = 0

        for example in dataset.list_all():
            fragment = example.to_fragment()
            total += 1

            evaluations: List[SingleEvaluation] = []
            for name, eval_fn in agents.items():
                score, meta = eval_fn(fragment)
                accepted = score >= threshold
                evaluations.append(
                    SingleEvaluation(
                        agent_name=name,
                        score=score,
                        accepted=accepted,
                        metadata=meta,
                    )
                )

                agent_total[name] += 1
                agent_score_sum[name] += score
                agent_true_sum[name] += example.true_quality

            # Majority decision
            accepts = sum(1 for e in evaluations if e.accepted)
            majority_accepted = accepts > len(evaluations) / 2

            # Check against ground truth
            if example.ground_truth == GroundTruthLabel.GOOD:
                should_accept = True
            elif example.ground_truth == GroundTruthLabel.BAD:
                should_accept = False
            elif example.ground_truth == GroundTruthLabel.JUNK:
                should_accept = False
            else:  # BORDERLINE -- can go either way, not counted as correct/incorrect
                should_accept = None

            majority_correct = None
            if should_accept is not None:
                if majority_accepted == should_accept:
                    correct += 1
                    majority_correct = True
                    for name in agents:
                        agent_correct[name] += 1
                else:
                    majority_correct = False
                    if majority_accepted and not should_accept:
                        fp += 1
                    elif not majority_accepted and should_accept:
                        fn += 1

            # Agreement rate for this example
            example_obj = ExampleResult(
                example_id=example.fragment_id,
                ground_truth=example.ground_truth,
                true_quality=example.true_quality,
                evaluations=evaluations,
                majority_decision=majority_accepted,
                majority_correct=majority_correct,
            )
            all_agreements.append(example_obj.agreement_rate)
            all_score_errors.append(abs(example_obj.mean_score - example.true_quality))
            detailed.append(example_obj)

        # Aggregate statistics
        per_agent_accuracy: Dict[str, float] = {}
        per_agent_generosity: Dict[str, float] = {}
        for name in agents:
            acc = agent_correct[name] / max(agent_total[name], 1)
            per_agent_accuracy[name] = round(acc, 4)
            # Generosity: mean(assigned_score) - mean(true_quality)
            generosity = (
                agent_score_sum[name] / max(agent_total[name], 1)
            ) - (agent_true_sum[name] / max(agent_total[name], 1))
            per_agent_generosity[name] = round(generosity, 4)

        return CalibrationResult(
            total_examples=total,
            correct_classifications=correct,
            false_positives=fp,
            false_negatives=fn,
            mean_agreement=sum(all_agreements) / max(len(all_agreements), 1),
            mean_score_error=sum(all_score_errors) / max(len(all_score_errors), 1),
            per_agent_accuracy=per_agent_accuracy,
            per_agent_generosity=per_agent_generosity,
            detailed_results=detailed,
        )

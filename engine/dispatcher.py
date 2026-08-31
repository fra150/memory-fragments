"""Dispatcher — strategic decision maker for query response paths.

The Dispatcher decides how to respond to a query:

- **compose**: existing certified fragments fully cover the query
- **partial**: partial coverage — compose what we have, generate the gap
- **generate**: insufficient coverage — generate from scratch

Uses the Intake Verifier for pre-filtering to find certified candidates
and extract query aspects. Then performs coverage analysis and cost-benefit
estimation to select the optimal path.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from memory_fragments.config import DispatcherConfig, default_config
from memory_fragments.models import Fragment
from memory_fragments.engine.intake import IntakeVerifier, IntakeResult


class DispatchPath(str, Enum):
    """The decision path for handling a query."""

    COMPOSE = "compose"
    """Full fragment composition — all query aspects are covered."""

    PARTIAL = "partial"
    """Partial composition — cover what we have, generate the gaps."""

    GENERATE = "generate"
    """Generate from scratch — insufficient certified coverage."""


@dataclass
class DispatchPlan:
    """The full plan produced by the Dispatcher for a query.

    Carries all context needed by downstream consumers (Modellatore,
    Generator, etc.) to execute the chosen path.
    """

    query: str
    """The original input query."""

    path: DispatchPath
    """The chosen execution path."""

    candidates: List[Fragment] = field(default_factory=list)
    """Certified fragments to use in composition."""

    aspects: List[str] = field(default_factory=list)
    """Query aspects extracted from the query."""

    covered_aspects: List[str] = field(default_factory=list)
    """Aspects covered by existing certified fragments."""

    missing_aspects: List[str] = field(default_factory=list)
    """Aspects not covered — need generation."""

    coverage_ratio: float = 0.0
    """Fraction of query aspects covered (0.0–1.0)."""

    estimated_compose_cost: float = 0.0
    """Estimated token cost of composition."""

    estimated_generate_cost: float = 0.0
    """Estimated token cost of generation from scratch."""

    path_reason: str = ""
    """Human-readable reason for the decision."""

    gap_description: str = ""
    """Description of gaps when path is PARTIAL."""

    intake_result: Optional[IntakeResult] = None
    """The IntakeResult that fed into this decision."""

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a plain dictionary for logging or API responses."""
        return {
            "query": self.query,
            "path": self.path.value,
            "candidates": [f.fragment_id for f in self.candidates],
            "candidate_count": len(self.candidates),
            "aspects": self.aspects,
            "covered_aspects": self.covered_aspects,
            "missing_aspects": self.missing_aspects,
            "coverage_ratio": round(self.coverage_ratio, 4),
            "estimated_compose_cost": round(self.estimated_compose_cost, 2),
            "estimated_generate_cost": round(self.estimated_generate_cost, 2),
            "path_reason": self.path_reason,
            "gap_description": self.gap_description,
        }

    def __repr__(self) -> str:
        return (
            f"DispatchPlan(path={self.path.value}, "
            f"coverage={self.coverage_ratio:.0%}, "
            f"candidates={len(self.candidates)}, "
            f"aspects={len(self.aspects)})"
        )


class Dispatcher:
    """Strategic decision maker for query response paths.

    Determines whether to compose from certified fragments, compose
    partially and generate gaps, or generate entirely from scratch.

    Usage::

        dispatcher = Dispatcher(intake_verifier)
        plan = dispatcher.dispatch(query)

        if plan.path == DispatchPath.COMPOSE:
            result = modellatore.compose(plan.candidates, query)
        elif plan.path == DispatchPath.PARTIAL:
            result = modellatore.compose_partial(
                plan.candidates, query, plan.gap_description
            )
        else:
            result = generate_from_scratch(query)
    """

    def __init__(
        self,
        intake: IntakeVerifier,
        config: Optional[DispatcherConfig] = None,
    ) -> None:
        """Initialise the Dispatcher.

        Args:
            intake: Intake Verifier for pre-filtering and aspect extraction.
            config: Configuration overrides. Falls back to default config.
        """
        self._intake = intake
        self._config = config or default_config.dispatcher

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def dispatch(self, query: str) -> DispatchPlan:
        """Produce a DispatchPlan for the given query.

        The plan includes the chosen path, candidates, coverage analysis,
        and cost-benefit estimates.

        Args:
            query: The input query to evaluate.

        Returns:
            A DispatchPlan with the decision and supporting data.
        """
        # 1. Run Intake Verifier to get candidates and aspects
        intake_result = self._intake.scan(query, top_k=20)

        if not intake_result.should_proceed or not intake_result.candidates:
            return self._plan_generate(
                query=query,
                intake_result=intake_result,
                reason="No certified candidates found by Intake Verifier",
            )

        # 2. Compute coverage ratio
        total_aspects = len(intake_result.aspects) or 1
        covered = len(intake_result.matched_aspects)
        coverage_ratio = covered / total_aspects

        # 3. Cost-benefit analysis
        compose_cost = self._estimate_compose_cost(
            intake_result.candidates, query
        )
        generate_cost = self._estimate_generate_cost(
            query, intake_result.aspects
        )

        # 4. Decide path based on coverage thresholds
        if coverage_ratio >= self._config.coverage_threshold_compose:
            return self._plan_compose(
                query=query,
                intake_result=intake_result,
                coverage_ratio=coverage_ratio,
                compose_cost=compose_cost,
                generate_cost=generate_cost,
            )

        elif coverage_ratio >= self._config.coverage_threshold_partial:
            return self._plan_partial(
                query=query,
                intake_result=intake_result,
                coverage_ratio=coverage_ratio,
                compose_cost=compose_cost,
                generate_cost=generate_cost,
            )

        else:
            return self._plan_generate(
                query=query,
                intake_result=intake_result,
                reason=(
                    f"Coverage too low "
                    f"({coverage_ratio:.1%} < "
                    f"{self._config.coverage_threshold_partial:.0%})"
                ),
            )

    def should_compose(self, query: str) -> bool:
        """Quick check: should this query be composed from fragments?

        Args:
            query: The input query.

        Returns:
            True if the path is COMPOSE or PARTIAL.
        """
        plan = self.dispatch(query)
        return plan.path in (DispatchPath.COMPOSE, DispatchPath.PARTIAL)

    def should_generate(self, query: str) -> bool:
        """Quick check: should this query be generated from scratch?

        Args:
            query: The input query.

        Returns:
            True if the path is GENERATE.
        """
        plan = self.dispatch(query)
        return plan.path == DispatchPath.GENERATE

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def config(self) -> DispatcherConfig:
        """The DispatcherConfig currently in use."""
        return self._config

    # ------------------------------------------------------------------
    # Path planners
    # ------------------------------------------------------------------

    def _plan_compose(
        self,
        query: str,
        intake_result: IntakeResult,
        coverage_ratio: float,
        compose_cost: float,
        generate_cost: float,
    ) -> DispatchPlan:
        """Plan for full composition — all aspects covered."""
        return DispatchPlan(
            query=query,
            path=DispatchPath.COMPOSE,
            candidates=intake_result.candidates,
            aspects=intake_result.aspects,
            covered_aspects=intake_result.matched_aspects,
            missing_aspects=[],
            coverage_ratio=coverage_ratio,
            estimated_compose_cost=compose_cost,
            estimated_generate_cost=generate_cost,
            path_reason=(
                f"Full composition: "
                f"{len(intake_result.matched_aspects)}/"
                f"{len(intake_result.aspects)} aspects covered "
                f"({coverage_ratio:.0%}), "
                f"compose cost ({compose_cost:.1f}) < "
                f"generate cost ({generate_cost:.1f})"
            ),
            intake_result=intake_result,
        )

    def _plan_partial(
        self,
        query: str,
        intake_result: IntakeResult,
        coverage_ratio: float,
        compose_cost: float,
        generate_cost: float,
    ) -> DispatchPlan:
        """Plan for partial composition with gap generation."""
        gap_desc = self._describe_gaps(intake_result.missing_aspects)

        return DispatchPlan(
            query=query,
            path=DispatchPath.PARTIAL,
            candidates=intake_result.candidates,
            aspects=intake_result.aspects,
            covered_aspects=intake_result.matched_aspects,
            missing_aspects=intake_result.missing_aspects,
            coverage_ratio=coverage_ratio,
            estimated_compose_cost=compose_cost,
            estimated_generate_cost=generate_cost,
            gap_description=gap_desc,
            path_reason=(
                f"Partial composition: "
                f"{len(intake_result.matched_aspects)}/"
                f"{len(intake_result.aspects)} aspects covered "
                f"({coverage_ratio:.0%}). "
                f"Missing: {intake_result.missing_aspects}. "
                f"Compose cost ({compose_cost:.1f}) < "
                f"generate cost ({generate_cost:.1f})"
            ),
            intake_result=intake_result,
        )

    def _plan_generate(
        self,
        query: str,
        intake_result: IntakeResult,
        reason: str,
    ) -> DispatchPlan:
        """Plan for full generation from scratch."""
        return DispatchPlan(
            query=query,
            path=DispatchPath.GENERATE,
            candidates=[],
            aspects=intake_result.aspects,
            covered_aspects=[],
            missing_aspects=intake_result.aspects,
            coverage_ratio=0.0,
            estimated_compose_cost=0.0,
            estimated_generate_cost=0.0,
            path_reason=f"Full generation: {reason}",
            intake_result=intake_result,
        )

    # ------------------------------------------------------------------
    # Cost-benefit estimation
    # ------------------------------------------------------------------

    def _estimate_compose_cost(
        self, candidates: List[Fragment], query: str
    ) -> float:
        """Estimate token cost of composition.

        Based on: number of fragments × average fragment token length
        + query overhead. This is a rough order-of-magnitude estimate,
        NOT a precise calculation.

        Args:
            candidates: The fragment candidates to compose.
            query: The original query.

        Returns:
            Estimated token cost as a float.
        """
        if not candidates:
            return float("inf")

        avg_fragment_len = (
            sum(len(f.content.split()) for f in candidates) / len(candidates)
        )
        query_len = len(query.split())

        # Composition cost = query + fragments + structural overhead
        cost = query_len + (avg_fragment_len * len(candidates) * 1.1)
        return cost * self._config.cost_token_estimate_factor

    def _estimate_generate_cost(
        self, query: str, aspects: List[str]
    ) -> float:
        """Estimate token cost of generating from scratch.

        Based on: query length × domain factor + aspects overhead.
        Generation is typically more expensive than composition.

        Args:
            query: The original query.
            aspects: The query aspects to cover.

        Returns:
            Estimated token cost as a float.
        """
        query_len = len(query.split())
        aspect_len = sum(len(a.split()) for a in aspects)

        # Generation cost = query + aspects + generation overhead.
        # Generation is ~3x more expensive than composition per unit.
        cost = (query_len + aspect_len) * 3.0
        return (
            cost
            * self._config.cost_token_estimate_factor
            * self._config.generate_cost_penalty
        )

    # ------------------------------------------------------------------
    # Gap description
    # ------------------------------------------------------------------

    @staticmethod
    def _describe_gaps(missing_aspects: List[str]) -> str:
        """Create a human-readable description of missing aspects.

        Args:
            missing_aspects: List of aspect names not covered.

        Returns:
            A natural-language description of the gaps.
        """
        if not missing_aspects:
            return ""
        if len(missing_aspects) == 1:
            return f"Missing aspect: {missing_aspects[0]}"
        *rest, last = missing_aspects
        aspects_str = ", ".join(rest)
        return f"Missing aspects: {aspects_str}, and {last}"

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"Dispatcher(compose≥{self._config.coverage_threshold_compose:.0%}, "
            f"partial≥{self._config.coverage_threshold_partial:.0%})"
        )

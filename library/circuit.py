"""3-Agent Majority Voting Circuit — production pipeline for quality verification.

The circuit implements the grey zone approach:

    quality < 0.60  →  REJECTED directly
    0.60 ≤ quality ≤ 0.90  →  GREY ZONE → 3 agents vote (majority wins)
    quality > 0.90  →  ACCEPTED (only if QualitySource != USER_CLAIMED)

    USER_CLAIMED fragments ALWAYS go through the 3-agent circuit,
    regardless of claimed score.
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple

from memory_fragments.models import Fragment, FragmentMetadata
from memory_fragments.models.quality import (
    QualityEvaluation,
    QualityProvenance,
    QualitySource,
)
from memory_fragments.calibration.agents import AgentConfig, MockQualityAgent
from memory_fragments.calibration.validator import AgentEvaluator

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

HARD_REJECT_THRESHOLD: float = 0.60
"""Fragments below this are rejected without agent voting."""

GREY_ZONE_LOW: float = 0.60
"""Start of the grey zone where agent voting is triggered."""

GREY_ZONE_HIGH: float = 0.90
"""End of the grey zone; above this, fragments pass without voting."""

VOTE_THRESHOLD: float = 0.80
"""Quality score that an agent must assign for a 'pass' vote."""

MIN_AGENTS: int = 2
"""Minimum number of agents that must respond for a valid vote."""

TOTAL_AGENTS: int = 3
"""Total number of agents in the circuit."""


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------


@dataclass
class VoteResult:
    """Result of a 3-agent majority vote on a fragment."""

    fragment_id: str
    accepted: bool
    """Whether the majority accepted the fragment."""

    votes_for: int = 0
    """Number of agents voting to accept."""

    votes_against: int = 0
    """Number of agents voting to reject."""

    agents_responded: int = 0
    """How many agents actually responded (may be < TOTAL_AGENTS on timeout/error)."""

    mean_score: float = 0.0
    """Average quality score across all responding agents."""

    min_score: float = 0.0
    """Minimum quality score assigned."""

    max_score: float = 0.0
    """Maximum quality score assigned."""

    consensus_reached: bool = False
    """True if agents agreed (all votes same direction, or 2-1 majority)."""

    fallback_used: bool = False
    """True if any agent fell back to mock due to unavailability."""

    agent_scores: Dict[str, float] = field(default_factory=dict)
    """Individual agent scores: {agent_name: score}."""

    agent_metadata: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    """Individual agent metadata: {agent_name: metadata_dict}."""

    quality_provenance: Optional[QualityProvenance] = None
    """Full provenance record for the vote."""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "fragment_id": self.fragment_id,
            "accepted": self.accepted,
            "votes_for": self.votes_for,
            "votes_against": self.votes_against,
            "agents_responded": self.agents_responded,
            "mean_score": round(self.mean_score, 4),
            "min_score": round(self.min_score, 4),
            "max_score": round(self.max_score, 4),
            "consensus_reached": self.consensus_reached,
            "fallback_used": self.fallback_used,
            "agent_scores": self.agent_scores,
        }

    def __repr__(self) -> str:
        return (
            f"VoteResult(fragment={self.fragment_id}, "
            f"accepted={self.accepted}, "
            f"votes={self.votes_for}/{self.votes_against}, "
            f"mean={self.mean_score:.2f})"
        )


# ---------------------------------------------------------------------------
# Circuit Exceptions
# ---------------------------------------------------------------------------


class CircuitError(Exception):
    """Base exception for circuit errors."""
    pass


class InsufficientAgentsError(CircuitError):
    """Raised when too few agents respond to form a majority."""
    pass


# ---------------------------------------------------------------------------
# Agent Circuit
# ---------------------------------------------------------------------------


class AgentCircuit:
    """Production 3-agent voting circuit with grey zone.

    The circuit handles the complete voting pipeline:
    1. Pre-filter (hard reject below threshold)
    2. Grey zone check (determine if voting is needed)
    3. Parallel-ish agent voting
    4. Majority decision
    5. Provenance tracking

    Usage::

        agents = create_mock_agents()
        circuit = AgentCircuit(agents)
        result = circuit.evaluate(fragment)

        if result.accepted:
            fragment = circuit.create_accepted_fragment(fragment, result)
            cassetto.add(fragment)
    """

    def __init__(
        self,
        agents: Dict[str, AgentEvaluator],
        hard_reject_threshold: float = HARD_REJECT_THRESHOLD,
        grey_zone_low: float = GREY_ZONE_LOW,
        grey_zone_high: float = GREY_ZONE_HIGH,
        vote_threshold: float = VOTE_THRESHOLD,
        min_agents: int = MIN_AGENTS,
        require_user_claimed_always_vote: bool = True,
    ) -> None:
        """
        Args:
            agents: Dict of {agent_name: evaluator_function}.
            hard_reject_threshold: Quality below this → instant reject.
            grey_zone_low: Start of grey zone (default 0.60).
            grey_zone_high: End of grey zone (default 0.90).
            vote_threshold: Minimum score per agent for a 'pass' vote (default 0.80).
            min_agents: Minimum agents needed for valid vote (default 2).
            require_user_claimed_always_vote: If True, USER_CLAIMED fragments
                always go through voting regardless of score.
        """
        if len(agents) < min_agents:
            raise ValueError(
                f"Need at least {min_agents} agents, got {len(agents)}"
            )

        self._agents = agents
        self._hard_reject_threshold = hard_reject_threshold
        self._grey_zone_low = grey_zone_low
        self._grey_zone_high = grey_zone_high
        self._vote_threshold = vote_threshold
        self._min_agents = min_agents
        self._require_user_claimed_always_vote = require_user_claimed_always_vote

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(self, fragment: Fragment) -> VoteResult:
        """Run the full 3-agent voting pipeline.

        Args:
            fragment: The fragment to evaluate.

        Returns:
            VoteResult with the majority decision and provenance.

        Raises:
            CircuitError: If the pipeline encounters a critical error.
        """
        # Step 1: Pre-filter — hard reject
        if not self._should_vote(fragment):
            return self._pre_filter_result(fragment)

        # Step 2: Run agents
        votes = self._run_agents(fragment)

        # Step 3: Determine outcome
        return self._determine_outcome(fragment, votes)

    def should_accept(self, fragment: Fragment) -> bool:
        """Quick check: should this fragment be accepted?

        Returns True if the fragment passes the circuit without needing
        to inspect the full VoteResult.
        """
        result = self.evaluate(fragment)
        return result.accepted

    def create_accepted_fragment(
        self,
        original: Fragment,
        vote_result: VoteResult,
    ) -> Fragment:
        """Create a new Fragment with QualityProvenance attached.

        The accepted fragment gets:
        - Its quality set to the vote's mean score
        - QualitySource.MAJORITY_VOTE as the provenance source
        - Full provenance trail from the vote
        - A new ID (original_id + '_vetted')

        Args:
            original: The original fragment that was voted on.
            vote_result: The vote result from evaluate().

        Returns:
            A new Fragment ready for storage, or the original Fragment
            modified with provenance if it was accepted without voting.
        """
        # If the fragment was accepted without voting (pre-filter pass),
        # just attach basic provenance
        if not vote_result.agent_scores:
            fragment = copy.deepcopy(original)
            provenance = QualityProvenance(
                final_quality=fragment.metadata.quality,
                final_source=QualitySource.INDEPENDENTLY_VERIFIED
                if fragment.metadata.quality > self._grey_zone_high
                else QualitySource.EVALUATOR_COMPUTED,
            )
            provenance.add_evaluation(QualityEvaluation(
                score=fragment.metadata.quality,
                source=provenance.final_source,
                model_id="pre_filter",
                model_version="1.0",
                metadata={"threshold": self._grey_zone_high},
            ))
            fragment.metadata.provenance = provenance
            return fragment

        # Voting was used — attach full provenance
        fragment = copy.deepcopy(original)
        fragment.fragment_id = f"{original.fragment_id}_vetted"
        fragment.metadata.quality = vote_result.mean_score

        provenance = QualityProvenance(
            final_quality=vote_result.mean_score,
            final_source=QualitySource.MAJORITY_VOTE,
            consensus_score=vote_result.mean_score,
            consensus_method=f"majority_vote_{TOTAL_AGENTS}",
        )

        for agent_name, score in vote_result.agent_scores.items():
            meta = vote_result.agent_metadata.get(agent_name, {})
            provenance.add_evaluation(QualityEvaluation(
                score=score,
                source=QualitySource.LLM_SELF_REPORTED,
                model_id=agent_name,
                model_version=meta.get("model_version", ""),
                metadata=meta,
            ))

        fragment.metadata.provenance = provenance
        return fragment

    # ------------------------------------------------------------------
    # Internal pipeline
    # ------------------------------------------------------------------

    def _should_vote(self, fragment: Fragment) -> bool:
        """Determine if the fragment needs 3-agent voting.

        Returns True if voting is needed, False if it can be decided
        without voting (hard reject or fast accept).
        """
        quality = fragment.metadata.quality

        # USER_CLAIMED always triggers voting (regardless of score)
        if self._is_user_claimed(fragment):
            return True

        # Below hard reject → no voting needed
        if quality < self._hard_reject_threshold:
            return False

        # Above grey zone → no voting needed (fast accept)
        if quality > self._grey_zone_high:
            return False

        # In the grey zone → voting needed
        return True

    def _is_user_claimed(self, fragment: Fragment) -> bool:
        """Check if the fragment has QualitySource.USER_CLAIMED."""
        if fragment.metadata.provenance is not None:
            return fragment.metadata.provenance.final_source == QualitySource.USER_CLAIMED
        return False

    def _pre_filter_result(self, fragment: Fragment) -> VoteResult:
        """Return a result when no voting is needed (pre-filter decision)."""
        quality = fragment.metadata.quality
        accepted = quality >= self._grey_zone_high

        if self._is_user_claimed(fragment):
            # Should never reach here if require_user_claimed_always_vote
            accepted = False

        return VoteResult(
            fragment_id=fragment.fragment_id,
            accepted=accepted,
            votes_for=1 if accepted else 0,
            votes_against=0 if accepted else 1,
            agents_responded=0,
            mean_score=quality,
            min_score=quality,
            max_score=quality,
            consensus_reached=True,
        )

    def _run_agents(
        self, fragment: Fragment
    ) -> Dict[str, Tuple[float, Dict[str, Any]]]:
        """Run all agents against the fragment.

        Each agent is called independently. Timeouts and errors are caught
        per-agent (not per-call — each agent is a single __call__).

        Returns:
            Dict of {agent_name: (score, metadata)} for responding agents.
        """
        votes: Dict[str, Tuple[float, Dict[str, Any]]] = {}

        for name, agent_fn in self._agents.items():
            try:
                score, meta = agent_fn(fragment)
                score = max(0.0, min(1.0, score))
                votes[name] = (score, meta)
            except Exception as e:
                logger.warning(
                    "Agent %s failed for fragment %s: %s",
                    name, fragment.fragment_id, e,
                )
                # Agent failure = skip (counts as non-response)
                continue

        if len(votes) < self._min_agents:
            raise InsufficientAgentsError(
                f"Only {len(votes)} of {len(self._agents)} agents responded "
                f"for fragment '{fragment.fragment_id}'. "
                f"Need at least {self._min_agents}."
            )

        return votes

    def _determine_outcome(
        self,
        fragment: Fragment,
        votes: Dict[str, Tuple[float, Dict[str, Any]]],
    ) -> VoteResult:
        """Determine majority outcome from agent votes."""
        votes_for = 0
        votes_against = 0
        scores: List[float] = []
        agent_scores: Dict[str, float] = {}
        agent_metadata: Dict[str, Dict[str, Any]] = {}
        fallback_used = False

        for name, (score, meta) in votes.items():
            agent_scores[name] = round(score, 4)
            agent_metadata[name] = meta
            scores.append(score)

            if score >= self._vote_threshold:
                votes_for += 1
            else:
                votes_against += 1

            if meta.get("parse_failed") or meta.get("fallback"):
                fallback_used = True

        accepted = votes_for > votes_against
        consensus = votes_for == 0 or votes_against == 0 or votes_for >= 2

        return VoteResult(
            fragment_id=fragment.fragment_id,
            accepted=accepted,
            votes_for=votes_for,
            votes_against=votes_against,
            agents_responded=len(votes),
            mean_score=round(sum(scores) / len(scores), 4) if scores else 0.0,
            min_score=round(min(scores), 4) if scores else 0.0,
            max_score=round(max(scores), 4) if scores else 0.0,
            consensus_reached=consensus,
            fallback_used=fallback_used,
            agent_scores=agent_scores,
            agent_metadata=agent_metadata,
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def agent_count(self) -> int:
        return len(self._agents)

    @property
    def agent_names(self) -> List[str]:
        return list(self._agents.keys())

    @property
    def config(self) -> Dict[str, Any]:
        return {
            "hard_reject_threshold": self._hard_reject_threshold,
            "grey_zone_low": self._grey_zone_low,
            "grey_zone_high": self._grey_zone_high,
            "vote_threshold": self._vote_threshold,
            "min_agents": self._min_agents,
            "require_user_claimed_always_vote": self._require_user_claimed_always_vote,
            "agents": list(self._agents.keys()),
        }

    def __repr__(self) -> str:
        return (
            f"AgentCircuit(agents={len(self._agents)}, "
            f"grey_zone=[{self._grey_zone_low}, {self._grey_zone_high}])"
        )

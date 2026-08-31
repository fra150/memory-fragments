"""Test dell'AgentCircuit — circuito di voto a 3 agenti con grey zone."""

import pytest

from memory_fragments.calibration.agents import create_mock_agents
from memory_fragments.library.circuit import (
    AgentCircuit,
    InsufficientAgentsError,
)
from memory_fragments.models import Fragment, FragmentMetadata
from memory_fragments.models.quality import QualityProvenance, QualitySource


def _fragment(quality: float, source: QualitySource = None) -> Fragment:
    fragment = Fragment(
        fragment_id="F-1",
        content="contenuto di esempio sufficientemente lungo per il voto degli agenti",
        metadata=FragmentMetadata(quality=quality),
    )
    if source is not None:
        fragment.metadata.provenance = QualityProvenance(
            final_quality=quality, final_source=source
        )
    return fragment


def _make_agent(score: float):
    def agent(fragment):
        return score, {"agent": "deterministic"}
    return agent


def _deterministic_agents() -> dict:
    return {
        "a": _make_agent(0.90),
        "b": _make_agent(0.85),
        "c": _make_agent(0.70),
    }


class TestCircuit:
    def test_grey_zone_majority_accept(self):
        circuit = AgentCircuit(_deterministic_agents())
        # 2 su 3 votano >= 0.80 -> accettato
        result = circuit.evaluate(_fragment(0.75))
        assert result.accepted is True
        assert result.votes_for == 2
        assert result.votes_against == 1

    def test_grey_zone_majority_reject(self):
        agents = {"a": _make_agent(0.70), "b": _make_agent(0.65), "c": _make_agent(0.60)}
        circuit = AgentCircuit(agents)
        result = circuit.evaluate(_fragment(0.75))
        assert result.accepted is False
        assert result.votes_against == 3

    def test_hard_reject_below_threshold(self):
        circuit = AgentCircuit(_deterministic_agents())
        result = circuit.evaluate(_fragment(0.50))
        assert result.accepted is False
        assert result.agents_responded == 0  # nessun voto, pre-filter

    def test_fast_accept_above_grey_zone(self):
        circuit = AgentCircuit(_deterministic_agents())
        result = circuit.evaluate(_fragment(0.95))
        assert result.accepted is True
        assert result.agents_responded == 0

    def test_user_claimed_always_votes(self):
        circuit = AgentCircuit(_deterministic_agents())
        # USER_CLAIMED con qualità alta deve comunque passare dal voto
        result = circuit.evaluate(_fragment(0.95, QualitySource.USER_CLAIMED))
        assert result.agents_responded == 3

    def test_insufficient_agents(self):
        def failing(fragment):
            raise RuntimeError("agent down")

        circuit = AgentCircuit(
            {"a": failing, "b": failing, "c": failing}, min_agents=2
        )
        with pytest.raises(InsufficientAgentsError):
            circuit.evaluate(_fragment(0.75))

    def test_too_few_agents_at_init(self):
        with pytest.raises(ValueError):
            AgentCircuit({"solo": _make_agent(0.8)}, min_agents=2)

    def test_should_accept(self):
        circuit = AgentCircuit(_deterministic_agents())
        assert circuit.should_accept(_fragment(0.95)) is True
        assert circuit.should_accept(_fragment(0.30)) is False


class TestCreateAcceptedFragment:
    def test_with_voting_attaches_provenance(self):
        circuit = AgentCircuit(_deterministic_agents())
        original = _fragment(0.75)
        result = circuit.evaluate(original)
        accepted = circuit.create_accepted_fragment(original, result)
        assert accepted.fragment_id == "F-1_vetted"
        assert accepted.metadata.provenance is not None
        assert accepted.metadata.provenance.final_source == QualitySource.MAJORITY_VOTE
        assert accepted.metadata.provenance.consensus_method == "majority_vote_3"

    def test_without_voting_keeps_id(self):
        circuit = AgentCircuit(_deterministic_agents())
        original = _fragment(0.95)
        result = circuit.evaluate(original)
        accepted = circuit.create_accepted_fragment(original, result)
        assert accepted.fragment_id == "F-1"
        assert accepted.metadata.provenance is not None


class TestMockAgents:
    def test_create_mock_agents(self):
        agents = create_mock_agents()
        assert len(agents) == 3
        circuit = AgentCircuit(agents)
        result = circuit.evaluate(_fragment(0.75))
        assert result.agents_responded == 3

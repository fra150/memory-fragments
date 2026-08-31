"""Tests for the AgentCircuit 3-agent majority voting pipeline."""

import pytest

from memory_fragments.calibration.agents import create_mock_agents
from memory_fragments.library.circuit import (
    AgentCircuit,
    InsufficientAgentsError,
    HARD_REJECT_THRESHOLD,
    GREY_ZONE_HIGH,
    VOTE_THRESHOLD,
)
from memory_fragments.models import Fragment, FragmentMetadata
from memory_fragments.models.quality import QualityProvenance, QualitySource


def _fragment(
    fid: str,
    quality: float,
    source: QualitySource | None = None,
    long: bool = True,
) -> Fragment:
    content = (
        "Questo frammento contiene un numero di parole sufficientemente elevato "
        "da attivare il bonus di lunghezza nella valutazione degli agenti mock, "
        "in modo che il voto sia stabile e ripetibile."
        if long
        else "Contenuto breve."
    )
    provenance = None
    if source is not None:
        provenance = QualityProvenance(final_quality=quality, final_source=source)
    return Fragment(
        fragment_id=fid,
        content=content,
        metadata=FragmentMetadata(topic="test", quality=quality, provenance=provenance),
    )


def _circuit(**kwargs) -> AgentCircuit:
    return AgentCircuit(create_mock_agents(), **kwargs)


def test_constructor_rejects_too_few_agents():
    with pytest.raises(ValueError):
        AgentCircuit({})


def test_agent_count_and_names():
    circuit = _circuit()
    assert circuit.agent_count == 3
    assert set(circuit.agent_names) == {
        "agent-strict",
        "agent-balanced",
        "agent-generous",
    }


def test_hard_reject_below_threshold():
    circuit = _circuit()
    frag = _fragment("r1", quality=HARD_REJECT_THRESHOLD - 0.1)
    result = circuit.evaluate(frag)
    assert result.accepted is False
    assert result.agents_responded == 0
    assert result.agent_scores == {}


def test_fast_accept_above_grey_zone():
    circuit = _circuit()
    frag = _fragment("r2", quality=GREY_ZONE_HIGH + 0.05)
    result = circuit.evaluate(frag)
    assert result.accepted is True
    assert result.agents_responded == 0


def test_grey_zone_triggers_voting():
    circuit = _circuit()
    frag = _fragment("r3", quality=0.75)
    result = circuit.evaluate(frag)
    assert result.agents_responded == 3
    assert len(result.agent_scores) == 3
    assert result.mean_score > 0.0


def test_user_claimed_always_triggers_voting():
    circuit = _circuit()
    frag = _fragment(
        "r4",
        quality=GREY_ZONE_HIGH + 0.05,
        source=QualitySource.USER_CLAIMED,
    )
    result = circuit.evaluate(frag)
    assert result.agents_responded == 3
    assert result.agent_scores != {}


def test_votes_are_deterministic():
    circuit = _circuit()
    frag = _fragment("r5", quality=0.75)
    first = circuit.evaluate(frag).agent_scores
    second = circuit.evaluate(frag).agent_scores
    assert first == second


def test_insufficient_agents_raises():
    def failing_agent(fragment):
        raise RuntimeError("agent down")

    circuit = AgentCircuit({"a1": failing_agent, "a2": failing_agent})
    with pytest.raises(InsufficientAgentsError):
        circuit.evaluate(_fragment("r6", quality=0.75))


def test_create_accepted_fragment_after_vote():
    circuit = _circuit()
    frag = _fragment("r7", quality=0.75)
    result = circuit.evaluate(frag)
    accepted = circuit.create_accepted_fragment(frag, result)
    assert accepted.fragment_id == "r7_vetted"
    assert accepted.metadata.quality == result.mean_score
    assert accepted.metadata.provenance is not None
    # The consensus result is stored on the provenance trail
    assert accepted.metadata.provenance.consensus_method.startswith("majority_vote")
    assert accepted.metadata.provenance.consensus_score == result.mean_score
    assert len(accepted.metadata.provenance.evaluations) == 3


def test_create_accepted_fragment_prefilter():
    circuit = _circuit()
    frag = _fragment("r8", quality=GREY_ZONE_HIGH + 0.05)
    result = circuit.evaluate(frag)
    assert result.agent_scores == {}
    accepted = circuit.create_accepted_fragment(frag, result)
    # Pre-filter pass keeps the original id and marks provenance
    assert accepted.fragment_id == "r8"
    assert accepted.metadata.provenance is not None
    assert (
        accepted.metadata.provenance.final_source
        == QualitySource.INDEPENDENTLY_VERIFIED
    )


def test_config_property():
    circuit = _circuit()
    cfg = circuit.config
    assert cfg["hard_reject_threshold"] == HARD_REJECT_THRESHOLD
    assert cfg["vote_threshold"] == VOTE_THRESHOLD
    assert len(cfg["agents"]) == 3

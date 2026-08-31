"""Test del Rastrello — pattern discovery + fase multi-agente (Card 17)."""

from memory_fragments.archive.static import StaticArchive
from memory_fragments.calibration.agents import create_mock_agents
from memory_fragments.engine.intake import IntakeVerifier
from memory_fragments.engine.rastrello import (
    ExtractedPattern,
    FragmentType,
    Rastrello,
)
from memory_fragments.library.circuit import AgentCircuit
from memory_fragments.library.guardian import FragmentGuardian
from memory_fragments.models.quality import QualitySource


def _build(with_circuit: bool = False):
    archive = StaticArchive()
    guardian = FragmentGuardian()
    intake = IntakeVerifier(guardian, archive)
    circuit = AgentCircuit(create_mock_agents()) if with_circuit else None
    rastrello = Rastrello(guardian, archive, intake, circuit=circuit)
    return rastrello, archive


def _function_pattern() -> ExtractedPattern:
    return ExtractedPattern(
        fragment_type=FragmentType.FUNCTION,
        name="connect",
        content=(
            "def connect(timeout=30):\n"
            "    \"\"\"Connect to the service with a configurable timeout.\"\"\"\n"
            "    for attempt in range(3):\n"
            "        try:\n"
            "            return socket.socket()\n"
            "        except OSError:\n"
            "            continue\n"
            "    return None\n"
        ),
        context="auth_module",
    )


def _import_pattern() -> ExtractedPattern:
    return ExtractedPattern(
        fragment_type=FragmentType.LIBRARY,
        name="os",
        content="import os",
        context="auth_module",
    )


class TestRastrelloCircuitStage:
    def test_backward_compatible_without_circuit(self):
        rastrello, archive = _build(with_circuit=False)
        certified = rastrello.propose_candidates([_import_pattern()])
        assert len(certified) == 1
        assert certified[0].metadata.provenance.final_source == QualitySource.RASTRELLO_DISCOVERED
        assert len(archive.list_all()) == 1

    def test_circuit_filters_short_patterns(self):
        # I mock agenti giudicano troppo povero un pattern di 2 parole
        rastrello, archive = _build(with_circuit=True)
        certified = rastrello.propose_candidates([_import_pattern()])
        assert certified == []
        assert len(archive.list_all()) == 0

    def test_circuit_accepts_substantial_pattern_with_majority_vote(self):
        rastrello, archive = _build(with_circuit=True)
        certified = rastrello.propose_candidates([_function_pattern()])
        assert len(certified) == 1
        frag = certified[0]
        prov = frag.metadata.provenance
        assert frag.fragment_id.endswith("_vetted")
        assert prov.final_source == QualitySource.MAJORITY_VOTE
        assert prov.consensus_method == "majority_vote_3"
        assert len(prov.evaluations) == 3
        assert len(archive.list_all()) == 1

    def test_circuit_rejects_are_not_archived(self):
        rastrello, archive = _build(with_circuit=True)
        certified = rastrello.propose_candidates([_import_pattern(), _function_pattern()])
        # Solo la funzione passa il voto; l'import corto viene filtrato
        assert len(certified) == 1
        assert all(f.fragment_id.endswith("_vetted") for f in certified)
        assert len(archive.list_all()) == 1

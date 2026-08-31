"""Tests for the MemoryFragmentsModel unified orchestrator."""

import json

import pytest

from memory_fragments.models import (
    AppealOperation,
    AppealStatus,
    Fragment,
    FragmentMetadata,
    OperationType,
)
from memory_fragments.model import MemoryFragmentsModel
from memory_fragments.retrieval.indexer import EmbeddingIndexer


@pytest.fixture(autouse=True)
def _offline_embedding(monkeypatch):
    """Force deterministic fallback embeddings (no network / model download)."""

    def _no_model(self):
        self._fallback = True

    monkeypatch.setattr(EmbeddingIndexer, "_ensure_model", _no_model)


def _fragment(fid, content, quality=0.9, topic="t", parents=None):
    return Fragment(
        fragment_id=fid,
        content=content,
        metadata=FragmentMetadata(topic=topic, quality=quality),
        parents=parents or [],
    )


def test_ingest_and_statistics():
    model = MemoryFragmentsModel()
    res = model.ingest(_fragment("f1", "La fotosintesi usa la luce solare."))
    assert res.ok
    assert len(model.archive) == 1
    assert model.graph.node_count() == 1
    assert model.get_statistics()["model"]["archive_count"] == 1


def test_ingest_rejects_duplicate():
    model = MemoryFragmentsModel()
    assert model.ingest(_fragment("f1", "contenuto")).ok
    res = model.ingest(_fragment("f1", "contenuto"))
    assert not res.ok
    assert "already exists" in res.reason


def test_ingest_guardian_rejection():
    from memory_fragments.library.guardian import FragmentGuardian

    model = MemoryFragmentsModel(
        guardian=FragmentGuardian(threshold=0.9, improver=lambda f: None)
    )
    res = model.ingest(_fragment("low", "contenuto scadente", quality=0.5))
    assert not res.ok
    assert "guardian" in res.reason
    assert len(model.archive) == 0


def test_ingest_guardian_acceptance():
    from memory_fragments.library.guardian import FragmentGuardian

    model = MemoryFragmentsModel(
        guardian=FragmentGuardian(threshold=0.9, improver=lambda f: None)
    )
    res = model.ingest(_fragment("high", "contenuto eccellente", quality=0.95))
    assert res.ok
    assert len(model.archive) == 1


def test_ingest_parents_recorded_in_graph():
    model = MemoryFragmentsModel()
    model.ingest(_fragment("p1", "fonte originale"))
    res = model.ingest(_fragment("c1", "derivato da p1", parents=["p1"]))
    assert res.ok
    ancestors = model.graph.get_ancestors("c1")
    assert any(n.fragment_id == "p1" for n in ancestors)


def test_retrieve_and_query():
    model = MemoryFragmentsModel()
    model.ingest(_fragment("f1", "La fotosintesi usa la luce solare per produrre glucosio.", topic="biology"))
    model.ingest(_fragment("f2", "Il Sole genera energia tramite la fusione nucleare.", topic="physics"))

    results = model.retrieve("luce solare energia", top_k=2)
    assert len(results) >= 1
    assert all(isinstance(s, float) for _, s in results)

    q = model.query("fotosintesi glucosio", top_k=2)
    assert q.fragments
    assert len(q.scores) == len(q.fragments)
    assert "fragment" in q.response.lower()


def test_query_no_hits_returns_message():
    model = MemoryFragmentsModel()
    q = model.query("nessun frammento presente", top_k=2)
    assert q.fragments == []
    assert "No relevant fragments" in q.response


def test_propose_computes_metrics_and_diff():
    model = MemoryFragmentsModel()
    model.ingest(_fragment("f1", "La fotosintesi usa la luce solare per produrre glucosio.", topic="biology"))

    appeal = model.propose(
        appeal_id="A1",
        source_ids=["f1"],
        proposed_content="La fotosintesi usa la luce solare per produrre glucosio e ossigeno.",
        explanation="Aggiunge il prodotto ossigeno.",
        ops=[AppealOperation(op_type=OperationType.ANNOTATE)],
    )
    assert appeal.appeal_id == "A1"
    assert appeal.status == AppealStatus.DRAFT
    assert appeal.proposed_content.startswith("La fotosintesi")
    assert appeal.metrics.delta_token != 0
    assert appeal.metrics.coverage > 0.0
    assert len(appeal.diff.added) > 0
    # metrics and diff are persisted in the trial space
    stored = model.appeal_space.get_appeal("A1")
    assert stored is not None
    assert stored.metrics == appeal.metrics


def test_propose_unknown_source_raises():
    model = MemoryFragmentsModel()
    with pytest.raises(ValueError, match="Unknown source"):
        model.propose(
            appeal_id="A1",
            source_ids=["missing"],
            proposed_content="qualcosa",
        )


def test_governance_workflow():
    model = MemoryFragmentsModel()
    model.ingest(_fragment("f1", "La fotosintesi usa la luce solare per produrre glucosio.", topic="biology"))

    model.propose(
        appeal_id="A1",
        source_ids=["f1"],
        proposed_content="La fotosintesi usa la luce solare e produce ossigeno.",
        explanation="Estensione con prodotto.",
    )
    report = model.submit("A1")
    assert report.appeal_id == "A1"

    approved = model.approve("A1", approver="reviewer", notes="ok")
    assert approved.fragment_id != "f1"
    assert model.archive.get(approved.fragment_id) is not None
    assert model.appeal_space.get_appeal("A1").status == AppealStatus.APPROVED

    history = model.get_report("A1")
    assert history is not None


def test_reject_workflow():
    model = MemoryFragmentsModel()
    model.ingest(_fragment("f1", "contenuto fonte", topic="t"))
    model.propose(
        appeal_id="A1",
        source_ids=["f1"],
        proposed_content="proposta non desiderata",
        explanation="test",
    )
    model.submit("A1")
    assert model.reject("A1", reason="non pertinente") is True
    assert model.appeal_space.get_appeal("A1").status == AppealStatus.REJECTED


def test_export_import_roundtrip(tmp_path):
    model = MemoryFragmentsModel()
    model.ingest(_fragment("f1", "La fotosintesi usa la luce solare.", topic="biology"))
    model.propose(
        appeal_id="A1",
        source_ids=["f1"],
        proposed_content="La fotosintesi usa la luce solare e l'acqua.",
        explanation="aggiunta acqua",
    )

    state = tmp_path / "model.json"
    model.save_state(str(state))
    raw = json.loads(state.read_text(encoding="utf-8"))
    assert raw["schema"] == "memory-fragments-model"
    assert len(raw["archive"]["fragments"]) == 1

    restored = MemoryFragmentsModel.load_state(str(state))
    assert len(restored.archive) == 1
    assert restored.archive.get("f1") is not None
    assert restored.appeal_space.get_appeal("A1") is not None
    assert restored.graph.node_count() == 1

    # retriever rebuilt from the archive
    results = restored.retrieve("fotosintesi luce", top_k=2)
    assert any(f.fragment_id == "f1" for f, _ in results)


def test_export_state_schema():
    model = MemoryFragmentsModel()
    state = model.export_state()
    assert state["schema"] == "memory-fragments-model"
    assert set(state.keys()) == {
        "schema",
        "version",
        "exported_at",
        "archive",
        "appeal_space",
        "genealogy",
    }

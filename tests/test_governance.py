"""Tests for the GovernanceAPI approval / rejection workflow."""

import pytest

from memory_fragments.archive.appeal_space import AppealTrialSpace
from memory_fragments.archive.static import StaticArchive
from memory_fragments.engine.diff_explain import DiffExplainEngine
from memory_fragments.engine.evaluator import Evaluator
from memory_fragments.governance.api import GovernanceAPI
from memory_fragments.models import (
    AppealStatus,
    Fragment,
    FragmentMetadata,
    FragmentStatus,
)
from memory_fragments.models.graph import GenealogyGraph


@pytest.fixture
def governance():
    archive = StaticArchive()
    appeal_space = AppealTrialSpace()
    evaluator = Evaluator()
    diff_explain = DiffExplainEngine()
    graph = GenealogyGraph()
    api = GovernanceAPI(archive, appeal_space, evaluator, diff_explain, graph)
    return api, archive, appeal_space, graph


def _source_fragment(fid: str = "src-1") -> Fragment:
    return Fragment(
        fragment_id=fid,
        content=(
            "I sistemi di memoria per agenti intelligenti combinano archiviazione "
            "immutabile, un registro delle versioni e un processo di governance."
        ),
        metadata=FragmentMetadata(
            topic="architettura", quality=0.90, author="ricercatore", tags=["memo"]
        ),
    )


def _prepare_pending_appeal(api, archive, appeal_space, aid="A-1"):
    if "src-1" not in archive:
        archive.add(_source_fragment())
    appeal_space.create_appeal(aid, sources=["src-1"])
    appeal_space.update_proposal(
        aid,
        proposed_content=(
            "I sistemi di memoria per agenti intelligenti combinano archiviazione "
            "immutabile, un registro delle versioni e un processo di governance "
            "con approvazione umana."
        ),
        explanation="Estende il frammento sorgente con il requisito di approvazione umana.",
    )
    api.submit_for_review(aid)


def test_submit_for_review_changes_status(governance):
    api, archive, appeal_space, _ = governance
    _prepare_pending_appeal(api, archive, appeal_space)
    assert appeal_space.get("A-1").status == AppealStatus.PENDING_USER_APPROVAL


def test_submit_for_review_returns_report(governance):
    api, archive, appeal_space, _ = governance
    _prepare_pending_appeal(api, archive, appeal_space)
    report = api.get_report("A-1")
    assert report.appeal_id == "A-1"
    assert report.proposed_content != ""
    assert report.metrics.coverage >= 0.0
    assert isinstance(report.explanation, str)


def test_approve_promotes_fragment(governance):
    api, archive, appeal_space, graph = governance
    _prepare_pending_appeal(api, archive, appeal_space)

    fragment = api.approve("A-1", approver="reviewer", notes="ok")

    assert fragment.fragment_id == "F-0001"
    assert archive.get_fragment("F-0001") is not None
    assert appeal_space.get("A-1").status == AppealStatus.APPROVED
    assert appeal_space.get("A-1").resolved_at is not None
    assert graph.has_node("F-0001")
    assert graph.get_node("F-0001").parent_ids == ["src-1"]


def test_approve_merges_metadata(governance):
    api, archive, appeal_space, _ = governance
    _prepare_pending_appeal(api, archive, appeal_space)
    fragment = api.approve("A-1")
    assert fragment.metadata.quality == 0.90
    assert fragment.metadata.topic == "architettura"
    assert fragment.metadata.source == "governance:approve"


def test_approve_wrong_status_raises(governance):
    api, archive, appeal_space, _ = governance
    appeal_space.create_appeal("A-2", sources=["src-1"])
    with pytest.raises(ValueError):
        api.approve("A-2")


def test_approve_missing_appeal_raises(governance):
    api, *_ = governance
    with pytest.raises(ValueError):
        api.approve("A-404")


def test_reject_marks_appeal(governance):
    api, archive, appeal_space, _ = governance
    _prepare_pending_appeal(api, archive, appeal_space, aid="A-3")
    assert api.reject("A-3", reason="Contenuto non verificato") is True
    appeal = appeal_space.get("A-3")
    assert appeal.status == AppealStatus.REJECTED
    assert appeal.resolved_at is not None
    assert "non verificato" in appeal.explanation


def test_get_report_missing_returns_none(governance):
    api, *_ = governance
    assert api.get_report("A-404") is None


def test_get_version_history(governance):
    api, archive, appeal_space, graph = governance
    # Register the source fragment in the genealogy graph so that the
    # promoted node can link back to it as an ancestor.
    graph.add_node(
        fragment_id="src-1",
        checksum=_source_fragment().checksum,
        quality_source="evaluator_computed",
    )
    _prepare_pending_appeal(api, archive, appeal_space)
    api.approve("A-1")
    history = api.get_version_history("F-0001")
    assert history["fragment_id"] == "F-0001"
    assert "src-1" in history["ancestors"]
    assert history["approver"] == "user"
    assert history["depth"] == 1


def test_rollback_returns_parent_and_archives(governance):
    api, archive, appeal_space, _ = governance
    _prepare_pending_appeal(api, archive, appeal_space)
    api.approve("A-1")

    parent = api.rollback("F-0001")
    assert parent is not None
    assert parent.fragment_id == "src-1"
    archived = archive.get_fragment("F-0001")
    assert archived.status == FragmentStatus.ARCHIVED


def test_rollback_unknown_fragment_returns_none(governance):
    api, *_ = governance
    assert api.rollback("ghost") is None


def test_get_statistics(governance):
    api, archive, appeal_space, _ = governance
    _prepare_pending_appeal(api, archive, appeal_space, aid="A-1")
    _prepare_pending_appeal(api, archive, appeal_space, aid="A-2")
    api.approve("A-1")
    api.reject("A-2")

    stats = api.get_statistics()
    assert stats["total_appeals"] == 2
    assert stats["approved"] == 1
    assert stats["rejected"] == 1
    assert stats["pending"] == 0
    assert stats["avg_time_to_decision_seconds"] >= 0

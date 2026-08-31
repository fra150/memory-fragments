"""Test della GovernanceAPI — ciclo di vita completo di approvazione/rigetto."""

import pytest

from memory_fragments.archive.appeal_space import AppealTrialSpace
from memory_fragments.archive.static import StaticArchive
from memory_fragments.engine.diff_explain import DiffExplainEngine
from memory_fragments.engine.evaluator import Evaluator
from memory_fragments.governance.api import GovernanceAPI, GovernanceReport
from memory_fragments.models import (
    AppealOperation,
    AppealStatus,
    Fragment,
    FragmentMetadata,
    OperationType,
)
from memory_fragments.models.graph import GenealogyGraph


def _fragment(fragment_id: str, content: str, quality: float = 0.8) -> Fragment:
    return Fragment(
        fragment_id=fragment_id,
        content=content,
        metadata=FragmentMetadata(quality=quality, topic="test"),
    )


def _setup():
    archive = StaticArchive()
    # ID sorgenti senza prefisso "F-" così il primo fragment promosso è F-0001
    archive.add(_fragment("src-1", "il gatto dorme sul divano e ronfa felice"))
    archive.add(_fragment("src-2", "il cane abbaia nel cortile durante la notte"))

    appeal_space = AppealTrialSpace()
    evaluator = Evaluator()
    diff_explain = DiffExplainEngine()
    graph = GenealogyGraph()

    api = GovernanceAPI(
        archive=archive,
        appeal_space=appeal_space,
        evaluator=evaluator,
        diff_explain=diff_explain,
        graph=graph,
    )
    return api, archive, appeal_space, graph


def _create_pending_appeal(api, appeal_id="A-1") -> None:
    op = AppealOperation(op_type=OperationType.MERGE, params={}, description="merge")
    api._appeal_space.create_appeal(
        appeal_id, sources=["src-1", "src-2"], ops=[op]
    )
    api._appeal_space.update_proposal(
        appeal_id,
        proposed_content="il gatto dorme e ronfa felice mentre il cane abbaia",
        explanation="unire le informazioni sui due animali",
    )
    api.submit_for_review(appeal_id)


class TestApproveWorkflow:
    def test_full_approve_flow(self):
        api, archive, appeal_space, graph = _setup()
        _create_pending_appeal(api)

        fragment = api.approve("A-1", approver="francesco", notes="ok")

        assert fragment.fragment_id == "F-0001"
        assert fragment.fragment_id in archive
        assert appeal_space.get("A-1").status == AppealStatus.APPROVED
        assert graph.has_node("F-0001")
        node = graph.get_node("F-0001")
        assert node.parent_ids == ["src-1", "src-2"]
        assert node.appeal_id == "A-1"
        assert node.approver == "francesco"

    def test_approve_not_pending_raises(self):
        api, *_ = _setup()
        api._appeal_space.create_appeal("A-1")
        with pytest.raises(ValueError):
            api.approve("A-1")

    def test_approve_missing_appeal_raises(self):
        api, *_ = _setup()
        with pytest.raises(ValueError):
            api.approve("A-MANCANTE")


class TestRejectWorkflow:
    def test_reject(self):
        api, _, appeal_space, _ = _setup()
        _create_pending_appeal(api)
        assert api.reject("A-1", reason="non pertinente") is True
        appeal = appeal_space.get("A-1")
        assert appeal.status == AppealStatus.REJECTED
        assert appeal.explanation == "non pertinente"
        assert appeal.resolved_at is not None


class TestReport:
    def test_get_report(self):
        api, *_ = _setup()
        _create_pending_appeal(api)
        report = api.get_report("A-1")
        assert isinstance(report, GovernanceReport)
        assert report.appeal_id == "A-1"
        assert report.status == AppealStatus.PENDING_USER_APPROVAL
        assert report.explanation
        assert report.metrics.coverage > 0.0

    def test_get_report_missing_returns_none(self):
        api, *_ = _setup()
        assert api.get_report("A-MANCANTE") is None

    def test_get_reports_for_fragment(self):
        api, *_ = _setup()
        _create_pending_appeal(api)
        reports = api.get_reports_for_fragment("src-1")
        assert len(reports) == 1
        assert reports[0].appeal_id == "A-1"


class TestVersionHistory:
    def test_version_history(self):
        api, *_ = _setup()
        _create_pending_appeal(api)
        api.approve("A-1")
        history = api.get_version_history("F-0001")
        assert history["fragment_id"] == "F-0001"
        assert set(history["parent_ids"]) == {"src-1", "src-2"}
        assert history["appeal_id"] == "A-1"

    def test_version_history_unknown(self):
        api, *_ = _setup()
        assert api.get_version_history("F-NOPE") == {}


class TestStatistics:
    def test_statistics_counts(self):
        api, *_ = _setup()
        _create_pending_appeal(api, "A-1")
        api.approve("A-1")
        api._appeal_space.create_appeal("A-2")
        api._appeal_space.set_status("A-2", AppealStatus.REJECTED)

        stats = api.get_statistics()
        assert stats["total_appeals"] == 2
        assert stats["approved"] == 1
        assert stats["rejected"] == 1
        assert stats["quarantine_size"] == 0


class TestRollback:
    def test_rollback_archives_and_returns_parent(self):
        api, archive, _, _ = _setup()
        _create_pending_appeal(api)
        api.approve("A-1")

        previous = api.rollback("F-0001")
        assert previous is not None
        assert archive.get("F-0001").status.value == "archived"

    def test_rollback_unknown_returns_none(self):
        api, *_ = _setup()
        assert api.rollback("F-NOPE") is None

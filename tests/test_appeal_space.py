"""Tests for the AppealTrialSpace (sandbox for Appeal lifecycle)."""

import pytest

from memory_fragments.archive import AppealTrialSpace
from memory_fragments.config import default_config
from memory_fragments.models import (
    Appeal,
    AppealMetrics,
    AppealOperation,
    AppealStatus,
    AppealDiff,
    OperationType,
)


def make_op(op_type: OperationType = OperationType.REWRITE) -> AppealOperation:
    return AppealOperation(op_type=op_type, params={"field": "content"}, description="rewrite content")


class TestCreate:
    def test_create_appeal_draft(self):
        space = AppealTrialSpace()
        appeal = space.create_appeal("app-1", sources=["frag-1"], ops=[make_op()])

        assert appeal.appeal_id == "app-1"
        assert appeal.status == AppealStatus.DRAFT
        assert appeal.sources == ["frag-1"]
        assert len(appeal.ops) == 1

    def test_duplicate_rejected(self):
        space = AppealTrialSpace()
        space.create_appeal("dup")
        with pytest.raises(ValueError):
            space.create_appeal("dup")

    def test_max_active_appeals(self, monkeypatch):
        monkeypatch.setattr(default_config.appeal, "max_active_appeals", 2)
        space = AppealTrialSpace()
        space.create_appeal("a1")
        space.create_appeal("a2")
        with pytest.raises(ValueError):
            space.create_appeal("a3")

    def test_max_ops_per_appeal(self, monkeypatch):
        monkeypatch.setattr(default_config.appeal, "max_ops_per_appeal", 1)
        space = AppealTrialSpace()
        with pytest.raises(ValueError):
            space.create_appeal("too-many-ops", ops=[make_op(), make_op()])

    def test_get_returns_deep_copy(self):
        space = AppealTrialSpace()
        space.create_appeal("app-x")
        stored = space.get_appeal("app-x")
        stored.status = AppealStatus.APPROVED
        assert space.get_appeal("app-x").status == AppealStatus.DRAFT

    def test_get_missing_returns_none(self):
        space = AppealTrialSpace()
        assert space.get_appeal("nope") is None


class TestUpdates:
    def test_update_metrics(self):
        space = AppealTrialSpace()
        space.create_appeal("app-m")
        metrics = AppealMetrics(delta_token=5, coverage=0.8, risk=0.1, aggregate_score=0.9)
        space.update_metrics("app-m", metrics)
        assert space.get_appeal("app-m").metrics == metrics

    def test_update_diff(self):
        space = AppealTrialSpace()
        space.create_appeal("app-d")
        diff = AppealDiff(added=["new", "words"], removed=["old"])
        space.update_diff("app-d", diff)
        assert space.get_appeal("app-d").diff.added == ["new", "words"]

    def test_update_proposal(self):
        space = AppealTrialSpace()
        space.create_appeal("app-p")
        space.update_proposal("app-p", "new content here", "because it is better")
        appeal = space.get_appeal("app-p")
        assert appeal.proposed_content == "new content here"
        assert appeal.explanation == "because it is better"

    def test_update_missing_raises(self):
        space = AppealTrialSpace()
        with pytest.raises(KeyError):
            space.update_metrics("nope", AppealMetrics())
        with pytest.raises(KeyError):
            space.update_diff("nope", AppealDiff())
        with pytest.raises(KeyError):
            space.update_proposal("nope", "x", "y")


class TestStatusAndLifecycle:
    def test_set_status_terminal_sets_resolved_at(self):
        space = AppealTrialSpace()
        space.create_appeal("app-r")
        assert space.set_status("app-r", AppealStatus.APPROVED) is True
        appeal = space.get_appeal("app-r")
        assert appeal.resolved_at is not None
        assert appeal.status == AppealStatus.APPROVED

    def test_set_status_missing_returns_false(self):
        space = AppealTrialSpace()
        assert space.set_status("nope", AppealStatus.REJECTED) is False

    def test_remove_appeal(self):
        space = AppealTrialSpace()
        space.create_appeal("app-del")
        assert space.remove_appeal("app-del") is True
        assert space.get_appeal("app-del") is None
        assert space.remove_appeal("app-del") is False

    def test_list_active_and_count(self):
        space = AppealTrialSpace()
        space.create_appeal("a1")
        space.create_appeal("a2")
        space.set_status("a2", AppealStatus.REJECTED)
        assert space.count_active() == 1
        assert [a.appeal_id for a in space.list_active()] == ["a1"]

    def test_prune_old_appeals_by_count(self, monkeypatch):
        monkeypatch.setattr(default_config.appeal, "prune_after_generations", 2)
        space = AppealTrialSpace()
        for i in range(3):
            space.create_appeal(f"p{i}")
            space.set_status(f"p{i}", AppealStatus.APPROVED)
        pruned = space.prune_old_appeals(max_generations=2)
        assert pruned == 1
        assert len(space.list_appeals()) == 2


class TestSerialization:
    def test_roundtrip(self):
        space = AppealTrialSpace()
        space.create_appeal("app-1", sources=["frag-1"], ops=[make_op()])
        space.update_proposal("app-1", "proposed content", "explanation")
        space.set_status("app-1", AppealStatus.PENDING_USER_APPROVAL)

        restored = AppealTrialSpace.from_dict(space.to_dict())

        assert restored.get_appeal("app-1") is not None
        appeal = restored.get_appeal("app-1")
        assert appeal.status == AppealStatus.PENDING_USER_APPROVAL
        assert appeal.proposed_content == "proposed content"
        assert appeal.ops[0].op_type == OperationType.REWRITE

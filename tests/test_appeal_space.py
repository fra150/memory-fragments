"""Test dell'AppealTrialSpace — sandbox per gli Appeal."""

import pytest

from memory_fragments.archive.appeal_space import AppealTrialSpace
from memory_fragments.models import (
    AppealMetrics,
    AppealOperation,
    AppealStatus,
    OperationType,
)
from memory_fragments.models.appeal import AppealDiff


def _op(op_type=OperationType.MERGE) -> AppealOperation:
    return AppealOperation(op_type=op_type, params={}, description="op")


class TestCreate:
    def test_create_and_get(self):
        space = AppealTrialSpace()
        appeal = space.create_appeal("A-1", sources=["F-1"])
        assert appeal.status == AppealStatus.DRAFT
        assert space.get("A-1").appeal_id == "A-1"

    def test_get_returns_deep_copy(self):
        space = AppealTrialSpace()
        space.create_appeal("A-1")
        got = space.get("A-1")
        got.proposed_content = "MUTATO"
        assert space.get("A-1").proposed_content == ""

    def test_duplicate_id_raises(self):
        space = AppealTrialSpace()
        space.create_appeal("A-1")
        with pytest.raises(ValueError):
            space.create_appeal("A-1")

    def test_missing_returns_none(self):
        assert AppealTrialSpace().get("NOPE") is None

    def test_ops_limit_exceeded(self):
        space = AppealTrialSpace()
        ops = [_op() for _ in range(11)]  # max_ops_per_appeal = 10
        with pytest.raises(ValueError):
            space.create_appeal("A-1", ops=ops)

    def test_max_active_appeals(self):
        space = AppealTrialSpace()
        for i in range(20):  # max_active_appeals = 20
            space.create_appeal(f"A-{i}")
        with pytest.raises(ValueError):
            space.create_appeal("A-20")


class TestUpdate:
    def test_update_metrics(self):
        space = AppealTrialSpace()
        space.create_appeal("A-1")
        space.update_metrics("A-1", AppealMetrics(delta_token=5, coverage=0.8, risk=0.1, aggregate_score=0.7))
        assert space.get("A-1").metrics.coverage == 0.8

    def test_update_metrics_missing_raises(self):
        with pytest.raises(KeyError):
            AppealTrialSpace().update_metrics("NOPE", AppealMetrics())

    def test_update_diff(self):
        space = AppealTrialSpace()
        space.create_appeal("A-1")
        space.update_diff("A-1", AppealDiff(added=["x"]))
        assert space.get("A-1").diff.added == ["x"]

    def test_update_proposal(self):
        space = AppealTrialSpace()
        space.create_appeal("A-1")
        space.update_proposal("A-1", "testo proposto", "spiegazione")
        appeal = space.get("A-1")
        assert appeal.proposed_content == "testo proposto"
        assert appeal.explanation == "spiegazione"


class TestStatus:
    def test_set_status_resolved_at(self):
        space = AppealTrialSpace()
        space.create_appeal("A-1")
        assert space.set_status("A-1", AppealStatus.APPROVED) is True
        appeal = space.get("A-1")
        assert appeal.status == AppealStatus.APPROVED
        assert appeal.resolved_at is not None

    def test_set_status_missing_returns_false(self):
        assert AppealTrialSpace().set_status("NOPE", AppealStatus.APPROVED) is False

    def test_list_filter_by_status(self):
        space = AppealTrialSpace()
        space.create_appeal("A-1")
        space.create_appeal("A-2")
        space.set_status("A-2", AppealStatus.PENDING_USER_APPROVAL)
        assert {a.appeal_id for a in space.list(status=AppealStatus.PENDING_USER_APPROVAL)} == {"A-2"}

    def test_list_active_excludes_terminal(self):
        space = AppealTrialSpace()
        space.create_appeal("A-1")
        space.create_appeal("A-2")
        space.set_status("A-2", AppealStatus.REJECTED)
        assert {a.appeal_id for a in space.list_active()} == {"A-1"}
        assert space.count_active() == 1


class TestRemoval:
    def test_remove_appeal(self):
        space = AppealTrialSpace()
        space.create_appeal("A-1")
        assert space.remove_appeal("A-1") is True
        assert space.get("A-1") is None

    def test_remove_missing_returns_false(self):
        assert AppealTrialSpace().remove_appeal("NOPE") is False

    def test_prune_by_generations(self):
        space = AppealTrialSpace()
        for i in range(5):
            space.create_appeal(f"A-{i}")
            space.set_status(f"A-{i}", AppealStatus.APPROVED)
        pruned = space.prune_old_appeals(max_generations=2)
        assert pruned == 3
        assert len(space) == 2


class TestSerialization:
    def test_round_trip(self):
        space = AppealTrialSpace()
        space.create_appeal("A-1", sources=["F-1"])
        space.update_proposal("A-1", "testo", "spiegazione")
        restored = AppealTrialSpace.from_dict(space.to_dict())
        assert len(restored) == 1
        assert restored.get("A-1").proposed_content == "testo"

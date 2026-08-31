"""Tests for the core data models (Fragment, Appeal, provenance, serialisation)."""

from datetime import datetime, timezone

from memory_fragments.models import (
    Appeal,
    AppealDiff,
    AppealMetrics,
    AppealOperation,
    AppealStatus,
    Fragment,
    FragmentConditions,
    FragmentMetadata,
    FragmentStatus,
    OperationType,
)
from memory_fragments.models.quality import (
    QualityEvaluation,
    QualityProvenance,
    QualitySource,
)


# ---------------------------------------------------------------------------
# Fragment
# ---------------------------------------------------------------------------


def test_fragment_auto_checksum():
    frag = Fragment(fragment_id="f1", content="contenuto")
    assert len(frag.checksum) == 16


def test_fragment_checksum_kept_when_provided():
    frag = Fragment(fragment_id="f1", content="contenuto", checksum="custom")
    assert frag.checksum == "custom"


def test_fragment_quality_is_clamped():
    frag = Fragment(
        fragment_id="f1",
        content="contenuto",
        metadata=FragmentMetadata(quality=1.7),
    )
    assert frag.metadata.quality == 1.0

    frag2 = Fragment(
        fragment_id="f2",
        content="contenuto",
        metadata=FragmentMetadata(quality=-0.3),
    )
    assert frag2.metadata.quality == 0.0


def test_fragment_roundtrip():
    frag = Fragment(
        fragment_id="f1",
        content="contenuto di prova",
        metadata=FragmentMetadata(
            topic="tema",
            source="sorgente",
            quality=0.85,
            author="autore",
            tags=["a", "b"],
            date=datetime(2026, 1, 1, tzinfo=timezone.utc),
        ),
        conditions=FragmentConditions(
            keywords=["k1"], semantic_threshold=0.5, require_all_keywords=True
        ),
        parents=["p1"],
        status=FragmentStatus.ACTIVE,
    )
    restored = Fragment.from_dict(frag.to_dict())
    assert restored.fragment_id == frag.fragment_id
    assert restored.content == frag.content
    assert restored.metadata.topic == frag.metadata.topic
    assert restored.metadata.quality == frag.metadata.quality
    assert restored.metadata.tags == frag.metadata.tags
    assert restored.metadata.date == frag.metadata.date
    assert restored.conditions.keywords == ["k1"]
    assert restored.conditions.require_all_keywords is True
    assert restored.parents == ["p1"]
    assert restored.checksum == frag.checksum
    assert restored.status == FragmentStatus.ACTIVE


def test_fragment_to_json():
    frag = Fragment(fragment_id="f1", content="contenuto")
    assert '"f1"' in frag.to_json()


# ---------------------------------------------------------------------------
# Appeal
# ---------------------------------------------------------------------------


def test_appeal_roundtrip():
    appeal = Appeal(
        appeal_id="A-1",
        sources=["f1", "f2"],
        ops=[AppealOperation(op_type=OperationType.MERGE, description="merge")],
        diff=AppealDiff(added=["a"], removed=["r"]),
        metrics=AppealMetrics(delta_token=3, coverage=0.8, risk=0.1, aggregate_score=0.9),
        status=AppealStatus.PENDING_USER_APPROVAL,
        proposed_content="nuovo contenuto",
        explanation="motivo",
        created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        resolved_at=datetime(2026, 1, 2, tzinfo=timezone.utc),
    )
    restored = Appeal.from_dict(appeal.to_dict())
    assert restored.appeal_id == appeal.appeal_id
    assert restored.sources == ["f1", "f2"]
    assert restored.ops[0].op_type == OperationType.MERGE
    assert restored.diff.added == ["a"]
    assert restored.diff.removed == ["r"]
    assert restored.metrics.aggregate_score == appeal.metrics.aggregate_score
    assert restored.status == AppealStatus.PENDING_USER_APPROVAL
    assert restored.proposed_content == appeal.proposed_content
    assert restored.resolved_at == appeal.resolved_at


def test_appeal_metrics_roundtrip():
    metrics = AppealMetrics(delta_token=-2, coverage=0.42, risk=0.33, aggregate_score=0.61)
    restored = AppealMetrics.from_dict(metrics.to_dict())
    assert restored == metrics


def test_appeal_diff_roundtrip():
    diff = AppealDiff(added=["x"], removed=["y"], modified=[{"a": 1}], reordered=[0, 2])
    restored = AppealDiff.from_dict(diff.to_dict())
    assert restored == diff


def test_appeal_operation_roundtrip():
    op = AppealOperation(op_type=OperationType.SPLIT, params={"n": 2}, description="split")
    restored = AppealOperation.from_dict(op.to_dict())
    assert restored == op
    assert restored.op_type == OperationType.SPLIT


# ---------------------------------------------------------------------------
# Quality provenance
# ---------------------------------------------------------------------------


def test_provenance_add_evaluation_updates_final():
    prov = QualityProvenance()
    prov.add_evaluation(
        QualityEvaluation(
            score=0.9,
            source=QualitySource.INDEPENDENTLY_VERIFIED,
            model_id="heuristic",
        )
    )
    assert prov.final_quality == 0.9
    assert prov.final_source == QualitySource.INDEPENDENTLY_VERIFIED
    assert len(prov.evaluations) == 1


def test_provenance_set_consensus():
    prov = QualityProvenance()
    prov.set_consensus(0.82, "majority_vote_3")
    assert prov.consensus_score == 0.82
    assert prov.consensus_method == "majority_vote_3"
    assert prov.final_quality == 0.82
    assert prov.final_source == QualitySource.MAJORITY_VOTE


def test_provenance_roundtrip():
    prov = QualityProvenance()
    prov.add_evaluation(
        QualityEvaluation(
            score=0.8,
            source=QualitySource.MAJORITY_VOTE,
            model_id="agent-strict",
        )
    )
    prov.set_consensus(0.83, "majority_vote_3")
    restored = QualityProvenance.from_dict(prov.to_dict())
    assert restored.final_quality == prov.final_quality
    assert restored.final_source == prov.final_source
    assert restored.consensus_method == "majority_vote_3"
    # set_consensus records the result but does not append a new evaluation
    assert len(restored.evaluations) == 1


def test_quality_source_values():
    assert QualitySource.USER_CLAIMED.value == "user_claimed"
    assert QualitySource.MAJORITY_VOTE.value == "majority_vote"

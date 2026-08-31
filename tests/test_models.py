"""Test dei data model: Fragment, Appeal, operazioni e serializzazione."""

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


def _make_fragment(
    fragment_id: str = "F-0001",
    content: str = "Il gatto dorme sul divano e ronfa.",
    quality: float = 0.8,
    **kwargs,
) -> Fragment:
    metadata = FragmentMetadata(topic="animali", quality=quality, tags=["gatto"])
    conditions = FragmentConditions(keywords=["gatto", "divano"])
    return Fragment(
        fragment_id=fragment_id,
        content=content,
        metadata=metadata,
        conditions=conditions,
        **kwargs,
    )


class TestFragment:
    def test_auto_checksum(self):
        frag = _make_fragment()
        assert frag.checksum  # generato automaticamente in __post_init__

    def test_checksum_deterministic(self):
        # Il checksum include created_at.isoformat() per design → stesso timestamp
        from datetime import datetime, timezone

        ts = datetime(2026, 1, 1, tzinfo=timezone.utc)
        f1 = _make_fragment("F-1", created_at=ts)
        f2 = _make_fragment("F-1", created_at=ts)
        assert f1.checksum == f2.checksum
        assert f1.checksum != _make_fragment("F-2", created_at=ts).checksum

    def test_quality_clamping(self):
        frag = _make_fragment(quality=1.7)
        assert frag.metadata.quality == 1.0
        frag = _make_fragment(quality=-0.5)
        assert frag.metadata.quality == 0.0

    def test_default_status_active(self):
        assert _make_fragment().status == FragmentStatus.ACTIVE

    def test_round_trip_dict(self):
        original = _make_fragment(
            quality=0.9, status=FragmentStatus.PENDING_REVIEW
        )
        restored = Fragment.from_dict(original.to_dict())
        assert restored.fragment_id == original.fragment_id
        assert restored.content == original.content
        assert restored.metadata.quality == original.metadata.quality
        assert restored.metadata.topic == original.metadata.topic
        assert restored.conditions.keywords == original.conditions.keywords
        assert restored.status == original.status
        assert restored.checksum == original.checksum

    def test_to_json(self):
        import json

        data = json.loads(_make_fragment().to_json())
        assert data["id"] == "F-0001"


class TestAppeal:
    def test_default_metrics_zero(self):
        appeal = Appeal(appeal_id="A-1")
        assert appeal.metrics.delta_token == 0
        assert appeal.status == AppealStatus.DRAFT

    def test_round_trip_dict(self):
        op = AppealOperation(
            op_type=OperationType.MERGE, params={"keep": True}, description="merge"
        )
        original = Appeal(
            appeal_id="A-9",
            sources=["F-1", "F-2"],
            ops=[op],
            diff=AppealDiff(added=["nuova"], removed=["vecchia"]),
            metrics=AppealMetrics(delta_token=3, coverage=0.7, risk=0.2, aggregate_score=0.5),
            status=AppealStatus.PENDING_USER_APPROVAL,
            proposed_content="testo proposto",
            explanation="spiegazione",
        )
        restored = Appeal.from_dict(original.to_dict())
        assert restored.appeal_id == "A-9"
        assert restored.sources == ["F-1", "F-2"]
        assert restored.ops[0].op_type == OperationType.MERGE
        assert restored.diff.added == ["nuova"]
        assert restored.metrics.coverage == 0.7
        assert restored.status == AppealStatus.PENDING_USER_APPROVAL
        assert restored.proposed_content == "testo proposto"

    def test_operation_type_values(self):
        assert OperationType.MERGE.value == "merge"
        assert AppealStatus.APPROVED.value == "approved"

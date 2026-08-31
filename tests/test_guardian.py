"""Tests for the FragmentGuardian quality gatekeeper."""

from memory_fragments.library.guardian import (
    FragmentGuardian,
    QUALITY_THRESHOLD,
    QUALITY_THRESHOLD_OVERRIDES,
)
from memory_fragments.models import Fragment, FragmentMetadata
from memory_fragments.models.quality import QualityProvenance, QualitySource


def _fragment(fid: str, quality: float, source: QualitySource | None = None) -> Fragment:
    provenance = None
    if source is not None:
        provenance = QualityProvenance(final_quality=quality, final_source=source)
    return Fragment(
        fragment_id=fid,
        content=(
            "Questa è una porzione di conoscenza sufficientemente lunga da essere "
            "considerata completa e ben strutturata per scopi di testing."
        ),
        metadata=FragmentMetadata(topic="test", quality=quality, provenance=provenance),
    )


def test_default_threshold():
    assert FragmentGuardian().threshold == QUALITY_THRESHOLD == 0.80


def test_source_overrides_defaults_are_present():
    overrides = FragmentGuardian().source_overrides
    assert overrides[QualitySource.USER_CLAIMED] == 0.95
    assert overrides[QualitySource.INDEPENDENTLY_VERIFIED] == 0.70
    assert len(overrides) == len(QUALITY_THRESHOLD_OVERRIDES)


def test_high_quality_fragment_accepted():
    guardian = FragmentGuardian()
    frag = _fragment("f1", quality=0.90)
    ok, result = guardian.guard(frag)
    assert ok is True
    assert result is not None
    assert result.fragment_id == "f1"


def test_result_is_deep_copy():
    guardian = FragmentGuardian()
    frag = _fragment("f2", quality=0.90)
    ok, result = guardian.guard(frag)
    result.metadata.topic = "mutated"
    # The original must be untouched
    assert frag.metadata.topic == "test"


def test_below_threshold_with_default_improver_boosted():
    guardian = FragmentGuardian()
    frag = _fragment("f3", quality=0.70)
    ok, result = guardian.guard(frag)
    assert ok is True
    assert result is not None
    # The heuristic improver appends "_improved" and boosts quality by 0.15
    assert result.fragment_id == "f3_improved"
    assert result.metadata.quality >= 0.80


def test_low_quality_never_recoverable_rejected():
    guardian = FragmentGuardian()
    frag = _fragment("f4", quality=0.10)
    ok, result = guardian.guard(frag)
    assert ok is False
    assert result is None


def test_custom_improver_returning_none_rejects():
    guardian = FragmentGuardian(improver=lambda fragment: None)
    frag = _fragment("f5", quality=0.70)
    ok, result = guardian.guard(frag)
    assert ok is False
    assert result is None


def test_custom_improver_can_save_fragment():
    def improver(fragment):
        improved = _fragment(fragment.fragment_id, quality=0.99)
        return improved

    guardian = FragmentGuardian(improver=improver)
    frag = _fragment("f6", quality=0.30)
    ok, result = guardian.guard(frag)
    assert ok is True
    assert result is not None
    assert result.metadata.quality == 0.99


def test_guard_many_splits_accepted_rejected():
    guardian = FragmentGuardian()
    accepted, rejected = guardian.guard_many(
        [_fragment("g1", quality=0.95), _fragment("g2", quality=0.20)]
    )
    assert len(accepted) == 1
    assert len(rejected) == 1
    assert accepted[0].fragment_id == "g1"
    assert rejected[0].fragment_id == "g2"


def test_user_claimed_needs_very_high_quality():
    # No improver, so a 0.90 USER_CLAIMED fragment (threshold 0.95) is rejected
    guardian = FragmentGuardian(improver=lambda f: None)
    frag = _fragment("h1", quality=0.90, source=QualitySource.USER_CLAIMED)
    # USER_CLAIMED override is 0.95 -> 0.90 is not enough
    assert guardian.get_threshold_for(frag) == 0.95
    ok, _ = guardian.guard(frag)
    assert ok is False


def test_user_claimed_at_threshold_accepted():
    guardian = FragmentGuardian()
    frag = _fragment("h2", quality=0.95, source=QualitySource.USER_CLAIMED)
    ok, result = guardian.guard(frag)
    assert ok is True
    assert result is not None


def test_independently_verified_lower_threshold():
    guardian = FragmentGuardian()
    frag = _fragment("h3", quality=0.75, source=QualitySource.INDEPENDENTLY_VERIFIED)
    assert guardian.get_threshold_for(frag) == 0.70
    ok, _ = guardian.guard(frag)
    assert ok is True


def test_custom_source_overrides():
    guardian = FragmentGuardian(
        source_overrides={QualitySource.MAJORITY_VOTE: 0.99},
        improver=lambda f: None,
    )
    frag = _fragment("h4", quality=0.80, source=QualitySource.MAJORITY_VOTE)
    assert guardian.get_threshold_for(frag) == 0.99
    ok, _ = guardian.guard(frag)
    assert ok is False


def test_threshold_setter():
    guardian = FragmentGuardian()
    guardian.threshold = 0.5
    assert guardian.threshold == 0.5


def test_no_provenance_uses_default_threshold():
    guardian = FragmentGuardian()
    frag = _fragment("h5", quality=0.85)
    assert guardian.get_threshold_for(frag) == 0.80

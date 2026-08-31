"""Test del FragmentGuardian — quality gatekeeper con soglie per QualitySource."""

from memory_fragments.library.guardian import FragmentGuardian
from memory_fragments.models import Fragment, FragmentMetadata
from memory_fragments.models.quality import QualityProvenance, QualitySource


def _fragment(quality: float, source: QualitySource = None) -> Fragment:
    fragment = Fragment(
        fragment_id="F-1",
        content="contenuto di esempio per il test del guardiano con parole sufficienti",
        metadata=FragmentMetadata(quality=quality),
    )
    if source is not None:
        fragment.metadata.provenance = QualityProvenance(
            final_quality=quality, final_source=source
        )
    return fragment


class TestGuard:
    def test_pass_through_high_quality(self):
        guardian = FragmentGuardian()
        accepted, result = guardian.guard(_fragment(0.9))
        assert accepted is True
        assert result is not None
        assert result.metadata.quality == 0.9

    def test_reject_low_quality_below_improvement(self):
        guardian = FragmentGuardian()
        # quality 0.50 -> boost 0.15 = 0.65 < 0.80 -> None
        accepted, result = guardian.guard(_fragment(0.50))
        assert accepted is False
        assert result is None

    def test_improve_then_accept(self):
        guardian = FragmentGuardian()
        # quality 0.75 -> boost 0.15 = 0.90 >= 0.80 -> accettato
        accepted, result = guardian.guard(_fragment(0.75))
        assert accepted is True
        assert result.fragment_id == "F-1_improved"

    def test_user_claimed_requires_high_threshold(self):
        # improver disabilitato: la soglia 0.95 deve essere superata dalla
        # qualità già dichiarata, non dal boost euristico
        guardian = FragmentGuardian(improver=lambda f: None)
        accepted, _ = guardian.guard(_fragment(0.90, QualitySource.USER_CLAIMED))
        assert accepted is False
        accepted, _ = guardian.guard(_fragment(0.96, QualitySource.USER_CLAIMED))
        assert accepted is True

    def test_independently_verified_lower_threshold(self):
        guardian = FragmentGuardian()
        # INDEPENDENTLY_VERIFIED -> soglia 0.70
        accepted, _ = guardian.guard(_fragment(0.72, QualitySource.INDEPENDENTLY_VERIFIED))
        assert accepted is True

    def test_guard_many_splits(self):
        guardian = FragmentGuardian()
        fragments = [_fragment(0.9), _fragment(0.3), _fragment(0.85)]
        accepted, rejected = guardian.guard_many(fragments)
        assert len(accepted) == 2
        assert len(rejected) == 1


class TestThreshold:
    def test_default_threshold(self):
        guardian = FragmentGuardian()
        assert guardian.threshold == 0.80

    def test_get_threshold_for_no_source(self):
        guardian = FragmentGuardian()
        assert guardian.get_threshold_for(_fragment(0.5)) == 0.80

    def test_custom_source_overrides(self):
        guardian = FragmentGuardian(source_overrides={QualitySource.USER_CLAIMED: 0.99})
        assert guardian.get_threshold_for(_fragment(0.5, QualitySource.USER_CLAIMED)) == 0.99

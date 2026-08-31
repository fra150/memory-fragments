"""Test dell'Evaluator — metriche automatiche per gli Appeal."""

from memory_fragments.config import EvaluatorConfig
from memory_fragments.engine.evaluator import Evaluator
from memory_fragments.models import Appeal, Fragment, FragmentMetadata


def _fragment(fragment_id: str, content: str, quality: float = 0.8) -> Fragment:
    return Fragment(
        fragment_id=fragment_id,
        content=content,
        metadata=FragmentMetadata(quality=quality),
    )


def _appeal(proposed: str, sources=None) -> Appeal:
    return Appeal(
        appeal_id="A-1",
        sources=sources or [],
        proposed_content=proposed,
    )


class TestEvaluate:
    def test_basic_metrics(self):
        evaluator = Evaluator()
        source = _fragment("F-1", "il gatto dorme sul divano")
        metrics = evaluator.evaluate(_appeal("il gatto dorme sul divano", ["F-1"]), [source])
        assert metrics.delta_token == 0
        assert metrics.coverage > 0.9
        assert 0.0 <= metrics.risk <= 1.0
        assert -1.0 <= metrics.aggregate_score <= 1.0

    def test_delta_token_positive_when_expansion(self):
        evaluator = Evaluator()
        source = _fragment("F-1", "breve testo")
        metrics = evaluator.evaluate(
            _appeal("breve testo molto molto molto più lungo", ["F-1"]), [source]
        )
        assert metrics.delta_token > 0

    def test_coverage_complete(self):
        evaluator = Evaluator()
        source = _fragment("F-1", "parola uno due tre")
        metrics = evaluator.evaluate(_appeal("parola uno due tre", ["F-1"]), [source])
        assert metrics.coverage == 1.0

    def test_risk_increases_with_novelty(self):
        evaluator = Evaluator()
        source = _fragment("F-1", "contesto conosciuto verificato")
        low_risk = evaluator.evaluate(
            _appeal("contesto conosciuto verificato", ["F-1"]), [source]
        ).risk
        high_risk = evaluator.evaluate(
            _appeal("invenzione totalmente nuova inesistente assurda", ["F-1"]), [source]
        ).risk
        assert high_risk >= low_risk

    def test_vagueness_penalty(self):
        evaluator = Evaluator()
        source = _fragment("F-1", "contesto chiaro")
        metrics = evaluator.evaluate(
            _appeal("forse probabilmente potrebbe sembrare circa", ["F-1"]), [source]
        )
        assert metrics.risk > 0.0

    def test_deterministic(self):
        evaluator = Evaluator()
        source = _fragment("F-1", "il gatto dorme sul divano")
        a = evaluator.evaluate(_appeal("il gatto dorme sul divano", ["F-1"]), [source])
        b = evaluator.evaluate(_appeal("il gatto dorme sul divano", ["F-1"]), [source])
        assert a.to_dict() == b.to_dict()

    def test_no_source_fragments(self):
        evaluator = Evaluator()
        metrics = evaluator.evaluate(_appeal("contenuto proposto"), [])
        assert metrics.coverage == 0.0

    def test_embedding_coverage_disabled_by_default(self):
        # Deve funzionare senza errore anche con use_embedding_coverage=False
        evaluator = Evaluator()
        source = _fragment("F-1", "il gatto dorme sul divano")
        metrics = evaluator.evaluate(
            _appeal("il gatto dorme sul divano", ["F-1"]), [source]
        )
        assert metrics.coverage > 0.0


class TestConfig:
    def test_custom_config_used(self):
        config = EvaluatorConfig(w_delta_token=1.0, w_coverage=0.0, w_risk=0.0)
        evaluator = Evaluator(config)
        assert evaluator.config.w_delta_token == 1.0

    def test_risk_threshold_default(self):
        assert EvaluatorConfig().risk_threshold == 0.1

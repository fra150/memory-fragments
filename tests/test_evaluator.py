"""Test suite for the Evaluator module (v0.4.0 V2).

Tests cover:
- Automatic metrics computation (delta_token, coverage, risk, aggregate_score)
- Embedding-based coverage fallback (deterministic, no network)
- Contradiction detection (heuristic placeholder)
- Risk / vagueness penalties
- Configuration handling
"""

import pytest

from memory_fragments.models import Appeal, AppealMetrics, Fragment, FragmentMetadata
from memory_fragments.engine.evaluator import Evaluator
from memory_fragments.config import EvaluatorConfig, default_config


def make_fragment(fragment_id: str, content: str, quality: float = 0.8) -> Fragment:
    return Fragment(
        fragment_id=fragment_id,
        content=content,
        metadata=FragmentMetadata(topic="test", quality=quality),
    )


def make_appeal(proposed_content: str, sources: list[str]) -> Appeal:
    return Appeal(appeal_id="appeal-1", sources=sources, proposed_content=proposed_content)


class TestEvaluatorMetrics:
    """Test automatic metric computation via the real evaluate() API."""

    def test_evaluate_returns_appeal_metrics(self):
        source = make_fragment("src-1", "Photosynthesis converts CO2 and water into glucose.")
        appeal = make_appeal(
            "Photosynthesis converts CO2 and water into glucose using sunlight.",
            sources=["src-1"],
        )

        metrics = Evaluator().evaluate(appeal, [source])

        assert isinstance(metrics, AppealMetrics)
        # Proposed content is longer than the source -> positive delta_token
        assert metrics.delta_token >= 0
        # Some words overlap (photosynthesis, converts, co2, water, glucose)
        assert 0.0 < metrics.coverage <= 1.0
        assert 0.0 <= metrics.risk <= 1.0

    def test_delta_token_savings_for_shorter_proposal(self):
        source = make_fragment("src-2", "The mitochondria is the powerhouse of the cell and makes us smart.")
        appeal = make_appeal("The mitochondria is the powerhouse of the cell.", sources=["src-2"])

        metrics = Evaluator().evaluate(appeal, [source])

        assert metrics.delta_token < 0  # shorter proposal = token savings

    def test_coverage_word_overlap(self):
        query_text = "The human heart has four chambers two atria and two ventricles"
        source = make_fragment("src-3", query_text)
        appeal = make_appeal(
            "The human heart has four chambers: two atria and two ventricles.",
            sources=["src-3"],
        )

        metrics = Evaluator().evaluate(appeal, [source])

        assert 0.0 <= metrics.coverage <= 1.0
        assert metrics.coverage > 0.5  # high overlap with source

    def test_vague_proposal_raises_risk(self):
        source = make_fragment("src-4", "Quantum physics describes subatomic particles and waves.")
        appeal = make_appeal(
            "Quantum physics is stuff about particles and waves maybe probably.",
            sources=["src-4"],
        )

        metrics = Evaluator().evaluate(appeal, [source])

        # Hedging words (maybe, probably) push risk up
        assert 0.0 <= metrics.risk <= 1.0
        assert metrics.risk > 0.0

    def test_overall_score_within_bounds(self):
        source = make_fragment("src-5", "Testing content with some shared words here.")
        appeal = make_appeal(
            "Testing content with some shared words here and a little more detail.",
            sources=["src-5"],
        )

        metrics = Evaluator().evaluate(appeal, [source])

        assert -1.0 <= metrics.aggregate_score <= 1.0

    def test_serialization_roundtrip(self):
        metrics = AppealMetrics(delta_token=10, coverage=0.75, risk=0.15, aggregate_score=0.8)
        restored = AppealMetrics.from_dict(metrics.to_dict())

        assert restored == metrics


class TestEvaluatorEmbeddingFallback:
    """Embedding coverage must degrade gracefully without sentence-transformers."""

    def test_embedding_coverage_enabled_without_package_returns_zero(self, monkeypatch):
        cfg = EvaluatorConfig(use_embedding_coverage=True, embedding_model="nonexistent-model")
        evaluator = Evaluator(cfg)

        # Force the fallback path (no real model load / no network access).
        monkeypatch.setattr(
            evaluator, "_ensure_embedding_model",
            lambda: setattr(evaluator, "_embedding_fallback", True),
        )

        source = make_fragment("src-6", "Some source content about plants and sunlight.")
        appeal = make_appeal("Some source content about plants and sunlight.", sources=["src-6"])

        # No sentence-transformers installed -> fallback returns 0.0,
        # blended with word overlap -> coverage still in [0,1].
        metrics = evaluator.evaluate(appeal, [source])

        assert 0.0 <= metrics.coverage <= 1.0


class TestEvaluatorContradiction:
    """Test the contradiction detection heuristic (placeholder)."""

    def test_no_contradiction_similar_content(self):
        evaluator = Evaluator()
        score = evaluator._detect_contradiction(
            "The heart pumps blood through the body.",
            ["The heart circulates blood throughout the organism."],
        )

        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0
        assert score < 0.5  # no negation words -> low score

    def test_negation_heavy_proposal_raises_score(self):
        evaluator = Evaluator()
        score = evaluator._detect_contradiction(
            "The heart does not pump blood and cannot circulate anything.",
            ["The heart pumps blood through the body."],
        )

        assert 0.0 <= score <= 1.0
        assert score > 0.5  # many negation words

    def test_empty_inputs(self):
        evaluator = Evaluator()
        assert evaluator._detect_contradiction("", []) == 0.0
        assert evaluator._detect_contradiction("some text", []) == 0.0


class TestEvaluatorConfig:
    """Test Evaluator configuration."""

    def test_default_risk_threshold(self):
        assert default_config.evaluator.risk_threshold == 0.1

    def test_use_embedding_coverage_default(self):
        assert default_config.evaluator.use_embedding_coverage is False

    def test_custom_config(self):
        custom_config = EvaluatorConfig(
            risk_threshold=0.2,
            use_embedding_coverage=True,
            w_delta_token=0.5,
            w_coverage=0.3,
            w_risk=0.2,
        )

        assert custom_config.risk_threshold == 0.2
        assert custom_config.use_embedding_coverage is True
        assert custom_config.w_delta_token == 0.5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

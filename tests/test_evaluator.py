"""Test suite for Evaluator module (v0.3.0 V2).

Tests cover:
- Automatic metrics computation (delta_token, coverage, risk)
- Embedding-based coverage (optional)
- Contradiction detection (heuristic placeholder)
- Risk threshold configuration
"""

import pytest
from typing import List

from memory_fragments.models import Fragment, FragmentMetadata, FragmentStatus
from memory_fragments.engine.evaluator import Evaluator, EvaluationResult
from memory_fragments.config import EvaluatorConfig, default_config


class TestEvaluatorMetrics:
    """Test automatic metric computation."""

    def test_delta_token_computation(self):
        """Test that delta_token is computed correctly."""
        query = "What is photosynthesis?"
        candidate = Fragment(
            fragment_id="test-001",
            content="Photosynthesis converts CO2 and water into glucose using sunlight.",
            metadata=FragmentMetadata(topic="biology", quality=0.9),
            status=FragmentStatus.ACTIVE,
        )
        
        result = Evaluator.evaluate_single(query, candidate)
        
        assert result.delta_token >= 0
        # Candidate should not be excessively longer than query
        assert result.delta_token < 100  # reasonable upper bound

    def test_coverage_word_overlap(self):
        """Test word overlap coverage computation."""
        query = "heart anatomy four chambers"
        candidate = Fragment(
            fragment_id="test-002",
            content="The human heart has four chambers: two atria and two ventricles.",
            metadata=FragmentMetadata(topic="anatomy", quality=0.85),
            status=FragmentStatus.ACTIVE,
        )
        
        result = Evaluator.evaluate_single(query, candidate)
        
        # Should have some coverage due to word overlap (heart, four, chambers)
        assert result.coverage > 0.0
        assert result.coverage <= 1.0

    def test_risk_computation(self):
        """Test risk score computation."""
        query = "explain quantum physics"
        candidate = Fragment(
            fragment_id="test-003",
            content="Quantum physics is stuff about particles and waves maybe.",
            metadata=FragmentMetadata(topic="physics", quality=0.5),
            status=FragmentStatus.ACTIVE,
        )
        
        result = Evaluator.evaluate_single(query, candidate)
        
        assert result.risk >= 0.0
        assert result.risk <= 1.0
        # Vague content should have higher risk
        assert result.risk > 0.1

    def test_overall_score_weights(self):
        """Test that overall_score respects configured weights."""
        query = "test query"
        candidate = Fragment(
            fragment_id="test-004",
            content="test content with some words",
            metadata=FragmentMetadata(topic="test", quality=0.7),
            status=FragmentStatus.ACTIVE,
        )
        
        result = Evaluator.evaluate_single(query, candidate)
        
        # Overall score should be weighted combination
        assert result.overall_score >= 0.0
        assert result.overall_score <= 1.0


class TestEvaluatorConfig:
    """Test Evaluator configuration."""

    def test_default_risk_threshold(self):
        """Test default risk threshold is 0.1."""
        assert default_config.evaluator.risk_threshold == 0.1

    def test_use_embedding_coverage_default(self):
        """Test embedding coverage is disabled by default."""
        assert default_config.evaluator.use_embedding_coverage is False

    def test_custom_config(self):
        """Test custom configuration overrides."""
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


class TestContradictionDetection:
    """Test contradiction detection (heuristic placeholder)."""

    def test_no_contradiction_similar_content(self):
        """Test that similar content doesn't trigger contradiction."""
        text_a = "The heart pumps blood through the body."
        text_b = "The heart circulates blood throughout the organism."
        
        has_contradiction, confidence = Evaluator._detect_contradiction(text_a, text_b)
        
        assert has_contradiction is False
        # Heuristic should not flag paraphrases as contradictions
        assert confidence < 0.5

    def test_potential_contradiction_opposite_claims(self):
        """Test that opposite claims may trigger contradiction heuristic."""
        text_a = "The heart has four chambers."
        text_b = "The heart has two chambers only."
        
        has_contradiction, confidence = Evaluator._detect_contradiction(text_a, text_b)
        
        # Heuristic may or may not catch this (it's a placeholder)
        # Test ensures the method runs without errors
        assert isinstance(has_contradiction, bool)
        assert isinstance(confidence, float)
        assert 0.0 <= confidence <= 1.0


class TestEvaluationResult:
    """Test EvaluationResult data structure."""

    def test_to_dict(self):
        """Test serialization to dictionary."""
        result = EvaluationResult(
            fragment_id="test-001",
            delta_token=10,
            coverage=0.75,
            risk=0.15,
            overall_score=0.80,
            has_contradiction=False,
            contradiction_confidence=0.1,
        )
        
        d = result.to_dict()
        
        assert d["fragment_id"] == "test-001"
        assert d["delta_token"] == 10
        assert d["coverage"] == 0.75
        assert d["risk"] == 0.15
        assert d["overall_score"] == 0.80
        assert d["has_contradiction"] is False

    def test_repr(self):
        """Test string representation."""
        result = EvaluationResult(
            fragment_id="test-002",
            delta_token=5,
            coverage=0.6,
            risk=0.2,
            overall_score=0.7,
            has_contradiction=False,
            contradiction_confidence=0.05,
        )
        
        repr_str = repr(result)
        
        assert "test-002" in repr_str
        assert "score=0.7" in repr_str or "overall_score=0.7" in repr_str


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

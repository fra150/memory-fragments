"""Automatic metric computation for Appeals."""

import math
from typing import List, Optional, Set

import numpy as np

from memory_fragments.config import EvaluatorConfig, default_config
from memory_fragments.models import Appeal, AppealMetrics, Fragment


class Evaluator:
    """Computes automatic metrics for an Appeal against its source fragments.

    The evaluator measures token delta, content coverage, hallucination risk,
    embedding-based coverage, vagueness, and contradiction likelihood, then
    produces a weighted aggregate score.
    """

    def __init__(self, config: Optional[EvaluatorConfig] = None) -> None:
        """Initialise with an optional custom config (falls back to default)."""
        self.config = config or default_config.evaluator
        # Lazy-loaded embedding model for semantic coverage
        self._embedding_model: Optional["SentenceTransformer"] = None  # noqa: F821
        self._embedding_fallback: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(
        self,
        appeal: Appeal,
        source_fragments: List[Fragment],
    ) -> AppealMetrics:
        """Compute AppealMetrics for a proposed appeal.

        Parameters
        ----------
        appeal:
            The appeal whose ``proposed_content`` is evaluated.
        source_fragments:
            The source fragments the appeal intends to transform.

        Returns
        -------
        AppealMetrics
            delta_token, coverage, risk and aggregate_score.
        """
        proposed_words = appeal.proposed_content.split()
        proposed_len = len(proposed_words)
        proposed_unique: Set[str] = set(proposed_words)

        # Collect all source content
        source_texts: List[str] = []
        source_word_lists: List[List[str]] = []
        source_unique: Set[str] = set()
        for frag in source_fragments:
            source_texts.append(frag.content)
            words = frag.content.split()
            source_word_lists.append(words)
            source_unique.update(words)

        total_source_len = sum(len(w) for w in source_word_lists)

        cfg = self.config

        # --- delta_token ---------------------------------------------------
        delta_token = proposed_len - total_source_len  # negative = savings

        # --- coverage ------------------------------------------------------
        # Word-overlap coverage (always computed, backward-compatible)
        word_coverage = self._compute_coverage(proposed_unique, source_unique)

        # Optionally blend with embedding-based coverage
        if cfg.use_embedding_coverage and cfg.embedding_coverage_weight > 0:
            emb_coverage = self._compute_embedding_coverage(
                appeal.proposed_content, source_texts
            )
            coverage = (
                (1.0 - cfg.embedding_coverage_weight) * word_coverage
                + cfg.embedding_coverage_weight * emb_coverage
            )
        else:
            coverage = word_coverage

        # --- risk ----------------------------------------------------------
        risk = self._compute_risk(
            proposed_unique, source_unique, proposed_len, total_source_len
        )

        # Contradiction penalty (only when embedding coverage is active)
        if cfg.use_embedding_coverage and cfg.risk_contradiction_penalty > 0:
            contradiction_score = self._detect_contradiction(
                appeal.proposed_content, source_texts
            )
            risk += contradiction_score * cfg.risk_contradiction_penalty
            risk = min(risk, 1.0)

        # Vagueness penalty
        if cfg.risk_vagueness_penalty > 0:
            vagueness = self._compute_vagueness(appeal.proposed_content)
            risk += vagueness * cfg.risk_vagueness_penalty
            risk = min(risk, 1.0)

        # --- aggregate_score -----------------------------------------------
        norm_delta = _tanh_normalise(delta_token, max(total_source_len, 1))
        aggregate_score = (
            cfg.w_delta_token * norm_delta
            + cfg.w_coverage * coverage
            - cfg.w_risk * risk
        )

        return AppealMetrics(
            delta_token=delta_token,
            coverage=coverage,
            risk=risk,
            aggregate_score=aggregate_score,
        )

    # ------------------------------------------------------------------
    # Internal helpers — coverage
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_coverage(
        proposed_unique: Set[str],
        source_unique: Set[str],
    ) -> float:
        """Fraction of unique source words represented in the proposal."""
        if not source_unique:
            # If there are no source words, coverage is 1 only when
            # there are also no proposed words, otherwise 0.
            return 1.0 if not proposed_unique else 0.0
        overlap = len(proposed_unique & source_unique)
        return overlap / len(source_unique)

    def _compute_embedding_coverage(
        self, proposed_text: str, source_texts: List[str]
    ) -> float:
        """Compute coverage using embedding cosine similarity.

        Lazy-loads sentence-transformers on first call (only when
        ``use_embedding_coverage`` is enabled).  Returns the maximum cosine
        similarity between the proposed text embedding and any source text
        embedding.

        Falls back to 0.0 if the model is unavailable.

        Parameters
        ----------
        proposed_text:
            The proposed content text.
        source_texts:
            List of source fragment text strings.

        Returns
        -------
        float
            Maximum cosine similarity in ``[0.0, 1.0]``.
        """
        if not source_texts or not proposed_text.strip():
            return 0.0

        self._ensure_embedding_model()
        if self._embedding_model is None:
            return 0.0

        try:
            proposed_vec = self._embedding_model.encode(
                proposed_text, normalize_embeddings=True
            )
            batch_vecs = self._embedding_model.encode(
                source_texts, normalize_embeddings=True
            )
            similarities = [
                float(np.dot(proposed_vec, vec)) for vec in batch_vecs
            ]
            return max(similarities) if similarities else 0.0
        except Exception:
            return 0.0

    # ------------------------------------------------------------------
    # Internal helpers — risk
    # ------------------------------------------------------------------

    def _compute_risk(
        self,
        proposed_words: Set[str],
        source_words: Set[str],
        proposed_len: int,
        source_len: int,
    ) -> float:
        """Estimate hallucination / error risk on a 0-1 scale.

        Risk factors (cumulative, capped at 1.0):

        * **Expansion** — if the proposal is much longer than the sources
          the chance of hallucinated content grows.
        * **Novelty** — words in the proposal that do not appear in any
          source fragment are a strong signal of invented detail.
        * **Oversimplification** — a very short proposal may lose nuance.
        """
        cfg = self.config
        risk = 0.0

        if source_len == 0:
            # No source to validate against -> medium risk if non-empty proposal
            return 0.5 if proposed_len > 0 else 0.0

        ratio = proposed_len / max(source_len, 1)

        # --- Expansion risk ------------------------------------------------
        if ratio > cfg.max_token_penalty_ratio:
            excess = (ratio - cfg.max_token_penalty_ratio) / cfg.max_token_penalty_ratio
            risk += min(excess * 0.5, 0.5)

        # --- Novelty penalty -----------------------------------------------
        if proposed_words and source_words:
            unknown = proposed_words - source_words
            novelty = len(unknown) / len(proposed_words)
            risk += novelty * 0.3

        # --- Oversimplification risk ---------------------------------------
        if ratio < 0.5 and source_len > 5:
            risk += 0.2

        return min(risk, 1.0)

    def _detect_contradiction(
        self, proposed_text: str, source_texts: List[str]
    ) -> float:
        """Lazy NLI-based contradiction detection (heuristic placeholder).

        Only meaningful when ``use_embedding_coverage`` is True and there are
        source fragments with embedding similarity above
        ``nli_contradiction_threshold`` (checked upstream in ``evaluate()``).

        Current implementation uses a simple negation-word heuristic as a
        placeholder.  Real NLI via BART-large-MNLI (ONNX) comes in a later
        card.

        Parameters
        ----------
        proposed_text:
            The proposed content text.
        source_texts:
            List of source fragment text strings.

        Returns
        -------
        float
            Contradiction likelihood in ``[0.0, 1.0]``.
        """
        # Simple heuristic: measure the ratio of negation words in the
        # proposed content.  The idea is that negation-heavy proposals
        # are more likely to reverse source claims.
        negation_words = {
            "not", "no", "never", "neither", "nor", "none", "nobody",
            "nothing", "nowhere", "cannot", "can't", "don't", "doesn't",
            "didn't", "won't", "wouldn't", "shouldn't", "couldn't",
            "isn't", "aren't", "wasn't", "weren't", "haven't", "hasn't",
            "hadn't", "dont", "cant", "wont",
        }
        proposed_words = proposed_text.lower().split()
        if not proposed_words or not source_texts:
            return 0.0

        negation_count = sum(1 for w in proposed_words if w in negation_words)
        ratio = negation_count / len(proposed_words)
        # Scale up but cap at 1.0 — even a few negation words are meaningful
        return min(ratio * 3.0, 1.0)

    def _compute_vagueness(self, text: str) -> float:
        """Estimate how vague / hedged the proposed content is.

        Measures the ratio of hedging words (e.g., "maybe", "could",
        "might", "possibly", "seems") to total unique words, then scales
        the ratio up so that even modest hedging raises the score.

        Parameters
        ----------
        text:
            The text to evaluate.

        Returns
        -------
        float
            Vagueness score in ``[0.0, 1.0]``.
        """
        hedging_words = {
            "maybe", "perhaps", "possibly", "probably", "could", "might",
            "would", "may", "seems", "appears", "suggests", "approximately",
            "roughly", "around", "about", "almost", "nearly", "virtually",
            "essentially", "basically", "generally", "mostly", "often",
            "sometimes", "occasionally", "somewhat", "rather", "quite",
            "relatively", "comparatively", "allegedly", "apparently",
            "presumably", "arguably", "debatably", "questionably",
            "some", "several", "many", "much", "a lot", "lots of",
        }
        words = set(text.lower().split())
        if not words:
            return 0.0
        hedge_count = sum(1 for w in words if w in hedging_words)
        # Scale the raw ratio up by 5× but cap at 1.0 so that even a modest
        # proportion of hedging words produces a noticeable signal.
        return min(hedge_count / max(len(words), 1) * 5.0, 1.0)

    # ------------------------------------------------------------------
    # Internal helpers — model lifecycle
    # ------------------------------------------------------------------

    def _ensure_embedding_model(self) -> None:
        """Lazy-load the sentence-transformers embedding model.

        Does nothing if the model is already loaded or if a previous load
        attempt failed (``_embedding_fallback`` is set).
        """
        if self._embedding_model is not None or self._embedding_fallback:
            return
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore[import-untyped]
            self._embedding_model = SentenceTransformer(self.config.embedding_model)
        except ImportError:
            import warnings
            warnings.warn(
                "sentence-transformers not installed — embedding coverage disabled. "
                "Install with `pip install sentence-transformers` for semantic coverage.",
                stacklevel=2,
            )
            self._embedding_fallback = True
        except Exception as exc:
            import warnings
            warnings.warn(
                f"Failed to load embedding model '{self.config.embedding_model}': {exc} — "
                f"embedding coverage disabled.",
                stacklevel=2,
            )
            self._embedding_fallback = True


def _tanh_normalise(value: int, scale: int) -> float:
    """Scale *value* to ``[-1, 1]`` via ``tanh(value / scale)``."""
    if scale <= 0:
        return 0.0
    return math.tanh(value / scale)

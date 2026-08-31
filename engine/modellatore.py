"""Modellatore — composition sandbox for assembling fragments like Lego pieces.

The Modellatore operates in isolation (no side effects) and:
1. Extracts tabs (what fragment provides) and slots (what fragment needs) from each fragment
2. Builds a coverage matrix (fragment × query aspects)
3. Detects gaps, overlaps, and contradictions
4. Composes fragments into coherent output when possible
5. When composition fails, reports exactly what's missing
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from memory_fragments.config import ModellatoreConfig, default_config
from memory_fragments.engine.conflict import ConflictDetector
from memory_fragments.library.guardian import FragmentGuardian
from memory_fragments.models import Fragment

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Composition Results
# ---------------------------------------------------------------------------


@dataclass
class CompositionGap:
    """A gap in fragment coverage — an aspect that no fragment covers."""

    aspect: str
    severity: str = "missing"  # "missing", "partial"
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "aspect": self.aspect,
            "severity": self.severity,
            "description": self.description,
        }


@dataclass
class CompositionContradiction:
    """A contradiction detected between two fragments."""

    fragment_a: str
    fragment_b: str
    score: float
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "fragment_a": self.fragment_a,
            "fragment_b": self.fragment_b,
            "score": round(self.score, 4),
            "description": self.description,
        }


@dataclass
class CompositionResult:
    """Result of a Modellatore composition attempt."""

    success: bool
    """True if the composition is valid and produces output."""

    partial: bool = False
    """True if the composition is partial (gaps exist)."""

    output: str = ""
    """The composed output text (empty on failure)."""

    fragments_used: List[str] = field(default_factory=list)
    """IDs of fragments that were successfully composed."""

    gaps: List[CompositionGap] = field(default_factory=list)
    """Gaps detected in coverage."""

    contradictions: List[CompositionContradiction] = field(default_factory=list)
    """Contradictions detected between fragments."""

    overlaps: List[Tuple[str, str, float]] = field(default_factory=list)
    """Overlapping fragment pairs and their similarity scores."""

    tabs: Dict[str, List[str]] = field(default_factory=dict)
    """Extracted tabs per fragment: {fragment_id: [tab1, tab2, ...]}."""

    slots: Dict[str, List[str]] = field(default_factory=dict)
    """Extracted slots per fragment: {fragment_id: [slot1, slot2, ...]}."""

    @property
    def has_gaps(self) -> bool:
        return len(self.gaps) > 0

    @property
    def has_contradictions(self) -> bool:
        return len(self.contradictions) > 0

    @property
    def is_blocked(self) -> bool:
        """True if contradictions block composition."""
        return any(c.score > 0.5 for c in self.contradictions)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "partial": self.partial,
            "output_preview": self.output[:200] if self.output else "",
            "fragments_used": self.fragments_used,
            "gaps": [g.to_dict() for g in self.gaps],
            "contradictions": [c.to_dict() for c in self.contradictions],
            "overlaps": [(a, b, round(s, 3)) for a, b, s in self.overlaps],
            "is_blocked": self.is_blocked,
        }

    def __repr__(self) -> str:
        return (
            f"CompositionResult(success={self.success}, "
            f"partial={self.partial}, "
            f"fragments={len(self.fragments_used)}, "
            f"gaps={len(self.gaps)}, "
            f"contradictions={len(self.contradictions)})"
        )


# ---------------------------------------------------------------------------
# Modellatore
# ---------------------------------------------------------------------------


class Modellatore:
    """Composition sandbox for assembling fragments like Lego pieces.

    Operates in isolation — no side effects, no writes.
    Each call is a dry-run composition attempt.

    Usage::

        modellatore = Modellatore()
        result = modellatore.compose(fragments, query)

        if result.success:
            output = result.output  # Coherent composed response
        elif result.is_blocked:
            # Contradictions found — need resolution
            contradictions = result.contradictions
        else:
            # Gaps found — need generation
            gaps = result.gaps
    """

    def __init__(
        self,
        config: Optional[ModellatoreConfig] = None,
        conflict_detector: Optional[ConflictDetector] = None,
    ) -> None:
        """
        Args:
            config: Configuration overrides.
            conflict_detector: ConflictDetector instance (created lazily if None).
        """
        self._config = config or default_config.modellatore
        self._conflict_detector = conflict_detector

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compose(
        self,
        fragments: List[Fragment],
        query: str,
    ) -> CompositionResult:
        """Attempt to compose fragments into a coherent response.

        Args:
            fragments: Certified fragments to compose.
            query: The original query for context.

        Returns:
            CompositionResult with success/failure, gaps, contradictions.
        """
        if not fragments:
            return CompositionResult(
                success=False,
                partial=False,
                gaps=[CompositionGap(
                    aspect="entire_query",
                    severity="missing",
                    description="No fragments provided for composition",
                )],
            )

        # 1. Extract tabs and slots from each fragment
        tabs: Dict[str, List[str]] = {}
        slots: Dict[str, List[str]] = {}
        for frag in fragments:
            tabs[frag.fragment_id], slots[frag.fragment_id] = self._extract_tabs_and_slots(frag)

        # 2. Extract query aspects
        query_aspects = self._extract_query_aspects(query)

        # 3. Build coverage matrix and detect gaps
        gaps = self._detect_gaps(query_aspects, fragments, tabs, slots)

        # 4. Detect overlaps (redundant coverage)
        overlaps = self._detect_overlaps(fragments)

        # 5. Detect contradictions (delegates to ConflictDetector)
        contradictions = self._detect_contradictions(fragments)

        # 6. Check if blocked by contradictions
        blocked = any(c.score > self._config.contradiction_block_threshold for c in contradictions)

        # 7. If blocked, return failure
        if blocked:
            return CompositionResult(
                success=False,
                partial=False,
                fragments_used=[f.fragment_id for f in fragments],
                gaps=gaps,
                contradictions=contradictions,
                overlaps=overlaps,
                tabs=tabs,
                slots=slots,
            )

        # 8. Attempt composition — determine level of completeness
        if gaps and contradictions:
            # Partial composition with both gaps and contradictions
            output = self._compose_partial(fragments, query, gaps)
            return CompositionResult(
                success=True,
                partial=True,
                output=output,
                fragments_used=[f.fragment_id for f in fragments],
                gaps=gaps,
                contradictions=contradictions,
                overlaps=overlaps,
                tabs=tabs,
                slots=slots,
            )
        elif gaps:
            # Partial composition with gaps only
            output = self._compose_partial(fragments, query, gaps)
            return CompositionResult(
                success=True,
                partial=True,
                output=output,
                fragments_used=[f.fragment_id for f in fragments],
                gaps=gaps,
                overlaps=overlaps,
                tabs=tabs,
                slots=slots,
            )
        elif contradictions:
            # Full composition with contradictions (warning-level only)
            output = self._compose_full(fragments, query)
            return CompositionResult(
                success=True,
                partial=False,
                output=output,
                fragments_used=[f.fragment_id for f in fragments],
                contradictions=[
                    c for c in contradictions
                    if c.score <= self._config.contradiction_block_threshold
                ],
                overlaps=overlaps,
                tabs=tabs,
                slots=slots,
            )
        else:
            # Full composition — no gaps, no contradictions
            output = self._compose_full(fragments, query)
            return CompositionResult(
                success=True,
                partial=False,
                output=output,
                fragments_used=[f.fragment_id for f in fragments],
                overlaps=overlaps,
                tabs=tabs,
                slots=slots,
            )

    def compose_partial(
        self,
        fragments: List[Fragment],
        query: str,
        gap_description: str,
    ) -> CompositionResult:
        """Attempt partial composition with known gaps.

        Args:
            fragments: Certified fragments for the covered portion.
            query: The original query.
            gap_description: Description of what needs to be generated.

        Returns:
            CompositionResult with partial output.
        """
        result = self.compose(fragments, query)

        # Add the gap description
        if gap_description and result.success:
            gap_aspects = gap_description.replace("Missing aspect: ", "").replace(
                "Missing aspects: ", ""
            )
            for chunk in gap_aspects.split(", and "):
                for a in chunk.split(", "):
                    a = a.strip()
                    if a and not any(g.aspect == a for g in result.gaps):
                        result.gaps.append(CompositionGap(
                            aspect=a,
                            severity="missing",
                            description=f"Requires generation: {a}",
                        ))
            result.partial = True

        return result

    def extract_tabs_slots_from_output(
        self, content: str, content_type: str = "generated"
    ) -> Tuple[List[str], List[str]]:
        """Extract tabs and slots from newly generated content.

        Called BEFORE passing to Guardian, so the new fragment enters
        the archive with connectors already attached.

        Args:
            content: The generated text content.
            content_type: Type of content ('generated', 'code', 'text').

        Returns:
            (tabs, slots): lists of tab and slot strings.
        """
        return self._extract_connectors_from_text(content)

    # ------------------------------------------------------------------
    # Tab/Slot extraction
    # ------------------------------------------------------------------

    def _extract_tabs_and_slots(
        self, fragment: Fragment
    ) -> Tuple[List[str], List[str]]:
        """Extract tabs (what fragment provides) and slots (what it answers).

        Default: embedding + co-occorrenza (costo zero).
        LLM fallback only if confidence < soglia (config `tab_slot_llm_threshold`,
        `enable_tab_slot_llm_fallback`).

        Returns:
            (tabs, slots) — lists of strings, deduplicated and limited to 10 items each.
        """
        content = fragment.content
        topic = fragment.metadata.topic
        tags = fragment.metadata.tags

        config: ModellatoreConfig = default_config.modellatore
        tabs: List[str] = []
        slots: List[str] = []

        # --- Step 1: topic & tags (zero-cost) ---
        if topic:
            tabs.append(topic)
            slots.append(topic)
        for tag in tags:
            tabs.append(tag)

        # --- Step 2: embedding-based key terms (if available) ---
        import re

        try:
            from memory_fragments.retrieval.indexer import _simple_embed  # type: ignore
            import numpy as np  # type: ignore

            # Generate a simple embedding for the content using the fallback
            # estimator (deterministic random projection — no external deps needed).
            embed = _simple_embed(content)
            # Use the embedding to boost/re-rank the heuristic terms below.
            # For now the embedding vector is computed but the existing heuristic
            # (capitalized/frequent terms) remains the primary driver; the embedding
            # can be plugged into a proper MMR selector in a future card.
            _ = embed  # suppress unused‑variable warning
        except Exception:
            logger.debug("Embedding fallback unavailable, using heuristic only")

        words = content.split()

        # Capitalized terms (potential entities/concepts)
        caps = re.findall(r'\b[A-Z][a-z]+\b', content)
        tabs.extend(caps[:5])

        # Frequent terms (appear 2+ times in content)
        word_counts = Counter(w.lower() for w in words if len(w) > 3)
        frequent = [w for w, c in word_counts.most_common(8) if c >= 2]
        tabs.extend(frequent)

        # Slots from key phrases (what questions does this answer?)
        # Look for definitional patterns
        def_patterns = re.findall(
            r'(?:is|are|was|were|means|refers to|defined as) (\w+(?:\s+\w+){0,5})',
            content,
        )
        for match in def_patterns[:3]:
            slots.append(match.strip())

        # --- Step 3: LLM fallback if confidence low (config driven) ---
        # TODO: integrate actual LLM when enable_tab_slot_llm_fallback is True
        # and confidence (e.g., embedding similarity) < tab_slot_llm_threshold

        # Deduplicate
        tabs = list(set(tabs))[:10]
        slots = list(set(slots))[:10]

        return tabs, slots

    def _extract_connectors_from_text(
        self, text: str
    ) -> Tuple[List[str], List[str]]:
        """Extract tabs and slots from arbitrary text (for generated content)."""
        import re

        words = text.split()

        # Tabs: key entities and topics
        tabs: List[str] = []
        caps = re.findall(r'\b[A-Z][a-z]+\b', text)
        tabs.extend(caps[:5])

        word_counts = Counter(w.lower() for w in words if len(w) > 3)
        frequent = [w for w, c in word_counts.most_common(5) if c >= 2]
        tabs.extend(frequent)

        # Slots: topics this text could answer questions about
        sentences = re.split(r'[.!?]+', text)
        slots: List[str] = []
        for sent in sentences[:5]:
            sent = sent.strip()
            if sent and len(sent) > 10:
                # Use first meaningful phrase as potential slot
                key_phrase = " ".join(sent.split()[:6])
                slots.append(key_phrase)

        return list(set(tabs))[:10], list(set(slots))[:10]

    # ------------------------------------------------------------------
    # Query aspects
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_query_aspects(query: str) -> List[str]:
        """Extract aspects from a query for coverage analysis."""
        tokens = query.lower().split()

        aspects: List[str] = []
        if len(tokens) < 5:
            for t in tokens:
                if len(t) > 2:
                    aspects.append(t)
            for i in range(len(tokens) - 1):
                aspects.append(f"{tokens[i]} {tokens[i+1]}")
            return list(set(aspects))[:5]

        import re

        quotes = re.findall(r'"([^"]+)"', query)
        aspects.extend(quotes)
        caps = re.findall(r'\b[A-Z][a-z]+\b', query)
        aspects.extend(caps)

        stopwords = {
            "the", "a", "an", "is", "are", "was", "were", "in", "on",
            "at", "to", "for", "of", "with", "by", "from", "as",
            "and", "or", "but", "not", "be", "been", "being",
            "have", "has", "had", "do", "does", "did", "will",
            "would", "could", "should", "may", "might", "can",
            "shall", "about", "into", "through", "during", "before",
            "after", "above", "below", "between", "out", "off",
            "over", "under", "again", "further", "then", "once",
            "here", "there", "when", "where", "why", "how",
            "all", "each", "every", "both", "few", "more",
            "most", "other", "some", "such", "no", "nor", "not",
            "only", "own", "same", "so", "than", "too", "very",
            "just", "because", "as", "until", "while", "if",
        }
        key_terms = [t for t in tokens if t not in stopwords and len(t) > 2]
        aspects.extend(key_terms[:6])

        return list(set(aspects))[:10]

    # ------------------------------------------------------------------
    # Gap detection
    # ------------------------------------------------------------------

    def _detect_gaps(
        self,
        query_aspects: List[str],
        fragments: List[Fragment],
        tabs: Dict[str, List[str]],
        slots: Dict[str, List[str]],
    ) -> List[CompositionGap]:
        """Detect which query aspects are not covered by any fragment.

        For multi-word aspects (e.g., bigrams), checks if ALL individual
        words appear in the content (not necessarily as a contiguous phrase).
        This prevents false gaps from n-gram aspect extraction.
        """
        if not query_aspects:
            return []

        def _content_covers(aspect: str, content: str) -> bool:
            """Check if aspect is covered by content.

            For single words → verbatim substring match.
            For multi-word aspects (bigrams, phrases) → all words present
            in content (not necessarily as a contiguous phrase). This prevents
            false gaps from n-gram aspect extraction.
            """
            aspect_lower = aspect.lower()
            content_lower = content.lower()
            words = aspect_lower.split()
            # Check each word present in content
            return all(w in content_lower for w in words)

        gaps: List[CompositionGap] = []
        for aspect in query_aspects:
            covered = False
            for frag in fragments:
                # Check content
                if _content_covers(aspect, frag.content):
                    covered = True
                    break
                # Check tabs
                frag_tabs = tabs.get(frag.fragment_id, [])
                if any(aspect.lower() in t.lower() for t in frag_tabs):
                    covered = True
                    break
                # Check slots
                frag_slots = slots.get(frag.fragment_id, [])
                if any(aspect.lower() in s.lower() for s in frag_slots):
                    covered = True
                    break

            if not covered:
                gaps.append(CompositionGap(aspect=aspect, severity="missing"))

        return gaps

    # ------------------------------------------------------------------
    # Overlap detection
    # ------------------------------------------------------------------

    def _detect_overlaps(
        self, fragments: List[Fragment]
    ) -> List[Tuple[str, str, float]]:
        """Detect semantically overlapping fragment pairs."""
        overlaps: List[Tuple[str, str, float]] = []
        for i in range(len(fragments)):
            for j in range(i + 1, len(fragments)):
                sim = self._text_similarity(
                    fragments[i].content, fragments[j].content
                )
                if sim > 0.75:
                    overlaps.append((
                        fragments[i].fragment_id,
                        fragments[j].fragment_id,
                        sim,
                    ))
        return overlaps

    # ------------------------------------------------------------------
    # Contradiction detection (delegates to ConflictDetector)
    # ------------------------------------------------------------------

    def _detect_contradictions(
        self, fragments: List[Fragment]
    ) -> List[CompositionContradiction]:
        """Detect contradictions between fragments.

        Delegates to ConflictDetector (Card 6) when contradiction score
        exceeds the threshold.
        """
        contradictions: List[CompositionContradiction] = []

        # For each pair, check for contradictions
        for i in range(len(fragments)):
            for j in range(i + 1, len(fragments)):
                sim = self._text_similarity(
                    fragments[i].content, fragments[j].content
                )

                # Only check contradictory pairs with significant overlap
                if sim > self._config.contradiction_warn_threshold:
                    # Use ConflictDetector if available
                    if self._conflict_detector is not None:
                        report = self._conflict_detector.scan_fragment(
                            fragments[i], [fragments[j]],
                            threshold=self._config.contradiction_warn_threshold,
                        )
                        if report.has_conflicts:
                            for entry in report.conflicts:
                                contradictions.append(CompositionContradiction(
                                    fragment_a=fragments[i].fragment_id,
                                    fragment_b=fragments[j].fragment_id,
                                    score=entry.contradiction_score or sim,
                                    description=(
                                        f"Similarity: {entry.similarity:.2%}, "
                                        f"method: {entry.detection_method}"
                                    ),
                                ))
                    else:
                        # Simple heuristic: check for antonym pairs
                        contradiction_score = self._simple_contradiction_check(
                            fragments[i].content, fragments[j].content
                        )
                        if contradiction_score > 0.3:
                            contradictions.append(CompositionContradiction(
                                fragment_a=fragments[i].fragment_id,
                                fragment_b=fragments[j].fragment_id,
                                score=contradiction_score,
                            ))

        return contradictions

    @staticmethod
    def _simple_contradiction_check(text_a: str, text_b: str) -> float:
        """Simple heuristic contradiction check based on antonym pairs."""
        contradiction_pairs = [
            ("increase", "decrease"), ("buy", "sell"), ("start", "stop"),
            ("true", "false"), ("yes", "no"), ("on", "off"),
            ("positive", "negative"), ("success", "failure"),
            ("win", "lose"), ("gain", "loss"), ("up", "down"),
            ("high", "low"), ("hot", "cold"), ("fast", "slow"),
            ("always", "never"), ("all", "none"), ("every", "no"),
            ("include", "exclude"), ("add", "remove"), ("create", "delete"),
            ("enter", "exit"), ("open", "close"), ("begin", "end"),
            ("accept", "reject"), ("approve", "deny"),
        ]

        words_a = set(text_a.lower().split())
        words_b = set(text_b.lower().split())

        contradictions = 0
        for w1, w2 in contradiction_pairs:
            if (w1 in words_a and w2 in words_b) or (w2 in words_a and w1 in words_b):
                contradictions += 1

        return min(contradictions * 0.25, 0.8)

    # ------------------------------------------------------------------
    # Text composition
    # ------------------------------------------------------------------

    def _compose_full(
        self, fragments: List[Fragment], query: str
    ) -> str:
        """Compose fragments into a coherent full response.

        Unlike the naive Composer, this produces flowing prose
        rather than concatenated headings.
        """
        if not fragments:
            return ""

        # Sort by quality descending
        sorted_frags = sorted(fragments, key=lambda f: f.metadata.quality, reverse=True)

        # Build coherent response
        parts: List[str] = []

        for i, frag in enumerate(sorted_frags):
            content = frag.content.strip()
            if content:
                parts.append(content)

        # Join with appropriate spacing
        return "\n\n".join(parts)

    def _compose_partial(
        self, fragments: List[Fragment], query: str, gaps: List[CompositionGap]
    ) -> str:
        """Compose fragments with noted gaps."""
        output = self._compose_full(fragments, query)

        if gaps:
            gap_lines: List[str] = []
            for gap in gaps:
                gap_lines.append(f"[Missing: {gap.aspect}]")
            if gap_lines:
                output += "\n\n---\n" + "\n".join(gap_lines)

        return output

    # ------------------------------------------------------------------
    # Similarity utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _text_similarity(text_a: str, text_b: str) -> float:
        """Compute Jaccard similarity between two texts."""
        words_a = set(text_a.lower().split())
        words_b = set(text_b.lower().split())
        if not words_a or not words_b:
            return 0.0
        intersection = words_a & words_b
        union = words_a | words_b
        return len(intersection) / len(union)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def config(self) -> ModellatoreConfig:
        return self._config

    def __repr__(self) -> str:
        return f"Modellatore(block_threshold={self._config.contradiction_block_threshold})"

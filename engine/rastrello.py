"""Rastrello (Rake) — automatic pattern discovery from code, text, and documents.

Scans input to identify recurring patterns — functions, constants, libraries,
and phrases — and creates certified fragments from them. The Rastrello is
self-adaptive: its frequency thresholds adjust based on how often the
Dispatcher actually uses the fragments it discovers.

Feedback loop: Rastrello -> Guardian -> Archive -> Dispatcher uses
                                                    |
                                                    +-- feedback -> Rastrello adjusts threshold
"""

from __future__ import annotations

import hashlib
import logging
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

from memory_fragments.config import RastrelloConfig, default_config
from memory_fragments.models import (
    Fragment,
    FragmentConditions,
    FragmentMetadata,
    FragmentStatus,
)
from memory_fragments.models.quality import (
    QualityEvaluation,
    QualityProvenance,
    QualitySource,
)
from memory_fragments.archive.static import StaticArchive
from memory_fragments.library.guardian import FragmentGuardian
from memory_fragments.engine.intake import IntakeVerifier

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Fragment Type
# ---------------------------------------------------------------------------


class FragmentType(str, Enum):
    """Type of fragment discovered by the Rastrello."""

    FUNCTION = "function"    # Functions, methods, classes
    CONSTANT = "constant"    # Constants, configuration values
    LIBRARY = "library"      # Imports, dependencies, libraries
    PHRASE = "phrase"        # Textual phrases, boilerplate, patterns


# ---------------------------------------------------------------------------
# Pattern tracking
# ---------------------------------------------------------------------------


@dataclass
class PatternSignature:
    """Unique signature for a discovered pattern."""

    fragment_type: FragmentType
    content_hash: str
    """SHA-256 hash of the canonical content."""

    name: str = ""
    """Human-readable name (e.g., function name, constant name)."""

    context: str = ""
    """Context where the pattern was found (e.g., module name, file path)."""

    def to_key(self) -> str:
        return f"{self.fragment_type.value}:{self.content_hash[:12]}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.fragment_type.value,
            "content_hash": self.content_hash[:12],
            "name": self.name,
            "context": self.context,
        }


@dataclass
class PatternFrequency:
    """Frequency tracker for a single pattern."""

    signature: PatternSignature
    count: int = 0
    first_seen: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_seen: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    cluster_id: str = ""
    """ID for grouping similar patterns."""

    # Dynamic threshold
    current_threshold: int = 0
    times_used: int = 0
    times_ignored: int = 0

    def increment(self) -> None:
        self.count += 1
        self.last_seen = datetime.now(timezone.utc)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "signature": self.signature.to_dict(),
            "count": self.count,
            "first_seen": self.first_seen.isoformat(),
            "last_seen": self.last_seen.isoformat(),
            "cluster_id": self.cluster_id,
            "current_threshold": self.current_threshold,
            "times_used": self.times_used,
            "times_ignored": self.times_ignored,
            "ready": self.is_ready,
        }

    @property
    def is_ready(self) -> bool:
        """True if this pattern has been seen enough times to be a candidate."""
        return self.count >= self.current_threshold if self.current_threshold > 0 else False

    def __repr__(self) -> str:
        return (
            f"PatternFrequency(name={self.signature.name}, "
            f"type={self.signature.fragment_type.value}, "
            f"count={self.count}/{self.current_threshold})"
        )


# ---------------------------------------------------------------------------
# Extraction results
# ---------------------------------------------------------------------------


@dataclass
class ExtractedPattern:
    """A pattern extracted from input."""

    fragment_type: FragmentType
    name: str
    content: str
    context: str = ""
    tags: List[str] = field(default_factory=list)
    content_hash: str = ""

    def __post_init__(self) -> None:
        if not self.content_hash:
            self.content_hash = hashlib.sha256(
                self.content.encode("utf-8")
            ).hexdigest()

    def to_fragment(self) -> Fragment:
        """Convert to a Fragment for Guardian certification."""
        provenance = QualityProvenance(
            final_quality=0.80,
            final_source=QualitySource.RASTRELLO_DISCOVERED,
        )
        provenance.add_evaluation(QualityEvaluation(
            score=0.80,
            source=QualitySource.RASTRELLO_DISCOVERED,
            model_id="rastrello",
            model_version="1.0",
            metadata={
                "pattern_type": self.fragment_type.value,
                "pattern_name": self.name,
                "context": self.context,
            },
        ))

        return Fragment(
            fragment_id=f"R-{self.content_hash[:8]}",
            content=self.content,
            metadata=FragmentMetadata(
                topic=self.name,
                quality=0.80,
                tags=self.tags + [self.fragment_type.value],
                source=f"rastrello:{self.context}",
                provenance=provenance,
            ),
            conditions=FragmentConditions(
                keywords=[self.name.lower()] + self.tags,
                semantic_threshold=0.72,
            ),
            status=FragmentStatus.ACTIVE,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.fragment_type.value,
            "name": self.name,
            "content_preview": self.content[:100],
            "context": self.context,
            "tags": self.tags,
            "hash": self.content_hash[:12],
        }


# ---------------------------------------------------------------------------
# Dedup result
# ---------------------------------------------------------------------------


class DedupResult:
    """Result of a dedup check."""

    NEW = "new"
    EXISTS = "exists"
    SIMILAR_EXISTS = "similar_exists"


# ---------------------------------------------------------------------------
# Rastrello
# ---------------------------------------------------------------------------


class Rastrello:
    """Automatic pattern discovery engine.

    Discovers reusable patterns from input and creates certified fragments.
    Self-adaptive: adjusts frequency thresholds based on Dispatcher feedback.

    Usage::

        rastrello = Rastrello(guardian, archive, intake)

        # Scan a codebase
        results = rastrello.scan_code(code_string, context="auth_module")

        # Scan text
        results = rastrello.scan_text(document_text, context="api_docs")

        # Process results
        for result in results:
            print(f"Discovered: {result.name} ({result.fragment_type.value})")

        # Provide feedback from Dispatcher
        rastrello.record_usage(fragment_id)
        rastrello.record_ignored(fragment_id)
    """

    def __init__(
        self,
        guardian: FragmentGuardian,
        archive: StaticArchive,
        intake: IntakeVerifier,
        config: Optional[RastrelloConfig] = None,
    ) -> None:
        """
        Args:
            guardian: Guardian for certification.
            archive: Archive for storing discovered fragments.
            intake: Intake Verifier for dedup checks.
            config: Configuration overrides.
        """
        self._guardian = guardian
        self._archive = archive
        self._intake = intake
        self._config = config or default_config.rastrello

        # Frequency trackers: key -> PatternFrequency
        self._trackers: Dict[str, PatternFrequency] = {}

        # Feedback state
        self._feedback_log: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Public API -- Scanning
    # ------------------------------------------------------------------

    def scan_code(
        self,
        code: str,
        context: str = "",
        language: str = "python",
    ) -> List[ExtractedPattern]:
        """Scan source code for reusable patterns.

        Args:
            code: Source code to scan.
            context: Context label (e.g., module name, file path).
            language: Programming language (currently supports 'python').

        Returns:
            List of newly discovered patterns (empty if none new).
        """
        patterns: List[ExtractedPattern] = []

        if language == "python":
            patterns.extend(self._extract_python_functions(code, context))
            patterns.extend(self._extract_python_constants(code, context))
            patterns.extend(self._extract_python_imports(code, context))

        return self._process_patterns(patterns)

    def scan_text(
        self,
        text: str,
        context: str = "",
        min_phrase_length: int = 10,
    ) -> List[ExtractedPattern]:
        """Scan text for reusable phrases and patterns.

        Args:
            text: Text content to scan.
            context: Context label.
            min_phrase_length: Minimum words for a phrase pattern.

        Returns:
            List of newly discovered patterns.
        """
        patterns: List[ExtractedPattern] = []
        patterns.extend(self._extract_phrases(text, context, min_phrase_length))
        return self._process_patterns(patterns)

    def scan_document(
        self,
        content: str,
        context: str = "",
        content_type: str = "text",
    ) -> List[ExtractedPattern]:
        """Scan a document (auto-detects type).

        Args:
            content: Document content.
            context: Context label.
            content_type: 'code', 'text', or 'auto' (default).

        Returns:
            List of newly discovered patterns.
        """
        if content_type == "code" or (content_type == "auto" and self._looks_like_code(content)):
            return self.scan_code(content, context)
        return self.scan_text(content, context)

    # ------------------------------------------------------------------
    # Public API -- Feedback
    # ------------------------------------------------------------------

    def record_usage(self, fragment_id: str) -> None:
        """Record that a Rastrello-discovered fragment was used by the Dispatcher.

        This LOWERS the frequency threshold for similar patterns.
        """
        for tracker in self._trackers.values():
            if tracker.signature.content_hash[:8] in fragment_id:
                old_threshold = tracker.current_threshold
                tracker.times_used += 1
                self._adjust_threshold(tracker)
                self._feedback_log.append({
                    "fragment_id": fragment_id,
                    "action": "used",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "threshold_before": old_threshold,
                    "threshold_after": tracker.current_threshold,
                })
                return

        logger.debug("record_usage: fragment '%s' not found in trackers", fragment_id)

    def record_ignored(self, fragment_id: str) -> None:
        """Record that a Rastrello-discovered fragment was ignored by the Dispatcher.

        This RAISES the frequency threshold for similar patterns.
        """
        for tracker in self._trackers.values():
            if tracker.signature.content_hash[:8] in fragment_id:
                old_threshold = tracker.current_threshold
                tracker.times_ignored += 1
                self._adjust_threshold(tracker)
                self._feedback_log.append({
                    "fragment_id": fragment_id,
                    "action": "ignored",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "threshold_before": old_threshold,
                    "threshold_after": tracker.current_threshold,
                })
                return

        logger.debug("record_ignored: fragment '%s' not found in trackers", fragment_id)

    # ------------------------------------------------------------------
    # Public API -- Queries
    # ------------------------------------------------------------------

    def get_statistics(self) -> Dict[str, Any]:
        """Return Rastrello statistics."""
        by_type: Dict[str, int] = {}
        ready_count = 0
        for tracker in self._trackers.values():
            t = tracker.signature.fragment_type.value
            by_type[t] = by_type.get(t, 0) + 1
            if tracker.is_ready:
                ready_count += 1

        return {
            "total_patterns": len(self._trackers),
            "ready_candidates": ready_count,
            "by_type": by_type,
            "feedback_events": len(self._feedback_log),
        }

    def list_trackers(self) -> List[PatternFrequency]:
        """Return all frequency trackers."""
        return list(self._trackers.values())

    def list_ready_candidates(self) -> List[PatternFrequency]:
        """Return trackers ready for certification."""
        return [t for t in self._trackers.values() if t.is_ready]

    # ------------------------------------------------------------------
    # Pattern extraction -- Python code
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_python_functions(code: str, context: str) -> List[ExtractedPattern]:
        """Extract functions and classes from Python code via AST."""
        patterns: List[ExtractedPattern] = []

        try:
            import ast
            tree = ast.parse(code)

            for node in ast.walk(tree):
                # Function definitions
                if isinstance(node, ast.FunctionDef):
                    func_code = ast.get_source_segment(code, node) or ""
                    if len(func_code.splitlines()) >= 2:  # At least 2 lines
                        patterns.append(ExtractedPattern(
                            fragment_type=FragmentType.FUNCTION,
                            name=node.name,
                            content=func_code.strip(),
                            context=context,
                            tags=[context] if context else [],
                        ))

                # Class definitions
                elif isinstance(node, ast.ClassDef):
                    class_code = ast.get_source_segment(code, node) or ""
                    if len(class_code.splitlines()) >= 3:  # At least 3 lines
                        patterns.append(ExtractedPattern(
                            fragment_type=FragmentType.FUNCTION,
                            name=node.name,
                            content=class_code.strip(),
                            context=context,
                            tags=[context, "class"] if context else ["class"],
                        ))

        except SyntaxError:
            logger.debug("Syntax error parsing code in context '%s'", context)
        except Exception as e:
            logger.debug("Error extracting functions: %s", e)

        return patterns

    @staticmethod
    def _extract_python_constants(code: str, context: str) -> List[ExtractedPattern]:
        """Extract constants (module-level assignments of simple values)."""
        patterns: List[ExtractedPattern] = []

        try:
            import ast
            tree = ast.parse(code)

            for node in ast.walk(tree):
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            name = target.id
                            # Heuristic: UPPER_CASE names are constants
                            if name.isupper() and len(name) > 1:
                                const_code = ast.get_source_segment(code, node) or ""
                                patterns.append(ExtractedPattern(
                                    fragment_type=FragmentType.CONSTANT,
                                    name=name,
                                    content=const_code.strip(),
                                    context=context,
                                    tags=[context, "constant"] if context else ["constant"],
                                ))

        except SyntaxError:
            # Invalid Python syntax — skip constant extraction
            logger.debug("SyntaxError during constant extraction, skipping")
            pass
        except Exception as e:
            logger.debug("Error extracting constants: %s", e)

        return patterns

    @staticmethod
    def _extract_python_imports(code: str, context: str) -> List[ExtractedPattern]:
        """Extract import statements."""
        patterns: List[ExtractedPattern] = []

        try:
            import ast
            tree = ast.parse(code)

            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        import_code = ast.get_source_segment(code, node) or ""
                        patterns.append(ExtractedPattern(
                            fragment_type=FragmentType.LIBRARY,
                            name=alias.name,
                            content=import_code.strip(),
                            context=context,
                            tags=[context, "import"] if context else ["import"],
                        ))

                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ""
                    for alias in node.names:
                        import_code = ast.get_source_segment(code, node) or ""
                        patterns.append(ExtractedPattern(
                            fragment_type=FragmentType.LIBRARY,
                            name=f"{module}.{alias.name}",
                            content=import_code.strip(),
                            context=context,
                            tags=[context, "import"] if context else ["import"],
                        ))

        except SyntaxError:
            # Invalid Python syntax — skip import extraction
            logger.debug("SyntaxError during import extraction, skipping")
            pass
        except Exception as e:
            logger.debug("Error extracting imports: %s", e)

        return patterns

    # ------------------------------------------------------------------
    # Pattern extraction -- Text
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_phrases(
        text: str, context: str, min_length: int = 10
    ) -> List[ExtractedPattern]:
        """Extract reusable phrases from text.

        Uses n-gram analysis to find frequent, meaningful phrases.
        """
        patterns: List[ExtractedPattern] = []

        # Clean and tokenize
        text = re.sub(r'\s+', ' ', text).strip()
        sentences = re.split(r'[.!?]+', text)

        # Find meaningful sentences (not too short, not too long)
        for sent in sentences:
            sent = sent.strip()
            if not sent:
                continue
            words = sent.split()
            if min_length <= len(words) <= 50:  # Reasonable phrase length
                # Check if it contains substantive content (not all stopwords)
                stopwords = {"the", "a", "an", "is", "are", "was", "were",
                             "in", "on", "at", "to", "for", "of", "with",
                             "and", "or", "but", "not", "be", "it", "its",
                             "this", "that", "these", "those", "we", "you",
                             "they", "he", "she", "it", "i", "my", "our"}
                content_words = [w for w in words if w.lower() not in stopwords]
                if len(content_words) >= 3:  # At least 3 content words
                    name = " ".join(words[:5]) + ("..." if len(words) > 5 else "")
                    patterns.append(ExtractedPattern(
                        fragment_type=FragmentType.PHRASE,
                        name=name[:60],
                        content=sent,
                        context=context,
                        tags=[context, "phrase"] if context else ["phrase"],
                    ))

        return patterns

    # ------------------------------------------------------------------
    # Pattern processing
    # ------------------------------------------------------------------

    def _process_patterns(
        self, patterns: List[ExtractedPattern]
    ) -> List[ExtractedPattern]:
        """Process extracted patterns: dedup, frequency tracking, certification.

        Returns only the patterns that are new candidates for certification.
        """
        new_candidates: List[ExtractedPattern] = []

        for pattern in patterns:
            # Create signature
            sig = PatternSignature(
                fragment_type=pattern.fragment_type,
                content_hash=pattern.content_hash,
                name=pattern.name,
                context=pattern.context,
            )

            # Check if we already have this pattern
            existing = self._get_or_create_tracker(sig)
            existing.increment()

            # Set initial threshold if not set
            if existing.current_threshold == 0:
                existing.current_threshold = self._get_base_threshold(pattern.fragment_type)

            # Check if ready for certification
            if existing.is_ready:
                # Dedup: check if similar fragment already exists in archive
                dedup_result = self._dedup_check(pattern)
                if dedup_result == DedupResult.NEW:
                    new_candidates.append(pattern)

        return new_candidates

    def propose_candidates(
        self, patterns: List[ExtractedPattern]
    ) -> List[Fragment]:
        """Propose candidates to the Guardian for certification.

        Args:
            patterns: Extracted patterns ready for certification.

        Returns:
            List of certified fragments that were stored in the archive.
        """
        certified_fragments: List[Fragment] = []

        for pattern in patterns:
            fragment = pattern.to_fragment()

            # Pass through Guardian
            accepted, result = self._guardian.guard(fragment)
            if accepted and result is not None:
                try:
                    self._archive.add(result)
                    certified_fragments.append(result)
                    logger.info(
                        "Rastrello: certified '%s' as %s (quality=%.2f)",
                        pattern.name, pattern.fragment_type.value, result.metadata.quality,
                    )
                except (ValueError, Exception) as e:
                    logger.warning(
                        "Rastrello: failed to store '%s': %s", pattern.name, e
                    )
            else:
                logger.debug(
                    "Rastrello: '%s' rejected by Guardian (quality below threshold)",
                    pattern.name,
                )

        return certified_fragments

    # ------------------------------------------------------------------
    # Dedup
    # ------------------------------------------------------------------

    def _dedup_check(self, pattern: ExtractedPattern) -> str:
        """Check if a similar pattern already exists in the archive.

        Returns a ``DedupResult`` constant.
        """
        # 1. Exact hash match
        for frag in self._archive.list_all():
            if frag.checksum and pattern.content_hash[:8] in frag.checksum:
                return DedupResult.EXISTS

        # 2. Semantic similarity via Intake Verifier
        intake_result = self._intake.scan(pattern.content, top_k=3)
        if intake_result.max_similarity >= self._config.dedup_similarity_threshold:
            return DedupResult.SIMILAR_EXISTS

        return DedupResult.NEW

    # ------------------------------------------------------------------
    # Frequency management
    # ------------------------------------------------------------------

    def _get_or_create_tracker(
        self, signature: PatternSignature
    ) -> PatternFrequency:
        """Get existing tracker or create a new one."""
        key = signature.to_key()
        if key not in self._trackers:
            self._trackers[key] = PatternFrequency(signature=signature)
        return self._trackers[key]

    def _get_base_threshold(self, fragment_type: FragmentType) -> int:
        """Get the base frequency threshold for a fragment type."""
        thresholds = {
            FragmentType.FUNCTION: self._config.function_threshold,
            FragmentType.CONSTANT: self._config.constant_threshold,
            FragmentType.LIBRARY: self._config.library_threshold,
            FragmentType.PHRASE: self._config.phrase_threshold,
        }
        return thresholds.get(fragment_type, 4)

    def _adjust_threshold(self, tracker: PatternFrequency) -> None:
        """Adjust the frequency threshold based on feedback.

        Used by Dispatcher -> tracked externally -> threshold adjusted.
        """
        # Calculate adjustment
        adjustment = 0
        adjustment += tracker.times_used * self._config.feedback_used_delta
        adjustment += tracker.times_ignored * self._config.feedback_ignored_delta

        new_threshold = max(
            self._config.threshold_min,
            min(
                self._config.threshold_max,
                tracker.current_threshold + int(round(adjustment)),
            ),
        )

        # Ensure the threshold doesn't drop below the actual count (would trigger
        # immediate re-discovery without new evidence)
        if new_threshold <= tracker.count:
            new_threshold = tracker.count + 1  # Need one more occurrence to re-trigger

        tracker.current_threshold = new_threshold

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _looks_like_code(text: str) -> bool:
        """Heuristic check if text looks like source code."""
        code_indicators = [
            r'\bdef\b', r'\bclass\b', r'\bimport\b', r'\breturn\b',
            r'\bif\s+.*:', r'\bfor\s+.*:', r'\bwhile\s+.*:',
            r'^\s*#.*$', r'"""', r"'''", r'->\s*\w+',
        ]
        score = 0
        for indicator in code_indicators:
            if re.search(indicator, text, re.MULTILINE):
                score += 1
        return score >= 3

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def config(self) -> RastrelloConfig:
        return self._config

    def __repr__(self) -> str:
        stats = self.get_statistics()
        return (
            f"Rastrello(patterns={stats['total_patterns']}, "
            f"ready={stats['ready_candidates']})"
        )

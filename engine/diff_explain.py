"""Diff computation and explanation generation for Appeals."""

from typing import Any, Dict, List, Tuple

from memory_fragments.models import (
    Appeal,
    AppealMetrics,
    Fragment,
)
from memory_fragments.models.appeal import AppealDiff


class DiffExplainEngine:
    """Computes structured diffs and generates human-readable explanations.

    Two main responsibilities:

    1. **Diffing** — identify added / removed terms, modified lines, and
       content reordering between an appeal's proposal and its sources.
    2. **Explanation** — produce concise English explanations and
       pro/con lists for the governance review step.
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute_diff(
        self,
        appeal: Appeal,
        source_fragments: List[Fragment],
    ) -> AppealDiff:
        """Compute a structured diff between the appeal and its sources.

        Parameters
        ----------
        appeal:
            The appeal whose ``proposed_content`` is compared.
        source_fragments:
            The source fragments that the appeal originates from.

        Returns
        -------
        AppealDiff
            Added / removed terms, modified lines, and reordering hints.
        """
        proposed = appeal.proposed_content
        source_text = " ".join(f.content for f in source_fragments)

        proposed_words = set(proposed.split())
        source_words = set(source_text.split())

        # Added terms — in proposed but not in any source
        added = sorted(proposed_words - source_words)

        # Removed terms — in sources but not in proposed
        removed = sorted(source_words - proposed_words)

        # Modified lines — line-by-line comparison
        modified = self._compute_modified_lines(proposed, source_text)

        # Reordering hint — check whether fragment content appears
        # in a different order than the source list.
        reordered = self._detect_reordering(appeal, source_fragments)

        return AppealDiff(
            added=added,
            removed=removed,
            modified=modified,
            reordered=reordered,
        )

    def generate_explanation(
        self,
        appeal: Appeal,
        source_fragments: List[Fragment],
        metrics: AppealMetrics,
    ) -> str:
        """Return a 2-4 sentence human-readable explanation of the appeal.

        Describes what the appeal does, its token impact, coverage,
        and risk assessment.
        """
        op_types = [op.op_type.value for op in appeal.ops]
        ops_str = ", ".join(op_types) if op_types else "modify"

        delta = metrics.delta_token
        if delta < 0:
            tokens_str = f"saves {-delta} tokens"
        elif delta > 0:
            tokens_str = f"costs {delta} additional tokens"
        else:
            tokens_str = "does not change the token count"

        coverage_pct = round(metrics.coverage * 100)
        risk_pct = round(metrics.risk * 100)

        return (
            f"This appeal proposes to {ops_str} content from "
            f"{len(source_fragments)} source fragment(s). "
            f"It {tokens_str}, with {coverage_pct}% content coverage and "
            f"{risk_pct}% estimated risk. "
            f"The aggregate quality score is {round(metrics.aggregate_score, 3)}."
        )

    def generate_pros_cons(
        self,
        appeal: Appeal,
        metrics: AppealMetrics,
    ) -> Tuple[List[str], List[str]]:
        """Generate separate pro and con bullet points for governance review.

        Returns
        -------
        (pros, cons)
            Two lists of short human-readable statements.
        """
        pros: List[str] = []
        cons: List[str] = []

        # --- Token delta ---------------------------------------------------
        if metrics.delta_token < 0:
            pros.append(f"Token savings: {-metrics.delta_token} fewer tokens")
        elif metrics.delta_token > 0:
            cons.append(f"Token increase: {metrics.delta_token} additional tokens")
        else:
            pros.append("No change in token count")

        # --- Coverage ------------------------------------------------------
        coverage_pct = round(metrics.coverage * 100)
        if metrics.coverage >= 0.8:
            pros.append(f"High content coverage ({coverage_pct}%)")
        elif metrics.coverage >= 0.5:
            pros.append(f"Moderate content coverage ({coverage_pct}%)")
        else:
            cons.append(f"Low content coverage ({coverage_pct}%)")

        # --- Risk ----------------------------------------------------------
        risk_pct = round(metrics.risk * 100)
        if metrics.risk < 0.2:
            pros.append(f"Low risk ({risk_pct}%)")
        elif metrics.risk < 0.5:
            cons.append(f"Moderate risk ({risk_pct}%)")
        else:
            cons.append(f"High risk ({risk_pct}%)")

        # --- Aggregate score -----------------------------------------------
        if metrics.aggregate_score > 0.5:
            pros.append("Strong aggregate quality score")
        elif metrics.aggregate_score < 0.0:
            cons.append("Negative aggregate quality score")

        # --- Operation details ---------------------------------------------
        op_types = [op.op_type.value for op in appeal.ops]
        if op_types:
            pros.append(f"Operations proposed: {', '.join(op_types)}")

        return pros, cons

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_modified_lines(proposed: str, source: str) -> List[Dict[str, Any]]:
        """Compare *proposed* and *source* line-by-line.

        Returns up to 50 ``{line_num, old, new}`` entries for lines
        that differ between the two texts.
        """
        proposed_lines = proposed.splitlines()
        source_lines = source.splitlines()

        modified: List[Dict[str, Any]] = []
        max_lines = max(len(proposed_lines), len(source_lines))

        for i in range(max_lines):
            old_line = source_lines[i] if i < len(source_lines) else ""
            new_line = proposed_lines[i] if i < len(proposed_lines) else ""
            if old_line != new_line:
                modified.append(
                    {
                        "line_num": i + 1,
                        "old": old_line,
                        "new": new_line,
                    }
                )

        # Keep output manageable for downstream consumers
        return modified[:50]

    @staticmethod
    def _detect_reordering(
        appeal: Appeal,
        source_fragments: List[Fragment],
    ) -> List[int]:
        """Detect whether fragments appear in a different order in the proposal.

        For each source fragment whose content is found verbatim (case-
        insensitive) inside ``proposed_content``, we record its first
        occurrence position.  If the occurrence order does not match the
        source list order, the fragment indices that are out of place are
        returned.

        Returns
        -------
        List[int]
            Indices of source fragments whose content appears reordered.
        """
        if not source_fragments:
            return []

        proposed_lower = appeal.proposed_content.lower()
        id_to_idx = {f.fragment_id: i for i, f in enumerate(source_fragments)}

        # Record (position, fragment_id) for each fragment found verbatim
        appearances: List[Tuple[int, str]] = []
        for frag in source_fragments:
            content = frag.content.strip()
            if content and content.lower() in proposed_lower:
                pos = proposed_lower.index(content.lower())
                appearances.append((pos, frag.fragment_id))

        if len(appearances) < 2:
            return []

        appearances.sort(key=lambda x: x[0])  # order in proposed content

        expected = list(range(len(appearances)))
        actual = [id_to_idx[fid] for _, fid in appearances]

        if actual == expected:
            return []

        return [i for i, (exp, act) in enumerate(zip(expected, actual)) if exp != act]

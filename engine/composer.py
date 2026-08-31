"""Response composition from activated fragments."""

from __future__ import annotations

from typing import List, Optional, TYPE_CHECKING

from memory_fragments.models import Appeal, AppealStatus, Fragment

if TYPE_CHECKING:
    from memory_fragments.engine.conflict import ConflictReport


class Composer:
    """Generates coherent responses by composing activated fragments.

    The composer sorts fragments by quality, structures them with a
    preamble and per-fragment headings, and optionally substitutes
    an approved appeal's proposed content for the original sources.
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compose(self, fragments: List[Fragment], query: str) -> str:
        """Compose a response from the given fragments ranked by quality.

        Parameters
        ----------
        fragments:
            Activated fragments to compose into a response.
        query:
            The original user query, included for context.

        Returns
        -------
        str
            A formatted response string.
        """
        if not fragments:
            return f"No relevant fragments found for query: {query}"

        # Order by quality descending (highest first)
        sorted_frags = sorted(fragments, key=lambda f: f.metadata.quality, reverse=True)

        parts: List[str] = [
            f"Based on {len(sorted_frags)} relevant memory fragment(s), "
            "here is a synthesized response:",
            "",
        ]

        for i, frag in enumerate(sorted_frags, start=1):
            topic = frag.metadata.topic or "untitled"
            quality = round(frag.metadata.quality, 3)
            parts.append(f"--- Fragment {i}: [{topic}] (quality: {quality}) ---")
            parts.append(frag.content)
            parts.append("")

        parts.append(f"--- End of response for: {query} ---")

        return "\n".join(parts).strip()

    def compose_with_conflicts(
        self,
        fragments: List[Fragment],
        query: str,
        conflict_reports: Optional[List[ConflictReport]] = None,
    ) -> str:
        """Compose a response with optional conflict warnings appended.

        Parameters
        ----------
        fragments:
            Activated fragments to compose into a response.
        query:
            The original user query, included for context.
        conflict_reports:
            Optional conflict reports from
            :meth:`HybridRetriever.retrieve_with_conflicts`.
            When provided, warning lines are appended below the response.

        Returns
        -------
        str
            A formatted response string, with conflict warnings if any.
        """
        response = self.compose(fragments, query)

        if conflict_reports:
            warnings: List[str] = []
            for report in conflict_reports:
                for conflict in report.conflicts:
                    sim_pct = round(conflict.similarity * 100)
                    warnings.append(
                        f"⚠ Conflict detected: {report.fragment_id} vs "
                        f"{conflict.other_fragment_id} "
                        f"(similarity: {sim_pct}%, "
                        f"method: {conflict.detection_method})"
                    )
            if warnings:
                response += "\n\n--- Conflict Warnings ---\n"
                response += "\n".join(warnings)

        return response

    def compose_with_appeal(
        self,
        fragments: List[Fragment],
        appeal: Appeal,
        query: str,
    ) -> str:
        """Compose a response, using the appeal's content if approved.

        If the appeal is in ``APPROVED`` status and carries non-empty
        ``proposed_content``, that content is used directly (with a
        preamble noting the appeal).  Otherwise falls back to
        :meth:`compose`.

        Parameters
        ----------
        fragments:
            Original source fragments (used as fallback).
        appeal:
            The appeal that may or may not be approved.
        query:
            Original user query for context.

        Returns
        -------
        str
            Composed response.
        """
        if appeal.status == AppealStatus.APPROVED and appeal.proposed_content:
            preamble = (
                f"An approved appeal ({appeal.appeal_id}) was applied "
                f"to synthesise the following response for: {query}"
            )
            return preamble + "\n\n" + appeal.proposed_content.strip()

        return self.compose(fragments, query)

    @staticmethod
    def estimate_tokens(text: str) -> int:
        """Estimate the number of tokens in *text* via simple word count.

        This is a rough approximation (1 token ≈ 1 word) useful for
        quick cost estimation without a dedicated tokeniser.
        """
        return len(text.split())

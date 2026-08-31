"""MemoryFragmentsModel — unified V2 cognitive orchestrator.

Coordinates the full memory-fragment lifecycle on top of the existing
sub-systems::

    ingest ──► archive + retriever + genealogy
    query  ──► hybrid retrieval ──► composition
    propose ──► appeal trial ──► metrics + diff
    submit / approve / reject ──► governance (human-in-the-loop)

The orchestrator owns one ``StaticArchive``, one ``HybridRetriever``,
one ``AppealTrialSpace``, one ``GenealogyGraph`` and a ``GovernanceAPI``.
An optional ``FragmentGuardian`` enforces the quality gate at ingest.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from memory_fragments.archive.appeal_space import AppealTrialSpace
from memory_fragments.archive.static import StaticArchive
from memory_fragments.config import GovernanceConfig, RetrieverConfig
from memory_fragments.engine.composer import Composer
from memory_fragments.engine.diff_explain import DiffExplainEngine
from memory_fragments.engine.evaluator import Evaluator
from memory_fragments.governance.api import GovernanceAPI, GovernanceReport
from memory_fragments.models import (
    Appeal,
    AppealDiff,
    AppealMetrics,
    AppealOperation,
    AppealStatus,
    Fragment,
    FragmentConditions,
    FragmentMetadata,
    FragmentStatus,
    GenealogyGraph,
)
from memory_fragments.retrieval.retriever import HybridRetriever


@dataclass
class IngestResult:
    """Outcome of an ``ingest`` call."""

    accepted: bool
    fragment: Optional[Fragment]
    reason: str = ""

    @property
    def ok(self) -> bool:
        return self.accepted


@dataclass
class QueryResult:
    """Outcome of a ``query`` call: retrieved fragments plus a composed response."""

    query: str
    fragments: List[Fragment] = field(default_factory=list)
    scores: List[float] = field(default_factory=list)
    response: str = ""


class MemoryFragmentsModel:
    """A self-contained Memory Fragments V2 instance.

    Parameters
    ----------
    guardian:
        Optional :class:`~memory_fragments.library.guardian.FragmentGuardian`
        enforcing the quality gate at ingest.  When omitted every fragment
        is accepted as-is.
    retriever_config:
        Optional per-instance retrieval tuning.
    governance_config:
        Optional per-instance governance rules.
    """

    def __init__(
        self,
        guardian: Optional[Any] = None,
        retriever_config: Optional[RetrieverConfig] = None,
        governance_config: Optional[GovernanceConfig] = None,
    ) -> None:
        self.guardian = guardian

        # Core state
        self.archive = StaticArchive()
        self.retriever = HybridRetriever(retriever_config)
        self.appeal_space = AppealTrialSpace()
        self.graph = GenealogyGraph()

        # Engine components
        self.evaluator = Evaluator()
        self.diff_explain = DiffExplainEngine()
        self.composer = Composer()

        # Governance
        self.governance = GovernanceAPI(
            archive=self.archive,
            appeal_space=self.appeal_space,
            evaluator=self.evaluator,
            diff_explain=self.diff_explain,
            graph=self.graph,
            config=governance_config,
        )

    # ------------------------------------------------------------------
    # Ingest
    # ------------------------------------------------------------------

    def ingest(
        self,
        fragment: Fragment,
        parents: Optional[List[str]] = None,
        model_id: str = "",
    ) -> IngestResult:
        """Register a fragment in the model.

        The fragment passes through the optional guardian, is added to the
        archive and the retriever index, and its genealogy node is recorded.

        Returns
        -------
        IngestResult
            ``accepted`` is ``False`` when the guardian rejects the fragment.
        """
        if self.guardian is not None:
            accepted, maybe = self.guardian.guard(fragment)
            if not accepted:
                return IngestResult(
                    accepted=False,
                    fragment=fragment,
                    reason=f"guardian rejected quality {fragment.metadata.quality:.3f}",
                )
            if maybe is not None:
                fragment = maybe

        try:
            self.archive.add(fragment)
        except ValueError as exc:  # duplicate / oversized / empty checksum
            return IngestResult(
                accepted=False, fragment=fragment, reason=str(exc)
            )

        parent_ids = parents or (fragment.parents or [])
        self.retriever.add_fragment(fragment)
        self.graph.add_node(
            fragment_id=fragment.fragment_id,
            parent_ids=parent_ids,
            checksum=fragment.checksum,
            model_id=model_id,
            quality_source=(
                fragment.metadata.provenance.primary_source.value
                if fragment.metadata.provenance
                else ""
            ),
        )
        return IngestResult(accepted=True, fragment=fragment, reason="ok")

    # ------------------------------------------------------------------
    # Retrieval & composition
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
    ) -> List[Tuple[Fragment, float]]:
        """Hybrid (BM25 + embedding) retrieval over the archive."""
        return self.retriever.retrieve(query, top_k)

    def compose(self, query: str, top_k: Optional[int] = None) -> str:
        """Retrieve the best fragments for *query* and compose a response."""
        results = self.retrieve(query, top_k)
        fragments = [frag for frag, _ in results]
        return self.composer.compose(fragments, query)

    def query(
        self,
        query: str,
        top_k: Optional[int] = None,
    ) -> QueryResult:
        """Run retrieval and composition in one step."""
        results = self.retrieve(query, top_k)
        fragments = [frag for frag, _ in results]
        scores = [round(score, 4) for _, score in results]
        return QueryResult(
            query=query,
            fragments=fragments,
            scores=scores,
            response=self.composer.compose(fragments, query),
        )

    # ------------------------------------------------------------------
    # Appeal trial & governance
    # ------------------------------------------------------------------

    def propose(
        self,
        appeal_id: str,
        source_ids: List[str],
        proposed_content: str,
        explanation: str = "",
        ops: Optional[List[AppealOperation]] = None,
    ) -> Appeal:
        """Create a DRAFT appeal with automatic metrics and diff.

        Raises
        ------
        ValueError
            If a source ID does not exist in the archive.
        """
        sources = [self.archive.get(sid) for sid in source_ids]
        missing = [sid for sid, frag in zip(source_ids, sources) if frag is None]
        if missing:
            raise ValueError(
                f"Unknown source fragment(s): {', '.join(missing)}"
            )

        appeal = self.appeal_space.create_appeal(
            appeal_id=appeal_id,
            sources=source_ids,
            ops=ops,
        )
        self.appeal_space.update_proposal(
            appeal_id=appeal_id,
            proposed_content=proposed_content,
            explanation=explanation,
        )

        source_fragments = [f for f in sources if f is not None]
        # Re-read the stored appeal so metrics/diff see the updated proposal.
        current = self.appeal_space.get_appeal(appeal_id) or appeal
        diff = self.diff_explain.compute_diff(current, source_fragments)
        metrics = self.evaluator.evaluate(current, source_fragments)
        self.appeal_space.update_diff(appeal_id, diff)
        self.appeal_space.update_metrics(appeal_id, metrics)

        return self.appeal_space.get_appeal(appeal_id) or appeal

    def submit(self, appeal_id: str) -> GovernanceReport:
        """Send an appeal to human review (pending approval)."""
        return self.governance.submit_for_review(appeal_id)

    def approve(
        self,
        appeal_id: str,
        approver: str = "user",
        notes: str = "",
    ) -> Fragment:
        """Approve a pending appeal and promote it to the archive."""
        return self.governance.approve(appeal_id, approver=approver, notes=notes)

    def reject(self, appeal_id: str, reason: str = "") -> bool:
        """Reject a pending appeal."""
        return self.governance.reject(appeal_id, reason=reason)

    def get_report(self, appeal_id: str) -> Optional[GovernanceReport]:
        """Return the current governance report for an appeal."""
        return self.governance.get_report(appeal_id)

    # ------------------------------------------------------------------
    # Statistics & persistence
    # ------------------------------------------------------------------

    def get_statistics(self) -> Dict[str, Any]:
        """Aggregate statistics from archive, trial space and genealogy."""
        stats = self.governance.get_statistics()
        stats["model"] = {
            "archive_count": len(self.archive),
            "active_appeals": self.appeal_space.count_active(),
            "graph_nodes": self.graph.node_count(),
        }
        return stats

    def export_state(self) -> Dict[str, Any]:
        """Serialise the full model state (archive, appeals, genealogy)."""
        return {
            "schema": "memory-fragments-model",
            "version": 1,
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "archive": self.archive.to_dict(),
            "appeal_space": self.appeal_space.to_dict(),
            "genealogy": self.graph.to_dict(),
        }

    @classmethod
    def from_state(cls, data: Dict[str, Any]) -> "MemoryFragmentsModel":
        """Reconstruct a model from a dictionary produced by :meth:`export_state`."""
        model = cls()
        model.archive = StaticArchive.from_dict(data.get("archive", {}))
        model.appeal_space = AppealTrialSpace.from_dict(
            data.get("appeal_space", {})
        )
        model.graph = GenealogyGraph.from_dict(data.get("genealogy", {}))
        model.retriever.rebuild(model.archive.list_all())
        model.governance = GovernanceAPI(
            archive=model.archive,
            appeal_space=model.appeal_space,
            evaluator=model.evaluator,
            diff_explain=model.diff_explain,
            graph=model.graph,
        )
        return model

    def save_state(self, path: str) -> None:
        """Persist the full model state to a JSON file."""
        Path(path).write_text(
            json.dumps(self.export_state(), indent=2, default=str),
            encoding="utf-8",
        )

    @classmethod
    def load_state(cls, path: str) -> "MemoryFragmentsModel":
        """Load a model state previously written by :meth:`save_state`."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_state(data)

    def __repr__(self) -> str:
        return (
            f"MemoryFragmentsModel(fragments={len(self.archive)}, "
            f"appeals={len(self.appeal_space)}, graph={self.graph.node_count()})"
        )

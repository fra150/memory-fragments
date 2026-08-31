"""Genealogy DAG — tracks version ancestry and provenance of promoted fragments."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set


@dataclass
class GenealogyNode:
    """A single node in the genealogy DAG, representing one promoted fragment version."""

    fragment_id: str
    parent_ids: List[str] = field(default_factory=list)
    child_ids: List[str] = field(default_factory=list)
    appeal_id: Optional[str] = None
    approver: str = "user"
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    checksum: str = ""
    promotion_notes: str = ""
    depth: int = 0

    # -- model versioning & provenance (Card 8) ----------------------------
    model_id: str = ""
    """Identifier of the model that produced this node's content (e.g., 'gpt-4o', 'phi-3-mini')."""

    model_version: str = ""
    """Version of the model (e.g., '1.0', git hash, or '2024-11-20')."""

    evaluator_model_id: str = ""
    """Identifier of the model used to evaluate the quality of this node."""

    evaluator_model_version: str = ""
    """Version of the evaluator model."""

    quality_source: str = ""
    """String representation of QualitySource at time of promotion."""

    provenance_snapshot: str = ""
    """JSON snapshot of QualityProvenance at time of promotion (for historical reference)."""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "fragment_id": self.fragment_id,
            "parent_ids": self.parent_ids,
            "child_ids": self.child_ids,
            "appeal_id": self.appeal_id,
            "approver": self.approver,
            "timestamp": self.timestamp.isoformat(),
            "checksum": self.checksum,
            "promotion_notes": self.promotion_notes,
            "depth": self.depth,
            "model_id": self.model_id,
            "model_version": self.model_version,
            "evaluator_model_id": self.evaluator_model_id,
            "evaluator_model_version": self.evaluator_model_version,
            "quality_source": self.quality_source,
            "provenance_snapshot": self.provenance_snapshot,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> GenealogyNode:
        ts = data["timestamp"]
        if isinstance(ts, str):
            ts = datetime.fromisoformat(ts)
        return cls(
            fragment_id=data["fragment_id"],
            parent_ids=data.get("parent_ids", []),
            child_ids=data.get("child_ids", []),
            appeal_id=data.get("appeal_id"),
            approver=data.get("approver", "user"),
            timestamp=ts,
            checksum=data.get("checksum", ""),
            promotion_notes=data.get("promotion_notes", ""),
            depth=data.get("depth", 0),
            model_id=data.get("model_id", ""),
            model_version=data.get("model_version", ""),
            evaluator_model_id=data.get("evaluator_model_id", ""),
            evaluator_model_version=data.get("evaluator_model_version", ""),
            quality_source=data.get("quality_source", ""),
            provenance_snapshot=data.get("provenance_snapshot", ""),
        )


class GenealogyGraph:
    """
    Directed Acyclic Graph (DAG) tracking the full version history of fragments.

    Every promoted Appeal creates a new node linked to its source fragments
    (parents). The graph supports traversal, integrity verification, depth
    tracking, and pruning for long-running archives.
    """

    def __init__(self) -> None:
        self._nodes: Dict[str, GenealogyNode] = {}

    # ---- query ----

    def get_node(self, fragment_id: str) -> Optional[GenealogyNode]:
        return self._nodes.get(fragment_id)

    def has_node(self, fragment_id: str) -> bool:
        return fragment_id in self._nodes

    def all_ids(self) -> List[str]:
        return list(self._nodes.keys())

    def node_count(self) -> int:
        return len(self._nodes)

    # ---- mutation ----

    def add_node(
        self,
        fragment_id: str,
        parent_ids: Optional[List[str]] = None,
        appeal_id: Optional[str] = None,
        approver: str = "user",
        checksum: str = "",
        notes: str = "",
        model_id: str = "",
        model_version: str = "",
        evaluator_model_id: str = "",
        evaluator_model_version: str = "",
        quality_source: str = "",
        provenance_snapshot: str = "",
    ) -> GenealogyNode:
        parent_ids = parent_ids or []
        depth = 0
        for pid in parent_ids:
            pnode = self._nodes.get(pid)
            if pnode:
                depth = max(depth, pnode.depth + 1)

        node = GenealogyNode(
            fragment_id=fragment_id,
            parent_ids=parent_ids,
            appeal_id=appeal_id,
            approver=approver,
            checksum=checksum,
            promotion_notes=notes,
            depth=depth,
            model_id=model_id,
            model_version=model_version,
            evaluator_model_id=evaluator_model_id,
            evaluator_model_version=evaluator_model_version,
            quality_source=quality_source,
            provenance_snapshot=provenance_snapshot,
        )
        self._nodes[fragment_id] = node

        # Update child pointers on parents
        for pid in parent_ids:
            pnode = self._nodes.get(pid)
            if pnode and fragment_id not in pnode.child_ids:
                pnode.child_ids.append(fragment_id)

        return node

    def remove_node(self, fragment_id: str) -> bool:
        """Soft-delete: remove from the index. Does not cascade."""
        if fragment_id not in self._nodes:
            return False
        del self._nodes[fragment_id]
        return True

    # ---- traversal ----

    def get_ancestors(self, fragment_id: str, max_depth: Optional[int] = None) -> List[GenealogyNode]:
        """Walk up the DAG from *fragment_id* toward roots."""
        result: List[GenealogyNode] = []
        visited: Set[str] = set()

        def _walk(fid: str, d: int) -> None:
            if max_depth is not None and d > max_depth:
                return
            if fid in visited:
                return
            visited.add(fid)
            node = self._nodes.get(fid)
            if node:
                result.append(node)
                for pid in node.parent_ids:
                    _walk(pid, d + 1)

        _walk(fragment_id, 0)
        return result

    def get_descendants(self, fragment_id: str, max_depth: Optional[int] = None) -> List[GenealogyNode]:
        """Walk down the DAG from *fragment_id* toward leaves."""
        result: List[GenealogyNode] = []
        visited: Set[str] = set()

        def _walk(fid: str, d: int) -> None:
            if max_depth is not None and d > max_depth:
                return
            if fid in visited:
                return
            visited.add(fid)
            node = self._nodes.get(fid)
            if node:
                result.append(node)
                for cid in node.child_ids:
                    _walk(cid, d + 1)

        _walk(fragment_id, 0)
        return result

    def get_lineage(self, fragment_id: str) -> List[str]:
        """Return an ordered chain from root to *fragment_id* (breadth-first approximation)."""
        ancestors = list(reversed(self.get_ancestors(fragment_id)))
        seen: Set[str] = set()
        chain: List[str] = []
        for n in ancestors:
            if n.fragment_id not in seen:
                chain.append(n.fragment_id)
                seen.add(n.fragment_id)
        if fragment_id not in seen:
            chain.append(fragment_id)
        return chain

    def get_roots(self) -> List[GenealogyNode]:
        """Nodes with no parents — the origin fragments."""
        return [n for n in self._nodes.values() if not n.parent_ids]

    def get_leaves(self) -> List[GenealogyNode]:
        """Nodes with no children — the latest versions."""
        return [n for n in self._nodes.values() if not n.child_ids]

    # ---- model versioning & provenance queries (Card 8) ----

    def get_nodes_by_model(self, model_id: str) -> List[GenealogyNode]:
        """Return all nodes produced by a specific model."""
        return [n for n in self._nodes.values() if n.model_id == model_id]

    def get_nodes_by_evaluator(self, evaluator_model_id: str) -> List[GenealogyNode]:
        """Return all nodes evaluated by a specific model."""
        return [n for n in self._nodes.values() if n.evaluator_model_id == evaluator_model_id]

    def get_model_statistics(self) -> Dict[str, Any]:
        """Return statistics about which models have been used."""
        model_counts: Dict[str, int] = {}
        evaluator_counts: Dict[str, int] = {}
        quality_sources: Dict[str, int] = {}

        for node in self._nodes.values():
            if node.model_id:
                model_counts[node.model_id] = model_counts.get(node.model_id, 0) + 1
            if node.evaluator_model_id:
                evaluator_counts[node.evaluator_model_id] = (
                    evaluator_counts.get(node.evaluator_model_id, 0) + 1
                )
            if node.quality_source:
                quality_sources[node.quality_source] = (
                    quality_sources.get(node.quality_source, 0) + 1
                )

        return {
            "nodes_by_model": model_counts,
            "nodes_by_evaluator": evaluator_counts,
            "nodes_by_quality_source": quality_sources,
            "total_nodes": self.node_count(),
        }

    # ---- integrity ----

    def detect_cycles(self) -> List[str]:
        """Return any fragment_ids involved in a cycle (should be empty in a DAG)."""
        visited: Set[str] = set()
        path: Set[str] = set()
        cyclic: List[str] = []

        def _dfs(fid: str) -> None:
            if fid in path:
                cyclic.append(fid)
                return
            if fid in visited:
                return
            visited.add(fid)
            path.add(fid)
            node = self._nodes.get(fid)
            if node:
                for cid in node.child_ids:
                    _dfs(cid)
            path.remove(fid)

        for nid in self._nodes:
            _dfs(nid)
        return cyclic

    # ---- serialization ----

    def to_dict(self) -> Dict[str, Any]:
        return {
            "nodes": {nid: node.to_dict() for nid, node in self._nodes.items()},
            "roots": [n.fragment_id for n in self.get_roots()],
            "leaves": [n.fragment_id for n in self.get_leaves()],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, default=str)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> GenealogyGraph:
        g = cls()
        for nid, ndata in data.get("nodes", {}).items():
            g._nodes[nid] = GenealogyNode.from_dict(ndata)
        return g

    def __repr__(self) -> str:
        return f"GenealogyGraph(nodes={len(self._nodes)}, roots={len(self.get_roots())}, leaves={len(self.get_leaves())})"

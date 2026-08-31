"""LibrarySystem — manager of multiple domain-specific Cassetti (shelves)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from memory_fragments.library.cassetto import Cassetto, CassettoConfig
from memory_fragments.models import Fragment


class LibrarySystem:
    """Manages a collection of domain-specific Cassetti.

    Usage::

        lib = LibrarySystem()
        medical = lib.create_cassetto(CassettoConfig(name="medical", topic="medicine"))
        legal = lib.create_cassetto(CassettoConfig(name="legal", topic="law"))

        medical.add(my_fragment)          # guardian-enforced
        results = lib.query("heart", ["medical"])  # domain-scoped
    """

    def __init__(self) -> None:
        self._cassetti: Dict[str, Cassetto] = {}

    # ------------------------------------------------------------------
    # Cassetto lifecycle
    # ------------------------------------------------------------------

    def create_cassetto(self, config: CassettoConfig) -> Cassetto:
        """Create a new named shelf (raises on duplicate name)."""
        if config.name in self._cassetti:
            raise ValueError(
                f"Cassetto '{config.name}' already exists. "
                "Use get_cassetto() to access it."
            )
        cassetto = Cassetto(config)
        self._cassetti[config.name] = cassetto
        return cassetto

    def get_cassetto(self, name: str) -> Optional[Cassetto]:
        return self._cassetti.get(name)

    def has_cassetto(self, name: str) -> bool:
        return name in self._cassetti

    def remove_cassetto(self, name: str) -> bool:
        """Remove a shelf entirely (and all its fragments)."""
        if name not in self._cassetti:
            return False
        del self._cassetti[name]
        return True

    def list_cassetti(self) -> List[str]:
        return list(self._cassetti.keys())

    def count(self) -> int:
        return len(self._cassetti)

    # ------------------------------------------------------------------
    # Cross-shelf operations
    # ------------------------------------------------------------------

    def query(
        self,
        query_text: str,
        shelves: Optional[List[str]] = None,
        top_k_per_shelf: int = 3,
    ) -> Dict[str, List[Tuple[Fragment, float]]]:
        """Query one or more shelves independently.

        Parameters
        ----------
        query_text:
            The search query.
        shelves:
            Specific shelves to query (default: all).
        top_k_per_shelf:
            Results per shelf.

        Returns
        -------
        ``{shelf_name: [(Fragment, score), ...]}``
        """
        targets = shelves or self.list_cassetti()
        results: Dict[str, List[Tuple[Fragment, float]]] = {}
        for name in targets:
            cassetto = self._cassetti.get(name)
            if cassetto is not None:
                results[name] = cassetto.search(query_text, top_k_per_shelf)
            else:
                results[name] = []
        return results

    def query_all(
        self,
        query_text: str,
        top_k_per_shelf: int = 3,
    ) -> Dict[str, List[Tuple[Fragment, float]]]:
        """Convenience: query every shelf."""
        return self.query(query_text, shelves=None, top_k_per_shelf=top_k_per_shelf)

    def add_to_shelf(self, shelf_name: str, fragment: Fragment) -> bool:
        """Add a fragment to a specific shelf (guardian-enforced)."""
        cassetto = self._cassetti.get(shelf_name)
        if cassetto is None:
            raise ValueError(f"Cassetto '{shelf_name}' does not exist.")
        return cassetto.add(fragment)

    # ------------------------------------------------------------------
    # Aggregate stats
    # ------------------------------------------------------------------

    def statistics(self) -> Dict[str, Any]:
        """Aggregate statistics across all shelves."""
        total_fragments = 0
        total_rejected = 0
        shelf_details: List[Dict[str, Any]] = []

        for name, c in self._cassetti.items():
            total_fragments += c.count()
            total_rejected += c.rejected_count()
            shelf_details.append({
                "name": name,
                "fragments": c.count(),
                "rejected": c.rejected_count(),
                "active_appeals": c.appeal_space.count_active(),
            })

        return {
            "shelves": len(self._cassetti),
            "total_fragments": total_fragments,
            "total_rejected": total_rejected,
            "details": shelf_details,
        }

    def __repr__(self) -> str:
        return f"LibrarySystem(shelves={self.list_cassetti()})"

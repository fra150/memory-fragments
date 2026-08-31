"""Static Archive — the immutable fragment store.

The StaticArchive provides a write-once, read-many store for Fragment objects.
Once a fragment is added, its ``content`` and ``conditions`` are frozen; any
subsequent retrieval returns a deep copy so that mutations by callers never
taint the canonical store.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional

from memory_fragments.config import default_config
from memory_fragments.models import Fragment, FragmentConditions, FragmentStatus


class StaticArchive:
    """The immutable, in-memory fragment store.

    Manages the lifecycle of Fragment objects with a strict **add-once**
    policy.  Duplicate fragment IDs are rejected and the canonical content
    + conditions are kept read-only.

    Thread-safety is *not* guaranteed by this implementation; external
    synchronisation should be provided when used concurrently.
    """

    def __init__(self) -> None:
        self._fragments: Dict[str, Fragment] = {}
        self._config = default_config.archive

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add(self, fragment: Fragment) -> None:
        """Register a new fragment in the archive.

        Parameters
        ----------
        fragment : Fragment
            The fragment to store.  Must have a non-empty ``checksum`` and
            a ``fragment_id`` that does not already exist in the archive.

        Raises
        ------
        ValueError
            If *fragment* has no checksum, or its ``fragment_id`` is already
            present.
        """
        if not fragment.checksum:
            raise ValueError(
                f"Cannot add fragment '{fragment.fragment_id}': "
                "checksum is empty. Ensure the fragment has been finalised."
            )

        if fragment.fragment_id in self._fragments:
            raise ValueError(
                f"Fragment '{fragment.fragment_id}' already exists in the archive. "
                "Duplicates are not permitted."
            )

        if len(fragment.content) > self._config.max_fragment_length:
            raise ValueError(
                f"Fragment '{fragment.fragment_id}' content length "
                f"({len(fragment.content)}) exceeds the maximum allowed "
                f"({self._config.max_fragment_length})."
            )

        # Store a deep copy so the archive owns its data exclusively.
        self._fragments[fragment.fragment_id] = copy.deepcopy(fragment)

    def get(self, fragment_id: str) -> Optional[Fragment]:
        """Retrieve a fragment by its ID.

        Returns a **deep copy** so that callers cannot mutate the canonical
        store.  Returns ``None`` if no fragment with that ID exists.
        """
        fragment = self._fragments.get(fragment_id)
        if fragment is None:
            return None
        return copy.deepcopy(fragment)

    def delete(self, fragment_id: str) -> bool:
        """Soft-delete a fragment by setting its status to ``ARCHIVED``.

        Returns ``True`` if the fragment was found and archived, ``False``
        otherwise.
        """
        fragment = self._fragments.get(fragment_id)
        if fragment is None:
            return False
        fragment.status = FragmentStatus.ARCHIVED
        return True

    def list(
        self,
        topic: Optional[str] = None,
        status: Optional[FragmentStatus] = None,
        tags: Optional[List[str]] = None,
    ) -> List[Fragment]:
        """Return fragments matching the supplied filters.

        Parameters
        ----------
        topic : str, optional
            If given, only fragments whose ``metadata.topic`` equals this
            value are returned (case-sensitive).
        status : FragmentStatus, optional
            If given, only fragments whose ``status`` matches are returned.
        tags : list of str, optional
            If given, only fragments that contain **any** of these tags
            in ``metadata.tags`` are returned (case-sensitive).

        Returns
        -------
        List[Fragment]
            Deep copies of matching fragments.
        """
        results: List[Fragment] = []
        for frag in self._fragments.values():
            if topic is not None and frag.metadata.topic != topic:
                continue
            if status is not None and frag.status != status:
                continue
            if tags is not None:
                frag_tags_lower = {t.lower() for t in frag.metadata.tags}
                if not any(t.lower() in frag_tags_lower for t in tags):
                    continue
            results.append(copy.deepcopy(frag))
        return results

    def search_by_keywords(self, keywords: List[str]) -> List[Fragment]:
        """Search fragments whose conditions or content match any keyword.

        The match is **case-insensitive**: each keyword is tested against:

        * every entry in ``fragment.conditions.keywords``, and
        * the raw ``fragment.content`` string.

        A fragment is included if *any* keyword matches either source.

        Parameters
        ----------
        keywords : list of str
            The search terms.

        Returns
        -------
        List[Fragment]
            Deep copies of matching fragments.
        """
        if not keywords:
            return []

        lower_keywords = [kw.lower() for kw in keywords]
        results: List[Fragment] = []

        for frag in self._fragments.values():
            # Build a set of condition keywords (lowered).
            cond_keywords_lower = {kw.lower() for kw in frag.conditions.keywords}
            content_lower = frag.content.lower()

            for kw in lower_keywords:
                if kw in cond_keywords_lower or kw in content_lower:
                    results.append(copy.deepcopy(frag))
                    break

        return results

    def search_by_quality(self, min_quality: float) -> List[Fragment]:
        """Return fragments whose ``metadata.quality`` >= *min_quality*.

        Parameters
        ----------
        min_quality : float
            Minimum quality threshold (inclusive).

        Returns
        -------
        List[Fragment]
            Deep copies of matching fragments.
        """
        return [
            copy.deepcopy(frag)
            for frag in self._fragments.values()
            if frag.metadata.quality >= min_quality
        ]

    def count(self) -> int:
        """Return the total number of fragments in the archive."""
        return len(self._fragments)

    def list_all(self) -> List[Fragment]:
        """Return **all** fragments as deep copies."""
        return [copy.deepcopy(frag) for frag in self._fragments.values()]

    def all_fragment_ids(self) -> List[str]:
        """Return the fragment ID of every stored fragment."""
        return list(self._fragments.keys())

    # -- convenience aliases for compatibility with GovernanceAPI ------------

    def get_fragment(self, fragment_id: str) -> Optional[Fragment]:
        """Alias for :meth:`get`."""
        return self.get(fragment_id)

    def add_fragment(self, fragment: Fragment) -> None:
        """Alias for :meth:`add`."""
        return self.add(fragment)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Serialise the archive to a JSON-compatible dictionary.

        Returns
        -------
        dict
            A dictionary with a ``"fragments"`` key containing the list of
            serialised fragments.
        """
        return {
            "fragments": [frag.to_dict() for frag in self._fragments.values()],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> StaticArchive:
        """Deserialise an archive from a dictionary previously produced by
        :meth:`to_dict`.

        Parameters
        ----------
        data : dict
            A dictionary with a ``"fragments"`` key containing a list of
            serialised fragment dictionaries.

        Returns
        -------
        StaticArchive
            A new archive instance populated with the deserialised fragments.
        """
        instance = cls()
        for item in data.get("fragments", []):
            fragment = Fragment.from_dict(item)
            instance._fragments[fragment.fragment_id] = fragment
        return instance

    def __repr__(self) -> str:
        return f"StaticArchive(count={len(self._fragments)})"

    def __len__(self) -> int:
        return len(self._fragments)

    def __contains__(self, fragment_id: str) -> bool:
        return fragment_id in self._fragments

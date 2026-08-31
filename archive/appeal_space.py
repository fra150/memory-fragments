"""Appeal Trial Space — the sandbox workspace for Appeal management.

The AppealTrialSpace provides a temporary workspace where Appeals can be
created, evaluated, and either promoted (merged into the Static Archive)
or discarded.  It enforces config-driven limits on the number of active
appeals and operations per appeal.
"""

from __future__ import annotations

import copy
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from memory_fragments.config import default_config
from memory_fragments.models import Appeal, AppealMetrics, AppealOperation, AppealStatus
from memory_fragments.models.appeal import AppealDiff


class AppealTrialSpace:
    """An in-memory sandbox for creating, evaluating, and managing Appeals.

    Appeals live here through their lifecycle: **DRAFT** → evaluation →
    **PENDING_USER_APPROVAL** → **APPROVED** / **REJECTED** / **EXPIRED**.
    Once resolved (approved or rejected), appeals are retained for a
    configurable retention period but are excluded from "active" queries.
    """

    def __init__(self) -> None:
        self._appeals: Dict[str, Appeal] = {}
        self._config = default_config.appeal

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def create_appeal(
        self,
        appeal_id: str,
        sources: Optional[List[str]] = None,
        ops: Optional[List[AppealOperation]] = None,
    ) -> Appeal:
        """Create a new Appeal with status ``DRAFT``.

        Parameters
        ----------
        appeal_id : str
            Unique identifier for the appeal.  Must not already exist in the
            trial space.
        sources : list of str, optional
            Fragment IDs that this appeal references as source material.
        ops : list of AppealOperation, optional
            Transformation operations proposed by this appeal.

        Returns
        -------
        Appeal
            The newly created (and stored) appeal.

        Raises
        ------
        ValueError
            If *appeal_id* already exists, or if the number of active appeals
            or operations exceeds the configured limits.
        """
        if appeal_id in self._appeals:
            raise ValueError(
                f"Appeal '{appeal_id}' already exists in the trial space."
            )

        active_count = self.count_active()
        if active_count >= self._config.max_active_appeals:
            raise ValueError(
                f"Cannot create appeal '{appeal_id}': "
                f"active appeal limit ({self._config.max_active_appeals}) reached "
                f"({active_count} active)."
            )

        ops = ops or []
        if len(ops) > self._config.max_ops_per_appeal:
            raise ValueError(
                f"Cannot create appeal '{appeal_id}': "
                f"operation count ({len(ops)}) exceeds the maximum allowed "
                f"({self._config.max_ops_per_appeal})."
            )

        appeal = Appeal(
            appeal_id=appeal_id,
            sources=sources or [],
            ops=list(ops),
            status=AppealStatus.DRAFT,
        )
        self._appeals[appeal_id] = appeal
        return copy.deepcopy(appeal)

    def get_appeal(self, appeal_id: str) -> Optional[Appeal]:
        """Retrieve an appeal by its ID.

        Returns a deep copy so callers cannot mutate the stored state.
        Returns ``None`` if no such appeal exists.
        """
        appeal = self._appeals.get(appeal_id)
        if appeal is None:
            return None
        return copy.deepcopy(appeal)

    def list_appeals(
        self, status: Optional[AppealStatus] = None
    ) -> List[Appeal]:
        """Return appeals, optionally filtered by status.

        Parameters
        ----------
        status : AppealStatus, optional
            If given, only appeals whose status matches are returned.

        Returns
        -------
        List[Appeal]
            Deep copies of matching appeals.
        """
        if status is not None:
            return [
                copy.deepcopy(a)
                for a in self._appeals.values()
                if a.status == status
            ]
        return [copy.deepcopy(a) for a in self._appeals.values()]

    def update_metrics(
        self, appeal_id: str, metrics: AppealMetrics
    ) -> None:
        """Set the computed metrics for an appeal.

        Parameters
        ----------
        appeal_id : str
            The target appeal's ID.
        metrics : AppealMetrics
            The metrics to store (overwrites any previous metrics).

        Raises
        ------
        KeyError
            If *appeal_id* does not exist.
        """
        if appeal_id not in self._appeals:
            raise KeyError(f"Appeal '{appeal_id}' not found in trial space.")
        self._appeals[appeal_id].metrics = metrics

    def update_diff(self, appeal_id: str, diff: AppealDiff) -> None:
        """Set the structured diff for an appeal.

        Parameters
        ----------
        appeal_id : str
            The target appeal's ID.
        diff : AppealDiff
            The diff to store (overwrites any previous diff).

        Raises
        ------
        KeyError
            If *appeal_id* does not exist.
        """
        if appeal_id not in self._appeals:
            raise KeyError(f"Appeal '{appeal_id}' not found in trial space.")
        self._appeals[appeal_id].diff = diff

    def update_proposal(
        self, appeal_id: str, proposed_content: str, explanation: str
    ) -> None:
        """Set the proposed content and its explanation for an appeal.

        Parameters
        ----------
        appeal_id : str
            The target appeal's ID.
        proposed_content : str
            The new content proposed by this appeal.
        explanation : str
            A human-readable explanation of why this change is being proposed.

        Raises
        ------
        KeyError
            If *appeal_id* does not exist.
        """
        if appeal_id not in self._appeals:
            raise KeyError(f"Appeal '{appeal_id}' not found in trial space.")
        appeal = self._appeals[appeal_id]
        appeal.proposed_content = proposed_content
        appeal.explanation = explanation

    # -- convenience aliases for compatibility with GovernanceAPI ------------

    def get(self, appeal_id: str) -> Optional[Appeal]:
        """Alias for :meth:`get_appeal`."""
        return self.get_appeal(appeal_id)

    def list(self, status: Optional[AppealStatus] = None) -> List[Appeal]:
        """Alias for :meth:`list_appeals`."""
        return self.list_appeals(status)

    def update(self, appeal: Appeal) -> None:
        """Replace the stored copy of *appeal*.

        This is a convenience method for the GovernanceAPI which
        modifies an appeal in-place and then writes it back.
        """
        if appeal.appeal_id not in self._appeals:
            raise KeyError(f"Appeal '{appeal.appeal_id}' not found in trial space.")
        self._appeals[appeal.appeal_id] = copy.deepcopy(appeal)

    def set_status(self, appeal_id: str, status: AppealStatus) -> bool:
        """Transition an appeal to a new status.

        If the target status is a terminal state (``APPROVED`` or
        ``REJECTED``), the appeal's ``resolved_at`` timestamp is also set
        to the current UTC time.

        Parameters
        ----------
        appeal_id : str
            The target appeal's ID.
        status : AppealStatus
            The new status to assign.

        Returns
        -------
        bool
            ``True`` if the appeal was found and updated, ``False`` otherwise.
        """
        appeal = self._appeals.get(appeal_id)
        if appeal is None:
            return False

        appeal.status = status

        if status in (AppealStatus.APPROVED, AppealStatus.REJECTED):
            appeal.resolved_at = datetime.now(timezone.utc)

        return True

    def remove_appeal(self, appeal_id: str) -> bool:
        """Discard an appeal from the trial space without promotion.

        This performs a **hard delete** — the appeal is removed entirely
        and will not appear in any subsequent queries or serialisation.

        Parameters
        ----------
        appeal_id : str
            The target appeal's ID.

        Returns
        -------
        bool
            ``True`` if the appeal existed and was removed, ``False``
            otherwise.
        """
        if appeal_id not in self._appeals:
            return False
        del self._appeals[appeal_id]
        return True

    def prune_old_appeals(
        self,
        max_generations: Optional[int] = None,
        retention_days: Optional[int] = None,
    ) -> int:
        """Hard-delete appeals that exceed configured retention limits.

        Args:
            max_generations: Maximum number of resolved appeals to keep.
                Appeals beyond this count (oldest first) are pruned.
            retention_days: Maximum age in days for resolved appeals.
                Appeals older than this are pruned.

        Returns:
            Number of appeals pruned.
        """
        max_gen = max_generations or self._config.prune_after_generations
        ret_days = retention_days or self._config.retention_days

        resolved = [
            a
            for a in self._appeals.values()
            if a.status
            in (AppealStatus.APPROVED, AppealStatus.REJECTED, AppealStatus.EXPIRED)
        ]

        if not resolved:
            return 0

        pruned = 0
        now = datetime.now(timezone.utc)

        # Prune by age
        for appeal in resolved:
            if appeal.resolved_at and (now - appeal.resolved_at).days > ret_days:
                del self._appeals[appeal.appeal_id]
                pruned += 1

        # Prune by count (only if no age-based pruning happened)
        if pruned == 0 and len(resolved) > max_gen:
            # Sort by resolved_at ascending (oldest first)
            sorted_resolved = sorted(
                [a for a in resolved if a.appeal_id in self._appeals],
                key=lambda a: a.resolved_at or datetime.min.replace(tzinfo=timezone.utc),
            )
            to_remove = len(sorted_resolved) - max_gen
            for appeal in sorted_resolved[:to_remove]:
                if appeal.appeal_id in self._appeals:
                    del self._appeals[appeal.appeal_id]
                    pruned += 1

        return pruned

    def list_active(self) -> List[Appeal]:
        """Return appeals that are still in-flight (not yet resolved).

        Active statuses are ``DRAFT`` and ``PENDING_USER_APPROVAL``.
        Appeals that have been ``APPROVED``, ``REJECTED``, or ``EXPIRED``
        are excluded.

        Returns
        -------
        List[Appeal]
            Deep copies of active appeals.
        """
        terminal = {AppealStatus.APPROVED, AppealStatus.REJECTED, AppealStatus.EXPIRED}
        return [
            copy.deepcopy(a)
            for a in self._appeals.values()
            if a.status not in terminal
        ]

    def count_active(self) -> int:
        """Return the number of in-flight (non-terminal) appeals."""
        terminal = {AppealStatus.APPROVED, AppealStatus.REJECTED, AppealStatus.EXPIRED}
        return sum(1 for a in self._appeals.values() if a.status not in terminal)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Serialise the trial space to a JSON-compatible dictionary.

        Returns
        -------
        dict
            A dictionary with an ``"appeals"`` key containing the list of
            serialised appeals.
        """
        return {
            "appeals": [appeal.to_dict() for appeal in self._appeals.values()],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> AppealTrialSpace:
        """Deserialise a trial space from a dictionary previously produced
        by :meth:`to_dict`.

        Parameters
        ----------
        data : dict
            A dictionary with an ``"appeals"`` key containing a list of
            serialised appeal dictionaries.

        Returns
        -------
        AppealTrialSpace
            A new instance populated with the deserialised appeals.
        """
        instance = cls()
        for item in data.get("appeals", []):
            appeal = Appeal.from_dict(item)
            instance._appeals[appeal.appeal_id] = appeal
        return instance

    def __repr__(self) -> str:
        return (
            f"AppealTrialSpace(total={len(self._appeals)}, "
            f"active={self.count_active()})"
        )

    def __len__(self) -> int:
        return len(self._appeals)

    def __contains__(self, appeal_id: str) -> bool:
        return appeal_id in self._appeals

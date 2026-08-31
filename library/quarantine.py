"""Quarantine manager for fragments that fail the 3-agent circuit.

When a fragment enters quarantine, it is neither accepted nor rejected.
It stays in quarantine until manually reviewed. After 24h an admin
notification is triggered, but fragments are NEVER auto-deleted.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, List, Optional

from memory_fragments.models import Fragment, FragmentMetadata

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NOTIFICATION_HOURS: int = 24
"""After this many hours in quarantine, trigger admin notification."""

DEFAULT_MAX_QUARANTINE_SIZE: int = 100
"""Soft limit on quarantine size. When exceeded, warn but don't reject."""

ALERT_THRESHOLD: int = 50
"""When quarantine exceeds this size, an alert callback is triggered."""


# ---------------------------------------------------------------------------
# Quarantine entry
# ---------------------------------------------------------------------------


@dataclass
class QuarantineEntry:
    """A fragment held in quarantine with metadata about why it failed."""

    fragment: Fragment
    """The quarantined fragment."""

    reason: str
    """Why the fragment was quarantined (e.g., 'insufficient_agents', 'vote_rejected')."""

    details: Dict[str, Any] = field(default_factory=dict)
    """Additional details: agent_count, scores, error messages, etc."""

    quarantined_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    """When the fragment entered quarantine."""

    notified_at: Optional[datetime] = None
    """When the admin notification was last sent."""

    resolved: bool = False
    """Whether this quarantine entry has been resolved."""

    resolution: str = ""
    """Resolution note: 'accepted', 'rejected', or custom."""

    @property
    def age_hours(self) -> float:
        """Hours since this fragment was quarantined."""
        delta = datetime.now(timezone.utc) - self.quarantined_at
        return delta.total_seconds() / 3600

    @property
    def needs_notification(self) -> bool:
        """True if the fragment has been in quarantine > 24h and not notified."""
        return (
            self.age_hours >= NOTIFICATION_HOURS
            and not self.resolved
            and self.notified_at is None
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "fragment_id": self.fragment.fragment_id,
            "topic": self.fragment.metadata.topic,
            "quality": self.fragment.metadata.quality,
            "reason": self.reason,
            "details": self.details,
            "quarantined_at": self.quarantined_at.isoformat(),
            "notified_at": self.notified_at.isoformat() if self.notified_at else None,
            "age_hours": round(self.age_hours, 1),
            "needs_notification": self.needs_notification,
            "resolved": self.resolved,
            "resolution": self.resolution,
        }

    def __repr__(self) -> str:
        return (
            f"QuarantineEntry(id={self.fragment.fragment_id}, "
            f"reason={self.reason}, age={self.age_hours:.1f}h, "
            f"resolved={self.resolved})"
        )


# ---------------------------------------------------------------------------
# Notification callback type
# ---------------------------------------------------------------------------

NotificationCallback = Callable[[QuarantineEntry], None]
"""Signature for admin notification callbacks: (entry) -> None"""


# ---------------------------------------------------------------------------
# Quarantine Manager
# ---------------------------------------------------------------------------


class QuarantineManager:
    """Manages fragments that fail the 3-agent circuit.

    Key behaviors:
    - Fragments are stored indefinitely until manually resolved
    - After 24h, an admin notification is triggered (once)
    - Quarantine has a soft size limit (warns but does not reject)
    - Supports accept/reject resolution

    Usage:
        manager = QuarantineManager()

        # When AgentCircuit raises InsufficientAgentsError
        try:
            result = circuit.evaluate(fragment)
        except InsufficientAgentsError as e:
            entry = manager.quarantine(fragment, "insufficient_agents", {
                "error": str(e),
                "agent_count": 1,
            })

        # Check for notifications
        pending = manager.get_pending_notifications()
        for entry in pending:
            notify_admin(entry)
            manager.mark_notified(entry.fragment.fragment_id)
    """

    def __init__(
        self,
        notification_callback: Optional[NotificationCallback] = None,
        max_size: int = DEFAULT_MAX_QUARANTINE_SIZE,
        alert_threshold: int = ALERT_THRESHOLD,
    ) -> None:
        """
        Args:
            notification_callback: Called when a fragment needs admin attention.
            max_size: Soft limit — warn when exceeded (no hard rejection).
            alert_threshold: When quarantine exceeds this size, an alert is logged.
        """
        self._entries: Dict[str, QuarantineEntry] = {}
        self._notification_callback = notification_callback
        self._max_size = max_size
        self._alert_threshold = alert_threshold

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def quarantine(
        self,
        fragment: Fragment,
        reason: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> QuarantineEntry:
        """Place a fragment in quarantine.

        Args:
            fragment: The fragment to quarantine.
            reason: Why it was quarantined.
            details: Additional context (agent scores, errors, etc.).

        Returns:
            The new QuarantineEntry.

        Raises:
            ValueError: If the fragment is already in quarantine.
        """
        if fragment.fragment_id in self._entries:
            raise ValueError(
                f"Fragment '{fragment.fragment_id}' is already in quarantine."
            )

        entry = QuarantineEntry(
            fragment=fragment,
            reason=reason,
            details=details or {},
        )
        self._entries[fragment.fragment_id] = entry

        # Warn if quarantine is large
        if len(self._entries) > self._max_size:
            logger.warning(
                "Quarantine size (%d) exceeds soft limit (%d). "
                "Manual review recommended.",
                len(self._entries),
                self._max_size,
            )

        # Alert if quarantine exceeds threshold
        if len(self._entries) >= self._alert_threshold:
            logger.warning(
                "QUARANTINE ALERT: %d fragments in quarantine (threshold: %d). "
                "Manual review recommended.",
                len(self._entries),
                self._alert_threshold,
            )

        logger.info(
            "Fragment '%s' quarantined (reason: %s). Total in quarantine: %d",
            fragment.fragment_id,
            reason,
            len(self._entries),
        )

        return entry

    def resolve(
        self,
        fragment_id: str,
        resolution: str,
        admin: str = "system",
    ) -> bool:
        """Resolve a quarantined fragment (accept or reject).

        Args:
            fragment_id: The fragment to resolve.
            resolution: 'accepted', 'rejected', or custom note.
            admin: Who resolved it.

        Returns:
            True if the fragment was found and resolved.
        """
        entry = self._entries.get(fragment_id)
        if entry is None:
            logger.warning(
                "Cannot resolve fragment '%s': not in quarantine.", fragment_id
            )
            return False

        entry.resolved = True
        entry.resolution = resolution
        logger.info(
            "Fragment '%s' resolved as '%s' by %s.",
            fragment_id,
            resolution,
            admin,
        )
        return True

    def release(self, fragment_id: str) -> Optional[Fragment]:
        """Remove a fragment from quarantine and return it.

        This is used when a resolved fragment needs to be re-processed
        (e.g., accepted after manual review).

        Args:
            fragment_id: The fragment to release.

        Returns:
            The Fragment if found, None otherwise.
        """
        entry = self._entries.pop(fragment_id, None)
        if entry is None:
            return None
        return entry.fragment

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get(self, fragment_id: str) -> Optional[QuarantineEntry]:
        """Get a quarantine entry by fragment ID."""
        return self._entries.get(fragment_id)

    def list_all(self) -> List[QuarantineEntry]:
        """Return all quarantine entries."""
        return list(self._entries.values())

    def list_unresolved(self) -> List[QuarantineEntry]:
        """Return only unresolved quarantine entries."""
        return [e for e in self._entries.values() if not e.resolved]

    def list_resolved(self) -> List[QuarantineEntry]:
        """Return only resolved quarantine entries."""
        return [e for e in self._entries.values() if e.resolved]

    def count(self) -> int:
        """Return the total number of quarantine entries."""
        return len(self._entries)

    # ------------------------------------------------------------------
    # Notifications
    # ------------------------------------------------------------------

    def get_pending_notifications(self) -> List[QuarantineEntry]:
        """Return entries that need admin notification (quarantined > 24h)."""
        return [e for e in self._entries.values() if e.needs_notification]

    def mark_notified(self, fragment_id: str) -> bool:
        """Mark a quarantined fragment as having been notified.

        If a notification callback is registered, it is invoked here.

        Args:
            fragment_id: The fragment to mark as notified.

        Returns:
            True if the fragment was found and updated.
        """
        entry = self._entries.get(fragment_id)
        if entry is None:
            return False
        entry.notified_at = datetime.now(timezone.utc)

        # If a notification callback is registered, call it
        if self._notification_callback is not None:
            try:
                self._notification_callback(entry)
            except Exception as e:
                logger.error(
                    "Notification callback failed for '%s': %s",
                    fragment_id,
                    e,
                )

        return True

    def check_notifications(self) -> int:
        """Check all entries and trigger callbacks for pending ones.

        Returns the number of notifications triggered.
        """
        count = 0
        for entry in self.list_unresolved():
            if entry.needs_notification:
                self.mark_notified(entry.fragment.fragment_id)
                count += 1
        return count

    # ------------------------------------------------------------------
    # Alerts
    # ------------------------------------------------------------------

    def quarantine_alerts(self) -> List[Dict[str, Any]]:
        """Return actionable alerts about quarantine state.

        Returns a list of alert dicts with severity levels:
        - "critical": quarantine > alert_threshold * 2
        - "warning": quarantine > alert_threshold
        - "info": pending notifications
        - "ok": all clear
        """
        alerts: List[Dict[str, Any]] = []
        total = len(self._entries)
        unresolved = len(self.list_unresolved())
        pending = len(self.get_pending_notifications())

        if total >= self._alert_threshold * 2:
            alerts.append({
                "severity": "critical",
                "message": (
                    f"Quarantine has {total} fragments "
                    f"(>{self._alert_threshold * 2}x threshold)"
                ),
                "metric": "quarantine_size",
                "value": total,
            })
        elif total >= self._alert_threshold:
            alerts.append({
                "severity": "warning",
                "message": (
                    f"Quarantine has {total} fragments "
                    f"(exceeds threshold of {self._alert_threshold})"
                ),
                "metric": "quarantine_size",
                "value": total,
            })

        if pending > 0:
            alerts.append({
                "severity": "warning",
                "message": (
                    f"{pending} fragment(s) in quarantine >24h "
                    f"without admin review"
                ),
                "metric": "pending_notifications",
                "value": pending,
            })

        if not alerts:
            alerts.append({
                "severity": "ok",
                "message": "Quarantine is healthy",
                "metric": "quarantine_size",
                "value": total,
            })

        return alerts

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the quarantine manager state to a dictionary."""
        return {
            "entries": [e.to_dict() for e in self._entries.values()],
            "count": self.count(),
            "unresolved": len(self.list_unresolved()),
            "pending_notifications": len(self.get_pending_notifications()),
        }

    def __repr__(self) -> str:
        unresolved = len(self.list_unresolved())
        pending = len(self.get_pending_notifications())
        return (
            f"QuarantineManager(total={self.count()}, "
            f"unresolved={unresolved}, "
            f"pending_notifications={pending})"
        )

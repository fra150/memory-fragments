"""Archive package — Static Archive and Appeal Trial Space.

The ``archive`` sub-package provides the two core storage components of the
Memory Fragments V2 system:

* :class:`StaticArchive` — The immutable, write-once fragment store.
* :class:`AppealTrialSpace` — The sandbox workspace for creating and
  evaluating Appeals before they are promoted into the archive.
"""

from memory_fragments.archive.appeal_space import AppealTrialSpace
from memory_fragments.archive.static import StaticArchive

__all__ = [
    "AppealTrialSpace",
    "StaticArchive",
]

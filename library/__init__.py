"""Library module — domain-specific Cassetti with quality guardian."""

from memory_fragments.library.guardian import FragmentGuardian
from memory_fragments.library.cassetto import Cassetto, CassettoConfig
from memory_fragments.library.system import LibrarySystem
from memory_fragments.library.improver import (
    HeuristicImprover,
    AnthropicFragmentImprover,
    DeepSeekFragmentImprover,
)
from memory_fragments.library.circuit import (
    AgentCircuit,
    VoteResult,
    CircuitError,
    InsufficientAgentsError,
)
from memory_fragments.library.quarantine import (
    QuarantineManager,
    QuarantineEntry,
    NotificationCallback,
)

__all__ = [
    "FragmentGuardian",
    "Cassetto",
    "CassettoConfig",
    "LibrarySystem",
    "HeuristicImprover",
    "AnthropicFragmentImprover",
    "DeepSeekFragmentImprover",
    "AgentCircuit",
    "VoteResult",
    "CircuitError",
    "InsufficientAgentsError",
    "QuarantineManager",
    "QuarantineEntry",
    "NotificationCallback",
]

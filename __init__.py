"""Memory Fragments V2 — A Modular Cognitive Model with Appeal Trial and Human Governance.

Based on the paper by Francesco Bulla (2024-2025), this library implements:

- **Static Archive**: An immutable store of conditionally-activated knowledge fragments.
- **Hybrid Retriever**: BM25 keyword + semantic embedding retrieval.
- **Appeal Trial Space**: A sandbox where the AI can propose fragment modifications.
- **Evaluator**: Automatic metrics (delta token, coverage, risk, aggregate score).
- **Diff & Explain**: Structured diffs and human-readable explanations.
- **Composer**: Response generation from activated fragments.
- **Governance API**: Human-in-the-loop approval workflow with DAG genealogy tracking.
"""

# -- Core models ---------------------------------------------------------------
from memory_fragments.models import (
    Fragment,
    FragmentConditions,
    FragmentMetadata,
    FragmentStatus,
    Appeal,
    AppealOperation,
    AppealMetrics,
    AppealDiff,
    AppealStatus,
    OperationType,
    GenealogyGraph,
    GenealogyNode,
    QualityProvenance,
    QualityEvaluation,
    QualitySource,
)

# -- Archive -------------------------------------------------------------------
from memory_fragments.archive import StaticArchive, AppealTrialSpace

# -- Retrieval -----------------------------------------------------------------
from memory_fragments.retrieval import BM25Indexer, EmbeddingIndexer, HybridRetriever

# -- Engine --------------------------------------------------------------------
from memory_fragments.engine import (
    Evaluator,
    DiffExplainEngine,
    Composer,
    IntakeVerifier,
    IntakeResult,
    Dispatcher,
    DispatchPlan,
    DispatchPath,
    Modellatore,
    CompositionResult,
    Rastrello,
    FragmentType,
)

# -- Governance ----------------------------------------------------------------
from memory_fragments.governance import GovernanceAPI, GovernanceReport

# -- Model (unified orchestrator) ---------------------------------------------
from memory_fragments.model import MemoryFragmentsModel, IngestResult, QueryResult

# -- Library (Cassetti) --------------------------------------------------------
from memory_fragments.library import (
    Cassetto,
    CassettoConfig,
    FragmentGuardian,
    LibrarySystem,
    HeuristicImprover,
    AnthropicFragmentImprover,
    DeepSeekFragmentImprover,
    AgentCircuit,
    VoteResult,
    QuarantineManager,
    QuarantineEntry,
)

# -- Config --------------------------------------------------------------------
from memory_fragments.config import (
    MemoryFragmentsConfig,
    RetrieverConfig,
    EvaluatorConfig,
    ArchiveConfig,
    AppealConfig,
    GovernanceConfig,
    DispatcherConfig,
    ModellatoreConfig,
    RastrelloConfig,
    IntakeConfig,
    default_config,
)

__version__ = "0.4.0"
__all__ = [
    # Models
    "Fragment",
    "FragmentConditions",
    "FragmentMetadata",
    "FragmentStatus",
    "Appeal",
    "AppealOperation",
    "AppealMetrics",
    "AppealDiff",
    "AppealStatus",
    "OperationType",
    "GenealogyGraph",
    "GenealogyNode",
    "QualityProvenance",
    "QualityEvaluation",
    "QualitySource",
    # Archive
    "StaticArchive",
    "AppealTrialSpace",
    # Retrieval
    "BM25Indexer",
    "EmbeddingIndexer",
    "HybridRetriever",
    # Engine
    "Evaluator",
    "DiffExplainEngine",
    "Composer",
    "IntakeVerifier",
    "IntakeResult",
    "Dispatcher",
    "DispatchPlan",
    "DispatchPath",
    "Modellatore",
    "CompositionResult",
    "Rastrello",
    "FragmentType",
    # Governance
    "GovernanceAPI",
    "GovernanceReport",
    # Model (unified orchestrator)
    "MemoryFragmentsModel",
    "IngestResult",
    "QueryResult",
    # Library
    "Cassetto",
    "CassettoConfig",
    "FragmentGuardian",
    "LibrarySystem",
    "HeuristicImprover",
    "AnthropicFragmentImprover",
    "DeepSeekFragmentImprover",
    "AgentCircuit",
    "VoteResult",
    "QuarantineManager",
    "QuarantineEntry",
    # Config
    "MemoryFragmentsConfig",
    "RetrieverConfig",
    "EvaluatorConfig",
    "ArchiveConfig",
    "AppealConfig",
    "GovernanceConfig",
    "DispatcherConfig",
    "ModellatoreConfig",
    "RastrelloConfig",
    "IntakeConfig",
    "default_config",
]

"""Global configuration for the Memory Fragments system."""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class RetrieverConfig:
    """Configuration for the hybrid retriever."""

    top_k: int = 5
    semantic_threshold: float = 0.72
    bm25_weight: float = 0.4
    embedding_weight: float = 0.6
    bm25_k1: float = 1.5
    bm25_b: float = 0.75
    embedding_model: str = "all-MiniLM-L6-v2"


@dataclass
class EvaluatorConfig:
    """Configuration for the automatic evaluator."""

    w_delta_token: float = 0.3
    w_coverage: float = 0.4
    w_risk: float = 0.3
    risk_contradiction_penalty: float = 0.25
    risk_vagueness_penalty: float = 0.15
    max_token_penalty_ratio: float = 2.0

    # Embedding-based coverage settings
    use_embedding_coverage: bool = False       # Set True to enable (requires sentence-transformers)
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_coverage_weight: float = 0.5     # Blend: 0.0 = pure word overlap, 1.0 = pure embedding

    # NLI contradiction detection (lazy)
    nli_model: str = "facebook/bart-large-mnli"
    nli_contradiction_threshold: float = 0.85  # Embedding similarity threshold to trigger NLI
    nli_contradiction_penalty_multiplier: float = 1.0  # How much contradiction penalizes risk

    # Cross-fragment conflict detection
    conflict_check_enabled: bool = False
    conflict_similarity_threshold: float = 0.85

    # Risk threshold for auto-promote (replaces deprecated GovernanceConfig.allow_auto_promote_below_risk)
    risk_threshold: float = 0.1  # Threshold below which auto-promote is allowed; deprecated: use GovernanceConfig requirement instead


@dataclass
class ArchiveConfig:
    """Configuration for the static archive."""

    max_fragment_length: int = 10_000
    min_quality_threshold: float = 0.3
    anonymize_pii: bool = False
    audit_enabled: bool = True


@dataclass
class AppealConfig:
    """Configuration for the Appeal Trial Space."""

    max_active_appeals: int = 20
    max_ops_per_appeal: int = 10
    max_depth: int = 50
    prune_after_generations: int = 100
    retention_days: int = 30


@dataclass
class GovernanceConfig:
    """Configuration for the governance layer."""

    require_user_approval: bool = True
    # allow_auto_promote_below_risk: float = 0.1  # DEPRECATED — replaced by EvaluatorConfig.risk_threshold; kept for backward compatibility only
    signed_audit_trail: bool = True
    max_genealogy_depth: int = 20


@dataclass
class DispatcherConfig:
    """Configuration for the Dispatcher (Card 14)."""

    coverage_threshold_compose: float = 0.90
    """Minimum query coverage to attempt full composition."""

    coverage_threshold_partial: float = 0.40
    """Minimum query coverage to attempt partial composition (below this → generate)."""

    min_aspects: int = 2
    """Minimum query aspects to extract (fallback for short queries)."""

    aspect_chunk_model: str = "all-MiniLM-L6-v2"
    """Embedding model for aspect decomposition."""

    cost_token_estimate_factor: float = 1.3
    """Multiplier for token estimation in cost-benefit analysis."""

    generate_cost_penalty: float = 1.0
    """Penalty applied to generation cost (higher = prefer composition)."""


@dataclass
class ModellatoreConfig:
    """Configuration for the Modellatore (Card 15) — composition sandbox."""

    tab_slot_llm_threshold: float = 0.6
    """Minimum confidence for embedding-based tab/slot extraction.
    Below this → uses LLM for extraction."""

    contradiction_block_threshold: float = 0.5
    """Contradiction score above this blocks composition (delegates to ConflictDetector)."""

    contradiction_warn_threshold: float = 0.3
    """Contradiction score above this adds a warning but allows composition."""

    max_composition_fragments: int = 10
    """Maximum fragments to attempt composition with."""

    enable_tab_slot_llm_fallback: bool = True
    """Allow LLM fallback for tab/slot extraction when confidence is low."""


@dataclass
class RastrelloConfig:
    """Configuration for the Rastrello (Card 17) — pattern discovery."""

    enabled: bool = True
    """Whether the Rastrello scanner is active."""

    # Base thresholds per FragmentType
    function_threshold: int = 4
    constant_threshold: int = 1
    library_threshold: int = 1
    phrase_threshold: int = 12

    # Min/max bounds for dynamic threshold adjustment
    threshold_min: int = 1
    threshold_max: int = 30

    # Feedback loop adjustments
    feedback_used_delta: float = -0.5
    feedback_ignored_delta: float = 0.3
    feedback_confirmed_delta: float = -1.0

    dedup_similarity_threshold: float = 0.90
    """Embedding similarity above this means 'same' for dedup."""

    batch_size: int = 1000
    """Number of items to process in a batch scan."""


@dataclass
class IntakeConfig:
    """Configuration for the Intake Verifier (Card 16)."""

    candidate_threshold: float = 0.35
    """Minimum similarity for a fragment to be considered a candidate."""

    min_aspects: int = 2
    """Minimum query aspects to extract (fallback for short queries)."""

    top_k_candidates: int = 10
    """Maximum candidates to return."""

    coverage_ratio_to_proceed: float = 0.25
    """Minimum aspect coverage ratio to proceed with Dispatch."""


@dataclass
class MemoryFragmentsConfig:
    """Top-level configuration aggregating all sub-configs."""

    retriever: RetrieverConfig = field(default_factory=RetrieverConfig)
    evaluator: EvaluatorConfig = field(default_factory=EvaluatorConfig)
    archive: ArchiveConfig = field(default_factory=ArchiveConfig)
    appeal: AppealConfig = field(default_factory=AppealConfig)
    governance: GovernanceConfig = field(default_factory=GovernanceConfig)
    dispatcher: DispatcherConfig = field(default_factory=DispatcherConfig)
    modellatore: ModellatoreConfig = field(default_factory=ModellatoreConfig)
    rastrello: RastrelloConfig = field(default_factory=RastrelloConfig)
    intake: IntakeConfig = field(default_factory=IntakeConfig)


# Module-level default (mutable so callers can tweak at runtime)
default_config = MemoryFragmentsConfig()

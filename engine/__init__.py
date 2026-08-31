"""Engine module — evaluation, diff/explanation, composition, conflict detection,
intake verification, dispatch decision-making, and Modellatore sandbox."""

from .composer import Composer
from .conflict import ConflictDetector, ConflictEntry, ConflictReport
from .diff_explain import DiffExplainEngine
from .dispatcher import Dispatcher, DispatchPlan, DispatchPath
from .evaluator import Evaluator
from .intake import IntakeVerifier, IntakeResult
from .modellatore import CompositionContradiction, CompositionGap, CompositionResult, Modellatore
from .rastrello import Rastrello, FragmentType, ExtractedPattern, PatternFrequency

__all__ = [
    "Composer",
    "ConflictDetector",
    "ConflictEntry",
    "ConflictReport",
    "DiffExplainEngine",
    "Dispatcher",
    "DispatchPlan",
    "DispatchPath",
    "Evaluator",
    "IntakeVerifier",
    "IntakeResult",
    "Modellatore",
    "CompositionResult",
    "CompositionGap",
    "CompositionContradiction",
    "Rastrello",
    "FragmentType",
    "ExtractedPattern",
    "PatternFrequency",
]
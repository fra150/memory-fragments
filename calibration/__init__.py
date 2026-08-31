"""Calibration dataset and validation framework for 3-agent majority voting."""
from .dataset import CalibrationDataset, CalibrationExample, GroundTruthLabel
from .validator import AgentCalibrator, CalibrationResult
from .agents import (
    AgentConfig,
    MockQualityAgent,
    LocalModelAgent,
    OllamaQualityAgent,
    OpenAIApiAgent,
    create_mock_agents,
    QUALITY_EVALUATOR_PROMPT,
)

__all__ = [
    "CalibrationDataset",
    "CalibrationExample",
    "GroundTruthLabel",
    "AgentCalibrator",
    "CalibrationResult",
    "AgentConfig",
    "MockQualityAgent",
    "LocalModelAgent",
    "OllamaQualityAgent",
    "OpenAIApiAgent",
    "create_mock_agents",
    "QUALITY_EVALUATOR_PROMPT",
]

"""Evaluation components for ForesightBench."""

from .config import (
    EvaluationConfig,
    PenaltyConfig,
    SemanticWeights,
    ForesightScoreWeights,
    ReliabilityWeights,
    PlanningQualityWeights,
    CompletenessHeuristic,
    DriftConfig,
    PassThresholds,
    StepBounds,
    AlignmentConfig,
    DecomposedEvalConfig,
    DEFAULT_CONFIG,
    load_config_from_dict,
)
from .rule_validators import (
    RuleValidator,
    ValidationResult,
    ValidationIssue,
    ValidationSeverity,
    quick_validate,
)
from .semantic_evaluator import (
    SemanticEvaluator,
    SemanticEvaluationResult,
    StepEvaluation,
    EvaluationMethod,
    calculate_drift,
)
from .alignment import (
    SemanticAligner,
    AlignmentResult,
    StepAlignment,
    AlignmentType,
    EmbeddingClient,
    align_with_fallback,
)
from .metrics import (
    MetricsComputer,
    StepMetrics,
    TaskMetrics,
    GlobalMetrics,
    format_metrics_report,
    format_task_report,
)

__all__ = [
    # Configuration
    "EvaluationConfig",
    "PenaltyConfig",
    "SemanticWeights",
    "ForesightScoreWeights",
    "ReliabilityWeights",
    "PlanningQualityWeights",
    "CompletenessHeuristic",
    "DriftConfig",
    "PassThresholds",
    "StepBounds",
    "AlignmentConfig",
    "DecomposedEvalConfig",
    "DEFAULT_CONFIG",
    "load_config_from_dict",
    # Rule validation
    "RuleValidator",
    "ValidationResult",
    "ValidationIssue",
    "ValidationSeverity",
    "quick_validate",
    # Semantic evaluation
    "SemanticEvaluator",
    "SemanticEvaluationResult",
    "StepEvaluation",
    "EvaluationMethod",
    "calculate_drift",
    # Alignment
    "SemanticAligner",
    "AlignmentResult",
    "StepAlignment",
    "AlignmentType",
    "EmbeddingClient",
    "align_with_fallback",
    # Metrics
    "MetricsComputer",
    "StepMetrics",
    "TaskMetrics",
    "GlobalMetrics",
    "format_metrics_report",
    "format_task_report",
]

"""
Evaluation Configuration - Centralized configuration for all evaluation parameters.

This module provides a single source of truth for all evaluation weights,
thresholds, and parameters. All hardcoded values are now configurable.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class PenaltyConfig:
    """
    Penalty weights for rule validation issues.

    Higher values = more severe penalty (subtracted from 1.0 score).
    """
    step_count_mismatch: float = 0.1   # Per step difference between plan/execution
    skipped_step: float = 0.15          # Per skipped step (higher = worse to skip)
    extra_step: float = 0.05            # Per extra step (lower = extra work is okay)
    empty_step: float = 0.1             # Per empty step
    numbering_error: float = 0.05       # Per numbering issue
    parse_failure: float = 0.5          # Complete parsing failure (critical)

    def to_dict(self) -> dict[str, float]:
        """Convert to dictionary for RuleValidator compatibility."""
        return {
            "step_count_mismatch": self.step_count_mismatch,
            "skipped_step": self.skipped_step,
            "extra_step": self.extra_step,
            "empty_step": self.empty_step,
            "numbering_error": self.numbering_error,
            "parse_failure": self.parse_failure,
        }


@dataclass
class SemanticWeights:
    """
    Weights for combining semantic evaluation dimensions.

    Must sum to 1.0 for proper scoring.
    """
    step_match: float = 0.4             # How well execution matches plan
    completeness: float = 0.3           # Whether step was fully executed
    constraint_fidelity: float = 0.2    # Whether constraints were followed
    step_purity: float = 0.1            # No cross-step leakage

    def __post_init__(self):
        total = self.step_match + self.completeness + self.constraint_fidelity + self.step_purity
        if abs(total - 1.0) > 0.001:
            raise ValueError(f"SemanticWeights must sum to 1.0, got {total}")

    def to_dict(self) -> dict[str, float]:
        """Convert to dictionary for MetricsComputer compatibility."""
        return {
            "step_match": self.step_match,
            "completeness": self.completeness,
            "constraint_fidelity": self.constraint_fidelity,
            "step_purity": self.step_purity,
        }


@dataclass
class ForesightScoreWeights:
    """
    Weights for computing the final foresight score.

    foresight_score = rule_weight * rule_score + semantic_weight * semantic_score
    Must sum to 1.0.
    """
    rule_validation: float = 0.3        # Weight for rule-based validation
    semantic_evaluation: float = 0.7    # Weight for semantic (LLM) evaluation

    def __post_init__(self):
        total = self.rule_validation + self.semantic_evaluation
        if abs(total - 1.0) > 0.001:
            raise ValueError(f"ForesightScoreWeights must sum to 1.0, got {total}")


@dataclass
class ReliabilityWeights:
    """
    Weights for computing execution reliability score.

    execution_reliability = (1 - skip_rate) * skip_weight + step_match * match_weight + rule_score * rule_weight
    Must sum to 1.0.
    """
    skip_avoidance: float = 0.5         # Weight for not skipping steps
    step_match: float = 0.3             # Weight for matching plan steps
    rule_compliance: float = 0.2        # Weight for rule validation score

    def __post_init__(self):
        total = self.skip_avoidance + self.step_match + self.rule_compliance
        if abs(total - 1.0) > 0.001:
            raise ValueError(f"ReliabilityWeights must sum to 1.0, got {total}")


@dataclass
class PlanningQualityWeights:
    """
    Weights for computing planning quality score.

    planning_quality = completeness * completeness_weight + rule_score * rule_weight
    Must sum to 1.0.
    """
    completeness: float = 0.5           # Weight for step completeness
    rule_compliance: float = 0.5        # Weight for rule validation score

    def __post_init__(self):
        total = self.completeness + self.rule_compliance
        if abs(total - 1.0) > 0.001:
            raise ValueError(f"PlanningQualityWeights must sum to 1.0, got {total}")


@dataclass
class CompletenessHeuristic:
    """
    Thresholds for heuristic completeness evaluation (word count based).

    When no LLM judge is available, completeness is estimated from output length.
    """
    very_short_threshold: int = 10      # Words below this -> very_short_score
    short_threshold: int = 50           # Words below this -> short_score
    medium_threshold: int = 150         # Words below this -> medium_score

    very_short_score: float = 0.3       # Score for very short outputs
    short_score: float = 0.6            # Score for short outputs
    medium_score: float = 0.8           # Score for medium outputs
    long_score: float = 1.0             # Score for long outputs

    def get_score(self, word_count: int) -> float:
        """Get completeness score based on word count."""
        if word_count < self.very_short_threshold:
            return self.very_short_score
        elif word_count < self.short_threshold:
            return self.short_score
        elif word_count < self.medium_threshold:
            return self.medium_score
        else:
            return self.long_score


@dataclass
class DriftConfig:
    """
    Configuration for drift detection analysis.

    Drift measures performance degradation as execution progresses.
    """
    threshold: float = 0.1              # Magnitude above which drift is detected
    rolling_window_size: int = 3        # Window size for rolling average smoothing

    def classify_trend(self, drift_magnitude: float) -> str:
        """Classify drift trend based on magnitude."""
        if drift_magnitude > self.threshold:
            return "declining"
        elif drift_magnitude < -self.threshold:
            return "improving"
        else:
            return "stable"


@dataclass
class PassThresholds:
    """
    Thresholds for determining pass/fail status.
    """
    rule_validation: float = 0.5        # Minimum rule validation score to pass
    semantic_evaluation: float = 0.7    # Minimum semantic score to pass (default)
    foresight_score: float = 0.7        # Minimum foresight score to pass


@dataclass
class StepBounds:
    """
    Bounds for plan step count validation.
    """
    min_steps: int = 1                  # Minimum allowed steps in a plan
    max_steps: int = 20                 # Maximum allowed steps in a plan


@dataclass
class AlignmentConfig:
    """
    Configuration for semantic step alignment.
    """
    similarity_threshold: float = 0.5   # Minimum similarity for alignment
    merge_detection: bool = True        # Detect merged steps
    split_detection: bool = True        # Detect split steps
    use_semantic: bool = True           # Use semantic vs index-based alignment
    embedding_model: str = "text-embedding-3-small"  # Model for embeddings


@dataclass
class DecomposedEvalConfig:
    """
    Configuration for decomposed (Q&A-based) evaluation.
    """
    use_decomposed: bool = True         # Use decomposed prompts vs single-score
    intent_weight: float = 0.5          # Weight for intent in step_match
    quality_weight: float = 0.5         # Weight for quality in step_match
    merge_purity_penalty: float = 0.7   # Purity score for merged steps
    extra_step_score: float = 0.25      # Default score for extra steps


@dataclass
class EvaluationConfig:
    """
    Master configuration for the entire evaluation system.

    This is the single source of truth for all evaluation parameters.
    Use this to customize evaluation behavior without modifying code.

    Example:
        ```python
        # Create custom config with stricter skipped step penalties
        config = EvaluationConfig(
            penalties=PenaltyConfig(skipped_step=0.25),
            pass_thresholds=PassThresholds(foresight_score=0.8),
        )

        # Use in evaluation
        validator = RuleValidator(config=config)
        evaluator = SemanticEvaluator(config=config)
        computer = MetricsComputer(config=config)
        ```
    """
    # Component configurations
    penalties: PenaltyConfig = field(default_factory=PenaltyConfig)
    semantic_weights: SemanticWeights = field(default_factory=SemanticWeights)
    foresight_weights: ForesightScoreWeights = field(default_factory=ForesightScoreWeights)
    reliability_weights: ReliabilityWeights = field(default_factory=ReliabilityWeights)
    planning_weights: PlanningQualityWeights = field(default_factory=PlanningQualityWeights)
    completeness_heuristic: CompletenessHeuristic = field(default_factory=CompletenessHeuristic)
    drift: DriftConfig = field(default_factory=DriftConfig)
    pass_thresholds: PassThresholds = field(default_factory=PassThresholds)
    step_bounds: StepBounds = field(default_factory=StepBounds)

    # New alignment and decomposed evaluation configs
    alignment: AlignmentConfig = field(default_factory=AlignmentConfig)
    decomposed_eval: DecomposedEvalConfig = field(default_factory=DecomposedEvalConfig)

    # Additional settings
    late_step_bonus: float = 0.0        # Extra weight for later steps (0 = disabled)

    def summary(self) -> str:
        """Return a human-readable summary of the configuration."""
        return f"""
EvaluationConfig Summary
========================

Penalties (subtracted from 1.0):
  - Skipped step: {self.penalties.skipped_step} per step
  - Extra step: {self.penalties.extra_step} per step
  - Empty step: {self.penalties.empty_step} per step
  - Parse failure: {self.penalties.parse_failure} (critical)

Semantic Weights (must sum to 1.0):
  - Step match: {self.semantic_weights.step_match}
  - Completeness: {self.semantic_weights.completeness}
  - Constraint fidelity: {self.semantic_weights.constraint_fidelity}
  - Step purity: {self.semantic_weights.step_purity}

Foresight Score = {self.foresight_weights.rule_validation} x rule + {self.foresight_weights.semantic_evaluation} x semantic

Pass Thresholds:
  - Rule validation: {self.pass_thresholds.rule_validation}
  - Semantic evaluation: {self.pass_thresholds.semantic_evaluation}
  - Foresight score: {self.pass_thresholds.foresight_score}

Drift Detection:
  - Threshold: +/-{self.drift.threshold}
  - Rolling window: {self.drift.rolling_window_size} steps

Alignment:
  - Use semantic: {self.alignment.use_semantic}
  - Similarity threshold: {self.alignment.similarity_threshold}
  - Merge detection: {self.alignment.merge_detection}
  - Split detection: {self.alignment.split_detection}

Decomposed Evaluation:
  - Use decomposed: {self.decomposed_eval.use_decomposed}
  - Intent weight: {self.decomposed_eval.intent_weight}
  - Quality weight: {self.decomposed_eval.quality_weight}

Step Bounds: {self.step_bounds.min_steps} - {self.step_bounds.max_steps}
"""


# Default configuration instance
DEFAULT_CONFIG = EvaluationConfig()


def _filter_comments(d: dict) -> dict:
    """Remove _comment keys from a dictionary."""
    return {k: v for k, v in d.items() if not k.startswith("_")}


def load_config_from_dict(data: dict) -> EvaluationConfig:
    """
    Load configuration from a dictionary (e.g., from JSON/YAML file).

    Args:
        data: Dictionary with configuration values

    Returns:
        EvaluationConfig instance
    """
    config = EvaluationConfig()

    if "penalties" in data:
        config.penalties = PenaltyConfig(**_filter_comments(data["penalties"]))
    if "semantic_weights" in data:
        config.semantic_weights = SemanticWeights(**_filter_comments(data["semantic_weights"]))
    if "foresight_weights" in data:
        config.foresight_weights = ForesightScoreWeights(**_filter_comments(data["foresight_weights"]))
    if "reliability_weights" in data:
        config.reliability_weights = ReliabilityWeights(**_filter_comments(data["reliability_weights"]))
    if "planning_weights" in data:
        config.planning_weights = PlanningQualityWeights(**_filter_comments(data["planning_weights"]))
    if "completeness_heuristic" in data:
        config.completeness_heuristic = CompletenessHeuristic(**_filter_comments(data["completeness_heuristic"]))
    if "drift" in data:
        config.drift = DriftConfig(**_filter_comments(data["drift"]))
    if "pass_thresholds" in data:
        config.pass_thresholds = PassThresholds(**_filter_comments(data["pass_thresholds"]))
    if "step_bounds" in data:
        config.step_bounds = StepBounds(**_filter_comments(data["step_bounds"]))
    if "alignment" in data:
        config.alignment = AlignmentConfig(**_filter_comments(data["alignment"]))
    if "decomposed_eval" in data:
        config.decomposed_eval = DecomposedEvalConfig(**_filter_comments(data["decomposed_eval"]))
    if "late_step_bonus" in data:
        config.late_step_bonus = data["late_step_bonus"]

    return config

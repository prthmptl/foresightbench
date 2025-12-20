"""
Metrics Computation - Calculates step-level, task-level, and global metrics.

Computes:
- Foresight Score: Overall plan-execution faithfulness
- Execution Reliability: Consistency of execution
- Intent Accuracy: How well the model identified correct tasks
- Quality Score: How well tasks were executed
- Skipped/Extra Step Rates
- Degradation metrics
"""

from dataclasses import dataclass, field
from typing import Optional, TYPE_CHECKING
import statistics

from .rule_validators import ValidationResult
from .semantic_evaluator import SemanticEvaluationResult, StepEvaluation, calculate_drift
from .alignment import AlignmentResult, AlignmentType

if TYPE_CHECKING:
    from .config import EvaluationConfig


@dataclass
class StepMetrics:
    """Metrics for a single step."""
    step_index: int
    # Core metrics
    intent_match: float        # Did the model do the right thing?
    execution_quality: float   # How well was it done?
    step_match: float          # Combined (backwards compatibility)
    constraint_fidelity: float
    step_purity: float
    completeness: float
    combined_score: float
    # Validation
    rule_passed: bool
    # Alignment info
    alignment_type: str = "1:1"  # "1:1", "merge", "split", "skip", "extra"


@dataclass
class TaskMetrics:
    """Metrics for a single task run."""
    task_id: str
    model: str
    run_id: str

    # Core scores
    foresight_score: float       # Overall plan-execution faithfulness
    execution_reliability: float # How reliably execution followed plan
    planning_quality: float      # Quality of the generated plan

    # New decomposed scores
    intent_accuracy: float       # Average intent match across steps
    quality_score: float         # Average execution quality across steps

    # Step metrics
    step_metrics: list[StepMetrics]
    average_step_score: float

    # Structural metrics
    plan_step_count: int
    execution_step_count: int
    skipped_step_count: int
    extra_step_count: int
    skipped_step_rate: float
    extra_step_rate: float

    # Alignment metrics
    merge_count: int = 0
    split_count: int = 0
    reorder_detected: bool = False
    alignment_score: float = 1.0  # Overall alignment quality

    # Degradation
    degradation_curve: list[float] = field(default_factory=list)
    drift_detected: bool = False
    drift_magnitude: float = 0.0

    # Validation
    rule_validation_score: float = 0.0
    semantic_evaluation_score: float = 0.0

    # Metadata
    latency_ms: float = 0.0
    token_count: int = 0
    metadata: dict = field(default_factory=dict)


@dataclass
class GlobalMetrics:
    """Aggregated metrics across all tasks/runs."""
    model: str
    total_tasks: int
    total_runs: int

    # Aggregate scores
    mean_foresight_score: float
    std_foresight_score: float
    median_foresight_score: float

    mean_execution_reliability: float
    mean_planning_quality: float

    # New decomposed aggregates
    mean_intent_accuracy: float
    mean_quality_score: float

    # Rates
    overall_skipped_step_rate: float
    overall_extra_step_rate: float

    # Alignment aggregates
    total_merges: int = 0
    total_splits: int = 0
    reorder_rate: float = 0.0
    mean_alignment_score: float = 1.0

    # Pass rates
    rule_validation_pass_rate: float = 0.0
    semantic_threshold_pass_rate: float = 0.0  # % above threshold

    # Degradation
    tasks_with_drift: int = 0
    drift_rate: float = 0.0
    average_drift_magnitude: float = 0.0

    # Per-category breakdown (optional)
    category_scores: dict[str, float] = field(default_factory=dict)
    difficulty_scores: dict[str, float] = field(default_factory=dict)


class MetricsComputer:
    """
    Computes and aggregates metrics from evaluation results.
    """

    def __init__(
        self,
        step_weights: Optional[dict[str, float]] = None,
        late_step_bonus: float = 0.0,  # Extra weight for later steps
        pass_threshold: float = 0.7,
        config: Optional["EvaluationConfig"] = None,
    ):
        """
        Initialize metrics computer.

        Args:
            step_weights: Weights for combining step metrics - DEPRECATED, use config
            late_step_bonus: Additional weight per step index for later steps - DEPRECATED, use config
            pass_threshold: Threshold for "passing" a task - DEPRECATED, use config
            config: EvaluationConfig for centralized configuration
        """
        self.config = config

        if config is not None:
            # Use config if provided
            self.step_weights = config.semantic_weights.to_dict()
            self.late_step_bonus = config.late_step_bonus
            self.pass_threshold = config.pass_thresholds.semantic_evaluation
        else:
            # Legacy behavior
            self.step_weights = step_weights or {
                "step_match": 0.4,
                "completeness": 0.3,
                "constraint_fidelity": 0.2,
                "step_purity": 0.1,
            }
            self.late_step_bonus = late_step_bonus
            self.pass_threshold = pass_threshold

    def compute_step_metrics(
        self,
        step_eval: StepEvaluation,
        rule_passed: bool = True,
        total_steps: int = 1,
        alignment_type: str = "1:1",
    ) -> StepMetrics:
        """
        Compute combined metrics for a single step.

        Args:
            step_eval: Semantic evaluation for the step
            rule_passed: Whether rule validation passed
            total_steps: Total number of steps (for weighting)
            alignment_type: Type of alignment for this step

        Returns:
            StepMetrics with combined score
        """
        # Calculate weighted score
        combined = (
            self.step_weights["step_match"] * step_eval.step_match +
            self.step_weights["completeness"] * step_eval.completeness +
            self.step_weights["constraint_fidelity"] * step_eval.constraint_fidelity +
            self.step_weights["step_purity"] * step_eval.step_purity
        )

        # Apply late step bonus
        if self.late_step_bonus > 0 and total_steps > 1:
            position_weight = 1.0 + (step_eval.step_index / total_steps) * self.late_step_bonus
            combined *= position_weight

        return StepMetrics(
            step_index=step_eval.step_index,
            intent_match=step_eval.intent_match,
            execution_quality=step_eval.execution_quality,
            step_match=step_eval.step_match,
            constraint_fidelity=step_eval.constraint_fidelity,
            step_purity=step_eval.step_purity,
            completeness=step_eval.completeness,
            combined_score=min(1.0, combined),
            rule_passed=rule_passed,
            alignment_type=alignment_type,
        )

    def compute_task_metrics(
        self,
        task_id: str,
        model: str,
        run_id: str,
        rule_result: ValidationResult,
        semantic_result: SemanticEvaluationResult,
        plan_step_count: int,
        execution_step_count: int,
        latency_ms: float = 0.0,
        token_count: int = 0,
    ) -> TaskMetrics:
        """
        Compute all metrics for a single task run.

        Args:
            task_id: Task identifier
            model: Model name
            run_id: Run identifier
            rule_result: Rule validation result
            semantic_result: Semantic evaluation result
            plan_step_count: Number of steps in plan
            execution_step_count: Number of steps executed
            latency_ms: Total latency
            token_count: Total tokens used

        Returns:
            Complete TaskMetrics
        """
        # Get alignment info
        alignment_result = semantic_result.alignment_result

        # Compute step metrics with alignment type
        step_metrics = []
        for i, step_eval in enumerate(semantic_result.step_evaluations):
            # Determine alignment type for this step
            alignment_type = "1:1"
            if alignment_result:
                for alignment in alignment_result.alignments:
                    if step_eval.step_index in [
                        s.index if hasattr(s, 'index') else s
                        for s in (alignment.plan_indices if alignment.plan_indices else [])
                    ] or step_eval.step_index - 1 in alignment.plan_indices:
                        alignment_type = alignment.alignment_type.value
                        break

            step_metric = self.compute_step_metrics(
                step_eval,
                rule_passed=rule_result.passed,
                total_steps=plan_step_count,
                alignment_type=alignment_type,
            )
            step_metrics.append(step_metric)

        # Average step score
        average_step_score = (
            sum(s.combined_score for s in step_metrics) / len(step_metrics)
            if step_metrics else 0.0
        )

        # Extract alignment metrics
        if alignment_result:
            merge_count = alignment_result.merge_count
            split_count = alignment_result.split_count
            skip_count = alignment_result.skip_count
            extra_count = alignment_result.extra_count
            reorder_detected = alignment_result.reorder_detected
            alignment_score = alignment_result.overall_alignment_score
        else:
            # Fallback to simple calculation
            merge_count = 0
            split_count = 0
            skip_count = max(0, plan_step_count - execution_step_count)
            extra_count = max(0, execution_step_count - plan_step_count)
            reorder_detected = False
            alignment_score = 1.0

        # Rates
        skipped_rate = skip_count / plan_step_count if plan_step_count > 0 else 0.0
        extra_rate = extra_count / plan_step_count if plan_step_count > 0 else 0.0

        # Drift analysis - use config if available
        if self.config is not None:
            drift_info = calculate_drift(
                semantic_result.degradation_curve,
                drift_threshold=self.config.drift.threshold,
                rolling_window_size=self.config.drift.rolling_window_size,
            )
        else:
            drift_info = calculate_drift(semantic_result.degradation_curve)

        # Intent accuracy and quality score
        intent_accuracy = semantic_result.average_intent_match
        quality_score = semantic_result.average_execution_quality

        # Core scores - use config weights if available
        if self.config is not None:
            fw = self.config.foresight_weights
            foresight_score = (
                rule_result.score * fw.rule_validation +
                semantic_result.overall_score * fw.semantic_evaluation
            )

            rw = self.config.reliability_weights
            execution_reliability = (
                (1.0 - skipped_rate) * rw.skip_avoidance +
                semantic_result.average_step_match * rw.step_match +
                rule_result.score * rw.rule_compliance
            )

            pw = self.config.planning_weights
            planning_quality = (
                semantic_result.average_completeness * pw.completeness +
                rule_result.score * pw.rule_compliance
            )
        else:
            # Legacy hardcoded weights
            foresight_score = (
                rule_result.score * 0.3 +
                semantic_result.overall_score * 0.7
            )

            execution_reliability = (
                (1.0 - skipped_rate) * 0.5 +
                semantic_result.average_step_match * 0.3 +
                rule_result.score * 0.2
            )

            planning_quality = (
                semantic_result.average_completeness * 0.5 +
                rule_result.score * 0.5
            )

        return TaskMetrics(
            task_id=task_id,
            model=model,
            run_id=run_id,
            foresight_score=foresight_score,
            execution_reliability=execution_reliability,
            planning_quality=planning_quality,
            intent_accuracy=intent_accuracy,
            quality_score=quality_score,
            step_metrics=step_metrics,
            average_step_score=average_step_score,
            plan_step_count=plan_step_count,
            execution_step_count=execution_step_count,
            skipped_step_count=skip_count,
            extra_step_count=extra_count,
            skipped_step_rate=skipped_rate,
            extra_step_rate=extra_rate,
            merge_count=merge_count,
            split_count=split_count,
            reorder_detected=reorder_detected,
            alignment_score=alignment_score,
            degradation_curve=semantic_result.degradation_curve,
            drift_detected=drift_info["drift_detected"],
            drift_magnitude=drift_info["drift_magnitude"],
            rule_validation_score=rule_result.score,
            semantic_evaluation_score=semantic_result.overall_score,
            latency_ms=latency_ms,
            token_count=token_count,
            metadata={
                "evaluation_method": semantic_result.method.value,
                "alignment_method": alignment_result.method if alignment_result else "none",
            },
        )

    def compute_global_metrics(
        self,
        task_metrics: list[TaskMetrics],
        model: str,
    ) -> GlobalMetrics:
        """
        Compute aggregated global metrics across all tasks.

        Args:
            task_metrics: List of TaskMetrics from individual runs
            model: Model name

        Returns:
            GlobalMetrics with aggregated statistics
        """
        if not task_metrics:
            return GlobalMetrics(
                model=model,
                total_tasks=0,
                total_runs=0,
                mean_foresight_score=0.0,
                std_foresight_score=0.0,
                median_foresight_score=0.0,
                mean_execution_reliability=0.0,
                mean_planning_quality=0.0,
                mean_intent_accuracy=0.0,
                mean_quality_score=0.0,
                overall_skipped_step_rate=0.0,
                overall_extra_step_rate=0.0,
            )

        # Extract scores
        foresight_scores = [t.foresight_score for t in task_metrics]
        execution_reliabilities = [t.execution_reliability for t in task_metrics]
        planning_qualities = [t.planning_quality for t in task_metrics]
        intent_accuracies = [t.intent_accuracy for t in task_metrics]
        quality_scores = [t.quality_score for t in task_metrics]
        skipped_rates = [t.skipped_step_rate for t in task_metrics]
        extra_rates = [t.extra_step_rate for t in task_metrics]
        alignment_scores = [t.alignment_score for t in task_metrics]

        # Unique tasks
        unique_tasks = len(set(t.task_id for t in task_metrics))

        # Foresight score statistics
        mean_foresight = statistics.mean(foresight_scores)
        std_foresight = statistics.stdev(foresight_scores) if len(foresight_scores) > 1 else 0.0
        median_foresight = statistics.median(foresight_scores)

        # Pass rates - use config thresholds if available
        rule_threshold = self.config.pass_thresholds.rule_validation if self.config else 0.5
        rule_passes = sum(1 for t in task_metrics if t.rule_validation_score >= rule_threshold)
        semantic_passes = sum(1 for t in task_metrics if t.semantic_evaluation_score >= self.pass_threshold)

        # Drift
        tasks_with_drift = sum(1 for t in task_metrics if t.drift_detected)
        drift_magnitudes = [t.drift_magnitude for t in task_metrics if t.drift_detected]
        avg_drift = statistics.mean(drift_magnitudes) if drift_magnitudes else 0.0

        # Alignment aggregates
        total_merges = sum(t.merge_count for t in task_metrics)
        total_splits = sum(t.split_count for t in task_metrics)
        reorder_count = sum(1 for t in task_metrics if t.reorder_detected)

        return GlobalMetrics(
            model=model,
            total_tasks=unique_tasks,
            total_runs=len(task_metrics),
            mean_foresight_score=mean_foresight,
            std_foresight_score=std_foresight,
            median_foresight_score=median_foresight,
            mean_execution_reliability=statistics.mean(execution_reliabilities),
            mean_planning_quality=statistics.mean(planning_qualities),
            mean_intent_accuracy=statistics.mean(intent_accuracies),
            mean_quality_score=statistics.mean(quality_scores),
            overall_skipped_step_rate=statistics.mean(skipped_rates),
            overall_extra_step_rate=statistics.mean(extra_rates),
            total_merges=total_merges,
            total_splits=total_splits,
            reorder_rate=reorder_count / len(task_metrics),
            mean_alignment_score=statistics.mean(alignment_scores),
            rule_validation_pass_rate=rule_passes / len(task_metrics),
            semantic_threshold_pass_rate=semantic_passes / len(task_metrics),
            tasks_with_drift=tasks_with_drift,
            drift_rate=tasks_with_drift / len(task_metrics),
            average_drift_magnitude=avg_drift,
        )

    def compute_category_breakdown(
        self,
        task_metrics: list[TaskMetrics],
        task_categories: dict[str, str],  # task_id -> category
    ) -> dict[str, float]:
        """
        Compute average foresight score by category.

        Args:
            task_metrics: List of TaskMetrics
            task_categories: Mapping of task_id to category

        Returns:
            Dictionary of category -> average score
        """
        category_scores: dict[str, list[float]] = {}

        for metrics in task_metrics:
            category = task_categories.get(metrics.task_id, "unknown")
            if category not in category_scores:
                category_scores[category] = []
            category_scores[category].append(metrics.foresight_score)

        return {
            cat: statistics.mean(scores)
            for cat, scores in category_scores.items()
        }


def format_metrics_report(global_metrics: GlobalMetrics) -> str:
    """
    Format global metrics as a human-readable report.

    Args:
        global_metrics: Computed global metrics

    Returns:
        Formatted string report
    """
    report = f"""
================================================================================
                        ForesightBench Results: {global_metrics.model}
================================================================================

OVERVIEW
--------
Tasks Evaluated:     {global_metrics.total_tasks}
Total Runs:          {global_metrics.total_runs}

CORE METRICS
------------
Foresight Score:     {global_metrics.mean_foresight_score:.3f} (+/-{global_metrics.std_foresight_score:.3f})
  - Median:          {global_metrics.median_foresight_score:.3f}

Execution Reliability: {global_metrics.mean_execution_reliability:.3f}
Planning Quality:      {global_metrics.mean_planning_quality:.3f}

DECOMPOSED METRICS
------------------
Intent Accuracy:     {global_metrics.mean_intent_accuracy:.3f}  (Did the model do the right thing?)
Quality Score:       {global_metrics.mean_quality_score:.3f}  (How well was it done?)

STRUCTURAL METRICS
------------------
Skipped Step Rate:   {global_metrics.overall_skipped_step_rate:.1%}
Extra Step Rate:     {global_metrics.overall_extra_step_rate:.1%}

ALIGNMENT ANALYSIS
------------------
Total Merges:        {global_metrics.total_merges} (steps combined)
Total Splits:        {global_metrics.total_splits} (steps divided)
Reorder Rate:        {global_metrics.reorder_rate:.1%}
Mean Alignment:      {global_metrics.mean_alignment_score:.3f}

PASS RATES
----------
Rule Validation:     {global_metrics.rule_validation_pass_rate:.1%}
Semantic (>=0.7):    {global_metrics.semantic_threshold_pass_rate:.1%}

DRIFT ANALYSIS
--------------
Tasks with Drift:    {global_metrics.tasks_with_drift} ({global_metrics.drift_rate:.1%})
Avg Drift Magnitude: {global_metrics.average_drift_magnitude:.3f}

================================================================================
"""
    return report


def format_task_report(task_metrics: TaskMetrics) -> str:
    """
    Format task metrics as a human-readable report.

    Args:
        task_metrics: Computed task metrics

    Returns:
        Formatted string report
    """
    report = f"""
Task: {task_metrics.task_id}
Model: {task_metrics.model}
Run: {task_metrics.run_id}

SCORES
------
Foresight Score:      {task_metrics.foresight_score:.3f}
Execution Reliability: {task_metrics.execution_reliability:.3f}
Planning Quality:      {task_metrics.planning_quality:.3f}

Intent Accuracy:       {task_metrics.intent_accuracy:.3f}
Quality Score:         {task_metrics.quality_score:.3f}

STRUCTURE
---------
Steps: {task_metrics.plan_step_count} planned, {task_metrics.execution_step_count} executed
Skipped: {task_metrics.skipped_step_count} ({task_metrics.skipped_step_rate:.1%})
Extra: {task_metrics.extra_step_count} ({task_metrics.extra_step_rate:.1%})
Merges: {task_metrics.merge_count}, Splits: {task_metrics.split_count}
Alignment Score: {task_metrics.alignment_score:.3f}

STEP DETAILS
------------
"""
    for step in task_metrics.step_metrics:
        alignment_marker = f" [{step.alignment_type}]" if step.alignment_type != "1:1" else ""
        report += f"  Step {step.step_index}{alignment_marker}: {step.combined_score:.3f} "
        report += f"(intent={step.intent_match:.2f}, quality={step.execution_quality:.2f}, complete={step.completeness:.2f})\n"

    if task_metrics.drift_detected:
        report += f"\n!! Drift detected (magnitude: {task_metrics.drift_magnitude:.3f})\n"

    if task_metrics.reorder_detected:
        report += f"\n!! Step reordering detected\n"

    return report

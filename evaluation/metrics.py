"""
Metrics Computation - Calculates step-level, task-level, and global metrics.

Computes:
- Foresight Score: Overall plan-execution faithfulness
- Execution Reliability: Consistency of execution
- Skipped/Extra Step Rates
- Degradation metrics
"""

from dataclasses import dataclass, field
from typing import Optional
import statistics

from .rule_validators import ValidationResult
from .semantic_evaluator import SemanticEvaluationResult, StepEvaluation, calculate_drift


@dataclass
class StepMetrics:
    """Metrics for a single step."""
    step_index: int
    step_match: float
    constraint_fidelity: float
    step_purity: float
    completeness: float
    combined_score: float
    rule_passed: bool


@dataclass
class TaskMetrics:
    """Metrics for a single task run."""
    task_id: str
    model: str
    run_id: str
    
    # Core scores
    foresight_score: float  # Overall plan-execution faithfulness
    execution_reliability: float  # How reliably execution followed plan
    planning_quality: float  # Quality of the generated plan
    
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
    
    # Degradation
    degradation_curve: list[float]
    drift_detected: bool
    drift_magnitude: float
    
    # Validation
    rule_validation_score: float
    semantic_evaluation_score: float
    
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
    
    # Rates
    overall_skipped_step_rate: float
    overall_extra_step_rate: float
    
    # Pass rates
    rule_validation_pass_rate: float
    semantic_threshold_pass_rate: float  # % above threshold
    
    # Degradation
    tasks_with_drift: int
    drift_rate: float
    average_drift_magnitude: float
    
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
    ):
        """
        Initialize metrics computer.
        
        Args:
            step_weights: Weights for combining step metrics
            late_step_bonus: Additional weight per step index for later steps
            pass_threshold: Threshold for "passing" a task
        """
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
    ) -> StepMetrics:
        """
        Compute combined metrics for a single step.
        
        Args:
            step_eval: Semantic evaluation for the step
            rule_passed: Whether rule validation passed
            total_steps: Total number of steps (for weighting)
            
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
            step_match=step_eval.step_match,
            constraint_fidelity=step_eval.constraint_fidelity,
            step_purity=step_eval.step_purity,
            completeness=step_eval.completeness,
            combined_score=min(1.0, combined),
            rule_passed=rule_passed,
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
        # Compute step metrics
        step_metrics = []
        for step_eval in semantic_result.step_evaluations:
            step_metric = self.compute_step_metrics(
                step_eval,
                rule_passed=rule_result.passed,
                total_steps=plan_step_count,
            )
            step_metrics.append(step_metric)

        # Average step score
        average_step_score = (
            sum(s.combined_score for s in step_metrics) / len(step_metrics)
            if step_metrics else 0.0
        )

        # Skipped and extra steps
        skipped = max(0, plan_step_count - execution_step_count)
        extra = max(0, execution_step_count - plan_step_count)
        skipped_rate = skipped / plan_step_count if plan_step_count > 0 else 0.0
        extra_rate = extra / plan_step_count if plan_step_count > 0 else 0.0

        # Drift analysis
        drift_info = calculate_drift(semantic_result.degradation_curve)

        # Core scores
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
            step_metrics=step_metrics,
            average_step_score=average_step_score,
            plan_step_count=plan_step_count,
            execution_step_count=execution_step_count,
            skipped_step_count=skipped,
            extra_step_count=extra,
            skipped_step_rate=skipped_rate,
            extra_step_rate=extra_rate,
            degradation_curve=semantic_result.degradation_curve,
            drift_detected=drift_info["drift_detected"],
            drift_magnitude=drift_info["drift_magnitude"],
            rule_validation_score=rule_result.score,
            semantic_evaluation_score=semantic_result.overall_score,
            latency_ms=latency_ms,
            token_count=token_count,
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
                overall_skipped_step_rate=0.0,
                overall_extra_step_rate=0.0,
                rule_validation_pass_rate=0.0,
                semantic_threshold_pass_rate=0.0,
                tasks_with_drift=0,
                drift_rate=0.0,
                average_drift_magnitude=0.0,
            )

        # Extract scores
        foresight_scores = [t.foresight_score for t in task_metrics]
        execution_reliabilities = [t.execution_reliability for t in task_metrics]
        planning_qualities = [t.planning_quality for t in task_metrics]
        skipped_rates = [t.skipped_step_rate for t in task_metrics]
        extra_rates = [t.extra_step_rate for t in task_metrics]

        # Unique tasks
        unique_tasks = len(set(t.task_id for t in task_metrics))

        # Foresight score statistics
        mean_foresight = statistics.mean(foresight_scores)
        std_foresight = statistics.stdev(foresight_scores) if len(foresight_scores) > 1 else 0.0
        median_foresight = statistics.median(foresight_scores)

        # Pass rates
        rule_passes = sum(1 for t in task_metrics if t.rule_validation_score >= 0.5)
        semantic_passes = sum(1 for t in task_metrics if t.semantic_evaluation_score >= self.pass_threshold)

        # Drift
        tasks_with_drift = sum(1 for t in task_metrics if t.drift_detected)
        drift_magnitudes = [t.drift_magnitude for t in task_metrics if t.drift_detected]
        avg_drift = statistics.mean(drift_magnitudes) if drift_magnitudes else 0.0

        return GlobalMetrics(
            model=model,
            total_tasks=unique_tasks,
            total_runs=len(task_metrics),
            mean_foresight_score=mean_foresight,
            std_foresight_score=std_foresight,
            median_foresight_score=median_foresight,
            mean_execution_reliability=statistics.mean(execution_reliabilities),
            mean_planning_quality=statistics.mean(planning_qualities),
            overall_skipped_step_rate=statistics.mean(skipped_rates),
            overall_extra_step_rate=statistics.mean(extra_rates),
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
Foresight Score:     {global_metrics.mean_foresight_score:.3f} (±{global_metrics.std_foresight_score:.3f})
  - Median:          {global_metrics.median_foresight_score:.3f}

Execution Reliability: {global_metrics.mean_execution_reliability:.3f}
Planning Quality:      {global_metrics.mean_planning_quality:.3f}

STRUCTURAL METRICS
------------------
Skipped Step Rate:   {global_metrics.overall_skipped_step_rate:.1%}
Extra Step Rate:     {global_metrics.overall_extra_step_rate:.1%}

PASS RATES
----------
Rule Validation:     {global_metrics.rule_validation_pass_rate:.1%}
Semantic (≥0.7):     {global_metrics.semantic_threshold_pass_rate:.1%}

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

Foresight Score:      {task_metrics.foresight_score:.3f}
Execution Reliability: {task_metrics.execution_reliability:.3f}
Planning Quality:      {task_metrics.planning_quality:.3f}

Steps: {task_metrics.plan_step_count} planned, {task_metrics.execution_step_count} executed
Skipped: {task_metrics.skipped_step_count} ({task_metrics.skipped_step_rate:.1%})
Extra: {task_metrics.extra_step_count} ({task_metrics.extra_step_rate:.1%})

Step Scores:
"""
    for step in task_metrics.step_metrics:
        report += f"  Step {step.step_index}: {step.combined_score:.3f} (match={step.step_match:.2f}, complete={step.completeness:.2f})\n"

    if task_metrics.drift_detected:
        report += f"\n⚠️  Drift detected (magnitude: {task_metrics.drift_magnitude:.3f})\n"

    return report

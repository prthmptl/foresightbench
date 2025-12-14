"""Tests for ForesightBench evaluation components."""

import pytest
import sys
sys.path.insert(0, '..')

from foresight_bench.core.capture import PlanCapture, ExecutionCapture
from foresight_bench.evaluation.rule_validators import RuleValidator, ValidationSeverity
from foresight_bench.evaluation.semantic_evaluator import SemanticEvaluator, calculate_drift
from foresight_bench.evaluation.metrics import MetricsComputer


class TestRuleValidator:
    """Tests for rule-based validation."""

    def test_validate_good_plan(self):
        plan_text = """Step 1: First
Step 2: Second
Step 3: Third"""
        
        plan = PlanCapture(plan_text)
        validator = RuleValidator()
        result = validator.validate_plan(plan, min_steps=2, max_steps=5)
        
        assert result.passed
        assert result.score > 0.9

    def test_validate_plan_too_few_steps(self):
        plan_text = """Step 1: Only one"""
        
        plan = PlanCapture(plan_text)
        validator = RuleValidator()
        result = validator.validate_plan(plan, min_steps=3, max_steps=5)
        
        assert not result.passed or result.score < 1.0
        assert any(i.code == "TOO_FEW_STEPS" for i in result.issues)

    def test_validate_execution_against_plan(self):
        plan_text = """Step 1: Do A
Step 2: Do B
Step 3: Do C"""
        
        exec_text = """Step 1: Did A
Step 2: Did B
Step 3: Did C"""
        
        plan = PlanCapture(plan_text)
        execution = ExecutionCapture(exec_text, expected_steps=3)
        
        validator = RuleValidator()
        result = validator.validate_execution(execution, plan)
        
        assert result.passed
        assert result.score > 0.9

    def test_detect_skipped_steps(self):
        plan_text = """Step 1: A
Step 2: B
Step 3: C"""
        
        exec_text = """Step 1: Did A
Step 3: Did C"""
        
        plan = PlanCapture(plan_text)
        execution = ExecutionCapture(exec_text, expected_steps=3)
        
        validator = RuleValidator()
        result = validator.validate_execution(execution, plan)
        
        assert any(i.code == "SKIPPED_STEP" for i in result.issues)

    def test_detect_extra_steps(self):
        plan_text = """Step 1: A
Step 2: B"""
        
        exec_text = """Step 1: Did A
Step 2: Did B
Step 3: Did extra"""
        
        plan = PlanCapture(plan_text)
        execution = ExecutionCapture(exec_text, expected_steps=2)
        
        validator = RuleValidator()
        result = validator.validate_execution(execution, plan)
        
        assert any(i.code == "EXTRA_STEP" for i in result.issues)


class TestSemanticEvaluator:
    """Tests for semantic evaluation (heuristic mode)."""

    def test_step_match_heuristic(self):
        evaluator = SemanticEvaluator(llm_client=None)
        
        # Good match - overlapping keywords
        score = evaluator._evaluate_step_match_heuristic(
            "Analyze the data and find patterns",
            "I analyzed the data to discover patterns in the dataset"
        )
        assert score > 0.3
        
        # Poor match - no overlap
        score = evaluator._evaluate_step_match_heuristic(
            "Write the introduction",
            "Calculate the final numbers"
        )
        assert score < 0.3

    def test_completeness_heuristic(self):
        evaluator = SemanticEvaluator(llm_client=None)
        
        # Long response - more complete
        long_response = " ".join(["word"] * 100)
        score = evaluator._evaluate_completeness_heuristic("Step", long_response)
        assert score > 0.7
        
        # Short response - less complete
        short_response = "Done"
        score = evaluator._evaluate_completeness_heuristic("Step", short_response)
        assert score < 0.5
        
        # Empty - not complete
        score = evaluator._evaluate_completeness_heuristic("Step", "")
        assert score == 0.0


class TestDriftCalculation:
    """Tests for drift/degradation analysis."""

    def test_no_drift_stable(self):
        curve = [0.8, 0.8, 0.8, 0.8]
        result = calculate_drift(curve)
        
        assert not result["drift_detected"]
        assert result["trend"] == "stable"

    def test_declining_drift(self):
        curve = [0.9, 0.85, 0.7, 0.5, 0.3]
        result = calculate_drift(curve)
        
        assert result["drift_detected"]
        assert result["trend"] == "declining"
        assert result["drift_magnitude"] > 0

    def test_improving_trend(self):
        curve = [0.4, 0.5, 0.7, 0.9]
        result = calculate_drift(curve)
        
        assert result["trend"] == "improving"


class TestMetricsComputer:
    """Tests for metrics computation."""

    def test_compute_step_metrics(self):
        from foresight_bench.evaluation.semantic_evaluator import StepEvaluation, EvaluationMethod
        
        step_eval = StepEvaluation(
            step_index=1,
            step_match=0.8,
            constraint_fidelity=1.0,
            step_purity=1.0,
            completeness=0.9,
            overall_score=0.9,
            method=EvaluationMethod.HEURISTIC,
        )
        
        computer = MetricsComputer()
        metrics = computer.compute_step_metrics(step_eval)
        
        assert metrics.step_index == 1
        assert 0 <= metrics.combined_score <= 1

    def test_global_metrics_aggregation(self):
        from foresight_bench.evaluation.metrics import TaskMetrics, StepMetrics
        
        # Create sample task metrics
        task_metrics = [
            TaskMetrics(
                task_id="task1",
                model="test-model",
                run_id="run1",
                foresight_score=0.8,
                execution_reliability=0.85,
                planning_quality=0.9,
                step_metrics=[],
                average_step_score=0.8,
                plan_step_count=4,
                execution_step_count=4,
                skipped_step_count=0,
                extra_step_count=0,
                skipped_step_rate=0.0,
                extra_step_rate=0.0,
                degradation_curve=[0.8, 0.8, 0.8, 0.8],
                drift_detected=False,
                drift_magnitude=0.0,
                rule_validation_score=1.0,
                semantic_evaluation_score=0.8,
            ),
            TaskMetrics(
                task_id="task2",
                model="test-model",
                run_id="run2",
                foresight_score=0.7,
                execution_reliability=0.75,
                planning_quality=0.8,
                step_metrics=[],
                average_step_score=0.7,
                plan_step_count=5,
                execution_step_count=4,
                skipped_step_count=1,
                extra_step_count=0,
                skipped_step_rate=0.2,
                extra_step_rate=0.0,
                degradation_curve=[0.9, 0.7, 0.6, 0.5],
                drift_detected=True,
                drift_magnitude=0.25,
                rule_validation_score=0.8,
                semantic_evaluation_score=0.7,
            ),
        ]
        
        computer = MetricsComputer()
        global_metrics = computer.compute_global_metrics(task_metrics, "test-model")
        
        assert global_metrics.total_tasks == 2
        assert global_metrics.mean_foresight_score == 0.75
        assert global_metrics.tasks_with_drift == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

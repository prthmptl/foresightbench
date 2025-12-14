"""
Semantic Step Evaluator - Deep evaluation using LLM-as-judge and embeddings.

Evaluates:
- Step Match: How well execution matches plan
- Constraint Fidelity: Whether constraints were followed
- Step Purity: Cross-step leakage detection
- Completeness: Whether step was fully executed
"""

import re
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum

from ..core.capture import PlanStep, ExecutionStep, PlanCapture, ExecutionCapture, align_plan_and_execution
from ..core.llm_interface import LLMClient, GenerationConfig


class EvaluationMethod(str, Enum):
    """Method used for semantic evaluation."""
    LLM_JUDGE = "llm_judge"
    EMBEDDING = "embedding"
    HEURISTIC = "heuristic"


@dataclass
class StepEvaluation:
    """Evaluation result for a single step."""
    step_index: int
    step_match: float  # 0-1: How well execution matches plan
    constraint_fidelity: float  # 0-1: Whether constraints were followed
    step_purity: float  # 0-1: No cross-step leakage
    completeness: float  # 0-1: Step fully executed
    overall_score: float  # Combined score
    method: EvaluationMethod
    rationale: str = ""
    details: dict = field(default_factory=dict)


@dataclass
class SemanticEvaluationResult:
    """Complete semantic evaluation result."""
    step_evaluations: list[StepEvaluation]
    overall_score: float
    degradation_curve: list[float]  # Score by step index
    average_step_match: float
    average_completeness: float
    method: EvaluationMethod
    metadata: dict = field(default_factory=dict)


# Prompt templates for LLM-as-judge
STEP_MATCH_PROMPT = """You are evaluating whether an execution step matches its planned step.

Planned step:
{plan_step}

Executed output:
{exec_step}

Rate the alignment from 0.0 to 1.0:
- 1.0: Perfect match - execution does exactly what was planned
- 0.7-0.9: Good match - execution addresses the plan with minor deviations
- 0.4-0.6: Partial match - execution partially addresses the plan
- 0.1-0.3: Poor match - execution barely relates to the plan
- 0.0: No match - execution is completely different from plan

Return ONLY a number between 0.0 and 1.0, nothing else."""

COMPLETENESS_PROMPT = """You are evaluating whether an execution step was completed fully.

Planned step:
{plan_step}

Executed output:
{exec_step}

Rate the completeness from 0.0 to 1.0:
- 1.0: Fully complete - all aspects of the step were addressed
- 0.7-0.9: Mostly complete - most aspects addressed with minor gaps
- 0.4-0.6: Partially complete - some aspects addressed
- 0.1-0.3: Barely started - very little completed
- 0.0: Not started - no meaningful output

Return ONLY a number between 0.0 and 1.0, nothing else."""

STEP_PURITY_PROMPT = """You are checking if an execution step contains content that belongs to OTHER steps.

The plan (all steps):
{full_plan}

Current step being evaluated (Step {step_index}):
{exec_step}

Rate step purity from 0.0 to 1.0:
- 1.0: Pure - content is only about this step, no leakage from other steps
- 0.7-0.9: Mostly pure - minor references to other steps
- 0.4-0.6: Mixed - significant content from other steps
- 0.0-0.3: Impure - mostly content from other steps

Return ONLY a number between 0.0 and 1.0, nothing else."""


class SemanticEvaluator:
    """
    Performs semantic evaluation of plan-execution alignment.
    
    Uses LLM-as-judge for deep understanding of step matching,
    with fallback to heuristic methods.
    """

    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        use_embeddings: bool = False,
        config: Optional[GenerationConfig] = None,
    ):
        """
        Initialize the semantic evaluator.
        
        Args:
            llm_client: LLM client for judge prompts (optional)
            use_embeddings: Whether to use embedding similarity
            config: Generation config for LLM calls
        """
        self.llm_client = llm_client
        self.use_embeddings = use_embeddings
        self.config = config or GenerationConfig(temperature=0.0, max_tokens=50)

    def _parse_score(self, text: str) -> float:
        """Parse a score from LLM output."""
        text = text.strip()
        # Try to find a number
        match = re.search(r"(\d+\.?\d*)", text)
        if match:
            score = float(match.group(1))
            # Normalize if needed
            if score > 1.0:
                score = score / 100.0 if score <= 100 else 1.0
            return min(1.0, max(0.0, score))
        return 0.5  # Default if parsing fails

    def _evaluate_step_match_llm(self, plan_step: str, exec_step: str) -> float:
        """Evaluate step match using LLM-as-judge."""
        if self.llm_client is None:
            return self._evaluate_step_match_heuristic(plan_step, exec_step)

        prompt = STEP_MATCH_PROMPT.format(
            plan_step=plan_step,
            exec_step=exec_step,
        )
        
        result = self.llm_client.generate(prompt, config=self.config)
        return self._parse_score(result.text)

    def _evaluate_completeness_llm(self, plan_step: str, exec_step: str) -> float:
        """Evaluate completeness using LLM-as-judge."""
        if self.llm_client is None:
            return self._evaluate_completeness_heuristic(plan_step, exec_step)

        prompt = COMPLETENESS_PROMPT.format(
            plan_step=plan_step,
            exec_step=exec_step,
        )
        
        result = self.llm_client.generate(prompt, config=self.config)
        return self._parse_score(result.text)

    def _evaluate_purity_llm(self, full_plan: str, step_index: int, exec_step: str) -> float:
        """Evaluate step purity using LLM-as-judge."""
        if self.llm_client is None:
            return 1.0  # Assume pure if no LLM

        prompt = STEP_PURITY_PROMPT.format(
            full_plan=full_plan,
            step_index=step_index,
            exec_step=exec_step,
        )
        
        result = self.llm_client.generate(prompt, config=self.config)
        return self._parse_score(result.text)

    def _evaluate_step_match_heuristic(self, plan_step: str, exec_step: str) -> float:
        """Heuristic step match evaluation based on keyword overlap."""
        plan_words = set(plan_step.lower().split())
        exec_words = set(exec_step.lower().split())

        # Remove common words
        stopwords = {"the", "a", "an", "is", "are", "to", "and", "of", "in", "for", "on", "with"}
        plan_words -= stopwords
        exec_words -= stopwords

        if not plan_words:
            return 0.5

        overlap = len(plan_words & exec_words)
        score = overlap / len(plan_words)
        return min(1.0, score)

    def _evaluate_completeness_heuristic(self, plan_step: str, exec_step: str) -> float:
        """Heuristic completeness evaluation based on output length."""
        if not exec_step.strip():
            return 0.0

        # Simple heuristic: longer outputs are more likely complete
        word_count = len(exec_step.split())
        if word_count < 10:
            return 0.3
        elif word_count < 50:
            return 0.6
        elif word_count < 150:
            return 0.8
        else:
            return 1.0

    def _evaluate_constraint_fidelity(
        self,
        exec_step: str,
        constraints: list[str],
    ) -> float:
        """Evaluate how well constraints were followed."""
        if not constraints:
            return 1.0  # No constraints means perfect fidelity

        # Simple keyword-based constraint checking
        exec_lower = exec_step.lower()
        satisfied = 0

        for constraint in constraints:
            # Extract key terms from constraint
            key_terms = [w for w in constraint.lower().split() 
                        if len(w) > 3 and w not in {"should", "must", "with", "from"}]
            
            if any(term in exec_lower for term in key_terms):
                satisfied += 1

        return satisfied / len(constraints) if constraints else 1.0

    def evaluate_step(
        self,
        plan_step: PlanStep,
        exec_step: ExecutionStep,
        full_plan: str = "",
        constraints: list[str] = None,
    ) -> StepEvaluation:
        """
        Evaluate a single step pair.
        
        Args:
            plan_step: The planned step
            exec_step: The executed step
            full_plan: Full plan text (for purity check)
            constraints: Optional constraints to check
            
        Returns:
            StepEvaluation with all metrics
        """
        constraints = constraints or []
        method = EvaluationMethod.LLM_JUDGE if self.llm_client else EvaluationMethod.HEURISTIC

        # Calculate each metric
        step_match = self._evaluate_step_match_llm(plan_step.text, exec_step.content)
        completeness = self._evaluate_completeness_llm(plan_step.text, exec_step.content)
        constraint_fidelity = self._evaluate_constraint_fidelity(exec_step.content, constraints)
        
        # Step purity (only if full plan provided)
        if full_plan and self.llm_client:
            step_purity = self._evaluate_purity_llm(full_plan, plan_step.index, exec_step.content)
        else:
            step_purity = 1.0

        # Calculate overall score (weighted average)
        overall_score = (
            step_match * 0.4 +
            completeness * 0.3 +
            constraint_fidelity * 0.2 +
            step_purity * 0.1
        )

        return StepEvaluation(
            step_index=plan_step.index,
            step_match=step_match,
            constraint_fidelity=constraint_fidelity,
            step_purity=step_purity,
            completeness=completeness,
            overall_score=overall_score,
            method=method,
        )

    def evaluate_all(
        self,
        plan: PlanCapture,
        execution: ExecutionCapture,
        constraints: list[str] = None,
    ) -> SemanticEvaluationResult:
        """
        Evaluate complete plan-execution alignment.
        
        Args:
            plan: Parsed plan
            execution: Parsed execution
            constraints: Optional task constraints
            
        Returns:
            SemanticEvaluationResult with all evaluations
        """
        constraints = constraints or []
        step_evaluations = []
        degradation_curve = []

        # Align steps
        aligned = align_plan_and_execution(plan, execution)
        full_plan = plan.raw_text

        for plan_step, exec_step in aligned:
            if plan_step is None or exec_step is None:
                # Unaligned step - score as 0
                idx = plan_step.index if plan_step else exec_step.index
                step_evaluations.append(StepEvaluation(
                    step_index=idx,
                    step_match=0.0,
                    constraint_fidelity=0.0,
                    step_purity=1.0,
                    completeness=0.0,
                    overall_score=0.0,
                    method=EvaluationMethod.HEURISTIC,
                    rationale="Step not aligned (missing plan or execution)",
                ))
                degradation_curve.append(0.0)
            else:
                evaluation = self.evaluate_step(
                    plan_step,
                    exec_step,
                    full_plan,
                    constraints,
                )
                step_evaluations.append(evaluation)
                degradation_curve.append(evaluation.overall_score)

        # Calculate aggregate metrics
        if step_evaluations:
            overall_score = sum(e.overall_score for e in step_evaluations) / len(step_evaluations)
            average_step_match = sum(e.step_match for e in step_evaluations) / len(step_evaluations)
            average_completeness = sum(e.completeness for e in step_evaluations) / len(step_evaluations)
        else:
            overall_score = 0.0
            average_step_match = 0.0
            average_completeness = 0.0

        method = step_evaluations[0].method if step_evaluations else EvaluationMethod.HEURISTIC

        return SemanticEvaluationResult(
            step_evaluations=step_evaluations,
            overall_score=overall_score,
            degradation_curve=degradation_curve,
            average_step_match=average_step_match,
            average_completeness=average_completeness,
            method=method,
        )


def calculate_drift(degradation_curve: list[float]) -> dict:
    """
    Calculate drift metrics from degradation curve.
    
    Args:
        degradation_curve: List of scores by step index
        
    Returns:
        Dictionary with drift metrics
    """
    if len(degradation_curve) < 2:
        return {
            "drift_detected": False,
            "drift_magnitude": 0.0,
            "trend": "stable",
        }

    # Calculate trend
    first_half = degradation_curve[:len(degradation_curve)//2]
    second_half = degradation_curve[len(degradation_curve)//2:]

    first_avg = sum(first_half) / len(first_half) if first_half else 0
    second_avg = sum(second_half) / len(second_half) if second_half else 0

    drift_magnitude = first_avg - second_avg

    if drift_magnitude > 0.1:
        trend = "declining"
    elif drift_magnitude < -0.1:
        trend = "improving"
    else:
        trend = "stable"

    # Rolling average
    window_size = min(3, len(degradation_curve))
    rolling_avg = []
    for i in range(len(degradation_curve) - window_size + 1):
        window = degradation_curve[i:i + window_size]
        rolling_avg.append(sum(window) / len(window))

    return {
        "drift_detected": abs(drift_magnitude) > 0.1,
        "drift_magnitude": drift_magnitude,
        "trend": trend,
        "first_half_avg": first_avg,
        "second_half_avg": second_avg,
        "rolling_average": rolling_avg,
    }

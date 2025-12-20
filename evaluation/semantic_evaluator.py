"""
Semantic Step Evaluator - Deep evaluation using LLM-as-judge and embeddings.

Evaluates:
- Intent Match: Did the model attempt the right task?
- Execution Quality: How well was the task performed?
- Constraint Fidelity: Whether constraints were followed
- Step Purity: Cross-step leakage detection
- Completeness: Whether step was fully executed
"""

import re
import json
from dataclasses import dataclass, field
from typing import Optional, TYPE_CHECKING
from enum import Enum

from core.capture import PlanStep, ExecutionStep, PlanCapture, ExecutionCapture
from core.llm_interface import LLMClient, GenerationConfig
from .alignment import SemanticAligner, AlignmentResult, AlignmentType, StepAlignment

if TYPE_CHECKING:
    from .config import EvaluationConfig


class EvaluationMethod(str, Enum):
    """Method used for semantic evaluation."""
    LLM_JUDGE = "llm_judge"
    LLM_DECOMPOSED = "llm_decomposed"
    EMBEDDING = "embedding"
    HEURISTIC = "heuristic"


@dataclass
class StepEvaluation:
    """Evaluation result for a single step."""
    step_index: int
    # Core metrics (0-1 each)
    intent_match: float       # Did execution attempt the right task?
    execution_quality: float  # How well was it executed?
    completeness: float       # Was step fully executed?
    constraint_fidelity: float  # Were constraints followed?
    step_purity: float        # No cross-step leakage

    # Combined scores
    step_match: float         # Combined intent + quality (for backwards compatibility)
    overall_score: float      # Weighted combination of all metrics

    # Metadata
    method: EvaluationMethod
    rationale: str = ""
    decomposed_answers: dict = field(default_factory=dict)  # Raw Q&A responses
    details: dict = field(default_factory=dict)


@dataclass
class SemanticEvaluationResult:
    """Complete semantic evaluation result."""
    step_evaluations: list[StepEvaluation]
    overall_score: float
    degradation_curve: list[float]  # Score by step index
    average_step_match: float
    average_intent_match: float
    average_execution_quality: float
    average_completeness: float
    method: EvaluationMethod
    alignment_result: Optional[AlignmentResult] = None
    metadata: dict = field(default_factory=dict)


# ============================================================================
# DECOMPOSED EVALUATION PROMPTS
# ============================================================================

DECOMPOSED_STEP_EVAL_PROMPT = """You are evaluating how well an execution step matches its planned step.

PLAN STEP:
{plan_step}

EXECUTION OUTPUT:
{exec_step}

Answer these questions with Yes, No, or Partial:

Q1 (Intent): Does the execution attempt the same task/action described in the plan?
Q2 (Approach): Does the execution follow the approach or method implied by the plan?
Q3 (Completeness): Is the execution complete for this step (all aspects addressed)?
Q4 (Quality): Is the execution done well (accurate, clear, thorough)?

Return your answers in this exact JSON format:
{{"Q1": "Yes/No/Partial", "Q2": "Yes/No/Partial", "Q3": "Yes/No/Partial", "Q4": "Yes/No/Partial", "rationale": "brief explanation"}}"""

STEP_PURITY_EVAL_PROMPT = """You are checking if an execution step contains content that belongs to OTHER steps.

THE FULL PLAN:
{full_plan}

CURRENT STEP BEING EVALUATED (Step {step_index}):
{exec_step}

Does this execution step contain significant content that belongs to OTHER plan steps (not Step {step_index})?

Answer: Yes (contains other steps' content), No (pure - only this step's content), or Partial (minor overlap)

Return in JSON format:
{{"answer": "Yes/No/Partial", "leaked_from_steps": [], "rationale": "brief explanation"}}"""

CONSTRAINT_EVAL_PROMPT = """You are evaluating whether task constraints were followed.

TASK CONSTRAINTS:
{constraints}

EXECUTION OUTPUT:
{execution}

For each constraint, determine if it was satisfied.

Return in JSON format:
{{"results": [{{"constraint": "constraint text", "satisfied": true/false, "evidence": "brief quote or explanation"}}], "overall_satisfied": true/false}}"""

# Legacy single-score prompts (kept for backwards compatibility)
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


class SemanticEvaluator:
    """
    Performs semantic evaluation of plan-execution alignment.

    Uses decomposed LLM-as-judge for fine-grained understanding of step matching,
    with fallback to heuristic methods.
    """

    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        use_embeddings: bool = False,
        use_decomposed: bool = True,  # Use decomposed prompts by default
        generation_config: Optional[GenerationConfig] = None,
        eval_config: Optional["EvaluationConfig"] = None,
    ):
        """
        Initialize the semantic evaluator.

        Args:
            llm_client: LLM client for judge prompts (optional)
            use_embeddings: Whether to use embedding similarity
            use_decomposed: Use decomposed Q&A evaluation (recommended)
            generation_config: Generation config for LLM calls
            eval_config: EvaluationConfig for centralized configuration
        """
        self.llm_client = llm_client
        self.use_embeddings = use_embeddings
        self.use_decomposed = use_decomposed
        self.generation_config = generation_config or GenerationConfig(temperature=0.0, max_tokens=500)
        self.eval_config = eval_config

        # Aligner for semantic step matching
        self.aligner = SemanticAligner(
            llm_client=llm_client,
            similarity_threshold=0.5,
            merge_detection=True,
            split_detection=True,
            generation_config=generation_config,
        )

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

    def _parse_json_response(self, text: str) -> dict:
        """Parse JSON from LLM response, handling common issues."""
        text = text.strip()

        # Try to extract JSON from response
        json_match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass

        # Try the whole text
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return {}

    def _answer_to_score(self, answer: str) -> float:
        """Convert Yes/No/Partial answer to numeric score."""
        answer = answer.lower().strip()
        if answer in ["yes", "y", "true"]:
            return 1.0
        elif answer in ["partial", "partially", "somewhat"]:
            return 0.5
        elif answer in ["no", "n", "false"]:
            return 0.0
        else:
            return 0.5  # Default for unclear answers

    def _evaluate_step_decomposed(
        self,
        plan_step: str,
        exec_step: str,
    ) -> dict:
        """
        Evaluate step using decomposed questions.

        Returns dict with intent_match, execution_quality, completeness, and raw answers.
        """
        if self.llm_client is None:
            return self._evaluate_step_decomposed_heuristic(plan_step, exec_step)

        prompt = DECOMPOSED_STEP_EVAL_PROMPT.format(
            plan_step=plan_step,
            exec_step=exec_step,
        )

        result = self.llm_client.generate(prompt, config=self.generation_config)
        parsed = self._parse_json_response(result.text)

        if not parsed:
            # Fallback to heuristic if parsing fails
            return self._evaluate_step_decomposed_heuristic(plan_step, exec_step)

        # Convert answers to scores
        intent_match = self._answer_to_score(parsed.get("Q1", "Partial"))
        approach_match = self._answer_to_score(parsed.get("Q2", "Partial"))
        completeness = self._answer_to_score(parsed.get("Q3", "Partial"))
        quality = self._answer_to_score(parsed.get("Q4", "Partial"))

        # Execution quality is combination of approach and quality
        execution_quality = (approach_match + quality) / 2

        return {
            "intent_match": intent_match,
            "execution_quality": execution_quality,
            "completeness": completeness,
            "rationale": parsed.get("rationale", ""),
            "raw_answers": parsed,
        }

    def _evaluate_step_decomposed_heuristic(
        self,
        plan_step: str,
        exec_step: str,
    ) -> dict:
        """Heuristic decomposed evaluation when no LLM available."""
        # Use keyword-based analysis
        plan_words = set(re.findall(r'\b[a-zA-Z]{3,}\b', plan_step.lower()))
        exec_words = set(re.findall(r'\b[a-zA-Z]{3,}\b', exec_step.lower()))

        stopwords = {
            "the", "a", "an", "is", "are", "to", "and", "of", "in", "for",
            "on", "with", "that", "this", "be", "it", "step", "will", "should"
        }
        plan_words -= stopwords
        exec_words -= stopwords

        # Intent match: key words from plan appear in execution
        if plan_words:
            intent_overlap = len(plan_words & exec_words) / len(plan_words)
        else:
            intent_overlap = 0.5

        # Execution quality: based on output characteristics
        word_count = len(exec_step.split())
        if word_count < 10:
            quality = 0.3
        elif word_count < 50:
            quality = 0.6
        else:
            quality = 0.8

        # Completeness: use config thresholds if available
        if self.eval_config is not None:
            completeness = self.eval_config.completeness_heuristic.get_score(word_count)
        else:
            if word_count < 10:
                completeness = 0.3
            elif word_count < 50:
                completeness = 0.6
            elif word_count < 150:
                completeness = 0.8
            else:
                completeness = 1.0

        return {
            "intent_match": min(1.0, intent_overlap * 1.2),  # Slight boost
            "execution_quality": quality,
            "completeness": completeness,
            "rationale": "Heuristic evaluation based on keyword overlap and output length",
            "raw_answers": {},
        }

    def _evaluate_purity_llm(
        self,
        full_plan: str,
        step_index: int,
        exec_step: str,
    ) -> tuple[float, dict]:
        """Evaluate step purity using LLM-as-judge."""
        if self.llm_client is None:
            return 1.0, {}  # Assume pure if no LLM

        prompt = STEP_PURITY_EVAL_PROMPT.format(
            full_plan=full_plan,
            step_index=step_index,
            exec_step=exec_step,
        )

        result = self.llm_client.generate(prompt, config=self.generation_config)
        parsed = self._parse_json_response(result.text)

        if not parsed:
            return 1.0, {}

        answer = parsed.get("answer", "No")
        purity_score = self._answer_to_score(answer)
        # Invert: "Yes" (contains leakage) = low purity
        if answer.lower() in ["yes", "y"]:
            purity_score = 0.2
        elif answer.lower() in ["partial", "partially"]:
            purity_score = 0.6
        else:
            purity_score = 1.0

        return purity_score, parsed

    def _evaluate_constraints_llm(
        self,
        constraints: list[str],
        execution: str,
    ) -> tuple[float, dict]:
        """Evaluate constraint fidelity using LLM."""
        if not constraints:
            return 1.0, {"no_constraints": True}

        if self.llm_client is None:
            return self._evaluate_constraints_heuristic(constraints, execution)

        constraints_text = "\n".join(f"- {c}" for c in constraints)
        prompt = CONSTRAINT_EVAL_PROMPT.format(
            constraints=constraints_text,
            execution=execution,
        )

        result = self.llm_client.generate(prompt, config=self.generation_config)
        parsed = self._parse_json_response(result.text)

        if not parsed:
            return self._evaluate_constraints_heuristic(constraints, execution)

        results = parsed.get("results", [])
        if results:
            satisfied_count = sum(1 for r in results if r.get("satisfied", False))
            fidelity = satisfied_count / len(results)
        else:
            fidelity = 1.0 if parsed.get("overall_satisfied", True) else 0.0

        return fidelity, parsed

    def _evaluate_constraints_heuristic(
        self,
        constraints: list[str],
        execution: str,
    ) -> tuple[float, dict]:
        """Heuristic constraint evaluation based on keyword presence."""
        if not constraints:
            return 1.0, {}

        exec_lower = execution.lower()
        satisfied = 0
        results = []

        for constraint in constraints:
            # Extract key terms from constraint
            key_terms = [w for w in re.findall(r'\b[a-zA-Z]{4,}\b', constraint.lower())
                        if w not in {"should", "must", "with", "from", "that", "have"}]

            term_found = any(term in exec_lower for term in key_terms)
            satisfied += 1 if term_found else 0
            results.append({
                "constraint": constraint,
                "satisfied": term_found,
                "method": "keyword_heuristic",
            })

        fidelity = satisfied / len(constraints) if constraints else 1.0
        return fidelity, {"results": results}

    def _evaluate_step_match_llm(self, plan_step: str, exec_step: str) -> float:
        """Evaluate step match using legacy single-score LLM prompt."""
        if self.llm_client is None:
            return self._evaluate_step_match_heuristic(plan_step, exec_step)

        prompt = STEP_MATCH_PROMPT.format(
            plan_step=plan_step,
            exec_step=exec_step,
        )

        result = self.llm_client.generate(prompt, config=self.generation_config)
        return self._parse_score(result.text)

    def _evaluate_step_match_heuristic(self, plan_step: str, exec_step: str) -> float:
        """Heuristic step match evaluation based on keyword overlap."""
        plan_words = set(plan_step.lower().split())
        exec_words = set(exec_step.lower().split())

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

        word_count = len(exec_step.split())

        if self.eval_config is not None:
            return self.eval_config.completeness_heuristic.get_score(word_count)
        else:
            if word_count < 10:
                return 0.3
            elif word_count < 50:
                return 0.6
            elif word_count < 150:
                return 0.8
            else:
                return 1.0

    def evaluate_step(
        self,
        plan_step: PlanStep,
        exec_step: ExecutionStep,
        full_plan: str = "",
        constraints: list[str] = None,
    ) -> StepEvaluation:
        """
        Evaluate a single step pair with decomposed metrics.

        Args:
            plan_step: The planned step
            exec_step: The executed step
            full_plan: Full plan text (for purity check)
            constraints: Optional constraints to check

        Returns:
            StepEvaluation with all metrics
        """
        constraints = constraints or []

        # Determine evaluation method
        if self.llm_client and self.use_decomposed:
            method = EvaluationMethod.LLM_DECOMPOSED
        elif self.llm_client:
            method = EvaluationMethod.LLM_JUDGE
        else:
            method = EvaluationMethod.HEURISTIC

        # Get decomposed evaluation
        if self.use_decomposed:
            decomposed = self._evaluate_step_decomposed(plan_step.text, exec_step.content)
            intent_match = decomposed["intent_match"]
            execution_quality = decomposed["execution_quality"]
            completeness = decomposed["completeness"]
            rationale = decomposed["rationale"]
            decomposed_answers = decomposed["raw_answers"]
        else:
            # Legacy evaluation
            step_match_score = self._evaluate_step_match_llm(plan_step.text, exec_step.content)
            completeness = self._evaluate_completeness_heuristic(plan_step.text, exec_step.content)
            # Estimate intent and quality from single score
            intent_match = step_match_score
            execution_quality = step_match_score
            rationale = ""
            decomposed_answers = {}

        # Step purity
        if full_plan and self.llm_client:
            step_purity, purity_details = self._evaluate_purity_llm(
                full_plan, plan_step.index, exec_step.content
            )
        else:
            step_purity = 1.0
            purity_details = {}

        # Constraint fidelity
        constraint_fidelity, constraint_details = self._evaluate_constraints_llm(
            constraints, exec_step.content
        )

        # Combined step_match (backwards compatibility)
        step_match = (intent_match + execution_quality) / 2

        # Calculate overall score (weighted average)
        if self.eval_config is not None:
            weights = self.eval_config.semantic_weights
            overall_score = (
                step_match * weights.step_match +
                completeness * weights.completeness +
                constraint_fidelity * weights.constraint_fidelity +
                step_purity * weights.step_purity
            )
        else:
            overall_score = (
                step_match * 0.4 +
                completeness * 0.3 +
                constraint_fidelity * 0.2 +
                step_purity * 0.1
            )

        return StepEvaluation(
            step_index=plan_step.index,
            intent_match=intent_match,
            execution_quality=execution_quality,
            completeness=completeness,
            constraint_fidelity=constraint_fidelity,
            step_purity=step_purity,
            step_match=step_match,
            overall_score=overall_score,
            method=method,
            rationale=rationale,
            decomposed_answers=decomposed_answers,
            details={
                "purity_details": purity_details,
                "constraint_details": constraint_details,
            },
        )

    def evaluate_merged_steps(
        self,
        plan_steps: list[PlanStep],
        exec_step: ExecutionStep,
        full_plan: str = "",
        constraints: list[str] = None,
    ) -> StepEvaluation:
        """
        Evaluate a case where multiple plan steps were merged into one execution.

        Args:
            plan_steps: The plan steps that were merged
            exec_step: The single execution step
            full_plan: Full plan text
            constraints: Optional constraints

        Returns:
            StepEvaluation representing the merged evaluation
        """
        constraints = constraints or []

        # Combine plan step texts
        combined_plan = " AND ".join(f"[Step {s.index}] {s.text}" for s in plan_steps)

        # Use same evaluation logic
        method = EvaluationMethod.LLM_DECOMPOSED if self.llm_client and self.use_decomposed else EvaluationMethod.HEURISTIC

        if self.use_decomposed:
            decomposed = self._evaluate_step_decomposed(combined_plan, exec_step.content)
            intent_match = decomposed["intent_match"]
            execution_quality = decomposed["execution_quality"]
            completeness = decomposed["completeness"]
            rationale = decomposed["rationale"]
            decomposed_answers = decomposed["raw_answers"]
        else:
            step_match_score = self._evaluate_step_match_heuristic(combined_plan, exec_step.content)
            completeness = self._evaluate_completeness_heuristic(combined_plan, exec_step.content)
            intent_match = step_match_score
            execution_quality = step_match_score
            rationale = "Merged step evaluation"
            decomposed_answers = {}

        # For merged steps, purity is inherently lower (contains multiple steps' content)
        step_purity = 0.7  # Partial penalty for merging

        constraint_fidelity, constraint_details = self._evaluate_constraints_llm(
            constraints, exec_step.content
        )

        step_match = (intent_match + execution_quality) / 2

        if self.eval_config is not None:
            weights = self.eval_config.semantic_weights
            overall_score = (
                step_match * weights.step_match +
                completeness * weights.completeness +
                constraint_fidelity * weights.constraint_fidelity +
                step_purity * weights.step_purity
            )
        else:
            overall_score = (
                step_match * 0.4 +
                completeness * 0.3 +
                constraint_fidelity * 0.2 +
                step_purity * 0.1
            )

        # Use first plan step index as representative
        return StepEvaluation(
            step_index=plan_steps[0].index,
            intent_match=intent_match,
            execution_quality=execution_quality,
            completeness=completeness,
            constraint_fidelity=constraint_fidelity,
            step_purity=step_purity,
            step_match=step_match,
            overall_score=overall_score,
            method=method,
            rationale=rationale,
            decomposed_answers=decomposed_answers,
            details={
                "merged_from_steps": [s.index for s in plan_steps],
                "constraint_details": constraint_details,
            },
        )

    def evaluate_split_steps(
        self,
        plan_step: PlanStep,
        exec_steps: list[ExecutionStep],
        full_plan: str = "",
        constraints: list[str] = None,
    ) -> StepEvaluation:
        """
        Evaluate a case where one plan step was split into multiple execution steps.

        Args:
            plan_step: The single plan step
            exec_steps: The execution steps it was split into
            full_plan: Full plan text
            constraints: Optional constraints

        Returns:
            StepEvaluation representing the split evaluation
        """
        constraints = constraints or []

        # Combine execution step contents
        combined_exec = "\n---\n".join(f"[Part {i+1}] {s.content}" for i, s in enumerate(exec_steps))

        method = EvaluationMethod.LLM_DECOMPOSED if self.llm_client and self.use_decomposed else EvaluationMethod.HEURISTIC

        if self.use_decomposed:
            decomposed = self._evaluate_step_decomposed(plan_step.text, combined_exec)
            intent_match = decomposed["intent_match"]
            execution_quality = decomposed["execution_quality"]
            completeness = decomposed["completeness"]
            rationale = decomposed["rationale"]
            decomposed_answers = decomposed["raw_answers"]
        else:
            step_match_score = self._evaluate_step_match_heuristic(plan_step.text, combined_exec)
            completeness = self._evaluate_completeness_heuristic(plan_step.text, combined_exec)
            intent_match = step_match_score
            execution_quality = step_match_score
            rationale = "Split step evaluation"
            decomposed_answers = {}

        # Purity is fine for splits (each part contributes to one plan step)
        step_purity = 1.0

        constraint_fidelity, constraint_details = self._evaluate_constraints_llm(
            constraints, combined_exec
        )

        step_match = (intent_match + execution_quality) / 2

        if self.eval_config is not None:
            weights = self.eval_config.semantic_weights
            overall_score = (
                step_match * weights.step_match +
                completeness * weights.completeness +
                constraint_fidelity * weights.constraint_fidelity +
                step_purity * weights.step_purity
            )
        else:
            overall_score = (
                step_match * 0.4 +
                completeness * 0.3 +
                constraint_fidelity * 0.2 +
                step_purity * 0.1
            )

        return StepEvaluation(
            step_index=plan_step.index,
            intent_match=intent_match,
            execution_quality=execution_quality,
            completeness=completeness,
            constraint_fidelity=constraint_fidelity,
            step_purity=step_purity,
            step_match=step_match,
            overall_score=overall_score,
            method=method,
            rationale=rationale,
            decomposed_answers=decomposed_answers,
            details={
                "split_into_steps": [s.index for s in exec_steps],
                "constraint_details": constraint_details,
            },
        )

    def evaluate_all(
        self,
        plan: PlanCapture,
        execution: ExecutionCapture,
        constraints: list[str] = None,
        use_semantic_alignment: bool = True,
    ) -> SemanticEvaluationResult:
        """
        Evaluate complete plan-execution alignment.

        Args:
            plan: Parsed plan
            execution: Parsed execution
            constraints: Optional task constraints
            use_semantic_alignment: Whether to use semantic alignment

        Returns:
            SemanticEvaluationResult with all evaluations
        """
        constraints = constraints or []
        step_evaluations = []
        degradation_curve = []

        plan.parse()
        execution.parse()
        full_plan = plan.raw_text

        # Get alignment
        if use_semantic_alignment:
            alignment_result = self.aligner.align(plan, execution)
        else:
            from .alignment import align_with_fallback
            alignment_result = align_with_fallback(plan, execution, use_semantic=False)

        # Evaluate based on alignment
        for alignment in alignment_result.alignments:
            if alignment.alignment_type == AlignmentType.SKIP:
                # Skipped step - score as 0
                step_idx = alignment.plan_indices[0] if alignment.plan_indices else 0
                step_evaluations.append(StepEvaluation(
                    step_index=plan.steps[step_idx].index if step_idx < len(plan.steps) else step_idx + 1,
                    intent_match=0.0,
                    execution_quality=0.0,
                    completeness=0.0,
                    constraint_fidelity=0.0,
                    step_purity=1.0,
                    step_match=0.0,
                    overall_score=0.0,
                    method=EvaluationMethod.HEURISTIC,
                    rationale="Step was skipped (not executed)",
                ))
                degradation_curve.append(0.0)

            elif alignment.alignment_type == AlignmentType.EXTRA:
                # Extra step - score but mark as extra
                exec_idx = alignment.exec_indices[0]
                exec_step = execution.steps[exec_idx]
                # Evaluate against empty plan (will get low intent match)
                step_evaluations.append(StepEvaluation(
                    step_index=exec_step.index,
                    intent_match=0.0,  # No matching plan
                    execution_quality=0.5,  # May still be quality content
                    completeness=0.5,
                    constraint_fidelity=1.0,  # N/A for extra
                    step_purity=0.5,  # May contain other steps' content
                    step_match=0.25,
                    overall_score=0.25,
                    method=EvaluationMethod.HEURISTIC,
                    rationale="Extra step (not in plan)",
                    details={"is_extra": True},
                ))
                degradation_curve.append(0.25)

            elif alignment.alignment_type == AlignmentType.MERGE:
                # Merged steps
                plan_indices = alignment.plan_indices
                exec_idx = alignment.exec_indices[0]
                plan_steps_for_merge = [plan.steps[i] for i in plan_indices]
                exec_step = execution.steps[exec_idx]

                evaluation = self.evaluate_merged_steps(
                    plan_steps_for_merge,
                    exec_step,
                    full_plan,
                    constraints,
                )
                step_evaluations.append(evaluation)
                degradation_curve.append(evaluation.overall_score)

            elif alignment.alignment_type == AlignmentType.SPLIT:
                # Split step
                plan_idx = alignment.plan_indices[0]
                exec_indices = alignment.exec_indices
                plan_step = plan.steps[plan_idx]
                exec_steps_for_split = [execution.steps[i] for i in exec_indices]

                evaluation = self.evaluate_split_steps(
                    plan_step,
                    exec_steps_for_split,
                    full_plan,
                    constraints,
                )
                step_evaluations.append(evaluation)
                degradation_curve.append(evaluation.overall_score)

            else:
                # 1:1 or REORDER alignment
                plan_idx = alignment.plan_indices[0]
                exec_idx = alignment.exec_indices[0]
                plan_step = plan.steps[plan_idx]
                exec_step = execution.steps[exec_idx]

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
            average_intent_match = sum(e.intent_match for e in step_evaluations) / len(step_evaluations)
            average_execution_quality = sum(e.execution_quality for e in step_evaluations) / len(step_evaluations)
            average_completeness = sum(e.completeness for e in step_evaluations) / len(step_evaluations)
        else:
            overall_score = 0.0
            average_step_match = 0.0
            average_intent_match = 0.0
            average_execution_quality = 0.0
            average_completeness = 0.0

        method = step_evaluations[0].method if step_evaluations else EvaluationMethod.HEURISTIC

        return SemanticEvaluationResult(
            step_evaluations=step_evaluations,
            overall_score=overall_score,
            degradation_curve=degradation_curve,
            average_step_match=average_step_match,
            average_intent_match=average_intent_match,
            average_execution_quality=average_execution_quality,
            average_completeness=average_completeness,
            method=method,
            alignment_result=alignment_result,
            metadata={
                "merge_count": alignment_result.merge_count,
                "split_count": alignment_result.split_count,
                "skip_count": alignment_result.skip_count,
                "extra_count": alignment_result.extra_count,
                "reorder_detected": alignment_result.reorder_detected,
            },
        )


def calculate_drift(
    degradation_curve: list[float],
    drift_threshold: float = 0.1,
    rolling_window_size: int = 3,
) -> dict:
    """
    Calculate drift metrics from degradation curve.

    Args:
        degradation_curve: List of scores by step index
        drift_threshold: Magnitude threshold for detecting drift (default: 0.1)
        rolling_window_size: Window size for rolling average (default: 3)

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

    # Classify trend based on threshold
    if drift_magnitude > drift_threshold:
        trend = "declining"
    elif drift_magnitude < -drift_threshold:
        trend = "improving"
    else:
        trend = "stable"

    # Rolling average
    window_size = min(rolling_window_size, len(degradation_curve))
    rolling_avg = []
    for i in range(len(degradation_curve) - window_size + 1):
        window = degradation_curve[i:i + window_size]
        rolling_avg.append(sum(window) / len(window))

    return {
        "drift_detected": abs(drift_magnitude) > drift_threshold,
        "drift_magnitude": drift_magnitude,
        "trend": trend,
        "first_half_avg": first_avg,
        "second_half_avg": second_avg,
        "rolling_average": rolling_avg,
    }

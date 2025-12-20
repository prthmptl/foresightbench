"""
Rule-Based Validators - Fast, deterministic validation checks.

These run BEFORE semantic judging to catch structural issues:
- Step count match
- Correct numbering
- No skipped or extra steps
- Output present for every step
"""

from dataclasses import dataclass, field
from typing import Optional, TYPE_CHECKING
from enum import Enum

from core.capture import PlanCapture, ExecutionCapture, align_plan_and_execution

if TYPE_CHECKING:
    from .config import EvaluationConfig

class ValidationSeverity(str, Enum):
    """Severity level of validation issues."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class ValidationIssue:
    """A single validation issue."""
    code: str
    message: str
    severity: ValidationSeverity
    step_index: Optional[int] = None
    details: dict = field(default_factory=dict)


@dataclass
class ValidationResult:
    """Result of validation checks."""
    passed: bool
    score: float  # 0.0 to 1.0
    issues: list[ValidationIssue] = field(default_factory=list)
    checks_performed: list[str] = field(default_factory=list)
    penalties: dict[str, float] = field(default_factory=dict)

    @property
    def error_count(self) -> int:
        return sum(1 for i in self.issues if i.severity in [ValidationSeverity.ERROR, ValidationSeverity.CRITICAL])

    @property
    def warning_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == ValidationSeverity.WARNING)


class RuleValidator:
    """
    Performs rule-based validation on plans and executions.

    This validator runs fast, deterministic checks that don't require
    semantic understanding.
    """

    # Penalty weights for different issues (legacy - use EvaluationConfig instead)
    DEFAULT_PENALTIES = {
        "step_count_mismatch": 0.1,  # Per step difference
        "skipped_step": 0.15,        # Per skipped step
        "extra_step": 0.05,          # Per extra step
        "empty_step": 0.1,           # Per empty step
        "numbering_error": 0.05,     # Per numbering issue
        "parse_failure": 0.5,        # If parsing completely failed
    }

    def __init__(
        self,
        penalties: Optional[dict[str, float]] = None,
        config: Optional["EvaluationConfig"] = None,
    ):
        """
        Initialize validator.

        Args:
            penalties: Custom penalty weights (overrides defaults) - DEPRECATED, use config
            config: EvaluationConfig instance for centralized configuration
        """
        if config is not None:
            # Use config if provided
            self.penalties = config.penalties.to_dict()
            self.config = config
        else:
            # Legacy behavior: use DEFAULT_PENALTIES with optional overrides
            self.penalties = {**self.DEFAULT_PENALTIES}
            if penalties:
                self.penalties.update(penalties)
            self.config = None

    def validate_plan(
        self,
        plan: PlanCapture,
        min_steps: Optional[int] = None,
        max_steps: Optional[int] = None,
    ) -> ValidationResult:
        """
        Validate a parsed plan.

        Args:
            plan: The parsed plan to validate
            min_steps: Minimum expected steps (defaults to config or 1)
            max_steps: Maximum allowed steps (defaults to config or 20)

        Returns:
            ValidationResult with issues and score
        """
        # Use config bounds if available, otherwise use defaults
        if min_steps is None:
            min_steps = self.config.step_bounds.min_steps if self.config else 1
        if max_steps is None:
            max_steps = self.config.step_bounds.max_steps if self.config else 20
        issues = []
        penalties = {}
        checks = []

        # Ensure plan is parsed
        parse_result = plan.parse()

        # Check 1: Parse success
        checks.append("parse_success")
        if parse_result.status.value == "failed":
            issues.append(ValidationIssue(
                code="PLAN_PARSE_FAILED",
                message="Failed to parse any steps from plan",
                severity=ValidationSeverity.CRITICAL,
            ))
            penalties["parse_failure"] = self.penalties["parse_failure"]

        # Check 2: Step count bounds
        checks.append("step_count_bounds")
        step_count = plan.step_count
        if step_count < min_steps:
            issues.append(ValidationIssue(
                code="TOO_FEW_STEPS",
                message=f"Plan has {step_count} steps, minimum is {min_steps}",
                severity=ValidationSeverity.WARNING,
                details={"actual": step_count, "minimum": min_steps},
            ))
            penalties["step_count_low"] = self.penalties["step_count_mismatch"] * (min_steps - step_count)

        if step_count > max_steps:
            issues.append(ValidationIssue(
                code="TOO_MANY_STEPS",
                message=f"Plan has {step_count} steps, maximum is {max_steps}",
                severity=ValidationSeverity.WARNING,
                details={"actual": step_count, "maximum": max_steps},
            ))
            penalties["step_count_high"] = self.penalties["step_count_mismatch"] * (step_count - max_steps)

        # Check 3: Step numbering sequence
        checks.append("step_numbering")
        if plan.steps:
            indices = [s.index for s in plan.steps]
            expected = list(range(1, len(indices) + 1))

            if indices[0] != 1:
                issues.append(ValidationIssue(
                    code="NUMBERING_NOT_FROM_ONE",
                    message=f"Steps should start from 1, got {indices[0]}",
                    severity=ValidationSeverity.WARNING,
                    step_index=indices[0],
                ))
                penalties["numbering_start"] = self.penalties["numbering_error"]

            # Check for gaps
            for i, (exp, act) in enumerate(zip(expected, indices)):
                if exp != act:
                    issues.append(ValidationIssue(
                        code="NUMBERING_GAP",
                        message=f"Expected step {exp} at position {i+1}, got step {act}",
                        severity=ValidationSeverity.WARNING,
                        step_index=act,
                    ))
                    penalties[f"numbering_gap_{i}"] = self.penalties["numbering_error"]

        # Check 4: Empty steps
        checks.append("empty_steps")
        for step in plan.steps:
            if not step.text.strip():
                issues.append(ValidationIssue(
                    code="EMPTY_STEP",
                    message=f"Step {step.index} has no content",
                    severity=ValidationSeverity.ERROR,
                    step_index=step.index,
                ))
                penalties[f"empty_step_{step.index}"] = self.penalties["empty_step"]

        # Calculate score
        total_penalty = min(1.0, sum(penalties.values()))
        score = max(0.0, 1.0 - total_penalty)
        passed = score >= 0.5 and not any(
            i.severity == ValidationSeverity.CRITICAL for i in issues
        )

        return ValidationResult(
            passed=passed,
            score=score,
            issues=issues,
            checks_performed=checks,
            penalties=penalties,
        )

    def validate_execution(
        self,
        execution: ExecutionCapture,
        plan: Optional[PlanCapture] = None,
    ) -> ValidationResult:
        """
        Validate execution output, optionally against a plan.
        
        Args:
            execution: The parsed execution to validate
            plan: Optional plan to compare against
            
        Returns:
            ValidationResult with issues and score
        """
        issues = []
        penalties = {}
        checks = []

        # Ensure execution is parsed
        parse_result = execution.parse()

        # Check 1: Parse success
        checks.append("parse_success")
        if parse_result.status.value == "failed":
            issues.append(ValidationIssue(
                code="EXEC_PARSE_FAILED",
                message="Failed to parse any steps from execution",
                severity=ValidationSeverity.CRITICAL,
            ))
            penalties["parse_failure"] = self.penalties["parse_failure"]

        # Check 2: Empty steps
        checks.append("empty_steps")
        for step in execution.steps:
            if not step.content.strip():
                issues.append(ValidationIssue(
                    code="EMPTY_EXECUTION",
                    message=f"Step {step.index} execution is empty",
                    severity=ValidationSeverity.ERROR,
                    step_index=step.index,
                ))
                penalties[f"empty_step_{step.index}"] = self.penalties["empty_step"]

        # Checks against plan (if provided)
        if plan is not None:
            plan.parse()
            
            # Check 3: Step count match
            checks.append("step_count_match")
            plan_count = plan.step_count
            exec_count = execution.step_count

            if exec_count != plan_count:
                issues.append(ValidationIssue(
                    code="STEP_COUNT_MISMATCH",
                    message=f"Execution has {exec_count} steps, plan has {plan_count}",
                    severity=ValidationSeverity.WARNING,
                    details={"execution": exec_count, "plan": plan_count},
                ))
                diff = abs(exec_count - plan_count)
                penalties["step_count_mismatch"] = self.penalties["step_count_mismatch"] * diff

            # Check 4: Skipped steps
            checks.append("skipped_steps")
            plan_indices = {s.index for s in plan.steps}
            exec_indices = {s.index for s in execution.steps}
            
            skipped = plan_indices - exec_indices
            for idx in skipped:
                issues.append(ValidationIssue(
                    code="SKIPPED_STEP",
                    message=f"Step {idx} from plan was not executed",
                    severity=ValidationSeverity.ERROR,
                    step_index=idx,
                ))
                penalties[f"skipped_{idx}"] = self.penalties["skipped_step"]

            # Check 5: Extra steps
            checks.append("extra_steps")
            extra = exec_indices - plan_indices
            for idx in extra:
                issues.append(ValidationIssue(
                    code="EXTRA_STEP",
                    message=f"Step {idx} was executed but not in plan",
                    severity=ValidationSeverity.WARNING,
                    step_index=idx,
                ))
                penalties[f"extra_{idx}"] = self.penalties["extra_step"]

            # Check 6: Execution order
            checks.append("execution_order")
            exec_order = [s.index for s in execution.steps]
            expected_order = sorted(exec_order)
            if exec_order != expected_order:
                issues.append(ValidationIssue(
                    code="ORDER_VIOLATION",
                    message="Steps were not executed in sequential order",
                    severity=ValidationSeverity.WARNING,
                    details={"actual_order": exec_order, "expected_order": expected_order},
                ))
                penalties["order_violation"] = self.penalties["numbering_error"] * 2

        # Calculate score
        total_penalty = min(1.0, sum(penalties.values()))
        score = max(0.0, 1.0 - total_penalty)
        passed = score >= 0.5 and not any(
            i.severity == ValidationSeverity.CRITICAL for i in issues
        )

        return ValidationResult(
            passed=passed,
            score=score,
            issues=issues,
            checks_performed=checks,
            penalties=penalties,
        )

    def validate_alignment(
        self,
        plan: PlanCapture,
        execution: ExecutionCapture,
    ) -> ValidationResult:
        """
        Validate the alignment between plan and execution.
        
        This performs comprehensive checks on how well execution
        matches the plan structure.
        
        Args:
            plan: The parsed plan
            execution: The parsed execution
            
        Returns:
            ValidationResult with alignment issues
        """
        # Run individual validations
        plan_result = self.validate_plan(plan)
        exec_result = self.validate_execution(execution, plan)

        # Combine issues
        all_issues = plan_result.issues + exec_result.issues
        all_penalties = {**plan_result.penalties, **exec_result.penalties}
        all_checks = list(set(plan_result.checks_performed + exec_result.checks_performed))

        # Additional alignment checks
        aligned = align_plan_and_execution(plan, execution)
        
        # Check for completely unaligned pairs
        unaligned_count = sum(
            1 for p, e in aligned
            if p is None or e is None
        )

        if unaligned_count > 0:
            all_issues.append(ValidationIssue(
                code="UNALIGNED_STEPS",
                message=f"{unaligned_count} steps could not be aligned between plan and execution",
                severity=ValidationSeverity.WARNING,
                details={"unaligned_count": unaligned_count},
            ))

        # Calculate combined score
        total_penalty = min(1.0, sum(all_penalties.values()))
        score = max(0.0, 1.0 - total_penalty)
        passed = plan_result.passed and exec_result.passed

        return ValidationResult(
            passed=passed,
            score=score,
            issues=all_issues,
            checks_performed=all_checks,
            penalties=all_penalties,
        )


def quick_validate(plan_text: str, execution_text: str) -> dict:
    """
    Quick validation helper function.
    
    Args:
        plan_text: Raw plan text
        execution_text: Raw execution text
        
    Returns:
        Dictionary with validation summary
    """
    plan = PlanCapture(plan_text)
    execution = ExecutionCapture(execution_text, expected_steps=plan.step_count)
    
    validator = RuleValidator()
    result = validator.validate_alignment(plan, execution)

    return {
        "passed": result.passed,
        "score": result.score,
        "error_count": result.error_count,
        "warning_count": result.warning_count,
        "plan_steps": plan.step_count,
        "execution_steps": execution.step_count,
        "issues": [
            {"code": i.code, "message": i.message, "severity": i.severity.value}
            for i in result.issues
        ],
    }

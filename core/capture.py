"""
Plan & Execution Capture Layer - Parses and stores plans and execution outputs.

This layer:
- Stores raw plan and execution text
- Parses steps from plans
- Stores execution outputs per step
- Validates structure and format
"""

import re
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum


class ParseStatus(str, Enum):
    """Status of parsing operation."""
    SUCCESS = "success"
    PARTIAL = "partial"  # Some steps parsed, but with warnings
    FAILED = "failed"


@dataclass
class PlanStep:
    """A single step from a plan."""
    index: int
    text: str
    raw_text: str  # Original text including step label
    
    def __repr__(self) -> str:
        preview = self.text[:50] + "..." if len(self.text) > 50 else self.text
        return f"PlanStep({self.index}: {preview})"


@dataclass
class ExecutionStep:
    """A single step from execution output."""
    index: int
    content: str
    raw_text: str
    planned_step_index: Optional[int] = None  # Which plan step this corresponds to
    
    def __repr__(self) -> str:
        preview = self.content[:50] + "..." if len(self.content) > 50 else self.content
        return f"ExecutionStep({self.index}: {preview})"


@dataclass
class ParseResult:
    """Result of parsing operation."""
    status: ParseStatus
    steps: list
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


class PlanCapture:
    """
    Captures and parses planning phase output.
    
    Parses numbered steps from plan text and validates structure.
    """

    # Patterns for step detection
    STEP_PATTERNS = [
        r"^Step\s+(\d+)[:\.\)]\s*(.+)",  # Step 1: ..., Step 1. ..., Step 1) ...
        r"^(\d+)[:\.\)]\s*(.+)",          # 1: ..., 1. ..., 1) ...
        r"^\*\*Step\s+(\d+)[:\.\)]?\*\*\s*(.+)",  # **Step 1:** ...
        r"^-\s*Step\s+(\d+)[:\.\)]\s*(.+)",  # - Step 1: ...
    ]

    def __init__(self, raw_text: str):
        """
        Initialize with raw plan text.
        
        Args:
            raw_text: The raw output from the planning phase
        """
        self.raw_text = raw_text
        self._steps: list[PlanStep] = []
        self._parse_result: Optional[ParseResult] = None

    def parse(self) -> ParseResult:
        """
        Parse the raw text into steps.
        
        Returns:
            ParseResult with extracted steps and any warnings/errors
        """
        if self._parse_result is not None:
            return self._parse_result

        steps = []
        warnings = []
        errors = []
        
        lines = self.raw_text.split("\n")
        current_step_index = None
        current_step_text = []
        current_raw_lines = []

        for line in lines:
            line_stripped = line.strip()
            if not line_stripped:
                continue

            # Try to match step patterns
            step_match = None
            for pattern in self.STEP_PATTERNS:
                match = re.match(pattern, line_stripped, re.IGNORECASE)
                if match:
                    step_match = match
                    break

            if step_match:
                # Save previous step if exists
                if current_step_index is not None:
                    steps.append(PlanStep(
                        index=current_step_index,
                        text=" ".join(current_step_text).strip(),
                        raw_text="\n".join(current_raw_lines),
                    ))

                # Start new step
                current_step_index = int(step_match.group(1))
                current_step_text = [step_match.group(2).strip()]
                current_raw_lines = [line]

            elif current_step_index is not None:
                # Continue current step (multi-line step content)
                current_step_text.append(line_stripped)
                current_raw_lines.append(line)

        # Don't forget last step
        if current_step_index is not None:
            steps.append(PlanStep(
                index=current_step_index,
                text=" ".join(current_step_text).strip(),
                raw_text="\n".join(current_raw_lines),
            ))

        # Validate step sequence
        if steps:
            expected_indices = list(range(1, len(steps) + 1))
            actual_indices = [s.index for s in steps]
            
            if actual_indices != expected_indices:
                if actual_indices[0] != 1:
                    warnings.append(f"Steps don't start at 1 (starts at {actual_indices[0]})")
                
                # Check for gaps
                for i, (expected, actual) in enumerate(zip(expected_indices, actual_indices)):
                    if expected != actual:
                        warnings.append(f"Step sequence issue at position {i+1}: expected {expected}, got {actual}")
                        break

        # Determine status
        if not steps:
            status = ParseStatus.FAILED
            errors.append("No steps found in plan")
        elif warnings:
            status = ParseStatus.PARTIAL
        else:
            status = ParseStatus.SUCCESS

        self._steps = steps
        self._parse_result = ParseResult(
            status=status,
            steps=steps,
            warnings=warnings,
            errors=errors,
        )

        return self._parse_result

    @property
    def steps(self) -> list[PlanStep]:
        """Get parsed steps (parses if not already done)."""
        if self._parse_result is None:
            self.parse()
        return self._steps

    @property
    def step_count(self) -> int:
        """Get number of steps."""
        return len(self.steps)

    def get_step(self, index: int) -> Optional[PlanStep]:
        """Get a specific step by index (1-based)."""
        for step in self.steps:
            if step.index == index:
                return step
        return None

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "raw_text": self.raw_text,
            "steps": [
                {"index": s.index, "text": s.text, "raw_text": s.raw_text}
                for s in self.steps
            ],
            "parse_status": self._parse_result.status.value if self._parse_result else None,
            "warnings": self._parse_result.warnings if self._parse_result else [],
            "errors": self._parse_result.errors if self._parse_result else [],
        }


class ExecutionCapture:
    """
    Captures and parses execution phase output.
    
    Parses step outputs and maps them to plan steps.
    """

    # Patterns for execution step detection
    EXEC_PATTERNS = [
        r"^Step\s+(\d+)[:\.\)]?\s*(.*)$",  # Step 1: ...
        r"^\*\*Step\s+(\d+)[:\.\)]?\*\*\s*(.*)$",  # **Step 1:** ...
        r"^#{1,3}\s*Step\s+(\d+)[:\.\)]?\s*(.*)$",  # ### Step 1: ...
    ]

    def __init__(self, raw_text: str, expected_steps: int = 0):
        """
        Initialize with raw execution text.
        
        Args:
            raw_text: The raw output from the execution phase
            expected_steps: Expected number of steps (from plan)
        """
        self.raw_text = raw_text
        self.expected_steps = expected_steps
        self._steps: list[ExecutionStep] = []
        self._parse_result: Optional[ParseResult] = None

    def parse(self) -> ParseResult:
        """
        Parse the raw text into execution steps.
        
        Returns:
            ParseResult with extracted steps and any warnings/errors
        """
        if self._parse_result is not None:
            return self._parse_result

        steps = []
        warnings = []
        errors = []

        lines = self.raw_text.split("\n")
        current_step_index = None
        current_content = []
        current_raw_lines = []

        for line in lines:
            line_stripped = line.strip()

            # Try to match execution step patterns
            step_match = None
            for pattern in self.EXEC_PATTERNS:
                match = re.match(pattern, line_stripped, re.IGNORECASE)
                if match:
                    step_match = match
                    break

            if step_match:
                # Save previous step if exists
                if current_step_index is not None:
                    steps.append(ExecutionStep(
                        index=current_step_index,
                        content="\n".join(current_content).strip(),
                        raw_text="\n".join(current_raw_lines),
                        planned_step_index=current_step_index,
                    ))

                # Start new step
                current_step_index = int(step_match.group(1))
                first_line_content = step_match.group(2).strip()
                current_content = [first_line_content] if first_line_content else []
                current_raw_lines = [line]

            elif current_step_index is not None:
                # Continue current step
                current_content.append(line)
                current_raw_lines.append(line)

        # Don't forget last step
        if current_step_index is not None:
            steps.append(ExecutionStep(
                index=current_step_index,
                content="\n".join(current_content).strip(),
                raw_text="\n".join(current_raw_lines),
                planned_step_index=current_step_index,
            ))

        # Validate against expected steps
        if self.expected_steps > 0:
            actual_count = len(steps)
            if actual_count < self.expected_steps:
                warnings.append(f"Fewer steps executed ({actual_count}) than planned ({self.expected_steps})")
            elif actual_count > self.expected_steps:
                warnings.append(f"More steps executed ({actual_count}) than planned ({self.expected_steps})")

            # Check for skipped steps
            executed_indices = {s.index for s in steps}
            expected_indices = set(range(1, self.expected_steps + 1))
            skipped = expected_indices - executed_indices
            if skipped:
                warnings.append(f"Skipped steps: {sorted(skipped)}")

            # Check for extra steps
            extra = executed_indices - expected_indices
            if extra:
                warnings.append(f"Extra steps: {sorted(extra)}")

        # Check for empty steps
        for step in steps:
            if not step.content.strip():
                warnings.append(f"Step {step.index} has empty content")

        # Determine status
        if not steps:
            status = ParseStatus.FAILED
            errors.append("No execution steps found")
        elif errors:
            status = ParseStatus.FAILED
        elif warnings:
            status = ParseStatus.PARTIAL
        else:
            status = ParseStatus.SUCCESS

        self._steps = steps
        self._parse_result = ParseResult(
            status=status,
            steps=steps,
            warnings=warnings,
            errors=errors,
        )

        return self._parse_result

    @property
    def steps(self) -> list[ExecutionStep]:
        """Get parsed execution steps."""
        if self._parse_result is None:
            self.parse()
        return self._steps

    @property
    def step_count(self) -> int:
        """Get number of executed steps."""
        return len(self.steps)

    def get_step(self, index: int) -> Optional[ExecutionStep]:
        """Get a specific execution step by index (1-based)."""
        for step in self.steps:
            if step.index == index:
                return step
        return None

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "raw_text": self.raw_text,
            "expected_steps": self.expected_steps,
            "steps": [
                {
                    "index": s.index,
                    "content": s.content,
                    "raw_text": s.raw_text,
                    "planned_step_index": s.planned_step_index,
                }
                for s in self.steps
            ],
            "parse_status": self._parse_result.status.value if self._parse_result else None,
            "warnings": self._parse_result.warnings if self._parse_result else [],
            "errors": self._parse_result.errors if self._parse_result else [],
        }


def align_plan_and_execution(
    plan: PlanCapture,
    execution: ExecutionCapture,
) -> list[tuple[Optional[PlanStep], Optional[ExecutionStep]]]:
    """
    Align plan steps with execution steps.
    
    Returns a list of (plan_step, execution_step) pairs.
    Unmatched steps will have None for the missing side.
    
    Args:
        plan: Parsed plan
        execution: Parsed execution
        
    Returns:
        List of aligned step pairs
    """
    plan_steps = {s.index: s for s in plan.steps}
    exec_steps = {s.index: s for s in execution.steps}

    all_indices = sorted(set(plan_steps.keys()) | set(exec_steps.keys()))

    aligned = []
    for idx in all_indices:
        aligned.append((
            plan_steps.get(idx),
            exec_steps.get(idx),
        ))

    return aligned

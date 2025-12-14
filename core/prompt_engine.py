"""
Prompt Engine - Constructs canonical prompts for planning and execution phases.

Key principles:
- Planning and execution are SEPARATE LLM calls
- Plans must be numbered and explicit
- Execution must follow the exact plan order
"""

from dataclasses import dataclass
from typing import Optional

from .task_store import Task


@dataclass
class PromptConfig:
    """Configuration for prompt generation."""
    # Planning phase
    planning_system_prompt: str = ""
    require_numbered_steps: bool = True
    step_prefix: str = "Step"
    max_steps_instruction: bool = True
    
    # Execution phase
    execution_system_prompt: str = ""
    enforce_step_labels: bool = True
    allow_step_modification: bool = False


# Default planning prompt template
DEFAULT_PLANNING_TEMPLATE = """You are tasked with creating a detailed, step-by-step plan for the following task.

TASK:
{task_description}

INSTRUCTIONS:
1. Create a numbered plan with {min_steps} to {max_steps} steps
2. Each step must be clearly actionable and specific
3. Steps should be sequential - each step builds on the previous
4. Use the format: "Step N: [Description of what this step accomplishes]"
5. Do not execute the plan yet - only create the plan
{constraints_section}
OUTPUT FORMAT:
Provide ONLY the numbered steps. Do not include any preamble, explanation, or execution.
Begin with "Step 1:" and continue sequentially.

YOUR PLAN:"""

# Default execution prompt template
DEFAULT_EXECUTION_TEMPLATE = """You previously created the following plan for a task. Now you must execute this plan exactly as written.

ORIGINAL TASK:
{task_description}

YOUR PLAN:
{plan}

INSTRUCTIONS:
1. Execute each step in the EXACT order specified
2. Label each step's output with "Step N:" matching your plan
3. Do not skip any steps
4. Do not add extra steps beyond your plan
5. Do not modify the order of steps
6. Complete each step fully before moving to the next

Execute your plan now, labeling each step's output:"""


class PromptEngine:
    """
    Generates prompts for the planning and execution phases.
    
    This engine ensures:
    - Consistent prompt formatting across runs
    - Clear separation between planning and execution
    - Proper constraint injection
    - Step numbering enforcement
    """

    def __init__(
        self,
        config: Optional[PromptConfig] = None,
        planning_template: Optional[str] = None,
        execution_template: Optional[str] = None,
    ):
        """
        Initialize the prompt engine.
        
        Args:
            config: Configuration for prompt generation
            planning_template: Custom planning prompt template
            execution_template: Custom execution prompt template
        """
        self.config = config or PromptConfig()
        self.planning_template = planning_template or DEFAULT_PLANNING_TEMPLATE
        self.execution_template = execution_template or DEFAULT_EXECUTION_TEMPLATE

    def generate_planning_prompt(self, task: Task) -> str:
        """
        Generate the Phase 1 planning prompt.
        
        This prompt asks the model to create a numbered, step-by-step plan
        without executing it.
        
        Args:
            task: The task to generate a plan for
            
        Returns:
            The formatted planning prompt
        """
        # Build constraints section if task has constraints
        constraints_section = ""
        if task.constraints:
            constraints_list = "\n".join(f"   - {c}" for c in task.constraints)
            constraints_section = f"\nCONSTRAINTS:\n{constraints_list}\n"

        prompt = self.planning_template.format(
            task_description=task.task_description,
            min_steps=task.min_steps,
            max_steps=task.max_steps,
            constraints_section=constraints_section,
        )

        return prompt

    def generate_execution_prompt(self, task: Task, plan: str) -> str:
        """
        Generate the Phase 2 execution prompt.
        
        This prompt provides the model with its own plan and asks it to
        execute each step in order.
        
        Args:
            task: The original task
            plan: The model's plan from Phase 1
            
        Returns:
            The formatted execution prompt
        """
        prompt = self.execution_template.format(
            task_description=task.task_description,
            plan=plan.strip(),
        )

        return prompt

    def get_system_prompt(self, phase: str) -> str:
        """
        Get the system prompt for a given phase.
        
        Args:
            phase: Either "planning" or "execution"
            
        Returns:
            The system prompt string
        """
        if phase == "planning":
            return self.config.planning_system_prompt
        elif phase == "execution":
            return self.config.execution_system_prompt
        else:
            raise ValueError(f"Unknown phase: {phase}")


class PromptVariants:
    """
    Alternative prompt styles for robustness testing.
    """

    @staticmethod
    def concise_planning_template() -> str:
        """A more concise planning template."""
        return """Create a {min_steps}-{max_steps} step plan for: {task_description}
{constraints_section}
Format: Step N: [action]
Plan only, do not execute:"""

    @staticmethod
    def detailed_planning_template() -> str:
        """A more detailed planning template with examples."""
        return """You are a meticulous planner. Your task is to create a comprehensive, step-by-step plan.

TASK TO PLAN:
{task_description}

PLANNING GUIDELINES:
- Create between {min_steps} and {max_steps} numbered steps
- Each step should be atomic (one clear action)
- Steps must be logically ordered
- Use imperative language (e.g., "Identify...", "Analyze...", "Create...")
- Be specific enough that the steps can be followed exactly
{constraints_section}
EXAMPLE FORMAT:
Step 1: Define the scope and objectives
Step 2: Gather relevant information
Step 3: Analyze the key components
...

YOUR PLAN (numbered steps only):"""

    @staticmethod
    def strict_execution_template() -> str:
        """A stricter execution template."""
        return """CRITICAL: Execute your plan EXACTLY as written. Any deviation is a failure.

Task: {task_description}

Your Plan:
{plan}

Rules:
- Output "Step N:" before each step's content
- Complete steps in order: 1, 2, 3, ...
- No skipping, no additions, no reordering

Begin execution:"""

    @staticmethod
    def flexible_execution_template() -> str:
        """A more flexible execution template for comparison."""
        return """Execute the following plan for the given task.

Task: {task_description}

Plan:
{plan}

Execute each step in order, labeling your output for each step:"""

"""
ForesightBench - A benchmark for evaluating LLM plan-execution faithfulness.

This benchmark measures how well language models can:
1. Create structured, multi-step plans
2. Execute their own plans faithfully
3. Maintain consistency across long-horizon tasks
"""

__version__ = "0.1.0"
__author__ = "ForesightBench Team"

from .core.task_store import TaskStore, Task
from .core.prompt_engine import PromptEngine
from .core.llm_interface import LLMClient, OpenAIClient, AnthropicClient
from .core.capture import PlanCapture, ExecutionCapture
from .evaluation.rule_validators import RuleValidator
from .evaluation.semantic_evaluator import SemanticEvaluator
from .evaluation.metrics import MetricsComputer
from .storage.experiment_tracker import ExperimentTracker
from .runner import ForesightBenchRunner

__all__ = [
    "TaskStore",
    "Task",
    "PromptEngine",
    "LLMClient",
    "OpenAIClient",
    "AnthropicClient",
    "PlanCapture",
    "ExecutionCapture",
    "RuleValidator",
    "SemanticEvaluator",
    "MetricsComputer",
    "ExperimentTracker",
    "ForesightBenchRunner",
]

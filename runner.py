"""
ForesightBench Runner - End-to-end benchmark execution.

Execution flow:
1. Load task
2. Generate plan (Phase 1)
3. Parse plan
4. Generate execution (Phase 2)
5. Parse execution
6. Run rule checks
7. Run semantic evaluation
8. Aggregate metrics
9. Store results
"""

import uuid
import logging
from dataclasses import dataclass, field
from typing import Optional, Callable
from pathlib import Path

from core.task_store import Task, TaskStore, create_default_tasks
from core.prompt_engine import PromptEngine, PromptConfig
from core.llm_interface import LLMClient, GenerationConfig, GenerationResult
from core.capture import PlanCapture, ExecutionCapture
from evaluation.rule_validators import RuleValidator, ValidationResult
from evaluation.semantic_evaluator import SemanticEvaluator, SemanticEvaluationResult
from evaluation.metrics import MetricsComputer, TaskMetrics, GlobalMetrics, format_metrics_report
from storage.experiment_tracker import ExperimentTracker

logger = logging.getLogger(__name__)


@dataclass
class RunConfig:
    """Configuration for a benchmark run."""
    temperature: float = 0.0
    max_tokens: int = 4096
    num_runs: int = 1  # Multiple runs for variance analysis
    use_semantic_evaluation: bool = True
    semantic_judge_client: Optional[LLMClient] = None  # Separate client for judging
    save_results: bool = True
    verbose: bool = True


@dataclass
class SingleRunResult:
    """Result from a single task run."""
    task: Task
    run_id: str
    
    # Raw outputs
    plan_text: str
    execution_text: str
    
    # Generation results
    plan_generation: GenerationResult
    execution_generation: GenerationResult
    
    # Parsed data
    plan_capture: PlanCapture
    execution_capture: ExecutionCapture
    
    # Evaluation results
    rule_validation: ValidationResult
    semantic_evaluation: Optional[SemanticEvaluationResult]
    
    # Final metrics
    task_metrics: TaskMetrics


@dataclass
class BenchmarkResult:
    """Result from running the full benchmark."""
    model: str
    run_results: list[SingleRunResult]
    task_metrics: list[TaskMetrics]
    global_metrics: GlobalMetrics
    experiment_id: str


class ForesightBenchRunner:
    """
    Main runner for ForesightBench.
    
    Orchestrates the complete benchmark execution pipeline.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        task_store: Optional[TaskStore] = None,
        prompt_engine: Optional[PromptEngine] = None,
        experiment_tracker: Optional[ExperimentTracker] = None,
        config: Optional[RunConfig] = None,
    ):
        """
        Initialize the benchmark runner.
        
        Args:
            llm_client: LLM client for generation
            task_store: Task store (uses default tasks if None)
            prompt_engine: Prompt engine (uses default if None)
            experiment_tracker: Tracker for results (creates new if None)
            config: Run configuration
        """
        self.llm_client = llm_client
        self.task_store = task_store or create_default_tasks()
        self.prompt_engine = prompt_engine or PromptEngine()
        self.config = config or RunConfig()
        
        # Initialize tracker
        self.tracker = experiment_tracker or ExperimentTracker()
        
        # Initialize evaluators
        self.rule_validator = RuleValidator()
        self.semantic_evaluator = SemanticEvaluator(
            llm_client=self.config.semantic_judge_client,
        ) if self.config.use_semantic_evaluation else None
        self.metrics_computer = MetricsComputer()
        
        # Progress callback
        self.progress_callback: Optional[Callable[[str, int, int], None]] = None

    def _log(self, message: str) -> None:
        """Log a message if verbose mode is on."""
        if self.config.verbose:
            logger.info(message)
            print(message)

    def run_single_task(self, task: Task) -> SingleRunResult:
        """
        Run the benchmark on a single task.
        
        Args:
            task: The task to run
            
        Returns:
            SingleRunResult with all outputs and metrics
        """
        run_id = str(uuid.uuid4())[:8]
        self._log(f"Running task {task.task_id[:8]}... (run: {run_id})")

        # Phase 1: Planning
        self._log("  Phase 1: Generating plan...")
        planning_prompt = self.prompt_engine.generate_planning_prompt(task)
        plan_config = GenerationConfig(
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )
        plan_result = self.llm_client.generate(planning_prompt, config=plan_config)
        plan_text = plan_result.text

        # Parse plan
        plan_capture = PlanCapture(plan_text)
        plan_capture.parse()
        self._log(f"  Parsed {plan_capture.step_count} steps from plan")

        # Phase 2: Execution
        self._log("  Phase 2: Generating execution...")
        execution_prompt = self.prompt_engine.generate_execution_prompt(task, plan_text)
        exec_config = GenerationConfig(
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens * 2,  # More tokens for execution
        )
        exec_result = self.llm_client.generate(execution_prompt, config=exec_config)
        execution_text = exec_result.text

        # Parse execution
        execution_capture = ExecutionCapture(execution_text, expected_steps=plan_capture.step_count)
        execution_capture.parse()
        self._log(f"  Parsed {execution_capture.step_count} steps from execution")

        # Rule validation
        self._log("  Running rule validation...")
        rule_result = self.rule_validator.validate_alignment(plan_capture, execution_capture)
        self._log(f"  Rule validation: {'PASSED' if rule_result.passed else 'FAILED'} (score: {rule_result.score:.3f})")

        # Semantic evaluation
        semantic_result = None
        if self.semantic_evaluator and self.config.use_semantic_evaluation:
            self._log("  Running semantic evaluation...")
            semantic_result = self.semantic_evaluator.evaluate_all(
                plan_capture,
                execution_capture,
                constraints=task.constraints,
            )
            self._log(f"  Semantic score: {semantic_result.overall_score:.3f}")
        else:
            # Create a minimal semantic result for metrics computation
            from evaluation.semantic_evaluator import SemanticEvaluationResult, StepEvaluation, EvaluationMethod
            step_evals = [
                StepEvaluation(
                    step_index=i + 1,
                    step_match=0.5,
                    constraint_fidelity=1.0,
                    step_purity=1.0,
                    completeness=0.5,
                    overall_score=0.5,
                    method=EvaluationMethod.HEURISTIC,
                )
                for i in range(plan_capture.step_count)
            ]
            semantic_result = SemanticEvaluationResult(
                step_evaluations=step_evals,
                overall_score=0.5,
                degradation_curve=[0.5] * plan_capture.step_count,
                average_step_match=0.5,
                average_completeness=0.5,
                method=EvaluationMethod.HEURISTIC,
            )

        # Compute metrics
        total_latency = plan_result.latency_ms + exec_result.latency_ms
        total_tokens = plan_result.total_tokens + exec_result.total_tokens

        task_metrics = self.metrics_computer.compute_task_metrics(
            task_id=task.task_id,
            model=self.llm_client.model,
            run_id=run_id,
            rule_result=rule_result,
            semantic_result=semantic_result,
            plan_step_count=plan_capture.step_count,
            execution_step_count=execution_capture.step_count,
            latency_ms=total_latency,
            token_count=total_tokens,
        )

        self._log(f"  Foresight Score: {task_metrics.foresight_score:.3f}")

        # Store results
        if self.config.save_results:
            self.tracker.log_run(
                task_id=task.task_id,
                model=self.llm_client.model,
                plan_text=plan_text,
                execution_text=execution_text,
                task_metrics=task_metrics,
                plan_steps=[s.__dict__ for s in plan_capture.steps],
                execution_steps=[s.__dict__ for s in execution_capture.steps],
                temperature=self.config.temperature,
            )

        return SingleRunResult(
            task=task,
            run_id=run_id,
            plan_text=plan_text,
            execution_text=execution_text,
            plan_generation=plan_result,
            execution_generation=exec_result,
            plan_capture=plan_capture,
            execution_capture=execution_capture,
            rule_validation=rule_result,
            semantic_evaluation=semantic_result,
            task_metrics=task_metrics,
        )

    def run_benchmark(
        self,
        tasks: Optional[list[Task]] = None,
        task_ids: Optional[list[str]] = None,
        categories: Optional[list[str]] = None,
        max_tasks: Optional[int] = None,
    ) -> BenchmarkResult:
        """
        Run the full benchmark.
        
        Args:
            tasks: Specific tasks to run (overrides other filters)
            task_ids: Specific task IDs to run
            categories: Filter by categories
            max_tasks: Maximum number of tasks to run
            
        Returns:
            BenchmarkResult with all metrics
        """
        # Determine tasks to run
        if tasks is None:
            tasks = list(self.task_store)
            
            if task_ids:
                tasks = [t for t in tasks if t.task_id in task_ids]
            
            if categories:
                tasks = [t for t in tasks if t.category.value in categories]
            
            if max_tasks:
                tasks = tasks[:max_tasks]

        self._log(f"\n{'='*60}")
        self._log(f"ForesightBench - Running {len(tasks)} tasks")
        self._log(f"Model: {self.llm_client.model}")
        self._log(f"{'='*60}\n")

        run_results = []
        all_task_metrics = []

        for i, task in enumerate(tasks):
            if self.progress_callback:
                self.progress_callback(task.task_id, i + 1, len(tasks))

            # Run multiple times if configured
            for run_num in range(self.config.num_runs):
                if self.config.num_runs > 1:
                    self._log(f"\n[Run {run_num + 1}/{self.config.num_runs}]")
                
                result = self.run_single_task(task)
                run_results.append(result)
                all_task_metrics.append(result.task_metrics)

        # Compute global metrics
        global_metrics = self.metrics_computer.compute_global_metrics(
            all_task_metrics,
            model=self.llm_client.model,
        )

        # Print summary
        self._log(format_metrics_report(global_metrics))

        return BenchmarkResult(
            model=self.llm_client.model,
            run_results=run_results,
            task_metrics=all_task_metrics,
            global_metrics=global_metrics,
            experiment_id=self.tracker.experiment_id,
        )

    def run_comparison(
        self,
        clients: list[LLMClient],
        tasks: Optional[list[Task]] = None,
        max_tasks: Optional[int] = None,
    ) -> list[BenchmarkResult]:
        """
        Run benchmark comparison across multiple models.
        
        Args:
            clients: List of LLM clients to compare
            tasks: Tasks to run (uses all if None)
            max_tasks: Maximum tasks per model
            
        Returns:
            List of BenchmarkResults, one per model
        """
        results = []
        
        for client in clients:
            self._log(f"\n{'#'*60}")
            self._log(f"# Running model: {client.model}")
            self._log(f"{'#'*60}")
            
            # Swap client
            original_client = self.llm_client
            self.llm_client = client
            
            result = self.run_benchmark(tasks=tasks, max_tasks=max_tasks)
            results.append(result)
            
            # Restore client
            self.llm_client = original_client

        # Print comparison
        self._log("\n" + "="*60)
        self._log("COMPARISON SUMMARY")
        self._log("="*60)
        
        for result in sorted(results, key=lambda r: r.global_metrics.mean_foresight_score, reverse=True):
            gm = result.global_metrics
            self._log(f"{gm.model}: {gm.mean_foresight_score:.3f} (±{gm.std_foresight_score:.3f})")

        return results


def run_quick_benchmark(
    model: str = "mock",
    provider: str = "mock",
    api_key: Optional[str] = None,
    max_tasks: int = 3,
    verbose: bool = True,
) -> BenchmarkResult:
    """
    Quick helper to run a benchmark with minimal setup.
    
    Args:
        model: Model name
        provider: Provider (openai, anthropic, mock)
        api_key: API key
        max_tasks: Number of tasks to run
        verbose: Print progress
        
    Returns:
        BenchmarkResult
    """
    from core.llm_interface import create_client
    
    client = create_client(provider, model, api_key)
    
    config = RunConfig(
        verbose=verbose,
        use_semantic_evaluation=False,  # Skip for quick runs
    )
    
    runner = ForesightBenchRunner(client, config=config)
    return runner.run_benchmark(max_tasks=max_tasks)

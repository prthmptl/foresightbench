# ForesightBench

**A benchmark for evaluating LLM plan-execution faithfulness.**

ForesightBench measures how well language models can create structured plans and then faithfully execute those plans step-by-step. It quantifies "foresight" — the ability to think ahead and follow through.

## Key Features

- **Two-Phase Evaluation**: Separate planning and execution phases reveal planning vs. execution capabilities
- **Step-Level Analysis**: Fine-grained metrics for each step, not just overall scores
- **Drift Detection**: Identifies when models lose track of their plans over time
- **Rule + Semantic Evaluation**: Fast structural checks plus deep LLM-as-judge analysis
- **Multi-Model Comparison**: Compare reliability across different models
- **Experiment Tracking**: Full reproducibility with JSONL traces and SQLite metrics

## Installation

```bash
# Basic installation
pip install -e .

# With OpenAI support
pip install -e ".[openai]"

# With Anthropic support
pip install -e ".[anthropic]"

# With all providers
pip install -e ".[all]"
```

## Quick Start

### Run a Demo

```bash
python -m foresight_bench demo
```

### Run on a Model

```python
from foresight_bench import ForesightBenchRunner, create_client, RunConfig

# Create an LLM client
client = create_client("openai", model="gpt-4", api_key="your-key")

# Configure the run
config = RunConfig(
    temperature=0.0,
    use_semantic_evaluation=True,
)

# Run the benchmark
runner = ForesightBenchRunner(client, config=config)
result = runner.run_benchmark(max_tasks=10)

# Print results
print(f"Foresight Score: {result.global_metrics.mean_foresight_score:.3f}")
```

### CLI Usage

```bash
# Run benchmark
python -m foresight_bench run --model gpt-4 --provider openai --max-tasks 10

# Compare models
python -m foresight_bench compare --models "gpt-4,claude-3-opus" --max-tasks 5

# Generate report
python -m foresight_bench report --experiment-name my_experiment
```

## How It Works

### Two-Phase Protocol

1. **Planning Phase**: The model receives a task and must create a numbered, step-by-step plan
2. **Execution Phase**: The model receives its own plan and must execute each step in order

### Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **Foresight Score** | Overall plan-execution faithfulness (0-1) |
| **Execution Reliability** | How consistently the model follows its plan |
| **Step Match** | Per-step alignment between plan and execution |
| **Completeness** | Whether each step was fully executed |
| **Skip Rate** | Percentage of planned steps that were skipped |
| **Drift Magnitude** | How much quality degrades over long plans |

### Evaluation Pipeline

```
Task → Planning Prompt → [LLM] → Plan
                                   ↓
                        Execution Prompt → [LLM] → Execution
                                                      ↓
                                            Rule Validation
                                                      ↓
                                          Semantic Evaluation
                                                      ↓
                                            Metrics Computation
```

## Architecture

```
foresight_bench/
├── core/
│   ├── task_store.py      # Task definitions and storage
│   ├── prompt_engine.py   # Planning/execution prompts
│   ├── llm_interface.py   # LLM client abstraction
│   └── capture.py         # Plan/execution parsing
├── evaluation/
│   ├── rule_validators.py     # Fast structural checks
│   ├── semantic_evaluator.py  # LLM-as-judge evaluation
│   └── metrics.py             # Metrics computation
├── storage/
│   └── experiment_tracker.py  # Experiment persistence
├── runner.py              # Main benchmark runner
└── __main__.py           # CLI entry point
```

## Task Categories

ForesightBench includes tasks across multiple categories:

- **Explanation**: Breaking down complex concepts
- **Analysis**: Structured analysis and reasoning
- **Reasoning**: Logic puzzles and problem-solving
- **Generation**: Creating structured content
- **Coding**: Software design and implementation
- **Research**: Information synthesis
- **Creative**: Narrative and creative writing

## Adding Custom Tasks

```python
from foresight_bench import Task, TaskStore, TaskCategory, TaskDifficulty

# Create a custom task
task = Task(
    task_description="Design a user authentication system. Plan the components and implementation.",
    category=TaskCategory.CODING,
    difficulty=TaskDifficulty.HARD,
    max_steps=8,
    min_steps=5,
    constraints=["Consider security best practices", "Include error handling"],
)

# Add to store
store = TaskStore()
store.add_task(task)

# Or load from file
store.load_from_file("my_tasks.jsonl")
```

## Experiment Tracking

All runs are automatically tracked:

```python
from foresight_bench import ExperimentTracker

tracker = ExperimentTracker(storage_dir="./experiments")

# Query results
runs = tracker.query_runs(model="gpt-4", min_score=0.8)

# Compare models
comparison = tracker.get_model_comparison()

# Export leaderboard
tracker.export_leaderboard("leaderboard.json")
```

## Configuration Options

```python
from foresight_bench import RunConfig, PromptConfig

# Run configuration
run_config = RunConfig(
    temperature=0.0,          # Generation temperature
    max_tokens=4096,          # Max tokens per generation
    num_runs=3,               # Multiple runs for variance
    use_semantic_evaluation=True,  # Enable LLM-as-judge
    save_results=True,        # Save to tracker
    verbose=True,             # Print progress
)

# Prompt configuration
prompt_config = PromptConfig(
    require_numbered_steps=True,
    step_prefix="Step",
    enforce_step_labels=True,
)
```

## Metrics Interpretation

### Foresight Score (0-1)
- **0.9+**: Excellent - Model reliably follows its own plans
- **0.7-0.9**: Good - Minor deviations but generally faithful
- **0.5-0.7**: Fair - Noticeable drift or skipped steps
- **<0.5**: Poor - Significant plan-execution mismatch

### Drift Detection
A negative drift magnitude indicates the model's execution quality degrades for later steps in the plan — a sign of "losing track" of the overall plan.

## Contributing

Contributions welcome! Areas of interest:

- Additional task categories
- New evaluation metrics
- Embedding-based similarity
- Visualization tools
- Integration with evaluation frameworks

## License

MIT License

## Citation

```bibtex
@software{foresightbench2024,
  title = {ForesightBench: Evaluating LLM Plan-Execution Faithfulness},
  year = {2024},
  url = {https://github.com/your-org/foresight-bench}
}
```

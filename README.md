# ForesightBench

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Status: Alpha](https://img.shields.io/badge/status-alpha-orange.svg)](https://github.com/prthmptl/foresight-bench)

**A benchmark for evaluating LLM plan-execution faithfulness.**

ForesightBench measures how well language models can create structured plans and then faithfully execute those plans step-by-step. It quantifies "foresight" — the ability to think ahead and follow through with consistent execution.

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [How It Works](#how-it-works)
- [Architecture](#architecture)
- [Configuration](#configuration)
- [Task Categories](#task-categories)
- [Metrics Reference](#metrics-reference)
- [API Reference](#api-reference)
- [Examples](#examples)
- [Experiment Tracking](#experiment-tracking)
- [Contributing](#contributing)
- [License](#license)
- [Citation](#citation)

---

## Overview

Large Language Models often struggle with long-horizon tasks that require maintaining consistency between planning and execution. ForesightBench provides a rigorous framework to:

1. **Measure Planning Quality**: How well can a model decompose a task into clear, actionable steps?
2. **Evaluate Execution Faithfulness**: Does the model actually follow its own plan?
3. **Detect Drift**: Does execution quality degrade as the model progresses through steps?
4. **Compare Models**: Which models are most reliable for multi-step tasks?

### The Problem

When asked to perform complex tasks, LLMs may:
- Skip steps they planned to do
- Add unplanned steps mid-execution
- Drift from their original intent
- Lose track of earlier context

ForesightBench quantifies these failure modes with precise metrics.

---

## Key Features

- **Two-Phase Evaluation**: Separate planning and execution phases reveal distinct capabilities
- **Step-Level Analysis**: Fine-grained metrics for each step, not just overall scores
- **Semantic Alignment**: Intelligent matching handles merged, split, or reordered steps
- **Drift Detection**: Identifies when models lose track of their plans over time
- **Rule + Semantic Evaluation**: Fast structural checks plus deep LLM-as-judge analysis
- **Decomposed Evaluation**: Breaks quality assessment into specific intent vs. quality dimensions
- **Multi-Model Comparison**: Compare reliability across different providers and models
- **Experiment Tracking**: Full reproducibility with JSONL traces and SQLite metrics
- **Zero Core Dependencies**: Core functionality works without any external packages
- **Flexible Configuration**: All parameters customizable via code or JSON files

---

## Installation

### Basic Installation

```bash
# Clone the repository
git clone https://github.com/prthmptl/foresight-bench.git
cd foresight-bench

# Basic installation (no model provider support)
pip install -e .
```

### With LLM Provider Support

```bash
# With OpenAI support (GPT-4, GPT-3.5-turbo, etc.)
pip install -e ".[openai]"

# With Anthropic support (Claude models)
pip install -e ".[anthropic]"

# With all providers
pip install -e ".[all]"
```

### Development Installation

```bash
# With development tools (pytest, black, mypy)
pip install -e ".[dev]"
```

### Requirements

- Python 3.10 or higher
- Optional: `openai>=1.0.0` for OpenAI models
- Optional: `anthropic>=0.18.0` for Anthropic models

### Environment Setup

Create a `.env` file in the project root with your API keys:

```bash
OPENAI_API_KEY=your-openai-api-key
ANTHROPIC_API_KEY=your-anthropic-api-key
```

---

## Quick Start

### Run a Demo

```bash
python -m foresight_bench demo
```

This runs the benchmark with a mock LLM client to demonstrate the framework.

### CLI Usage

```bash
# Run benchmark on a single model
python -m foresight_bench run --model gpt-4 --provider openai --max-tasks 10

# Compare multiple models
python -m foresight_bench compare --models "gpt-4,claude-3-opus" --max-tasks 5

# Generate report from saved experiment
python -m foresight_bench report --experiment-name my_experiment
```

### Python API

```python
from foresight_bench import ForesightBenchRunner, create_client, RunConfig

# 1. Create an LLM client
client = create_client("openai", model="gpt-4", api_key="your-key")

# 2. Configure the benchmark run
config = RunConfig(
    temperature=0.0,              # Deterministic generation
    max_tokens=4096,              # Max tokens per generation
    num_runs=1,                   # Runs per task
    use_semantic_evaluation=True, # Enable LLM-as-judge
    save_results=True,            # Persist results
    verbose=True,                 # Print progress
)

# 3. Run the benchmark
runner = ForesightBenchRunner(client, config=config)
result = runner.run_benchmark(max_tasks=10)

# 4. Access results
print(f"Foresight Score: {result.global_metrics.mean_foresight_score:.3f}")
print(f"Execution Reliability: {result.global_metrics.mean_execution_reliability:.3f}")
print(f"Skip Rate: {result.global_metrics.overall_skipped_step_rate:.1%}")
```

---

## How It Works

### Two-Phase Protocol

ForesightBench evaluates models using a two-phase protocol:

```
┌─────────────────────────────────────────────────────────────────────┐
│                        PHASE 1: PLANNING                            │
├─────────────────────────────────────────────────────────────────────┤
│  Input: Task description + constraints                              │
│  Output: Numbered step-by-step plan                                 │
│                                                                     │
│  Example Output:                                                    │
│    Step 1: Analyze the requirements                                 │
│    Step 2: Design the solution architecture                         │
│    Step 3: Implement the core functionality                         │
│    Step 4: Write tests and validate                                 │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       PHASE 2: EXECUTION                            │
├─────────────────────────────────────────────────────────────────────┤
│  Input: Original task + model's own plan                            │
│  Output: Step-by-step execution following the plan                  │
│                                                                     │
│  Example Output:                                                    │
│    Step 1: [Detailed analysis of requirements...]                   │
│    Step 2: [Architecture design with components...]                 │
│    Step 3: [Implementation code and explanations...]                │
│    Step 4: [Test cases and validation results...]                   │
└─────────────────────────────────────────────────────────────────────┘
```

### Evaluation Pipeline

```
Task → Planning Prompt → [LLM] → Plan
                                   ↓
                        Execution Prompt → [LLM] → Execution
                                                      ↓
                                        ┌─────────────┴─────────────┐
                                        ▼                           ▼
                              Rule Validation              Semantic Alignment
                              (Fast, Structural)           (Content Matching)
                                        │                           │
                                        └─────────────┬─────────────┘
                                                      ▼
                                          Semantic Evaluation
                                          (LLM-as-Judge)
                                                      ↓
                                            Metrics Computation
                                                      ↓
                                             Final Scores
```

### Evaluation Layers

1. **Rule Validation** (Fast, Deterministic)
   - Step count matching
   - Correct numbering
   - No skipped or empty steps
   - Structural compliance

2. **Semantic Alignment** (Content-Based)
   - Matches plan steps to execution steps by content similarity
   - Handles merged steps (multiple plan steps → one execution)
   - Handles split steps (one plan step → multiple executions)
   - Detects reordering and skips

3. **Semantic Evaluation** (LLM-as-Judge)
   - Intent accuracy: Did execution attempt the right task?
   - Execution quality: How well was it done?
   - Completeness: Was the step fully executed?
   - Constraint fidelity: Were constraints followed?
   - Step purity: No cross-step leakage?

---

## Architecture

```
foresight_bench/
├── core/                          # Core LLM interaction & task management
│   ├── task_store.py              # Task definitions and storage
│   ├── prompt_engine.py           # Planning/execution prompt generation
│   ├── llm_interface.py           # Abstract LLM client interface
│   └── capture.py                 # Plan and execution parsing
│
├── evaluation/                    # Evaluation and metrics computation
│   ├── rule_validators.py         # Fast structural validation checks
│   ├── semantic_evaluator.py      # LLM-as-judge deep evaluation
│   ├── alignment.py               # Semantic step alignment
│   ├── metrics.py                 # Metrics computation and aggregation
│   └── config.py                  # Centralized configuration
│
├── storage/                       # Experiment tracking and persistence
│   └── experiment_tracker.py      # JSONL traces + SQLite metrics
│
├── runner.py                      # Main benchmark orchestration engine
├── __main__.py                    # CLI entry point
├── setup.py                       # Package installation
│
├── examples/                      # Usage examples
│   ├── basic_usage.py             # Simple benchmark run
│   └── model_comparison.py        # Multi-model comparison
│
├── tests/                         # Unit tests
│   ├── test_core.py               # Core module tests
│   └── test_evaluation.py         # Evaluation tests
│
└── experiments/                   # Generated results (gitignored)
    ├── exp_*_traces.jsonl         # Raw execution traces
    └── exp_*_metrics.db           # SQLite metrics database
```

### Core Components

| Component | Purpose |
|-----------|---------|
| `LLMClient` | Abstract interface for LLM providers (OpenAI, Anthropic, Mock) |
| `Task` | Dataclass representing a benchmark task with metadata |
| `TaskStore` | In-memory task storage with filtering capabilities |
| `PromptEngine` | Generates canonical prompts for planning and execution |
| `PlanCapture` | Parses numbered steps from plan text |
| `ExecutionCapture` | Parses execution output and aligns with plan |
| `RuleValidator` | Fast structural validation checks |
| `SemanticEvaluator` | Deep evaluation using LLM-as-judge |
| `SemanticAligner` | Content-based step alignment |
| `MetricsComputer` | Aggregates scores into final metrics |
| `ExperimentTracker` | Persists results to JSONL and SQLite |
| `ForesightBenchRunner` | Orchestrates the entire pipeline |

---

## Configuration

ForesightBench provides extensive configuration options through the `EvaluationConfig` system.

### Run Configuration

```python
from foresight_bench import RunConfig

config = RunConfig(
    temperature=0.0,              # Generation temperature (0 = deterministic)
    max_tokens=4096,              # Max tokens per generation
    num_runs=3,                   # Multiple runs for variance estimation
    use_semantic_evaluation=True, # Enable LLM-as-judge (slower but more accurate)
    save_results=True,            # Save to experiment tracker
    verbose=True,                 # Print progress
)
```

### Evaluation Configuration

```python
from foresight_bench.evaluation import (
    EvaluationConfig,
    PenaltyConfig,
    SemanticWeights,
    ForesightScoreWeights,
)

# Customize evaluation parameters
eval_config = EvaluationConfig(
    # Adjust penalty weights
    penalties=PenaltyConfig(
        skipped_step=0.25,    # Higher penalty for skipping
        extra_step=0.05,      # Lower penalty for extra work
    ),

    # Adjust semantic dimension weights (must sum to 1.0)
    semantic_weights=SemanticWeights(
        step_match=0.4,
        completeness=0.3,
        constraint_fidelity=0.2,
        step_purity=0.1,
    ),

    # Adjust final score composition (must sum to 1.0)
    foresight_weights=ForesightScoreWeights(
        rule_validation=0.3,
        semantic_evaluation=0.7,
    ),
)
```

### Configuration from JSON

```python
import json
from foresight_bench.evaluation import load_config_from_dict

# Load from file
with open("my_config.json") as f:
    config = load_config_from_dict(json.load(f))
```

Example JSON configuration:

```json
{
  "penalties": {
    "skipped_step": 0.15,
    "extra_step": 0.05,
    "empty_step": 0.1,
    "parse_failure": 0.5
  },
  "semantic_weights": {
    "step_match": 0.4,
    "completeness": 0.3,
    "constraint_fidelity": 0.2,
    "step_purity": 0.1
  },
  "foresight_weights": {
    "rule_validation": 0.3,
    "semantic_evaluation": 0.7
  },
  "pass_thresholds": {
    "foresight_score": 0.7
  },
  "drift": {
    "threshold": 0.1,
    "rolling_window_size": 3
  }
}
```

---

## Task Categories

ForesightBench includes built-in tasks across multiple categories:

| Category | Description | Example Tasks |
|----------|-------------|---------------|
| **Explanation** | Breaking down complex concepts | Explain quantum computing to a beginner |
| **Analysis** | Structured analysis and reasoning | Analyze pros/cons of microservices |
| **Reasoning** | Logic puzzles and problem-solving | Solve constraint satisfaction problems |
| **Generation** | Creating structured content | Generate a project proposal |
| **Coding** | Software design and implementation | Design a REST API |
| **Research** | Information synthesis | Compare database technologies |
| **Creative** | Narrative and creative writing | Write a short story outline |

### Adding Custom Tasks

```python
from foresight_bench import Task, TaskStore, TaskCategory, TaskDifficulty

# Create a custom task
task = Task(
    task_description="""
    Design a user authentication system for a web application.
    Plan the components, security measures, and implementation approach.
    """,
    category=TaskCategory.CODING,
    difficulty=TaskDifficulty.HARD,
    max_steps=8,
    min_steps=5,
    constraints=[
        "Consider security best practices",
        "Include error handling",
        "Plan for scalability",
    ],
    expected_outputs=[
        "Authentication flow diagram",
        "Database schema",
        "API endpoint specifications",
    ],
)

# Add to store
store = TaskStore()
store.add_task(task)

# Or load multiple tasks from file
store.load_from_file("my_tasks.jsonl")
```

---

## Metrics Reference

### Core Metrics

| Metric | Range | Description |
|--------|-------|-------------|
| **Foresight Score** | 0-1 | Overall plan-execution faithfulness |
| **Execution Reliability** | 0-1 | How consistently the model follows its plan |
| **Planning Quality** | 0-1 | Quality of the generated plan |
| **Intent Accuracy** | 0-1 | Whether execution attempts the right tasks |
| **Quality Score** | 0-1 | How well tasks are executed |

### Structural Metrics

| Metric | Description |
|--------|-------------|
| **Plan Step Count** | Number of steps in the generated plan |
| **Execution Step Count** | Number of steps in the execution |
| **Skip Rate** | Percentage of planned steps that were skipped |
| **Extra Step Rate** | Percentage of extra (unplanned) steps |
| **Merge Count** | Number of merged step patterns detected |
| **Split Count** | Number of split step patterns detected |

### Step-Level Metrics

| Metric | Range | Description |
|--------|-------|-------------|
| **Step Match** | 0-1 | How well execution matches plan for this step |
| **Completeness** | 0-1 | Whether step was fully executed |
| **Constraint Fidelity** | 0-1 | Whether constraints were followed |
| **Step Purity** | 0-1 | No cross-step leakage |
| **Combined Score** | 0-1 | Weighted combination of above metrics |

### Drift Metrics

| Metric | Description |
|--------|-------------|
| **Drift Magnitude** | Slope of quality degradation over steps |
| **Drift Detected** | Boolean indicating significant drift |
| **Drift Trend** | "declining", "stable", or "improving" |

### Score Interpretation

| Foresight Score | Interpretation |
|-----------------|----------------|
| **0.9+** | Excellent - Model reliably follows its own plans |
| **0.7-0.9** | Good - Minor deviations but generally faithful |
| **0.5-0.7** | Fair - Noticeable drift or skipped steps |
| **<0.5** | Poor - Significant plan-execution mismatch |

---

## API Reference

### Creating LLM Clients

```python
from foresight_bench import create_client

# OpenAI
client = create_client("openai", model="gpt-4", api_key="...")

# Anthropic
client = create_client("anthropic", model="claude-3-opus-20240229", api_key="...")

# Mock (for testing)
client = create_client("mock", model="test")
```

### Running Benchmarks

```python
from foresight_bench import ForesightBenchRunner, RunConfig

runner = ForesightBenchRunner(client, config=RunConfig())

# Single task
result = runner.run_single_task(task)

# Multiple tasks
result = runner.run_benchmark(max_tasks=10)

# Compare models
results = runner.run_comparison(clients=[client1, client2], max_tasks=5)
```

### Accessing Results

```python
# Global metrics
gm = result.global_metrics
print(f"Mean Foresight: {gm.mean_foresight_score:.3f}")
print(f"Skip Rate: {gm.overall_skipped_step_rate:.1%}")
print(f"Drift Rate: {gm.drift_rate:.1%}")

# Individual task results
for run in result.run_results:
    tm = run.task_metrics
    print(f"Task {tm.task_id}: {tm.foresight_score:.3f}")

    # Step-level breakdown
    for step in tm.step_metrics:
        print(f"  Step {step.step_index}: {step.combined_score:.2f}")
```

---

## Examples

### Basic Usage

```python
from foresight_bench import (
    ForesightBenchRunner,
    RunConfig,
    create_client,
)

# Create client
client = create_client("openai", model="gpt-4")

# Configure
config = RunConfig(
    temperature=0.0,
    use_semantic_evaluation=True,
    num_runs=1,
    verbose=True,
)

# Run benchmark
runner = ForesightBenchRunner(client, config=config)
result = runner.run_benchmark(max_tasks=5)

# Analyze results
gm = result.global_metrics
print(f"Foresight Score: {gm.mean_foresight_score:.3f}")
print(f"Execution Reliability: {gm.mean_execution_reliability:.3f}")
print(f"Skipped Step Rate: {gm.overall_skipped_step_rate:.1%}")
```

### Model Comparison

```python
from foresight_bench import create_client, ForesightBenchRunner, RunConfig

# Create clients for different models
models = [
    ("openai", "gpt-4"),
    ("openai", "gpt-3.5-turbo"),
    ("anthropic", "claude-3-opus-20240229"),
]

clients = [create_client(provider, model=model) for provider, model in models]

# Run comparison
config = RunConfig(use_semantic_evaluation=True)
runner = ForesightBenchRunner(clients[0], config=config)
comparison = runner.run_comparison(clients=clients, max_tasks=10)

# Print results
for model, metrics in comparison.items():
    print(f"{model}: {metrics.mean_foresight_score:.3f}")
```

---

## Experiment Tracking

All benchmark runs are automatically tracked for reproducibility.

### Storage Format

- **JSONL Traces**: Detailed step-by-step execution traces
- **SQLite Database**: Structured metrics for efficient querying

### Querying Results

```python
from foresight_bench import ExperimentTracker

tracker = ExperimentTracker(storage_dir="./experiments")

# Query specific runs
runs = tracker.query_runs(
    model="gpt-4",
    min_score=0.8,
    category="CODING",
)

# Get model comparison statistics
comparison = tracker.get_model_comparison()

# Export leaderboard
tracker.export_leaderboard("leaderboard.json")
```

### Trace Viewer

```python
from foresight_bench import TraceViewer

viewer = TraceViewer(tracker)

# Browse detailed traces
viewer.show_run(run_id="abc123")
viewer.show_step(run_id="abc123", step_index=2)
```

---

## Contributing

Contributions are welcome! Please follow these guidelines to help maintain code quality and consistency.

### Getting Started

1. **Fork the repository**
   ```bash
   git clone https://github.com/prthmptl/foresight-bench.git
   cd foresight-bench
   ```

2. **Set up development environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   pip install -e ".[dev]"
   ```

3. **Create a branch for your changes**
   ```bash
   git checkout -b feature/your-feature-name
   ```

### Development Workflow

1. **Make your changes** following the code style guidelines below
2. **Add tests** for new functionality
3. **Run the test suite**
   ```bash
   pytest tests/
   ```
4. **Format your code**
   ```bash
   black foresight_bench/
   ```
5. **Check types**
   ```bash
   mypy foresight_bench/
   ```

### Code Style Guidelines

- **Python Version**: Target Python 3.10+
- **Formatting**: Use Black with default settings
- **Type Hints**: Use type annotations for all public functions
- **Docstrings**: Use Google-style docstrings for modules, classes, and functions
- **Imports**: Sort with `isort` (Black-compatible settings)

Example:

```python
from dataclasses import dataclass
from typing import Optional


@dataclass
class MyClass:
    """Short description of the class.

    Longer description with more details about the class
    and its purpose.

    Attributes:
        name: The name of the instance.
        value: Optional numeric value.
    """
    name: str
    value: Optional[float] = None

    def process(self, input_data: str) -> dict[str, any]:
        """Process the input data.

        Args:
            input_data: The data to process.

        Returns:
            A dictionary containing processed results.

        Raises:
            ValueError: If input_data is empty.
        """
        if not input_data:
            raise ValueError("Input data cannot be empty")
        return {"result": input_data.upper()}
```

### Testing Guidelines

- Write tests for all new functionality
- Place tests in the `tests/` directory
- Use descriptive test names: `test_<what>_<condition>_<expected>`
- Use fixtures for common setup

```python
import pytest
from foresight_bench import Task, TaskCategory, TaskDifficulty


@pytest.fixture
def sample_task():
    return Task(
        task_description="Test task",
        category=TaskCategory.EXPLANATION,
        difficulty=TaskDifficulty.EASY,
    )


def test_task_creation_with_valid_data_succeeds(sample_task):
    assert sample_task.category == TaskCategory.EXPLANATION
    assert sample_task.difficulty == TaskDifficulty.EASY
```

### Pull Request Process

1. **Update documentation** if you've changed functionality
2. **Add a clear description** of what your PR does and why
3. **Reference any related issues** using GitHub's linking syntax
4. **Ensure all checks pass** (tests, formatting, types)
5. **Request review** from maintainers

### Areas for Contribution

We especially welcome contributions in these areas:

| Area | Description |
|------|-------------|
| **New Task Categories** | Add tasks for new domains (legal, medical, scientific) |
| **Evaluation Metrics** | New ways to measure plan-execution faithfulness |
| **Embedding Integration** | Add embedding-based similarity for alignment |
| **Visualization** | Tools for visualizing benchmark results |
| **LLM Providers** | Support for additional LLM providers (Cohere, Mistral, etc.) |
| **Performance** | Optimizations for large-scale benchmarking |
| **Documentation** | Tutorials, examples, and API documentation |

### Reporting Issues

When reporting bugs, please include:

1. Python version and OS
2. ForesightBench version
3. Minimal reproduction code
4. Expected vs. actual behavior
5. Full error traceback (if applicable)

### Code of Conduct

- Be respectful and constructive in discussions
- Welcome newcomers and help them get started
- Focus on the technical merits of contributions
- Credit others for their work

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Citation

If you use ForesightBench in your research, please cite:

```bibtex
@software{foresightbench2025,
  title = {ForesightBench: Evaluating LLM Plan-Execution Faithfulness},
  author = {Pratham, Patel},
  year = {2025},
  url = {https://github.com/prthmptl/foresightbench},
  version = {0.1.0}
}
```

---

## Acknowledgments

ForesightBench was developed to address the growing need for rigorous evaluation of LLM planning and execution capabilities in real-world applications.

---

## Contact

- **Repository**: [github.com/prthmptl/foresightbench](https://github.com/prthmptl/foresightbench)
- **Issues**: [GitHub Issues](https://github.com/prthmptl/foresightbench/issues)

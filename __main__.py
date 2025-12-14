#!/usr/bin/env python3
"""
ForesightBench CLI - Command-line interface for running benchmarks.

Usage:
    python -m foresight_bench run --model gpt-4 --provider openai
    python -m foresight_bench compare --models gpt-4,claude-3-opus
    python -m foresight_bench report --experiment exp_20240101
"""

import argparse
import sys
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


def cmd_run(args):
    """Run benchmark on a single model."""
    from core.llm_interface import create_client
    from runner import ForesightBenchRunner, RunConfig
    from storage.experiment_tracker import ExperimentTracker

    # Create client
    client = create_client(
        provider=args.provider,
        model=args.model,
        api_key=args.api_key,
    )

    # Create config
    config = RunConfig(
        temperature=args.temperature,
        num_runs=args.num_runs,
        use_semantic_evaluation=args.semantic_eval,
        verbose=not args.quiet,
    )

    # Create tracker
    tracker = ExperimentTracker(
        storage_dir=args.output_dir,
        experiment_name=args.experiment_name,
    )

    # Run benchmark
    runner = ForesightBenchRunner(
        llm_client=client,
        experiment_tracker=tracker,
        config=config,
    )

    result = runner.run_benchmark(max_tasks=args.max_tasks)

    print(f"\nExperiment saved to: {tracker.storage_dir}")
    print(f"Experiment ID: {result.experiment_id}")

    return 0


def cmd_compare(args):
    """Run comparison across multiple models."""
    from core.llm_interface import create_client
    from runner import ForesightBenchRunner, RunConfig

    # Parse models
    model_specs = args.models.split(",")
    clients = []

    for spec in model_specs:
        if ":" in spec:
            provider, model = spec.split(":", 1)
        else:
            # Auto-detect provider
            if "gpt" in spec.lower():
                provider = "openai"
            elif "claude" in spec.lower():
                provider = "anthropic"
            else:
                provider = "mock"
            model = spec

        clients.append(create_client(provider, model))

    # Create config
    config = RunConfig(
        temperature=args.temperature,
        use_semantic_evaluation=args.semantic_eval,
        verbose=not args.quiet,
    )

    # Run comparison
    runner = ForesightBenchRunner(
        llm_client=clients[0],
        config=config,
    )

    results = runner.run_comparison(clients, max_tasks=args.max_tasks)

    # Print summary table
    print("\n" + "="*70)
    print("FINAL COMPARISON")
    print("="*70)
    print(f"{'Model':<30} {'Score':>10} {'Reliability':>12} {'Skip Rate':>10}")
    print("-"*70)

    for result in sorted(results, key=lambda r: r.global_metrics.mean_foresight_score, reverse=True):
        gm = result.global_metrics
        print(f"{gm.model:<30} {gm.mean_foresight_score:>10.3f} {gm.mean_execution_reliability:>12.3f} {gm.overall_skipped_step_rate:>10.1%}")

    return 0


def cmd_report(args):
    """Generate report from experiment data."""
    from storage.experiment_tracker import ExperimentTracker, TraceViewer
    from evaluation.metrics import format_metrics_report

    tracker = ExperimentTracker(
        storage_dir=args.output_dir,
        experiment_name=args.experiment_name,
    )

    # Get comparison
    comparison = tracker.get_model_comparison()

    if not comparison:
        print("No data found for this experiment.")
        return 1

    print("\n" + "="*60)
    print(f"Report: {args.experiment_name}")
    print("="*60)

    print(f"\n{'Model':<30} {'Runs':>6} {'Avg Score':>10}")
    print("-"*50)
    for row in comparison:
        print(f"{row['model']:<30} {row['run_count']:>6} {row['avg_foresight']:>10.3f}")

    # Export leaderboard if requested
    if args.export_leaderboard:
        leaderboard_path = Path(args.output_dir) / f"{args.experiment_name}_leaderboard.json"
        tracker.export_leaderboard(leaderboard_path)
        print(f"\nLeaderboard exported to: {leaderboard_path}")

    return 0


def cmd_demo(args):
    """Run a quick demo with mock model."""
    from runner import run_quick_benchmark
    from evaluation.metrics import format_metrics_report

    print("Running ForesightBench demo with mock model...\n")

    result = run_quick_benchmark(
        model="mock-demo",
        provider="mock",
        max_tasks=3,
        verbose=True,
    )

    print("\nDemo complete!")
    print(f"Tasks run: {result.global_metrics.total_tasks}")
    print(f"Average Foresight Score: {result.global_metrics.mean_foresight_score:.3f}")

    return 0


def main():
    parser = argparse.ArgumentParser(
        description="ForesightBench - Evaluate LLM plan-execution faithfulness",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Run command
    run_parser = subparsers.add_parser("run", help="Run benchmark on a model")
    run_parser.add_argument("--model", "-m", required=True, help="Model name")
    run_parser.add_argument("--provider", "-p", default="openai", 
                           choices=["openai", "anthropic", "mock"],
                           help="Model provider")
    run_parser.add_argument("--api-key", help="API key (or use env var)")
    run_parser.add_argument("--max-tasks", type=int, default=10, help="Max tasks to run")
    run_parser.add_argument("--num-runs", type=int, default=1, help="Runs per task")
    run_parser.add_argument("--temperature", type=float, default=0.0, help="Generation temperature")
    run_parser.add_argument("--semantic-eval", action="store_true", help="Enable semantic evaluation")
    run_parser.add_argument("--output-dir", default="./experiments", help="Output directory")
    run_parser.add_argument("--experiment-name", help="Experiment name")
    run_parser.add_argument("--quiet", "-q", action="store_true", help="Quiet mode")

    # Compare command
    compare_parser = subparsers.add_parser("compare", help="Compare multiple models")
    compare_parser.add_argument("--models", "-m", required=True, 
                               help="Models to compare (comma-separated, e.g., 'gpt-4,claude-3-opus')")
    compare_parser.add_argument("--max-tasks", type=int, default=10, help="Max tasks per model")
    compare_parser.add_argument("--temperature", type=float, default=0.0)
    compare_parser.add_argument("--semantic-eval", action="store_true")
    compare_parser.add_argument("--output-dir", default="./experiments")
    compare_parser.add_argument("--quiet", "-q", action="store_true")

    # Report command
    report_parser = subparsers.add_parser("report", help="Generate report from experiment")
    report_parser.add_argument("--experiment-name", "-e", required=True, help="Experiment name")
    report_parser.add_argument("--output-dir", default="./experiments")
    report_parser.add_argument("--export-leaderboard", action="store_true", help="Export leaderboard JSON")

    # Demo command
    demo_parser = subparsers.add_parser("demo", help="Run quick demo")

    args = parser.parse_args()

    if args.command == "run":
        return cmd_run(args)
    elif args.command == "compare":
        return cmd_compare(args)
    elif args.command == "report":
        return cmd_report(args)
    elif args.command == "demo":
        return cmd_demo(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())

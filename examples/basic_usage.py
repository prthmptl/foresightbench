#!/usr/bin/env python3
"""
Example: Basic ForesightBench Usage

This script demonstrates how to run ForesightBench on a model
and interpret the results.
"""

import sys
sys.path.insert(0, '..')

from foresight_bench import (
    ForesightBenchRunner,
    RunConfig,
    create_client,
    create_default_tasks,
    format_metrics_report,
)


def main():
    # 1. Create an LLM client
    # For testing, we use the mock client. Replace with real client for actual benchmarking:
    # client = create_client("openai", model="gpt-4", api_key="your-key")
    # client = create_client("anthropic", model="claude-3-opus-20240229")
    
    client = create_client("mock", model="mock-test")
    
    # 2. Configure the benchmark run
    config = RunConfig(
        temperature=0.0,           # Deterministic generation
        num_runs=1,                # Single run per task
        use_semantic_evaluation=False,  # Skip for demo (faster)
        save_results=True,         # Save to experiment tracker
        verbose=True,              # Print progress
    )
    
    # 3. Create the benchmark runner
    runner = ForesightBenchRunner(
        llm_client=client,
        config=config,
    )
    
    # 4. Run on a subset of tasks
    print("="*60)
    print("Running ForesightBench")
    print("="*60)
    
    result = runner.run_benchmark(max_tasks=3)
    
    # 5. Analyze results
    print("\n" + "="*60)
    print("RESULTS ANALYSIS")
    print("="*60)
    
    gm = result.global_metrics
    
    print(f"\nModel: {gm.model}")
    print(f"Tasks Evaluated: {gm.total_tasks}")
    print(f"\nCore Metrics:")
    print(f"  Foresight Score: {gm.mean_foresight_score:.3f}")
    print(f"  Execution Reliability: {gm.mean_execution_reliability:.3f}")
    print(f"  Planning Quality: {gm.mean_planning_quality:.3f}")
    
    print(f"\nStructural Metrics:")
    print(f"  Skipped Step Rate: {gm.overall_skipped_step_rate:.1%}")
    print(f"  Extra Step Rate: {gm.overall_extra_step_rate:.1%}")
    
    print(f"\nDrift Analysis:")
    print(f"  Tasks with Drift: {gm.tasks_with_drift}")
    print(f"  Drift Rate: {gm.drift_rate:.1%}")
    
    # 6. Look at individual task results
    print("\n" + "-"*60)
    print("Individual Task Results:")
    print("-"*60)
    
    for run_result in result.run_results[:3]:  # First 3 tasks
        tm = run_result.task_metrics
        print(f"\nTask: {tm.task_id[:8]}...")
        print(f"  Score: {tm.foresight_score:.3f}")
        print(f"  Steps: {tm.plan_step_count} planned, {tm.execution_step_count} executed")
        print(f"  Skipped: {tm.skipped_step_count}")
        
        # Show step-level breakdown
        if tm.step_metrics:
            print("  Step scores:", end=" ")
            for step in tm.step_metrics:
                print(f"S{step.step_index}:{step.combined_score:.2f}", end=" ")
            print()
    
    print("\n" + "="*60)
    print("Benchmark Complete!")
    print(f"Results saved to: {runner.tracker.storage_dir}")
    print("="*60)


if __name__ == "__main__":
    main()

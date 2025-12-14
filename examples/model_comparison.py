#!/usr/bin/env python3
"""
Example: Model Comparison

This script demonstrates how to compare multiple models using ForesightBench.
"""

import sys
sys.path.insert(0, '..')

from foresight_bench import (
    ForesightBenchRunner,
    RunConfig,
    MockLLMClient,
)


def main():
    # Create mock clients simulating different model behaviors
    
    # Good model - follows plans well
    good_model = MockLLMClient(
        model="good-model-v1",
        default_response="""Step 1: Understand the core requirements
Step 2: Break down into manageable components
Step 3: Design the solution architecture
Step 4: Implement the core functionality
Step 5: Test and validate the solution
Step 6: Document and finalize"""
    )
    
    # Average model - sometimes deviates
    average_model = MockLLMClient(
        model="average-model-v1",
        default_response="""Step 1: Look at the problem
Step 2: Think about solutions
Step 3: Try something
Step 4: See if it works"""
    )
    
    # Poor model - inconsistent
    poor_model = MockLLMClient(
        model="poor-model-v1",
        default_response="""Step 1: Start
Step 3: Skip ahead
Step 2: Go back"""
    )
    
    clients = [good_model, average_model, poor_model]
    
    # Configure run
    config = RunConfig(
        temperature=0.0,
        use_semantic_evaluation=False,
        verbose=True,
    )
    
    # Run comparison
    runner = ForesightBenchRunner(
        llm_client=clients[0],
        config=config,
    )
    
    print("="*70)
    print("ForesightBench Model Comparison")
    print("="*70)
    
    results = runner.run_comparison(clients, max_tasks=3)
    
    # Create comparison table
    print("\n" + "="*70)
    print("COMPARISON RESULTS")
    print("="*70)
    
    print(f"\n{'Model':<25} {'Foresight':>12} {'Reliability':>12} {'Skip Rate':>12}")
    print("-"*70)
    
    for result in sorted(results, key=lambda r: r.global_metrics.mean_foresight_score, reverse=True):
        gm = result.global_metrics
        print(f"{gm.model:<25} {gm.mean_foresight_score:>12.3f} {gm.mean_execution_reliability:>12.3f} {gm.overall_skipped_step_rate:>12.1%}")
    
    print("\n" + "="*70)
    print("INTERPRETATION")
    print("="*70)
    
    best = max(results, key=lambda r: r.global_metrics.mean_foresight_score)
    worst = min(results, key=lambda r: r.global_metrics.mean_foresight_score)
    
    print(f"\nBest Model: {best.model}")
    print(f"  - Highest foresight score indicates best plan-execution alignment")
    
    print(f"\nWorst Model: {worst.model}")
    print(f"  - Lower scores may indicate planning issues or execution drift")


if __name__ == "__main__":
    main()

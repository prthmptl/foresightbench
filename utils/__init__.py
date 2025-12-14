"""Utility functions for ForesightBench."""

import json
from pathlib import Path
from typing import Optional


def load_tasks_from_json(filepath: Path | str) -> list[dict]:
    """
    Load tasks from a JSON file.
    
    Args:
        filepath: Path to JSON file with task definitions
        
    Returns:
        List of task dictionaries
    """
    with open(filepath, "r") as f:
        return json.load(f)


def save_results_to_json(results: dict, filepath: Path | str) -> None:
    """
    Save results to a JSON file.
    
    Args:
        results: Results dictionary
        filepath: Output path
    """
    with open(filepath, "w") as f:
        json.dump(results, f, indent=2, default=str)


def format_duration(ms: float) -> str:
    """Format milliseconds as human-readable duration."""
    if ms < 1000:
        return f"{ms:.0f}ms"
    elif ms < 60000:
        return f"{ms/1000:.1f}s"
    else:
        return f"{ms/60000:.1f}m"


def truncate_text(text: str, max_length: int = 100) -> str:
    """Truncate text with ellipsis if too long."""
    if len(text) <= max_length:
        return text
    return text[:max_length - 3] + "..."


def calculate_token_cost(
    input_tokens: int,
    output_tokens: int,
    model: str,
) -> float:
    """
    Estimate cost based on token counts.
    
    Args:
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens
        model: Model name
        
    Returns:
        Estimated cost in USD
    """
    # Approximate pricing (update as needed)
    pricing = {
        "gpt-4": {"input": 0.03, "output": 0.06},
        "gpt-4-turbo": {"input": 0.01, "output": 0.03},
        "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
        "claude-3-opus": {"input": 0.015, "output": 0.075},
        "claude-3-sonnet": {"input": 0.003, "output": 0.015},
        "claude-3-haiku": {"input": 0.00025, "output": 0.00125},
    }
    
    # Find matching pricing
    for model_key, prices in pricing.items():
        if model_key in model.lower():
            return (
                (input_tokens / 1000) * prices["input"] +
                (output_tokens / 1000) * prices["output"]
            )
    
    return 0.0  # Unknown model

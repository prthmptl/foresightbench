"""
Task Store - Holds and manages benchmark tasks.

Each task is planable, multi-step, and produces verifiable step outputs.
"""

import json
import uuid
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Optional


class TaskCategory(str, Enum):
    """Categories for benchmark tasks."""
    EXPLANATION = "explanation"
    ANALYSIS = "analysis"
    GENERATION = "generation"
    REASONING = "reasoning"
    CODING = "coding"
    RESEARCH = "research"
    CREATIVE = "creative"


class TaskDifficulty(str, Enum):
    """Difficulty levels for tasks."""
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"
    EXPERT = "expert"


@dataclass
class Task:
    """
    A benchmark task that requires planning and execution.
    
    Attributes:
        task_id: Unique identifier for the task
        task_description: The full task description given to the model
        category: Type of task (explanation, analysis, etc.)
        difficulty: Difficulty level
        max_steps: Maximum number of steps expected in the plan
        min_steps: Minimum number of steps expected
        constraints: Optional list of constraints the plan must satisfy
        expected_outputs: Optional descriptions of what each step should produce
        metadata: Additional task-specific metadata
    """
    task_description: str
    category: TaskCategory
    difficulty: TaskDifficulty
    max_steps: int = 6
    min_steps: int = 3
    task_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    constraints: list[str] = field(default_factory=list)
    expected_outputs: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert task to dictionary for serialization."""
        data = asdict(self)
        data["category"] = self.category.value
        data["difficulty"] = self.difficulty.value
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "Task":
        """Create a Task from a dictionary."""
        data = data.copy()
        data["category"] = TaskCategory(data["category"])
        data["difficulty"] = TaskDifficulty(data["difficulty"])
        return cls(**data)

    def __repr__(self) -> str:
        return f"Task(id={self.task_id[:8]}..., category={self.category.value}, difficulty={self.difficulty.value})"


class TaskStore:
    """
    Storage and management for benchmark tasks.
    
    Supports loading from files, filtering, and iteration.
    """

    def __init__(self):
        self._tasks: dict[str, Task] = {}

    def add_task(self, task: Task) -> None:
        """Add a task to the store."""
        self._tasks[task.task_id] = task

    def get_task(self, task_id: str) -> Optional[Task]:
        """Get a task by ID."""
        return self._tasks.get(task_id)

    def remove_task(self, task_id: str) -> bool:
        """Remove a task by ID. Returns True if removed."""
        if task_id in self._tasks:
            del self._tasks[task_id]
            return True
        return False

    def list_tasks(
        self,
        category: Optional[TaskCategory] = None,
        difficulty: Optional[TaskDifficulty] = None,
        max_steps: Optional[int] = None,
    ) -> list[Task]:
        """
        List tasks with optional filtering.
        
        Args:
            category: Filter by category
            difficulty: Filter by difficulty
            max_steps: Filter by maximum step count
        
        Returns:
            List of matching tasks
        """
        tasks = list(self._tasks.values())

        if category is not None:
            tasks = [t for t in tasks if t.category == category]
        if difficulty is not None:
            tasks = [t for t in tasks if t.difficulty == difficulty]
        if max_steps is not None:
            tasks = [t for t in tasks if t.max_steps <= max_steps]

        return tasks

    def __len__(self) -> int:
        return len(self._tasks)

    def __iter__(self):
        return iter(self._tasks.values())

    def save_to_file(self, filepath: Path | str) -> None:
        """Save all tasks to a JSONL file."""
        filepath = Path(filepath)
        with open(filepath, "w") as f:
            for task in self._tasks.values():
                f.write(json.dumps(task.to_dict()) + "\n")

    def load_from_file(self, filepath: Path | str) -> int:
        """
        Load tasks from a JSONL file.
        
        Returns:
            Number of tasks loaded
        """
        filepath = Path(filepath)
        count = 0
        with open(filepath, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    task = Task.from_dict(json.loads(line))
                    self.add_task(task)
                    count += 1
        return count

    @classmethod
    def from_file(cls, filepath: Path | str) -> "TaskStore":
        """Create a TaskStore from a JSONL file."""
        store = cls()
        store.load_from_file(filepath)
        return store


def create_default_tasks() -> TaskStore:
    """Create a TaskStore with default benchmark tasks."""
    store = TaskStore()

    # Explanation tasks
    store.add_task(Task(
        task_description="Explain the concept of recursion in programming to someone who has never coded before. Break down the explanation into clear, sequential steps that build understanding progressively.",
        category=TaskCategory.EXPLANATION,
        difficulty=TaskDifficulty.MEDIUM,
        max_steps=6,
        min_steps=4,
        constraints=["Use analogies from everyday life", "Avoid jargon in initial steps"],
    ))

    store.add_task(Task(
        task_description="Explain how a neural network learns from data. Structure your explanation to cover the key concepts from basic to advanced.",
        category=TaskCategory.EXPLANATION,
        difficulty=TaskDifficulty.HARD,
        max_steps=8,
        min_steps=5,
        constraints=["Include mathematical intuition", "Use visual descriptions"],
    ))

    # Analysis tasks
    store.add_task(Task(
        task_description="Analyze the pros and cons of remote work for software development teams. Provide a structured analysis covering multiple dimensions.",
        category=TaskCategory.ANALYSIS,
        difficulty=TaskDifficulty.MEDIUM,
        max_steps=6,
        min_steps=4,
    ))

    store.add_task(Task(
        task_description="Analyze the potential impact of AI on the job market over the next decade. Consider multiple sectors and provide evidence-based reasoning.",
        category=TaskCategory.ANALYSIS,
        difficulty=TaskDifficulty.HARD,
        max_steps=8,
        min_steps=5,
    ))

    # Reasoning tasks
    store.add_task(Task(
        task_description="A farmer needs to transport a wolf, a goat, and a cabbage across a river. The boat can only hold the farmer and one item. If left alone, the wolf will eat the goat, and the goat will eat the cabbage. Solve this puzzle step by step.",
        category=TaskCategory.REASONING,
        difficulty=TaskDifficulty.EASY,
        max_steps=8,
        min_steps=6,
        constraints=["Each step must describe one river crossing", "Verify safety at each step"],
    ))

    store.add_task(Task(
        task_description="Design a system for fairly distributing limited vaccine doses across different age groups and risk categories. Walk through your reasoning process.",
        category=TaskCategory.REASONING,
        difficulty=TaskDifficulty.HARD,
        max_steps=7,
        min_steps=5,
    ))

    # Generation tasks
    store.add_task(Task(
        task_description="Create a comprehensive onboarding guide for new software engineers joining a team. Structure it as a series of steps covering their first two weeks.",
        category=TaskCategory.GENERATION,
        difficulty=TaskDifficulty.MEDIUM,
        max_steps=10,
        min_steps=6,
    ))

    store.add_task(Task(
        task_description="Write a short story about a time traveler. Plan the narrative arc first, then execute each part of the story.",
        category=TaskCategory.CREATIVE,
        difficulty=TaskDifficulty.MEDIUM,
        max_steps=6,
        min_steps=4,
        constraints=["Include a clear beginning, middle, and end", "Maintain consistent tone"],
    ))

    # Coding tasks
    store.add_task(Task(
        task_description="Design and implement a function that validates email addresses. Plan your approach including edge cases, then implement step by step.",
        category=TaskCategory.CODING,
        difficulty=TaskDifficulty.MEDIUM,
        max_steps=6,
        min_steps=4,
        constraints=["Consider common edge cases", "Explain regex patterns used"],
    ))

    store.add_task(Task(
        task_description="Design a REST API for a simple todo list application. Plan the endpoints, data models, and implementation approach.",
        category=TaskCategory.CODING,
        difficulty=TaskDifficulty.HARD,
        max_steps=8,
        min_steps=5,
    ))

    # Research tasks
    store.add_task(Task(
        task_description="Research and summarize the current state of quantum computing. Organize your research into clear topical sections.",
        category=TaskCategory.RESEARCH,
        difficulty=TaskDifficulty.HARD,
        max_steps=7,
        min_steps=5,
    ))

    return store

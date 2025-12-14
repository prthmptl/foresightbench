"""Tests for ForesightBench core components."""

import pytest
import sys
sys.path.insert(0, '..')

from foresight_bench.core.task_store import Task, TaskStore, TaskCategory, TaskDifficulty
from foresight_bench.core.prompt_engine import PromptEngine
from foresight_bench.core.capture import PlanCapture, ExecutionCapture, ParseStatus
from foresight_bench.core.llm_interface import MockLLMClient, GenerationConfig


class TestTask:
    """Tests for Task dataclass."""

    def test_task_creation(self):
        task = Task(
            task_description="Test task",
            category=TaskCategory.EXPLANATION,
            difficulty=TaskDifficulty.MEDIUM,
        )
        assert task.task_description == "Test task"
        assert task.category == TaskCategory.EXPLANATION
        assert task.difficulty == TaskDifficulty.MEDIUM
        assert task.task_id is not None

    def test_task_serialization(self):
        task = Task(
            task_description="Test",
            category=TaskCategory.CODING,
            difficulty=TaskDifficulty.HARD,
            max_steps=8,
        )
        
        data = task.to_dict()
        assert data["task_description"] == "Test"
        assert data["category"] == "coding"
        assert data["max_steps"] == 8
        
        restored = Task.from_dict(data)
        assert restored.task_description == task.task_description
        assert restored.category == task.category


class TestTaskStore:
    """Tests for TaskStore."""

    def test_add_and_get_task(self):
        store = TaskStore()
        task = Task(
            task_description="Test",
            category=TaskCategory.ANALYSIS,
            difficulty=TaskDifficulty.EASY,
        )
        
        store.add_task(task)
        retrieved = store.get_task(task.task_id)
        
        assert retrieved is not None
        assert retrieved.task_id == task.task_id

    def test_filter_by_category(self):
        store = TaskStore()
        
        store.add_task(Task("A", TaskCategory.CODING, TaskDifficulty.EASY))
        store.add_task(Task("B", TaskCategory.CODING, TaskDifficulty.MEDIUM))
        store.add_task(Task("C", TaskCategory.ANALYSIS, TaskDifficulty.EASY))
        
        coding_tasks = store.list_tasks(category=TaskCategory.CODING)
        assert len(coding_tasks) == 2


class TestPlanCapture:
    """Tests for plan parsing."""

    def test_parse_standard_format(self):
        plan_text = """Step 1: First step
Step 2: Second step
Step 3: Third step"""
        
        capture = PlanCapture(plan_text)
        result = capture.parse()
        
        assert result.status == ParseStatus.SUCCESS
        assert len(capture.steps) == 3
        assert capture.steps[0].index == 1
        assert capture.steps[0].text == "First step"

    def test_parse_with_numbering_only(self):
        plan_text = """1. First step
2. Second step
3. Third step"""
        
        capture = PlanCapture(plan_text)
        result = capture.parse()
        
        assert result.status == ParseStatus.SUCCESS
        assert len(capture.steps) == 3

    def test_parse_multiline_steps(self):
        plan_text = """Step 1: First step
with additional details
Step 2: Second step"""
        
        capture = PlanCapture(plan_text)
        capture.parse()
        
        assert len(capture.steps) == 2
        assert "additional details" in capture.steps[0].text

    def test_parse_empty_plan(self):
        capture = PlanCapture("")
        result = capture.parse()
        
        assert result.status == ParseStatus.FAILED
        assert len(result.errors) > 0


class TestExecutionCapture:
    """Tests for execution parsing."""

    def test_parse_execution(self):
        exec_text = """Step 1: Here is the output for step 1.
This step does X.

Step 2: Here is the output for step 2.
This step does Y."""
        
        capture = ExecutionCapture(exec_text, expected_steps=2)
        result = capture.parse()
        
        assert result.status == ParseStatus.SUCCESS
        assert len(capture.steps) == 2

    def test_detect_missing_steps(self):
        exec_text = """Step 1: Output 1
Step 3: Output 3"""
        
        capture = ExecutionCapture(exec_text, expected_steps=3)
        result = capture.parse()
        
        assert result.status == ParseStatus.PARTIAL
        assert any("Skipped" in w for w in result.warnings)


class TestPromptEngine:
    """Tests for prompt generation."""

    def test_planning_prompt(self):
        engine = PromptEngine()
        task = Task(
            task_description="Explain recursion",
            category=TaskCategory.EXPLANATION,
            difficulty=TaskDifficulty.MEDIUM,
            max_steps=5,
            min_steps=3,
        )
        
        prompt = engine.generate_planning_prompt(task)
        
        assert "recursion" in prompt
        assert "3" in prompt  # min steps
        assert "5" in prompt  # max steps

    def test_execution_prompt(self):
        engine = PromptEngine()
        task = Task(
            task_description="Test task",
            category=TaskCategory.CODING,
            difficulty=TaskDifficulty.EASY,
        )
        
        plan = "Step 1: Do X\nStep 2: Do Y"
        prompt = engine.generate_execution_prompt(task, plan)
        
        assert "Test task" in prompt
        assert "Step 1" in prompt
        assert "Step 2" in prompt


class TestMockLLMClient:
    """Tests for mock LLM client."""

    def test_generate_default_response(self):
        client = MockLLMClient()
        result = client.generate("Test prompt")
        
        assert result.text is not None
        assert "Step 1" in result.text

    def test_custom_responses(self):
        client = MockLLMClient(
            responses={"keyword": "Custom response"},
            default_response="Default",
        )
        
        result1 = client.generate("Contains keyword here")
        assert result1.text == "Custom response"
        
        result2 = client.generate("No match")
        assert result2.text == "Default"

    def test_call_history(self):
        client = MockLLMClient()
        client.generate("Prompt 1")
        client.generate("Prompt 2")
        
        assert len(client.call_history) == 2
        assert client.call_history[0]["prompt"] == "Prompt 1"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

"""
LLM Interface - Abstract interface to language models.

Supports multiple providers (OpenAI, Anthropic, local models) with a unified API.
"""

import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Any
import logging

logger = logging.getLogger(__name__)


@dataclass
class GenerationConfig:
    """Configuration for text generation."""
    temperature: float = 0.0
    max_tokens: int = 4096
    top_p: float = 1.0
    stop_sequences: list[str] = field(default_factory=list)
    seed: Optional[int] = None  # For reproducibility where supported


@dataclass
class GenerationResult:
    """Result from a generation call."""
    text: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    latency_ms: float = 0.0
    metadata: dict = field(default_factory=dict)

    @property
    def total_cost(self) -> float:
        """Estimate cost based on token counts (override in subclasses)."""
        return 0.0


class LLMClient(ABC):
    """
    Abstract base class for LLM clients.
    
    All model providers should implement this interface.
    """

    def __init__(self, model: str, config: Optional[GenerationConfig] = None):
        """
        Initialize the client.
        
        Args:
            model: Model identifier (e.g., "gpt-4", "claude-3-opus")
            config: Default generation configuration
        """
        self.model = model
        self.default_config = config or GenerationConfig()

    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        config: Optional[GenerationConfig] = None,
    ) -> GenerationResult:
        """
        Generate a response to the given prompt.
        
        Args:
            prompt: The user prompt
            system_prompt: Optional system prompt
            config: Generation config (uses default if not provided)
            
        Returns:
            GenerationResult with the response
        """
        pass

    def generate_multiple(
        self,
        prompt: str,
        n: int,
        system_prompt: Optional[str] = None,
        config: Optional[GenerationConfig] = None,
    ) -> list[GenerationResult]:
        """
        Generate multiple responses for variance analysis.
        
        Args:
            prompt: The user prompt
            n: Number of responses to generate
            system_prompt: Optional system prompt
            config: Generation config
            
        Returns:
            List of GenerationResults
        """
        results = []
        for i in range(n):
            result = self.generate(prompt, system_prompt, config)
            result.metadata["run_index"] = i
            results.append(result)
        return results


class OpenAIClient(LLMClient):
    """OpenAI API client."""

    def __init__(
        self,
        model: str = "gpt-4",
        api_key: Optional[str] = None,
        config: Optional[GenerationConfig] = None,
    ):
        super().__init__(model, config)
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self._client = None

    def _get_client(self):
        """Lazy initialization of OpenAI client."""
        if self._client is None:
            try:
                from openai import OpenAI
                self._client = OpenAI(api_key=self.api_key)
            except ImportError:
                raise ImportError("openai package not installed. Run: pip install openai")
        return self._client

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        config: Optional[GenerationConfig] = None,
    ) -> GenerationResult:
        """Generate using OpenAI API."""
        config = config or self.default_config
        client = self._get_client()

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        start_time = time.time()

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": config.temperature,
            "max_tokens": config.max_tokens,
            "top_p": config.top_p,
        }

        if config.stop_sequences:
            kwargs["stop"] = config.stop_sequences
        if config.seed is not None:
            kwargs["seed"] = config.seed

        response = client.chat.completions.create(**kwargs)

        latency_ms = (time.time() - start_time) * 1000

        return GenerationResult(
            text=response.choices[0].message.content or "",
            model=self.model,
            prompt_tokens=response.usage.prompt_tokens if response.usage else 0,
            completion_tokens=response.usage.completion_tokens if response.usage else 0,
            total_tokens=response.usage.total_tokens if response.usage else 0,
            latency_ms=latency_ms,
            metadata={"finish_reason": response.choices[0].finish_reason},
        )


class AnthropicClient(LLMClient):
    """Anthropic API client."""

    def __init__(
        self,
        model: str = "claude-3-opus-20240229",
        api_key: Optional[str] = None,
        config: Optional[GenerationConfig] = None,
    ):
        super().__init__(model, config)
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self._client = None

    def _get_client(self):
        """Lazy initialization of Anthropic client."""
        if self._client is None:
            try:
                from anthropic import Anthropic
                self._client = Anthropic(api_key=self.api_key)
            except ImportError:
                raise ImportError("anthropic package not installed. Run: pip install anthropic")
        return self._client

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        config: Optional[GenerationConfig] = None,
    ) -> GenerationResult:
        """Generate using Anthropic API."""
        config = config or self.default_config
        client = self._get_client()

        start_time = time.time()

        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": config.max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }

        if system_prompt:
            kwargs["system"] = system_prompt
        if config.temperature > 0:
            kwargs["temperature"] = config.temperature
        if config.stop_sequences:
            kwargs["stop_sequences"] = config.stop_sequences

        response = client.messages.create(**kwargs)

        latency_ms = (time.time() - start_time) * 1000

        text = ""
        for block in response.content:
            if hasattr(block, "text"):
                text += block.text

        return GenerationResult(
            text=text,
            model=self.model,
            prompt_tokens=response.usage.input_tokens,
            completion_tokens=response.usage.output_tokens,
            total_tokens=response.usage.input_tokens + response.usage.output_tokens,
            latency_ms=latency_ms,
            metadata={"stop_reason": response.stop_reason},
        )


class MockLLMClient(LLMClient):
    """
    Mock client for testing without API calls.
    
    Returns predetermined responses based on prompt patterns.
    """

    def __init__(
        self,
        model: str = "mock-model",
        responses: Optional[dict[str, str]] = None,
        default_response: Optional[str] = None,
    ):
        super().__init__(model)
        self.responses = responses or {}
        self.default_response = default_response or self._generate_mock_plan()
        self.call_history: list[dict] = []

    def _generate_mock_plan(self) -> str:
        return """Step 1: Understand the problem and identify key components
Step 2: Break down the problem into smaller sub-tasks
Step 3: Research relevant information and gather data
Step 4: Develop a solution approach
Step 5: Implement and test the solution
Step 6: Review and refine the output"""

    def _generate_mock_execution(self, plan: str) -> str:
        """Generate mock execution based on a plan."""
        import re
        steps = re.findall(r"Step \d+:[^\n]+", plan)
        
        execution = []
        for i, step in enumerate(steps, 1):
            step_text = step.split(":", 1)[1].strip() if ":" in step else step
            execution.append(f"Step {i}: [Executing: {step_text}]\nThis step has been completed successfully with the expected output.")
        
        return "\n\n".join(execution)

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        config: Optional[GenerationConfig] = None,
    ) -> GenerationResult:
        """Return mock response."""
        self.call_history.append({
            "prompt": prompt,
            "system_prompt": system_prompt,
            "config": config,
        })

        # Check for matching response
        for pattern, response in self.responses.items():
            if pattern in prompt:
                text = response
                break
        else:
            # Detect if this is planning or execution
            # Execution prompts typically ask to "execute your plan" or "execute the plan"
            # Planning prompts might contain "do not execute" which we should ignore
            prompt_lower = prompt.lower()
            
            # Check for execution markers - must have explicit execution instruction
            is_execution = (
                ("execute your plan" in prompt_lower or 
                 "execute the plan" in prompt_lower or
                 "execute each step" in prompt_lower or
                 "execute this plan" in prompt_lower) and
                "do not execute" not in prompt_lower
            )
            
            if is_execution and ("your plan" in prompt_lower or "YOUR PLAN:" in prompt):
                # Extract plan from prompt and generate execution
                # Try to find the plan section
                if "YOUR PLAN:" in prompt:
                    plan_match = prompt.split("YOUR PLAN:")[-1].split("INSTRUCTIONS:")[0]
                elif "Your Plan:" in prompt:
                    plan_match = prompt.split("Your Plan:")[-1].split("Instructions:")[0]
                else:
                    plan_match = prompt
                text = self._generate_mock_execution(plan_match)
            else:
                text = self.default_response

        return GenerationResult(
            text=text,
            model=self.model,
            prompt_tokens=len(prompt.split()),
            completion_tokens=len(text.split()),
            total_tokens=len(prompt.split()) + len(text.split()),
            latency_ms=10.0,
        )


def create_client(
    provider: str,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    config: Optional[GenerationConfig] = None,
) -> LLMClient:
    """
    Factory function to create an LLM client.
    
    Args:
        provider: One of "openai", "anthropic", "mock"
        model: Model name (uses default for provider if not specified)
        api_key: API key (uses environment variable if not specified)
        config: Generation configuration
        
    Returns:
        Configured LLMClient instance
    """
    provider = provider.lower()

    if provider == "openai":
        return OpenAIClient(
            model=model or "gpt-4",
            api_key=api_key,
            config=config,
        )
    elif provider == "anthropic":
        return AnthropicClient(
            model=model or "claude-3-opus-20240229",
            api_key=api_key,
            config=config,
        )
    elif provider == "mock":
        return MockLLMClient(model=model or "mock-model")
    else:
        raise ValueError(f"Unknown provider: {provider}")

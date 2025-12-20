"""
Semantic Step Alignment - Aligns plan steps with execution steps using content similarity.

This module provides intelligent alignment that handles:
- Mismatched step labels (model labels Step 2 content as Step 1)
- Merged steps (model combines multiple plan steps into one)
- Split steps (model breaks one plan step into multiple)
- Reordered execution
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, TYPE_CHECKING

from core.capture import PlanStep, ExecutionStep, PlanCapture, ExecutionCapture

if TYPE_CHECKING:
    from core.llm_interface import LLMClient, GenerationConfig


class AlignmentType(str, Enum):
    """Type of alignment between plan and execution steps."""
    ONE_TO_ONE = "1:1"           # Single plan step matches single exec step
    MERGE = "merge"              # Multiple plan steps merged into one exec step
    SPLIT = "split"              # One plan step split into multiple exec steps
    SKIP = "skip"                # Plan step was skipped (not executed)
    EXTRA = "extra"              # Exec step has no corresponding plan step
    REORDER = "reorder"          # Steps executed in different order


@dataclass
class StepAlignment:
    """Represents alignment between plan and execution steps."""
    plan_indices: list[int]      # Which plan step(s) - empty for EXTRA
    exec_indices: list[int]      # Which exec step(s) - empty for SKIP
    alignment_type: AlignmentType
    similarity_score: float      # Content similarity (0-1)
    confidence: float            # Confidence in this alignment (0-1)
    details: dict = field(default_factory=dict)

    @property
    def is_valid(self) -> bool:
        """Check if this is a valid (non-problematic) alignment."""
        return self.alignment_type == AlignmentType.ONE_TO_ONE and self.similarity_score >= 0.5


@dataclass
class AlignmentResult:
    """Complete alignment result for plan-execution pair."""
    alignments: list[StepAlignment]
    overall_alignment_score: float  # How well the execution aligns with plan
    merge_count: int                # Number of detected merges
    split_count: int                # Number of detected splits
    skip_count: int                 # Number of skipped steps
    extra_count: int                # Number of extra steps
    reorder_detected: bool          # Whether reordering was detected
    method: str                     # "semantic", "embedding", "index"
    metadata: dict = field(default_factory=dict)


# Prompts for LLM-based similarity
SIMILARITY_PROMPT = """Compare these two texts and rate their semantic similarity from 0.0 to 1.0.

Text A (Plan):
{text_a}

Text B (Execution):
{text_b}

Consider:
- Are they describing the same task/action?
- Does the execution attempt what the plan describes?
- Ignore differences in wording if the meaning is the same

Return ONLY a number between 0.0 and 1.0, nothing else."""

MERGE_DETECTION_PROMPT = """Analyze if this execution step contains content from multiple plan steps.

Plan steps:
{plan_steps}

Execution step:
{exec_step}

Does this execution step combine/merge content from multiple plan steps above?
If yes, which plan step numbers are merged?

Return in format: MERGED: [list of step numbers] or NOT_MERGED
Example: MERGED: [2, 3] or NOT_MERGED"""

SPLIT_DETECTION_PROMPT = """Analyze if these execution steps together represent a single plan step.

Plan step:
{plan_step}

Execution steps:
{exec_steps}

Do these execution steps together represent the single plan step split into parts?
If yes, which execution step numbers form the split?

Return in format: SPLIT: [list of exec step numbers] or NOT_SPLIT
Example: SPLIT: [2, 3, 4] or NOT_SPLIT"""


class SemanticAligner:
    """
    Aligns plan steps with execution steps using semantic similarity.

    Uses a combination of:
    1. LLM-based similarity scoring (if available)
    2. Embedding-based similarity (if embeddings available)
    3. Keyword overlap heuristics (fallback)
    """

    def __init__(
        self,
        llm_client: Optional["LLMClient"] = None,
        embedding_client: Optional["EmbeddingClient"] = None,
        similarity_threshold: float = 0.5,
        merge_detection: bool = True,
        split_detection: bool = True,
        generation_config: Optional["GenerationConfig"] = None,
    ):
        """
        Initialize the semantic aligner.

        Args:
            llm_client: LLM client for similarity scoring
            embedding_client: Client for computing embeddings
            similarity_threshold: Minimum similarity for alignment
            merge_detection: Whether to detect merged steps
            split_detection: Whether to detect split steps
            generation_config: Config for LLM calls
        """
        self.llm_client = llm_client
        self.embedding_client = embedding_client
        self.similarity_threshold = similarity_threshold
        self.merge_detection = merge_detection
        self.split_detection = split_detection
        self.generation_config = generation_config

    def _compute_similarity_llm(self, text_a: str, text_b: str) -> float:
        """Compute semantic similarity using LLM."""
        if self.llm_client is None:
            return self._compute_similarity_heuristic(text_a, text_b)

        prompt = SIMILARITY_PROMPT.format(text_a=text_a, text_b=text_b)
        result = self.llm_client.generate(prompt, config=self.generation_config)

        # Parse score from response
        try:
            score = float(re.search(r"(\d+\.?\d*)", result.text.strip()).group(1))
            if score > 1.0:
                score = score / 100.0 if score <= 100 else 1.0
            return min(1.0, max(0.0, score))
        except (ValueError, AttributeError):
            return self._compute_similarity_heuristic(text_a, text_b)

    def _compute_similarity_embedding(self, text_a: str, text_b: str) -> float:
        """Compute similarity using embeddings."""
        if self.embedding_client is None:
            return self._compute_similarity_heuristic(text_a, text_b)

        emb_a = self.embedding_client.embed(text_a)
        emb_b = self.embedding_client.embed(text_b)

        # Cosine similarity
        dot_product = sum(a * b for a, b in zip(emb_a, emb_b))
        norm_a = sum(a * a for a in emb_a) ** 0.5
        norm_b = sum(b * b for b in emb_b) ** 0.5

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return dot_product / (norm_a * norm_b)

    def _compute_similarity_heuristic(self, text_a: str, text_b: str) -> float:
        """Compute similarity using keyword overlap (fallback)."""
        # Normalize and tokenize
        stopwords = {
            "the", "a", "an", "is", "are", "to", "and", "of", "in", "for",
            "on", "with", "that", "this", "be", "it", "as", "at", "by",
            "from", "or", "was", "were", "been", "being", "have", "has",
            "had", "do", "does", "did", "will", "would", "could", "should",
            "step", "first", "then", "next", "finally", "also"
        }

        def tokenize(text: str) -> set[str]:
            words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
            return set(words) - stopwords

        tokens_a = tokenize(text_a)
        tokens_b = tokenize(text_b)

        if not tokens_a or not tokens_b:
            return 0.0

        # Jaccard similarity
        intersection = len(tokens_a & tokens_b)
        union = len(tokens_a | tokens_b)

        jaccard = intersection / union if union > 0 else 0.0

        # Also consider containment (useful for merged/split detection)
        containment_a = intersection / len(tokens_a) if tokens_a else 0.0
        containment_b = intersection / len(tokens_b) if tokens_b else 0.0

        # Weighted combination
        return 0.5 * jaccard + 0.25 * containment_a + 0.25 * containment_b

    def _compute_similarity(self, text_a: str, text_b: str) -> float:
        """Compute similarity using best available method."""
        if self.llm_client is not None:
            return self._compute_similarity_llm(text_a, text_b)
        elif self.embedding_client is not None:
            return self._compute_similarity_embedding(text_a, text_b)
        else:
            return self._compute_similarity_heuristic(text_a, text_b)

    def _build_similarity_matrix(
        self,
        plan_steps: list[PlanStep],
        exec_steps: list[ExecutionStep],
    ) -> list[list[float]]:
        """Build a similarity matrix between plan and execution steps."""
        matrix = []
        for plan_step in plan_steps:
            row = []
            for exec_step in exec_steps:
                similarity = self._compute_similarity(plan_step.text, exec_step.content)
                row.append(similarity)
            matrix.append(row)
        return matrix

    def _detect_merges(
        self,
        plan_steps: list[PlanStep],
        exec_steps: list[ExecutionStep],
        similarity_matrix: list[list[float]],
    ) -> list[tuple[list[int], int, float]]:
        """
        Detect cases where multiple plan steps were merged into one execution step.

        Returns list of (plan_indices, exec_index, confidence) tuples.
        """
        merges = []

        for exec_idx, exec_step in enumerate(exec_steps):
            # Find all plan steps that have reasonable similarity to this exec step
            matching_plans = []
            for plan_idx, plan_step in enumerate(plan_steps):
                sim = similarity_matrix[plan_idx][exec_idx]
                if sim >= self.similarity_threshold * 0.7:  # Lower threshold for merge detection
                    matching_plans.append((plan_idx, sim))

            # If multiple plan steps match this exec step, it might be a merge
            if len(matching_plans) >= 2:
                # Check if the combined content of plan steps matches better
                plan_indices = [p[0] for p in matching_plans]
                combined_plan_text = " ".join(plan_steps[i].text for i in plan_indices)
                combined_sim = self._compute_similarity(combined_plan_text, exec_step.content)

                # Only count as merge if combined similarity is higher
                individual_max = max(p[1] for p in matching_plans)
                if combined_sim > individual_max * 0.9:
                    merges.append((plan_indices, exec_idx, combined_sim))

        return merges

    def _detect_splits(
        self,
        plan_steps: list[PlanStep],
        exec_steps: list[ExecutionStep],
        similarity_matrix: list[list[float]],
    ) -> list[tuple[int, list[int], float]]:
        """
        Detect cases where one plan step was split into multiple execution steps.

        Returns list of (plan_index, exec_indices, confidence) tuples.
        """
        splits = []

        for plan_idx, plan_step in enumerate(plan_steps):
            # Find all exec steps that have some similarity to this plan step
            matching_execs = []
            for exec_idx, exec_step in enumerate(exec_steps):
                sim = similarity_matrix[plan_idx][exec_idx]
                if sim >= self.similarity_threshold * 0.5:  # Lower threshold for split detection
                    matching_execs.append((exec_idx, sim))

            # If multiple exec steps match this plan step, check for split
            if len(matching_execs) >= 2:
                exec_indices = [e[0] for e in matching_execs]
                # Check if consecutive (splits are usually sequential)
                exec_indices_sorted = sorted(exec_indices)
                is_consecutive = all(
                    exec_indices_sorted[i+1] - exec_indices_sorted[i] <= 2
                    for i in range(len(exec_indices_sorted) - 1)
                )

                if is_consecutive:
                    combined_exec_text = " ".join(exec_steps[i].content for i in exec_indices_sorted)
                    combined_sim = self._compute_similarity(plan_step.text, combined_exec_text)

                    individual_max = max(e[1] for e in matching_execs)
                    if combined_sim > individual_max * 0.8:
                        splits.append((plan_idx, exec_indices_sorted, combined_sim))

        return splits

    def _hungarian_align(
        self,
        similarity_matrix: list[list[float]],
    ) -> list[tuple[int, int, float]]:
        """
        Perform optimal 1:1 alignment using greedy assignment.

        Returns list of (plan_idx, exec_idx, similarity) tuples.
        """
        if not similarity_matrix or not similarity_matrix[0]:
            return []

        n_plan = len(similarity_matrix)
        n_exec = len(similarity_matrix[0])

        # Greedy assignment (for simplicity - could use scipy.optimize.linear_sum_assignment)
        assignments = []
        used_exec = set()

        # Create flat list of (plan_idx, exec_idx, similarity)
        candidates = []
        for p_idx in range(n_plan):
            for e_idx in range(n_exec):
                candidates.append((p_idx, e_idx, similarity_matrix[p_idx][e_idx]))

        # Sort by similarity (descending)
        candidates.sort(key=lambda x: x[2], reverse=True)

        used_plan = set()
        for p_idx, e_idx, sim in candidates:
            if p_idx not in used_plan and e_idx not in used_exec:
                if sim >= self.similarity_threshold:
                    assignments.append((p_idx, e_idx, sim))
                    used_plan.add(p_idx)
                    used_exec.add(e_idx)

        return assignments

    def align(
        self,
        plan: PlanCapture,
        execution: ExecutionCapture,
    ) -> AlignmentResult:
        """
        Align plan steps with execution steps using semantic similarity.

        Args:
            plan: Parsed plan
            execution: Parsed execution

        Returns:
            AlignmentResult with detailed alignment information
        """
        plan.parse()
        execution.parse()

        plan_steps = plan.steps
        exec_steps = execution.steps

        if not plan_steps or not exec_steps:
            return AlignmentResult(
                alignments=[],
                overall_alignment_score=0.0,
                merge_count=0,
                split_count=0,
                skip_count=len(plan_steps),
                extra_count=len(exec_steps),
                reorder_detected=False,
                method="empty",
            )

        # Build similarity matrix
        similarity_matrix = self._build_similarity_matrix(plan_steps, exec_steps)

        alignments = []
        aligned_plan_indices = set()
        aligned_exec_indices = set()

        # First, detect merges if enabled
        merges = []
        if self.merge_detection:
            merges = self._detect_merges(plan_steps, exec_steps, similarity_matrix)
            for plan_indices, exec_idx, confidence in merges:
                alignments.append(StepAlignment(
                    plan_indices=plan_indices,
                    exec_indices=[exec_idx],
                    alignment_type=AlignmentType.MERGE,
                    similarity_score=confidence,
                    confidence=confidence,
                    details={"merged_plan_steps": plan_indices},
                ))
                aligned_plan_indices.update(plan_indices)
                aligned_exec_indices.add(exec_idx)

        # Detect splits if enabled
        splits = []
        if self.split_detection:
            splits = self._detect_splits(plan_steps, exec_steps, similarity_matrix)
            for plan_idx, exec_indices, confidence in splits:
                if plan_idx not in aligned_plan_indices:
                    # Check exec indices aren't already used
                    if not any(e in aligned_exec_indices for e in exec_indices):
                        alignments.append(StepAlignment(
                            plan_indices=[plan_idx],
                            exec_indices=exec_indices,
                            alignment_type=AlignmentType.SPLIT,
                            similarity_score=confidence,
                            confidence=confidence,
                            details={"split_exec_steps": exec_indices},
                        ))
                        aligned_plan_indices.add(plan_idx)
                        aligned_exec_indices.update(exec_indices)

        # Perform 1:1 alignment for remaining steps
        remaining_plan = [i for i in range(len(plan_steps)) if i not in aligned_plan_indices]
        remaining_exec = [i for i in range(len(exec_steps)) if i not in aligned_exec_indices]

        if remaining_plan and remaining_exec:
            # Build sub-matrix for remaining steps
            sub_matrix = [
                [similarity_matrix[p][e] for e in remaining_exec]
                for p in remaining_plan
            ]

            sub_assignments = self._hungarian_align(sub_matrix)

            for sub_p, sub_e, sim in sub_assignments:
                p_idx = remaining_plan[sub_p]
                e_idx = remaining_exec[sub_e]

                # Check if this is a reorder (indices don't match)
                is_reorder = plan_steps[p_idx].index != exec_steps[e_idx].index

                alignments.append(StepAlignment(
                    plan_indices=[p_idx],
                    exec_indices=[e_idx],
                    alignment_type=AlignmentType.REORDER if is_reorder else AlignmentType.ONE_TO_ONE,
                    similarity_score=sim,
                    confidence=sim,
                ))
                aligned_plan_indices.add(p_idx)
                aligned_exec_indices.add(e_idx)

        # Mark remaining unaligned plan steps as SKIP
        for p_idx in range(len(plan_steps)):
            if p_idx not in aligned_plan_indices:
                alignments.append(StepAlignment(
                    plan_indices=[p_idx],
                    exec_indices=[],
                    alignment_type=AlignmentType.SKIP,
                    similarity_score=0.0,
                    confidence=1.0,  # Confident it was skipped
                ))

        # Mark remaining unaligned exec steps as EXTRA
        for e_idx in range(len(exec_steps)):
            if e_idx not in aligned_exec_indices:
                alignments.append(StepAlignment(
                    plan_indices=[],
                    exec_indices=[e_idx],
                    alignment_type=AlignmentType.EXTRA,
                    similarity_score=0.0,
                    confidence=1.0,  # Confident it's extra
                ))

        # Sort alignments by first plan index (or exec index for extras)
        alignments.sort(key=lambda a: (
            min(a.plan_indices) if a.plan_indices else float('inf'),
            min(a.exec_indices) if a.exec_indices else float('inf'),
        ))

        # Calculate summary statistics
        merge_count = sum(1 for a in alignments if a.alignment_type == AlignmentType.MERGE)
        split_count = sum(1 for a in alignments if a.alignment_type == AlignmentType.SPLIT)
        skip_count = sum(1 for a in alignments if a.alignment_type == AlignmentType.SKIP)
        extra_count = sum(1 for a in alignments if a.alignment_type == AlignmentType.EXTRA)
        reorder_detected = any(a.alignment_type == AlignmentType.REORDER for a in alignments)

        # Overall alignment score
        valid_alignments = [a for a in alignments if a.alignment_type in
                          [AlignmentType.ONE_TO_ONE, AlignmentType.MERGE, AlignmentType.SPLIT]]
        if valid_alignments:
            overall_score = sum(a.similarity_score for a in valid_alignments) / len(plan_steps)
        else:
            overall_score = 0.0

        # Determine method used
        if self.llm_client is not None:
            method = "semantic_llm"
        elif self.embedding_client is not None:
            method = "embedding"
        else:
            method = "heuristic"

        return AlignmentResult(
            alignments=alignments,
            overall_alignment_score=overall_score,
            merge_count=merge_count,
            split_count=split_count,
            skip_count=skip_count,
            extra_count=extra_count,
            reorder_detected=reorder_detected,
            method=method,
            metadata={
                "plan_step_count": len(plan_steps),
                "exec_step_count": len(exec_steps),
                "similarity_threshold": self.similarity_threshold,
            },
        )


class EmbeddingClient:
    """
    Client for computing text embeddings.

    Supports OpenAI embeddings API.
    """

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: Optional[str] = None,
    ):
        """
        Initialize embedding client.

        Args:
            model: Embedding model name
            api_key: API key (uses OPENAI_API_KEY env var if not provided)
        """
        self.model = model
        self.api_key = api_key
        self._client = None
        self._cache: dict[str, list[float]] = {}

    def _get_client(self):
        """Lazy initialization of OpenAI client."""
        if self._client is None:
            import os
            try:
                from openai import OpenAI
                self._client = OpenAI(api_key=self.api_key or os.environ.get("OPENAI_API_KEY"))
            except ImportError:
                raise ImportError("openai package not installed. Run: pip install openai")
        return self._client

    def embed(self, text: str) -> list[float]:
        """
        Get embedding for text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector
        """
        # Check cache
        if text in self._cache:
            return self._cache[text]

        client = self._get_client()
        response = client.embeddings.create(
            model=self.model,
            input=text,
        )

        embedding = response.data[0].embedding
        self._cache[text] = embedding

        return embedding

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Get embeddings for multiple texts.

        Args:
            texts: Texts to embed

        Returns:
            List of embedding vectors
        """
        # Check cache for each
        uncached_texts = [t for t in texts if t not in self._cache]

        if uncached_texts:
            client = self._get_client()
            response = client.embeddings.create(
                model=self.model,
                input=uncached_texts,
            )

            for text, data in zip(uncached_texts, response.data):
                self._cache[text] = data.embedding

        return [self._cache[t] for t in texts]


def align_with_fallback(
    plan: PlanCapture,
    execution: ExecutionCapture,
    llm_client: Optional["LLMClient"] = None,
    use_semantic: bool = True,
) -> AlignmentResult:
    """
    Convenience function to align with automatic fallback.

    Uses semantic alignment if LLM client provided, otherwise falls back to heuristic.

    Args:
        plan: Parsed plan
        execution: Parsed execution
        llm_client: Optional LLM client for semantic alignment
        use_semantic: Whether to use semantic alignment (vs index-based)

    Returns:
        AlignmentResult
    """
    if not use_semantic:
        # Simple index-based alignment (legacy behavior)
        from core.capture import align_plan_and_execution

        aligned = align_plan_and_execution(plan, execution)
        alignments = []

        for plan_step, exec_step in aligned:
            if plan_step is None:
                alignments.append(StepAlignment(
                    plan_indices=[],
                    exec_indices=[exec_step.index - 1],  # Convert to 0-based
                    alignment_type=AlignmentType.EXTRA,
                    similarity_score=0.0,
                    confidence=1.0,
                ))
            elif exec_step is None:
                alignments.append(StepAlignment(
                    plan_indices=[plan_step.index - 1],  # Convert to 0-based
                    exec_indices=[],
                    alignment_type=AlignmentType.SKIP,
                    similarity_score=0.0,
                    confidence=1.0,
                ))
            else:
                alignments.append(StepAlignment(
                    plan_indices=[plan_step.index - 1],
                    exec_indices=[exec_step.index - 1],
                    alignment_type=AlignmentType.ONE_TO_ONE,
                    similarity_score=1.0,  # Assume match for index-based
                    confidence=1.0,
                ))

        return AlignmentResult(
            alignments=alignments,
            overall_alignment_score=1.0 if alignments else 0.0,
            merge_count=0,
            split_count=0,
            skip_count=sum(1 for a in alignments if a.alignment_type == AlignmentType.SKIP),
            extra_count=sum(1 for a in alignments if a.alignment_type == AlignmentType.EXTRA),
            reorder_detected=False,
            method="index",
        )

    aligner = SemanticAligner(llm_client=llm_client)
    return aligner.align(plan, execution)

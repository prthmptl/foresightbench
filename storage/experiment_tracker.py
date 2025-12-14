"""
Experiment Tracker - Storage and experiment tracking for reproducibility.

Supports:
- JSONL for raw traces
- SQLite/DuckDB for metrics
- Cross-model comparison
- Ablation studies
"""

import json
import sqlite3
import uuid
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional, Iterator, Any

from evaluation.metrics import TaskMetrics, GlobalMetrics


@dataclass
class ExperimentRun:
    """A single experiment run record."""
    run_id: str
    experiment_id: str
    model: str
    task_id: str
    timestamp: str
    
    # Raw outputs
    plan_text: str
    execution_text: str
    
    # Parsed data
    plan_steps: list[dict]
    execution_steps: list[dict]
    
    # Metrics
    foresight_score: float
    execution_reliability: float
    planning_quality: float
    rule_validation_score: float
    semantic_evaluation_score: float
    
    # Structural
    plan_step_count: int
    execution_step_count: int
    skipped_step_count: int
    extra_step_count: int
    
    # Performance
    latency_ms: float
    token_count: int
    
    # Config
    temperature: float
    config: dict


class ExperimentTracker:
    """
    Tracks experiments with storage and querying capabilities.
    
    Supports both JSONL (raw traces) and SQLite (structured queries).
    """

    def __init__(
        self,
        storage_dir: Path | str = "./experiments",
        experiment_name: Optional[str] = None,
    ):
        """
        Initialize experiment tracker.
        
        Args:
            storage_dir: Directory for storing experiment data
            experiment_name: Name for this experiment (auto-generated if None)
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        self.experiment_name = experiment_name or f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.experiment_id = str(uuid.uuid4())
        
        # File paths
        self.traces_file = self.storage_dir / f"{self.experiment_name}_traces.jsonl"
        self.metrics_db = self.storage_dir / f"{self.experiment_name}_metrics.db"
        
        # Initialize SQLite
        self._init_database()

    def _init_database(self) -> None:
        """Initialize SQLite database schema."""
        conn = sqlite3.connect(self.metrics_db)
        cursor = conn.cursor()
        
        # Runs table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS runs (
                run_id TEXT PRIMARY KEY,
                experiment_id TEXT,
                model TEXT,
                task_id TEXT,
                timestamp TEXT,
                foresight_score REAL,
                execution_reliability REAL,
                planning_quality REAL,
                rule_validation_score REAL,
                semantic_evaluation_score REAL,
                plan_step_count INTEGER,
                execution_step_count INTEGER,
                skipped_step_count INTEGER,
                extra_step_count INTEGER,
                latency_ms REAL,
                token_count INTEGER,
                temperature REAL
            )
        """)
        
        # Step metrics table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS step_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT,
                step_index INTEGER,
                step_match REAL,
                constraint_fidelity REAL,
                step_purity REAL,
                completeness REAL,
                combined_score REAL,
                FOREIGN KEY (run_id) REFERENCES runs(run_id)
            )
        """)
        
        # Experiments metadata table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS experiments (
                experiment_id TEXT PRIMARY KEY,
                experiment_name TEXT,
                created_at TEXT,
                config TEXT
            )
        """)
        
        # Create indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_runs_model ON runs(model)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_runs_task ON runs(task_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_runs_experiment ON runs(experiment_id)")
        
        conn.commit()
        conn.close()

    def log_run(
        self,
        task_id: str,
        model: str,
        plan_text: str,
        execution_text: str,
        task_metrics: TaskMetrics,
        plan_steps: list[dict],
        execution_steps: list[dict],
        temperature: float = 0.0,
        config: Optional[dict] = None,
    ) -> str:
        """
        Log a single run to storage.
        
        Args:
            task_id: Task identifier
            model: Model name
            plan_text: Raw plan output
            execution_text: Raw execution output
            task_metrics: Computed metrics
            plan_steps: Parsed plan steps
            execution_steps: Parsed execution steps
            temperature: Generation temperature
            config: Additional configuration
            
        Returns:
            Run ID
        """
        run_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()
        config = config or {}

        # Create run record
        run = ExperimentRun(
            run_id=run_id,
            experiment_id=self.experiment_id,
            model=model,
            task_id=task_id,
            timestamp=timestamp,
            plan_text=plan_text,
            execution_text=execution_text,
            plan_steps=plan_steps,
            execution_steps=execution_steps,
            foresight_score=task_metrics.foresight_score,
            execution_reliability=task_metrics.execution_reliability,
            planning_quality=task_metrics.planning_quality,
            rule_validation_score=task_metrics.rule_validation_score,
            semantic_evaluation_score=task_metrics.semantic_evaluation_score,
            plan_step_count=task_metrics.plan_step_count,
            execution_step_count=task_metrics.execution_step_count,
            skipped_step_count=task_metrics.skipped_step_count,
            extra_step_count=task_metrics.extra_step_count,
            latency_ms=task_metrics.latency_ms,
            token_count=task_metrics.token_count,
            temperature=temperature,
            config=config,
        )

        # Write to JSONL
        with open(self.traces_file, "a") as f:
            f.write(json.dumps(asdict(run)) + "\n")

        # Write to SQLite
        conn = sqlite3.connect(self.metrics_db)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO runs (
                run_id, experiment_id, model, task_id, timestamp,
                foresight_score, execution_reliability, planning_quality,
                rule_validation_score, semantic_evaluation_score,
                plan_step_count, execution_step_count,
                skipped_step_count, extra_step_count,
                latency_ms, token_count, temperature
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            run_id, self.experiment_id, model, task_id, timestamp,
            task_metrics.foresight_score, task_metrics.execution_reliability,
            task_metrics.planning_quality, task_metrics.rule_validation_score,
            task_metrics.semantic_evaluation_score,
            task_metrics.plan_step_count, task_metrics.execution_step_count,
            task_metrics.skipped_step_count, task_metrics.extra_step_count,
            task_metrics.latency_ms, task_metrics.token_count, temperature,
        ))

        # Write step metrics
        for step in task_metrics.step_metrics:
            cursor.execute("""
                INSERT INTO step_metrics (
                    run_id, step_index, step_match, constraint_fidelity,
                    step_purity, completeness, combined_score
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                run_id, step.step_index, step.step_match,
                step.constraint_fidelity, step.step_purity,
                step.completeness, step.combined_score,
            ))

        conn.commit()
        conn.close()

        return run_id

    def get_run(self, run_id: str) -> Optional[dict]:
        """Get a specific run by ID."""
        conn = sqlite3.connect(self.metrics_db)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,))
        row = cursor.fetchone()
        
        conn.close()
        
        if row:
            return dict(row)
        return None

    def query_runs(
        self,
        model: Optional[str] = None,
        task_id: Optional[str] = None,
        min_score: Optional[float] = None,
        limit: int = 100,
    ) -> list[dict]:
        """
        Query runs with optional filters.
        
        Args:
            model: Filter by model
            task_id: Filter by task
            min_score: Minimum foresight score
            limit: Maximum results
            
        Returns:
            List of matching runs
        """
        conn = sqlite3.connect(self.metrics_db)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        query = "SELECT * FROM runs WHERE 1=1"
        params: list[Any] = []
        
        if model:
            query += " AND model = ?"
            params.append(model)
        if task_id:
            query += " AND task_id = ?"
            params.append(task_id)
        if min_score is not None:
            query += " AND foresight_score >= ?"
            params.append(min_score)
        
        query += f" ORDER BY timestamp DESC LIMIT {limit}"
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        
        conn.close()
        
        return [dict(row) for row in rows]

    def get_model_comparison(self) -> list[dict]:
        """
        Get aggregated metrics comparison across models.
        
        Returns:
            List of model statistics
        """
        conn = sqlite3.connect(self.metrics_db)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT 
                model,
                COUNT(*) as run_count,
                AVG(foresight_score) as avg_foresight,
                AVG(execution_reliability) as avg_reliability,
                AVG(planning_quality) as avg_planning,
                AVG(skipped_step_count * 1.0 / plan_step_count) as avg_skip_rate,
                AVG(latency_ms) as avg_latency
            FROM runs
            GROUP BY model
            ORDER BY avg_foresight DESC
        """)
        
        rows = cursor.fetchall()
        conn.close()
        
        return [dict(row) for row in rows]

    def get_task_breakdown(self, model: Optional[str] = None) -> list[dict]:
        """
        Get metrics breakdown by task.
        
        Args:
            model: Optional model filter
            
        Returns:
            List of task statistics
        """
        conn = sqlite3.connect(self.metrics_db)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        query = """
            SELECT 
                task_id,
                COUNT(*) as run_count,
                AVG(foresight_score) as avg_foresight,
                MIN(foresight_score) as min_foresight,
                MAX(foresight_score) as max_foresight
            FROM runs
        """
        
        params: list[Any] = []
        if model:
            query += " WHERE model = ?"
            params.append(model)
        
        query += " GROUP BY task_id ORDER BY avg_foresight DESC"
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()
        
        return [dict(row) for row in rows]

    def iterate_traces(self) -> Iterator[dict]:
        """
        Iterate over all raw traces from JSONL.
        
        Yields:
            Individual trace records
        """
        if not self.traces_file.exists():
            return
        
        with open(self.traces_file, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)

    def export_leaderboard(self, output_path: Optional[Path] = None) -> dict:
        """
        Export leaderboard data.
        
        Args:
            output_path: Optional path to save JSON
            
        Returns:
            Leaderboard data dictionary
        """
        comparison = self.get_model_comparison()
        
        leaderboard = {
            "experiment": self.experiment_name,
            "generated_at": datetime.now().isoformat(),
            "models": comparison,
            "top_model": comparison[0]["model"] if comparison else None,
            "top_score": comparison[0]["avg_foresight"] if comparison else 0.0,
        }
        
        if output_path:
            with open(output_path, "w") as f:
                json.dump(leaderboard, f, indent=2)
        
        return leaderboard


class TraceViewer:
    """Utility for viewing and analyzing traces."""

    def __init__(self, traces_file: Path | str):
        """
        Initialize trace viewer.
        
        Args:
            traces_file: Path to JSONL traces file
        """
        self.traces_file = Path(traces_file)
        self._traces: list[dict] = []
        self._loaded = False

    def load(self) -> int:
        """Load all traces into memory. Returns count."""
        self._traces = []
        with open(self.traces_file, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    self._traces.append(json.loads(line))
        self._loaded = True
        return len(self._traces)

    @property
    def traces(self) -> list[dict]:
        """Get all loaded traces."""
        if not self._loaded:
            self.load()
        return self._traces

    def filter_by_model(self, model: str) -> list[dict]:
        """Filter traces by model name."""
        return [t for t in self.traces if t["model"] == model]

    def filter_by_score(self, min_score: float, max_score: float = 1.0) -> list[dict]:
        """Filter traces by foresight score range."""
        return [
            t for t in self.traces
            if min_score <= t["foresight_score"] <= max_score
        ]

    def get_worst_runs(self, n: int = 10) -> list[dict]:
        """Get the n lowest scoring runs."""
        sorted_traces = sorted(self.traces, key=lambda t: t["foresight_score"])
        return sorted_traces[:n]

    def get_best_runs(self, n: int = 10) -> list[dict]:
        """Get the n highest scoring runs."""
        sorted_traces = sorted(self.traces, key=lambda t: t["foresight_score"], reverse=True)
        return sorted_traces[:n]

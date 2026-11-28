from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from src.core.project import ResolvedProject


class LakehousePaths(BaseModel):
    """Canonical directory and database paths inside an isolated sandbox workspace."""

    workspace_dir: Path
    raw_dir: Path
    prod_silver_dir: Path
    prod_mart_dir: Path
    prod_duckdb_path: Path
    dbt_project_dir: Path
    spark_jobs_dir: Path

    @classmethod
    def from_workspace(cls, workspace_dir: Path) -> LakehousePaths:
        root = workspace_dir.resolve()
        prod_mart_dir = root / "lakehouse" / "prod" / "mart"
        return cls(
            workspace_dir=root,
            raw_dir=root / "lakehouse" / "raw",
            prod_silver_dir=root / "lakehouse" / "prod" / "silver",
            prod_mart_dir=prod_mart_dir,
            prod_duckdb_path=prod_mart_dir / "warehouse.duckdb",
            dbt_project_dir=root / "dbt_project",
            spark_jobs_dir=root / "spark_jobs",
        )

    @classmethod
    def from_project(cls, project: ResolvedProject) -> LakehousePaths:
        """Build directory and database bindings from a discovered `ResolvedProject`."""
        root = project.repo_root.resolve()
        lakehouse_dir = project.lakehouse_dir or (root / "lakehouse")
        prod_mart_dir = lakehouse_dir / "prod" / "mart"
        return cls(
            workspace_dir=root,
            raw_dir=lakehouse_dir / "raw",
            prod_silver_dir=lakehouse_dir / "prod" / "silver",
            prod_mart_dir=prod_mart_dir,
            prod_duckdb_path=project.duckdb_path or (prod_mart_dir / "warehouse.duckdb"),
            dbt_project_dir=project.dbt_project_dir or (root / "dbt_project"),
            spark_jobs_dir=project.spark_jobs_dir or (root / "spark_jobs"),
        )


class SparkStageMetrics(BaseModel):
    """Per-stage task latency distribution parsed from Spark event logs."""

    stage_id: int
    stage_name: str = ""
    task_count: int
    failed_tasks: int = 0
    min_ms: float
    median_ms: float
    max_ms: float
    skew_ratio: float


class SparkEventSummary(BaseModel):
    """Deterministic diagnostic summary extracted from a Spark event log."""

    app_id: str | None = None
    app_name: str | None = None
    event_log_path: Path | None = None
    total_tasks: int = 0
    failed_tasks: int = 0
    max_skew_ratio: float = 1.0
    skewed_stages: list[int] = Field(default_factory=list)
    stages: list[SparkStageMetrics] = Field(default_factory=list)
    oom_detected: bool = False
    oom_reasons: list[str] = Field(default_factory=list)


class SparkRunResult(BaseModel):
    """Observable execution outcome of an isolated PySpark child subprocess."""

    job_name: str
    script_path: Path
    pid: int | None
    returncode: int
    duration_ms: float
    event_log_path: Path | None
    stdout: str
    stderr: str
    timed_out: bool = False
    event_summary: SparkEventSummary | None = None


class ResetSummary(BaseModel):
    """Verification summary emitted by the synthetic lakehouse reset command."""

    elapsed_seconds: float
    git_commits: int
    raw_orders: int
    raw_users: int
    raw_exchange_rates: int
    spark_runs: list[SparkRunResult]
    dbt_run_ok: bool
    dbt_test_ok: bool
    fct_daily_gmv_total: float
    dim_user_ltv_total: float
    daily_gmv_rows: int
    user_ltv_rows: int


class LineageTestRef(BaseModel):
    """Metadata describing a dbt test attached to a pipeline asset."""

    name: str
    test_type: str
    column_name: str | None = None
    file_path: str | None = None


class LineageNode(BaseModel):
    """Unified asset representation across raw, Spark, dbt, and external OpenLineage layers."""

    name: str
    engine: Literal["raw", "spark", "dbt", "external"]
    layer: str
    file_path: str | None = None
    namespace: str | None = None
    tags: list[str] = Field(default_factory=list)
    columns: list[str] = Field(default_factory=list)
    partition_columns: list[str] = Field(default_factory=list)
    pii_columns: list[str] = Field(default_factory=list)
    tests: list[LineageTestRef] = Field(default_factory=list)
    upstream: list[str] = Field(default_factory=list)
    downstream: list[str] = Field(default_factory=list)


class LineageEdge(BaseModel):
    """Directed dependency edge from an upstream asset to a downstream asset."""

    source: str
    target: str


class LineageSubgraph(BaseModel):
    """Bounded N-hop lineage neighborhood around a queried asset."""

    root: str
    direction: Literal["upstream", "downstream", "both"]
    depth: int
    nodes: dict[str, LineageNode]
    edges: list[LineageEdge]


class AffectedSubgraph(BaseModel):
    """Downstream Spark jobs and dbt models impacted by workspace file changes."""

    changed_paths: list[str]
    spark_jobs: list[str]
    dbt_models: list[str]
    affected_assets: list[str]


class SqlQueryResult(BaseModel):
    """Structured result returned by a read-only exploratory SQL query."""

    sql: str
    columns: list[str]
    rows: list[list[Any]]
    row_count: int
    duration_ms: float


class DataDiffSummary(BaseModel):
    """Row and key-level diff between production and shadow mart tables."""

    table_name: str
    key_column: str
    prod_row_count: int
    shadow_row_count: int
    added_keys: list[str] = Field(default_factory=list)
    removed_keys: list[str] = Field(default_factory=list)
    changed_keys: list[str] = Field(default_factory=list)
    is_identical: bool


class ComputeAdapter(ABC):
    """Abstract interface for isolated compute job execution."""

    @abstractmethod
    def run_job(
        self,
        script_path: Path,
        args: list[str] | None = None,
        driver_memory: str | None = None,
        timeout_seconds: int | None = None,
        env_overrides: dict[str, str] | None = None,
    ) -> SparkRunResult:
        """Execute a compute job in isolation and return its structured run result."""


class LineageAdapterBase(ABC):
    """Abstract interface for cross-engine pipeline lineage queries."""

    @abstractmethod
    def query_lineage(
        self,
        table_name: str,
        direction: Literal["upstream", "downstream", "both"] = "both",
        depth: int = 3,
    ) -> LineageSubgraph:
        """Return an N-hop lineage subgraph rooted at `table_name`."""

    @abstractmethod
    def shared_ancestors(self, table_names: list[str]) -> list[str]:
        """Return common ancestors across `table_names`, ordered lowest to highest."""

    @abstractmethod
    def lowest_common_ancestors(self, table_names: list[str]) -> list[str]:
        """Return minimal shared ancestors closest to `table_names`."""

    @abstractmethod
    def affected_downstream(self, changed_paths: list[str]) -> AffectedSubgraph:
        """Resolve changed workspace file paths to their impacted Spark and dbt subgraph."""

    def all_aliases(self) -> dict[str, str]:
        """Return known alias -> canonical asset name mappings when supported."""
        return {}

    def topological_order(self, assets: list[str] | None = None) -> list[str]:
        """Return assets in deterministic upstream-to-downstream order when supported."""
        return list(assets or [])


class StorageAdapter(ABC):
    """Abstract interface for read-only warehouse probing and shadow table diffing."""

    @abstractmethod
    def execute_readonly(self, sql: str) -> SqlQueryResult:
        """Run a read-only SQL query against the lakehouse."""

    @abstractmethod
    def data_diff(
        self,
        prod_db: Path,
        shadow_db: Path,
        table_name: str,
        key_column: str,
    ) -> DataDiffSummary:
        """Compare rows between production and shadow instances of a mart table."""

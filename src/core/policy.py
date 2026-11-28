from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal

import yaml
from pydantic import BaseModel, Field

from src.core.context import PrincipalRole, ResolutionMode

if TYPE_CHECKING:
    from src.adapters.base import LineageAdapterBase
    from src.core.project import ResolvedProject

_DIALECT_ALIASES: dict[str, str] = {
    "postgres": "postgres",
    "postgresql": "postgres",
    "pg": "postgres",
    "aurora-postgresql": "postgres",
    "rds-postgres": "postgres",
    "duckdb": "duckdb",
    "spark": "spark",
    "pyspark": "spark",
    "sparksql": "spark",
    "spark_sql": "spark",
    "databricks": "databricks",
    "snowflake": "snowflake",
    "bigquery": "bigquery",
    "bq": "bigquery",
    "clickhouse": "clickhouse",
    "trino": "trino",
    "presto": "trino",
    "athena": "trino",
    "sqlite": "sqlite",
    "sqlite3": "sqlite",
    "mysql": "mysql",
    "mariadb": "mysql",
    "redshift": "redshift",
    "tsql": "tsql",
    "sqlserver": "tsql",
    "mssql": "tsql",
}


def normalize_sql_dialect(raw: str | None) -> str:
    """Normalize SQL engine or dbt adapter names to canonical SQLGlot dialect identifiers."""
    if not raw or not raw.strip():
        return "duckdb"
    cleaned = raw.strip().lower()
    return _DIALECT_ALIASES.get(cleaned, cleaned)


class RolePolicy(BaseModel):
    """Permissions and execution ceiling for a single Principal Role."""

    resolution_mode: ResolutionMode
    allow_patch_paths: list[str] = Field(default_factory=list)
    allow_read_spark_event_logs: bool = False
    sql_scope: Literal["all_layers", "mart_only"] = "all_layers"
    allowed_tables: list[str] = Field(default_factory=list)
    sql_mode: Literal["rows_redacted", "aggregate_or_metadata_only"] = "rows_redacted"


class SqlGuardPolicy(BaseModel):
    """AST-level SQL safety, escape prevention, and scan limits."""

    sql_dialect: str = "duckdb"
    max_limit: int = 50
    allowed_top_level: list[str] = Field(
        default_factory=lambda: ["SELECT", "WITH", "UNION", "EXPLAIN", "DESCRIBE"]
    )
    blocked_functions: list[str] = Field(default_factory=list)
    blocked_statements: list[str] = Field(default_factory=list)


class LimitsPolicy(BaseModel):
    """Run turn, budget, and context-compaction thresholds."""

    max_turns: int = 20
    max_wall_clock_seconds: int = 300
    max_cost_usd: float = 2.0
    rejection_turn_extension: int = 8
    max_rejections: int = 2
    compactor_max_lines: int = 120
    compactor_max_chars: int = 4000
    compactor_fold_after_turns: int = 10
    compactor_fold_budget_ratio: float = 0.65


class AntiThrashPolicy(BaseModel):
    """Two-strike thrash detection thresholds."""

    repeat_call_threshold: int = 2
    error_stagnation_threshold: int = 2
    file_oscillation_threshold: int = 3
    strikes_before_abort: int = 2


class AggregatorPolicy(BaseModel):
    """Time/batch convergence window for Alert Storm deduplication."""

    window_seconds: int = 900


class PoliciesConfig(BaseModel):
    """Top-level governance and safety configuration loaded from configs/policies.yaml."""

    roles: dict[PrincipalRole, RolePolicy]
    pii_columns: list[str] = Field(default_factory=list)
    partitioned_tables: dict[str, list[str]] = Field(default_factory=dict)
    sql_guard: SqlGuardPolicy = Field(default_factory=SqlGuardPolicy)
    limits: LimitsPolicy = Field(default_factory=LimitsPolicy)
    anti_thrash: AntiThrashPolicy = Field(default_factory=AntiThrashPolicy)
    aggregator: AggregatorPolicy = Field(default_factory=AggregatorPolicy)


def load_policies(config_path: Path | None = None) -> PoliciesConfig:
    """Load and validate PoliciesConfig from YAML."""
    resolved = (
        config_path
        or Path(__file__).resolve().parent.parent.parent / "configs" / "policies.yaml"
    )
    raw = yaml.safe_load(resolved.read_text(encoding="utf-8")) or {}
    return PoliciesConfig.model_validate(raw)


def _merge_unique_ci(existing: list[str], additions: list[str]) -> list[str]:
    merged: list[str] = []
    seen_lower: set[str] = set()
    for item in [*existing, *additions]:
        cleaned = str(item).strip()
        lowered = cleaned.lower()
        if cleaned and lowered not in seen_lower:
            merged.append(cleaned)
            seen_lower.add(lowered)
    return merged


def resolve_policies(
    project: ResolvedProject,
    base_policies: PoliciesConfig | None = None,
    *,
    lineage: LineageAdapterBase | None = None,
    allow_manifestless_fix: bool = False,
) -> PoliciesConfig:
    """Merge discovered metadata and Project Manifest overrides onto base `PoliciesConfig`."""
    resolved = (base_policies or load_policies()).model_copy(deep=True)
    resolved.sql_guard.sql_dialect = normalize_sql_dialect(project.sql_dialect)
    gov = project.manifest.governance

    all_nodes_fn = getattr(lineage, "all_nodes", None) if lineage is not None else None
    lineage_nodes = all_nodes_fn() if callable(all_nodes_fn) else {}
    if not isinstance(lineage_nodes, dict):
        lineage_nodes = {}

    known_assets = set(project.discovered.discovered_assets) | set(lineage_nodes.keys())
    known_lower = {name.lower() for name in known_assets}

    lineage_pii = [col for node in lineage_nodes.values() for col in node.pii_columns]
    resolved.pii_columns = _merge_unique_ci(
        resolved.pii_columns,
        [*project.discovered.pii_columns, *lineage_pii, *gov.pii_columns],
    )

    is_external_repo = (
        project.has_manifest
        or project.dbt_project_dir is not None
        or project.spark_jobs_dir is not None
        or bool(project.openlineage_paths)
        or bool(known_assets)
    )
    if is_external_repo:
        partitioned: dict[str, list[str]] = {
            tbl: list(cols)
            for tbl, cols in resolved.partitioned_tables.items()
            if tbl.lower() in known_lower
        }
    else:
        partitioned = {tbl: list(cols) for tbl, cols in resolved.partitioned_tables.items()}

    def _apply_partition_rule(tbl_name: str, cols: list[str], *, replace: bool) -> None:
        cleaned = _merge_unique_ci([], list(cols))
        existing_key = next((k for k in partitioned if k.lower() == tbl_name.lower()), None)
        if replace:
            if existing_key is not None and existing_key != tbl_name:
                del partitioned[existing_key]
            if cleaned:
                partitioned[tbl_name] = cleaned
            else:
                partitioned.pop(tbl_name, None)
        elif cleaned:
            key = existing_key or tbl_name
            partitioned[key] = _merge_unique_ci(partitioned.get(key, []), cleaned)

    for tbl, cols in project.discovered.partitioned_tables.items():
        _apply_partition_rule(tbl, cols, replace=False)
    for node in lineage_nodes.values():
        if node.partition_columns:
            _apply_partition_rule(node.name, node.partition_columns, replace=False)
    for tbl, cols in gov.partitioned_tables.items():
        _apply_partition_rule(tbl, list(cols), replace=True)
    resolved.partitioned_tables = partitioned

    lineage_marts = [node.name for node in lineage_nodes.values() if node.layer == "mart"]
    discovered_marts = _merge_unique_ci(list(project.discovered.mart_tables), lineage_marts)

    spark_patch_paths: list[str] = []
    if project.spark_jobs_dir is not None and project.spark_jobs_dir.is_dir():
        resolved_root = project.repo_root.resolve()
        resolved_spark = project.spark_jobs_dir.resolve()
        if resolved_spark.is_relative_to(resolved_root):
            rel_spark = resolved_spark.relative_to(resolved_root).as_posix()
            if rel_spark and rel_spark != ".":
                spark_patch_paths = [f"{rel_spark}/"]

    if gov.max_limit is not None:
        resolved.sql_guard.max_limit = gov.max_limit
    if gov.blocked_functions:
        resolved.sql_guard.blocked_functions = _merge_unique_ci(
            resolved.sql_guard.blocked_functions, gov.blocked_functions
        )

    for role_name, override in gov.role_overrides.items():
        role_policy = resolved.roles.get(role_name)
        if role_policy is None:
            continue
        if override.resolution_mode is not None:
            role_policy.resolution_mode = override.resolution_mode
        if override.allow_patch_paths is not None:
            role_policy.allow_patch_paths = list(override.allow_patch_paths)
        if override.allow_read_spark_event_logs is not None:
            role_policy.allow_read_spark_event_logs = override.allow_read_spark_event_logs
        if override.sql_scope is not None:
            role_policy.sql_scope = override.sql_scope
        if override.allowed_tables is not None:
            role_policy.allowed_tables = list(override.allowed_tables)
        if override.sql_mode is not None:
            role_policy.sql_mode = override.sql_mode

    for role_name, role_policy in resolved.roles.items():
        override = gov.role_overrides.get(role_name)
        if role_policy.sql_scope == "mart_only" and (
            override is None or override.allowed_tables is None
        ):
            if gov.mart_tables:
                role_policy.allowed_tables = list(gov.mart_tables)
            elif discovered_marts:
                role_policy.allowed_tables = list(discovered_marts)
            elif is_external_repo:
                role_policy.allowed_tables = []

        if override is None or override.allow_patch_paths is None:
            if role_name == "de":
                if project.discovered.patchable_paths:
                    role_policy.allow_patch_paths = list(project.discovered.patchable_paths)
                elif is_external_repo:
                    role_policy.allow_patch_paths = []
            elif role_name == "sre" and is_external_repo:
                role_policy.allow_patch_paths = list(spark_patch_paths)

    if not project.has_manifest and not allow_manifestless_fix:
        for role_policy in resolved.roles.values():
            role_policy.resolution_mode = "diagnosis_only"
    elif project.default_resolution_mode == "diagnosis_only":
        for role_name, role_policy in resolved.roles.items():
            override = gov.role_overrides.get(role_name)
            if override is None or override.resolution_mode is None:
                role_policy.resolution_mode = "diagnosis_only"

    return resolved

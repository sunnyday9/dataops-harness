from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field

from src.core.context import PrincipalRole, ResolutionMode


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

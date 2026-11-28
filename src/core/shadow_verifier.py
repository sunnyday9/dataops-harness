from __future__ import annotations

import re
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol

import duckdb
import yaml
from pydantic import BaseModel

from src.adapters.base import AffectedSubgraph, DataDiffSummary, LakehousePaths, LineageAdapterBase
from src.adapters.lineage_adapter import LineageAdapter
from src.adapters.shadow_backend import (
    CompositeShadowBackend,
    DbtExecutor,
    DuckDBShadowBackend,
    ShadowBackend,
    ShadowContext,
    SparkSubprocessBackend,
    _default_dbt_executor,
    _discover_mart_diff_targets,
    _infer_key_column,
    _is_dbt_model_sql,
    _normalize_dbt_rel_paths,
    _read_dbt_subpaths,
    _resolve_dbt_subdir,
    _run_dbt_subset,
    capture_incremental_blocks,
    create_shadow_backend,
    extract_incremental_blocks,
    order_spark_jobs,
    touches_incremental_logic,
)
from src.adapters.spark_adapter import SparkJobRunner
from src.core.project import ResolvedProject, discover_project
from src.core.rca import OracleVerdict, ShadowVerificationResult
from src.core.worktree import RunWorktreeManager


class MetricReconciliationRule(BaseModel):
    """Declarative cross-table sum reconciliation check evaluated when both tables exist."""

    label: str = "metric_reconciliation"
    left_table: str
    left_column: str
    right_table: str
    right_column: str
    tolerance: float = 0.01
    require_positive: bool = True


_DEFAULT_OPTIONAL_RECONCILIATION_RULES: tuple[MetricReconciliationRule, ...] = (
    MetricReconciliationRule(
        label="Mart GMV",
        left_table="fct_daily_gmv",
        left_column="net_gmv_usd",
        right_table="dim_user_ltv",
        right_column="lifetime_gmv_usd",
        tolerance=0.01,
        require_positive=True,
    ),
)


def _quote_sql_ident(ident: str) -> str:
    return '"' + ident.replace('"', '""') + '"'


def _canonical_test_key(entry: Any) -> str:
    if isinstance(entry, str):
        return entry.strip()
    if isinstance(entry, dict) and entry:
        name = next(iter(entry))
        return str(name).strip()
    return repr(entry)


def _normalize_test_config(entry: Any) -> dict[str, Any]:
    if isinstance(entry, str):
        return {}
    if not isinstance(entry, dict) or not entry:
        return {}
    raw_params = next(iter(entry.values()))
    if not isinstance(raw_params, dict):
        return {}
    cfg: dict[str, Any] = {}
    for key in ("enabled", "severity", "where", "limit"):
        if key in raw_params:
            cfg[key] = raw_params[key]
    nested = raw_params.get("config")
    if isinstance(nested, dict):
        for key in ("enabled", "severity", "where", "limit"):
            if key in nested:
                cfg[key] = nested[key]
    return cfg


def _iter_schema_yaml_files(dbt_dir: Path) -> list[Path]:
    subpaths = _read_dbt_subpaths(dbt_dir)
    files: list[Path] = []
    seen: set[Path] = set()
    resolved_dbt = dbt_dir.resolve()
    skip_dirs = frozenset({"target", "logs", "dbt_packages", ".git"})
    for rel_dir in (
        subpaths["model-paths"] + subpaths["seed-paths"] + subpaths["snapshot-paths"]
    ):
        cand_dir = (dbt_dir / rel_dir) if rel_dir else dbt_dir
        if not cand_dir.is_dir():
            continue
        for yml_path in sorted({*cand_dir.rglob("*.yml"), *cand_dir.rglob("*.yaml")}):
            resolved = yml_path.resolve()
            if not yml_path.is_file() or resolved in seen:
                continue
            rel_parts = resolved.relative_to(resolved_dbt).parts if resolved.is_relative_to(resolved_dbt) else ()
            if any(part in skip_dirs for part in rel_parts):
                continue
            if rel_dir == "" and len(rel_parts) == 1 and rel_parts[0] in ("dbt_project.yml", "profiles.yml", "packages.yml"):
                continue
            seen.add(resolved)
            files.append(yml_path)
    return files


def _collect_schema_tests(dbt_dir: Path) -> Counter[tuple[str, str]]:
    counts: Counter[tuple[str, str]] = Counter()
    for yml_path in _iter_schema_yaml_files(dbt_dir):
        loaded = yaml.safe_load(yml_path.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            continue
        for model in loaded.get("models") or []:
            if not isinstance(model, dict):
                continue
            model_name = str(model.get("name") or "")
            for test_entry in model.get("tests") or model.get("data_tests") or []:
                counts[(model_name, _canonical_test_key(test_entry))] += 1
            for col in model.get("columns") or []:
                if not isinstance(col, dict):
                    continue
                col_name = str(col.get("name") or "")
                for test_entry in col.get("tests") or col.get("data_tests") or []:
                    counts[(f"{model_name}.{col_name}", _canonical_test_key(test_entry))] += 1
    return counts


def _collect_schema_test_configs(dbt_dir: Path) -> dict[tuple[str, str], list[dict[str, Any]]]:
    configs: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for yml_path in _iter_schema_yaml_files(dbt_dir):
        loaded = yaml.safe_load(yml_path.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            continue
        for model in loaded.get("models") or []:
            if not isinstance(model, dict):
                continue
            model_name = str(model.get("name") or "")
            for test_entry in model.get("tests") or model.get("data_tests") or []:
                configs.setdefault((model_name, _canonical_test_key(test_entry)), []).append(
                    _normalize_test_config(test_entry)
                )
            for col in model.get("columns") or []:
                if not isinstance(col, dict):
                    continue
                col_name = str(col.get("name") or "")
                for test_entry in col.get("tests") or col.get("data_tests") or []:
                    configs.setdefault(
                        (f"{model_name}.{col_name}", _canonical_test_key(test_entry)), []
                    ).append(_normalize_test_config(test_entry))
    return configs


_NEUTRALIZED_TEST_RE = re.compile(
    r"\b(?:where|and)\s+(?:1\s*=\s*0|0\s*=\s*1|false|null\s+is\s+not\s+null)\b|\blimit\s+0\b",
    re.IGNORECASE,
)
_DBT_REF_SOURCE_RE = re.compile(r"\{\{\s*(?:ref|source)\s*\(", re.IGNORECASE)


class HiddenRegressionOracle(Protocol):
    """Out-of-workspace integrity & regression check run against shadow execution."""

    def verify(
        self,
        *,
        prod_workspace: Path,
        worktree_dir: Path,
        shadow_silver_dir: Path,
        shadow_duckdb_path: Path,
        changed_files: list[str],
        unified_diff: str,
    ) -> OracleVerdict: ...


class DefaultIntegrityOracle:
    """Default regression oracle guarding test preservation and optional metric reconciliation."""

    def __init__(
        self,
        reconciliation_rules: Sequence[MetricReconciliationRule] | None = None,
    ) -> None:
        self.reconciliation_rules: tuple[MetricReconciliationRule, ...] = (
            _DEFAULT_OPTIONAL_RECONCILIATION_RULES
            if reconciliation_rules is None
            else tuple(reconciliation_rules)
        )

    def verify(
        self,
        *,
        prod_workspace: Path,
        worktree_dir: Path,
        shadow_silver_dir: Path,
        shadow_duckdb_path: Path,
        changed_files: list[str],
        unified_diff: str,
    ) -> OracleVerdict:
        del changed_files, unified_diff
        reasons: list[str] = []
        metrics: dict[str, Any] = {}

        dbt_rel, prod_dbt = _resolve_dbt_subdir(prod_workspace)
        wt_dbt = (worktree_dir / dbt_rel) if dbt_rel else worktree_dir

        # 1. Singular dbt test files must not be deleted, emptied, or neutralized.
        test_rel_dirs = _read_dbt_subpaths(prod_dbt)["test-paths"]
        seen_test_files: set[Path] = set()
        for test_rel_dir in test_rel_dirs:
            prod_tests_dir = prod_dbt / test_rel_dir
            wt_tests_dir = wt_dbt / test_rel_dir
            if not prod_tests_dir.is_dir():
                continue
            for test_sql in sorted(prod_tests_dir.rglob("*.sql")):
                resolved_sql = test_sql.resolve()
                if not test_sql.is_file() or resolved_sql in seen_test_files:
                    continue
                seen_test_files.add(resolved_sql)
                rel_sql = test_sql.relative_to(prod_tests_dir)
                wt_sql = wt_tests_dir / rel_sql
                display_rel = f"{test_rel_dir}/{rel_sql.as_posix()}"
                if not wt_sql.is_file() or not wt_sql.read_text(encoding="utf-8").strip():
                    reasons.append(f"Singular dbt test deleted or emptied: {display_rel}")
                    continue
                prod_text = test_sql.read_text(encoding="utf-8")
                wt_text = wt_sql.read_text(encoding="utf-8")
                if " ".join(prod_text.split()) != " ".join(wt_text.split()):
                    if _NEUTRALIZED_TEST_RE.search(wt_text):
                        reasons.append(
                            f"Singular dbt test neutralized with falsifying condition: {display_rel}"
                        )
                    elif _DBT_REF_SOURCE_RE.search(prod_text) and not _DBT_REF_SOURCE_RE.search(wt_text):
                        reasons.append(
                            f"Singular dbt test stripped model/source refs: {display_rel}"
                        )

        # 2. Schema tests in models/**/*.yml must not be removed or weakened.
        prod_counts = _collect_schema_tests(prod_dbt)
        wt_counts = _collect_schema_tests(wt_dbt)
        prod_cfgs = _collect_schema_test_configs(prod_dbt)
        wt_cfgs = _collect_schema_test_configs(wt_dbt)
        for key, expected_cnt in prod_counts.items():
            if wt_counts.get(key, 0) < expected_cnt:
                reasons.append(
                    f"Schema dbt test removed on {key[0]} ({key[1]}): "
                    f"expected >= {expected_cnt}, found {wt_counts.get(key, 0)}"
                )
                continue
            p_list = prod_cfgs.get(key, [])
            w_list = wt_cfgs.get(key, [])
            for idx, w_cfg in enumerate(w_list):
                p_cfg = p_list[idx] if idx < len(p_list) else {}
                if w_cfg.get("enabled") is False and p_cfg.get("enabled") is not False:
                    reasons.append(f"Schema dbt test disabled on {key[0]} ({key[1]}).")
                if (
                    str(w_cfg.get("severity") or "").lower() == "warn"
                    and str(p_cfg.get("severity") or "").lower() != "warn"
                ):
                    reasons.append(f"Schema dbt test downgraded to warn on {key[0]} ({key[1]}).")
                if w_cfg.get("limit") == 0 and p_cfg.get("limit") != 0:
                    reasons.append(f"Schema dbt test neutralized with limit=0 on {key[0]} ({key[1]}).")
                if "where" in w_cfg and w_cfg.get("where") != p_cfg.get("where"):
                    reasons.append(
                        f"Schema dbt test weakened with custom where filter on {key[0]} ({key[1]})."
                    )

        # 3. Declarative metric reconciliation check if matching tables exist in DuckDB.
        if self.reconciliation_rules and shadow_duckdb_path.is_file():
            con = duckdb.connect(str(shadow_duckdb_path), read_only=True)
            try:
                tables = {
                    row[0]
                    for row in con.execute(
                        "SELECT table_name FROM information_schema.tables "
                        "WHERE table_schema = 'main'"
                    ).fetchall()
                }
                for rule in self.reconciliation_rules:
                    if rule.left_table not in tables or rule.right_table not in tables:
                        continue
                    q_lt = _quote_sql_ident(rule.left_table)
                    q_lc = _quote_sql_ident(rule.left_column)
                    q_rt = _quote_sql_ident(rule.right_table)
                    q_rc = _quote_sql_ident(rule.right_column)
                    left_total = float(
                        con.execute(f"SELECT COALESCE(SUM({q_lc}), 0.0) FROM {q_lt};").fetchone()[0]
                    )
                    right_total = float(
                        con.execute(f"SELECT COALESCE(SUM({q_rc}), 0.0) FROM {q_rt};").fetchone()[0]
                    )
                    metrics[f"{rule.left_table}_total"] = round(left_total, 4)
                    metrics[f"{rule.right_table}_total"] = round(right_total, 4)
                    if rule.require_positive and (left_total <= 0 or right_total <= 0):
                        reasons.append(
                            f"{rule.label} totals must be positive (fct={left_total}, dim={right_total})."
                        )
                    elif abs(left_total - right_total) > rule.tolerance:
                        reasons.append(
                            f"{rule.label} reconciliation drift exceeded ${rule.tolerance:.2f} "
                            f"(fct={left_total:.2f}, dim={right_total:.2f})."
                        )
            finally:
                con.close()

        return OracleVerdict(
            passed=not reasons,
            oracle_name="default_integrity",
            reasons=reasons,
            metrics=metrics,
        )


def execute_pipeline_subset(
    *,
    code_workspace: Path,
    raw_dir: Path,
    prod_silver_dir: Path,
    target_silver_dir: Path,
    target_duckdb_path: Path,
    changed_files: list[str],
    affected_spark_jobs: list[str],
    affected_dbt_models: list[str],
    spark_runner: SparkJobRunner,
    dbt_executor: DbtExecutor,
    full_refresh: bool = True,
    lineage_adapter: LineageAdapterBase | None = None,
    dbt_project_rel: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Path], bool, bool, str, str, list[str]]:
    """Run affected Spark jobs then dbt models/tests against target silver and DuckDB paths."""
    resolved_ws = code_workspace.resolve()
    lineage = lineage_adapter
    if lineage is None and resolved_ws.is_dir():
        lineage = LineageAdapter.from_workspace(resolved_ws)

    project = (
        discover_project(resolved_ws)
        if resolved_ws.is_dir()
        else ResolvedProject(repo_root=resolved_ws)
    )
    prod_paths = LakehousePaths.from_project(project).model_copy(
        update={
            "raw_dir": raw_dir,
            "prod_silver_dir": prod_silver_dir,
        }
    )
    spark_backend = SparkSubprocessBackend(
        project,
        lineage_adapter=lineage,
        spark_runner=spark_runner,
    )
    spark_backend.prod_paths = prod_paths

    affected = AffectedSubgraph(
        changed_paths=[p.replace("\\", "/").lstrip("./") for p in changed_files],
        spark_jobs=list(affected_spark_jobs),
        dbt_models=list(affected_dbt_models),
        affected_assets=[],
    )
    events_dir = getattr(spark_runner, "event_log_dir", None) or (
        target_silver_dir.parent / "spark_events"
    )
    spark_res = spark_backend._execute_in_dirs(
        code_workspace=resolved_ws,
        target_silver_dir=target_silver_dir,
        events_dir=events_dir,
        changed_files=changed_files,
        affected=affected,
        require_match=False,
    )
    if not spark_res.passed:
        return (
            spark_res.spark_runs,
            spark_res.silver_overrides,
            False,
            False,
            "",
            "",
            spark_res.failure_reasons,
        )

    eff_dbt_rel = (
        dbt_project_rel
        if dbt_project_rel is not None
        else (project.dbt_project_rel if project.dbt_project_dir is not None else None)
    )
    dbt_res = _run_dbt_subset(
        code_workspace=resolved_ws,
        prod_paths=prod_paths,
        silver_overrides=spark_res.silver_overrides,
        target_duckdb_path=target_duckdb_path,
        changed_files=changed_files,
        affected_dbt_models=affected_dbt_models,
        dbt_executor=dbt_executor,
        full_refresh=full_refresh,
        dbt_project_rel=eff_dbt_rel,
        spark_runs=spark_res.spark_runs,
    )
    assert dbt_res is not None
    return (
        dbt_res.spark_runs,
        dbt_res.silver_overrides,
        dbt_res.dbt_run_ok,
        dbt_res.dbt_test_ok,
        dbt_res.dbt_stdout,
        dbt_res.dbt_stderr,
        dbt_res.failure_reasons,
    )


class ShadowOverlayVerifier:
    """Verify patched pipeline code in a pluggable `ShadowBackend` without mutating production."""

    def __init__(
        self,
        prod_workspace: Path,
        runs_dir: Path,
        *,
        lineage_adapter: LineageAdapterBase | None = None,
        oracle: HiddenRegressionOracle | None = None,
        spark_runner: SparkJobRunner | None = None,
        dbt_executor: DbtExecutor | None = None,
        project: ResolvedProject | None = None,
        backend: ShadowBackend | None = None,
        teardown_after_verify: bool | None = None,
    ) -> None:
        self.prod_workspace = prod_workspace.resolve()
        self.runs_dir = runs_dir.resolve()
        self._project = project or discover_project(self.prod_workspace)
        self.lineage_adapter = lineage_adapter or LineageAdapter.from_project(self._project)
        self.oracle: HiddenRegressionOracle = oracle or DefaultIntegrityOracle()
        self._custom_spark_runner = spark_runner
        self._dbt_executor: DbtExecutor = dbt_executor or _default_dbt_executor
        self.backend: ShadowBackend = backend or create_shadow_backend(
            self._project,
            lineage_adapter=self.lineage_adapter,
            spark_runner=self._custom_spark_runner,
            dbt_executor=dbt_executor,
        )
        self._teardown_after_verify: bool = (
            not getattr(self.backend, "preserve_artifacts", False)
            if teardown_after_verify is None
            else teardown_after_verify
        )

    def teardown(self, ctx: ShadowContext) -> None:
        """Delegate cleanup of shadow resources to the configured `ShadowBackend`."""
        self.backend.teardown(ctx)

    def verify(self, run_id: str, worktree_dir: Path) -> ShadowVerificationResult:
        """Run affected assets in the backend's Shadow Overlay and validate via Oracle."""
        resolved_wt = worktree_dir.resolve()
        changed_files = RunWorktreeManager.changed_files(resolved_wt)
        if not changed_files:
            prod_paths = LakehousePaths.from_project(self._project)
            shadow_root = self.runs_dir / run_id / "shadow"
            db_name = (
                prod_paths.prod_duckdb_path.name
                if prod_paths.prod_duckdb_path.name.endswith(".duckdb")
                else "warehouse.duckdb"
            )
            include_silver = isinstance(
                self.backend, (SparkSubprocessBackend, CompositeShadowBackend)
            )
            include_duckdb = isinstance(
                self.backend, (DuckDBShadowBackend, CompositeShadowBackend)
            )
            return ShadowVerificationResult(
                passed=False,
                run_id=run_id,
                worktree_path=str(resolved_wt),
                shadow_silver_dir=str(shadow_root / "silver") if include_silver else "",
                shadow_duckdb_path=str(shadow_root / "mart" / db_name) if include_duckdb else "",
                backend_name=self.backend.backend_name,
                failure_reasons=["No modified or added files found in Run worktree."],
            )

        ctx = self.backend.prepare_shadow(run_id, resolved_wt, self.runs_dir)
        try:
            affected = self.lineage_adapter.affected_downstream(changed_files)
            unified_diff = RunWorktreeManager.unified_diff(resolved_wt)
            dbt_rel = self._project.dbt_project_rel if self._project.dbt_project_dir is not None else None
            full_refresh = not touches_incremental_logic(
                self.prod_workspace,
                resolved_wt,
                changed_files,
                dbt_project_rel=dbt_rel,
            )

            exec_res = self.backend.execute_subset(
                ctx,
                changed_files=changed_files,
                affected=affected,
                full_refresh=full_refresh,
            )
            failure_reasons = list(exec_res.failure_reasons)
            oracle_verdict: OracleVerdict | None = None
            data_diffs: list[DataDiffSummary] = []

            if not failure_reasons:
                data_diffs = self.backend.diff_against_prod(ctx)
                oracle_verdict = self.oracle.verify(
                    prod_workspace=self.prod_workspace,
                    worktree_dir=resolved_wt,
                    shadow_silver_dir=ctx.shadow_silver_dir
                    or (ctx.shadow_root / ".no_silver_backend"),
                    shadow_duckdb_path=ctx.shadow_duckdb_path
                    or (ctx.shadow_root / ".no_duckdb_backend"),
                    changed_files=changed_files,
                    unified_diff=unified_diff,
                )
                if not oracle_verdict.passed:
                    failure_reasons.extend(oracle_verdict.reasons)

            return ShadowVerificationResult(
                passed=not failure_reasons,
                run_id=run_id,
                worktree_path=str(resolved_wt),
                shadow_silver_dir=str(ctx.shadow_silver_dir or ""),
                shadow_duckdb_path=str(ctx.shadow_duckdb_path or ""),
                backend_name=ctx.backend_name,
                shadow_schema=ctx.shadow_schema,
                shadow_metadata=dict(ctx.metadata),
                changed_files=changed_files,
                affected_spark_jobs=affected.spark_jobs,
                affected_dbt_models=affected.dbt_models,
                spark_runs=exec_res.spark_runs,
                dbt_run_ok=exec_res.dbt_run_ok,
                dbt_test_ok=exec_res.dbt_test_ok,
                dbt_stdout=exec_res.dbt_stdout,
                dbt_stderr=exec_res.dbt_stderr,
                data_diffs=data_diffs,
                oracle_verdict=oracle_verdict,
                failure_reasons=failure_reasons,
            )
        finally:
            if self._teardown_after_verify:
                self.backend.teardown(ctx)


__all__ = [
    "DbtExecutor",
    "DefaultIntegrityOracle",
    "HiddenRegressionOracle",
    "MetricReconciliationRule",
    "ShadowOverlayVerifier",
    "_default_dbt_executor",
    "_discover_mart_diff_targets",
    "_infer_key_column",
    "_is_dbt_model_sql",
    "_normalize_dbt_rel_paths",
    "_read_dbt_subpaths",
    "_resolve_dbt_subdir",
    "capture_incremental_blocks",
    "execute_pipeline_subset",
    "extract_incremental_blocks",
    "order_spark_jobs",
    "touches_incremental_logic",
]

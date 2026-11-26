from __future__ import annotations

import re
import shutil
import subprocess
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol

import duckdb
import yaml

from src.adapters.base import DataDiffSummary, LakehousePaths, LineageAdapterBase
from src.adapters.duckdb_adapter import DuckDBAdapter
from src.adapters.lineage_adapter import LineageAdapter
from src.adapters.spark_adapter import SparkJobRunner
from src.core.rca import OracleVerdict, ShadowVerificationResult
from src.core.worktree import RunWorktreeManager

SPARK_JOB_ORDER: tuple[str, ...] = (
    "spark_jobs/build_silver_users.py",
    "spark_jobs/build_silver_exchange_rates.py",
    "spark_jobs/build_silver_orders.py",
)
SPARK_JOB_TO_TABLE: dict[str, str] = {
    "spark_jobs/build_silver_users.py": "silver_users",
    "spark_jobs/build_silver_exchange_rates.py": "silver_exchange_rates",
    "spark_jobs/build_silver_orders.py": "silver_orders",
}

DbtExecutor = Callable[
    [LakehousePaths, dict[str, Path], Path, list[str]],
    subprocess.CompletedProcess[str],
]


def _default_dbt_executor(
    dbt_paths: LakehousePaths,
    silver_overrides: dict[str, Path],
    mart_duckdb_path: Path,
    args: list[str],
) -> subprocess.CompletedProcess[str]:
    adapter = DuckDBAdapter(
        dbt_paths,
        silver_overrides=silver_overrides,
        mart_duckdb_path=mart_duckdb_path,
    )
    return adapter.run_dbt(args)


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


def _collect_schema_tests(dbt_dir: Path) -> Counter[tuple[str, str]]:
    counts: Counter[tuple[str, str]] = Counter()
    models_dir = dbt_dir / "models"
    if not models_dir.is_dir():
        return counts
    for yml_path in sorted(models_dir.rglob("*.yml")):
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
    models_dir = dbt_dir / "models"
    if not models_dir.is_dir():
        return configs
    for yml_path in sorted(models_dir.rglob("*.yml")):
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
    """Default regression oracle guarding test preservation and mart GMV conservation."""

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
        reasons: list[str] = []
        metrics: dict[str, Any] = {}

        prod_dbt = prod_workspace / "dbt_project"
        wt_dbt = worktree_dir / "dbt_project"

        # 1. Singular dbt test files must not be deleted, emptied, or neutralized.
        prod_tests_dir = prod_dbt / "tests"
        wt_tests_dir = wt_dbt / "tests"
        if prod_tests_dir.is_dir():
            for test_sql in sorted(prod_tests_dir.glob("*.sql")):
                wt_sql = wt_tests_dir / test_sql.name
                if not wt_sql.is_file() or not wt_sql.read_text(encoding="utf-8").strip():
                    reasons.append(f"Singular dbt test deleted or emptied: tests/{test_sql.name}")
                    continue
                prod_text = test_sql.read_text(encoding="utf-8")
                wt_text = wt_sql.read_text(encoding="utf-8")
                if " ".join(prod_text.split()) != " ".join(wt_text.split()):
                    if _NEUTRALIZED_TEST_RE.search(wt_text):
                        reasons.append(
                            f"Singular dbt test neutralized with falsifying condition: tests/{test_sql.name}"
                        )
                    elif _DBT_REF_SOURCE_RE.search(prod_text) and not _DBT_REF_SOURCE_RE.search(wt_text):
                        reasons.append(
                            f"Singular dbt test stripped model/source refs: tests/{test_sql.name}"
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
                if str(w_cfg.get("severity") or "").lower() == "warn" and str(p_cfg.get("severity") or "").lower() != "warn":
                    reasons.append(f"Schema dbt test downgraded to warn on {key[0]} ({key[1]}).")
                if w_cfg.get("limit") == 0 and p_cfg.get("limit") != 0:
                    reasons.append(f"Schema dbt test neutralized with limit=0 on {key[0]} ({key[1]}).")
                if "where" in w_cfg and w_cfg.get("where") != p_cfg.get("where"):
                    reasons.append(f"Schema dbt test weakened with custom where filter on {key[0]} ({key[1]}).")

        # 3. Mart GMV conservation check if both mart tables exist in DuckDB.
        if shadow_duckdb_path.is_file():
            con = duckdb.connect(str(shadow_duckdb_path), read_only=True)
            try:
                tables = {
                    row[0]
                    for row in con.execute(
                        "SELECT table_name FROM information_schema.tables "
                        "WHERE table_schema = 'main'"
                    ).fetchall()
                }
                if "fct_daily_gmv" in tables and "dim_user_ltv" in tables:
                    fct_total = float(
                        con.execute("SELECT COALESCE(SUM(net_gmv_usd), 0.0) FROM fct_daily_gmv;")
                        .fetchone()[0]
                    )
                    dim_total = float(
                        con.execute(
                            "SELECT COALESCE(SUM(lifetime_gmv_usd), 0.0) FROM dim_user_ltv;"
                        ).fetchone()[0]
                    )
                    metrics["fct_daily_gmv_total"] = round(fct_total, 4)
                    metrics["dim_user_ltv_total"] = round(dim_total, 4)
                    if fct_total <= 0 or dim_total <= 0:
                        reasons.append(
                            f"Mart GMV totals must be positive (fct={fct_total}, dim={dim_total})."
                        )
                    elif abs(fct_total - dim_total) > 0.01:
                        reasons.append(
                            f"Mart GMV reconciliation drift exceeded $0.01 "
                            f"(fct={fct_total:.2f}, dim={dim_total:.2f})."
                        )
            finally:
                con.close()

        return OracleVerdict(
            passed=not reasons,
            oracle_name="default_integrity",
            reasons=reasons,
            metrics=metrics,
        )


def order_spark_jobs(spark_jobs: list[str]) -> list[str]:
    """Sort affected Spark jobs in deterministic upstream-to-downstream order."""
    ordered = [job for job in SPARK_JOB_ORDER if job in spark_jobs]
    for job in spark_jobs:
        if job not in ordered:
            ordered.append(job)
    return ordered


_INCREMENTAL_BLOCK_RE = re.compile(
    r"\{%[-\s]*if\s+is_incremental\(\)\s*[-\s]*%\}(.*?)\{%[-\s]*endif\s*[-\s]*%\}",
    re.DOTALL | re.IGNORECASE,
)


def extract_incremental_blocks(text: str) -> list[str]:
    """Return normalized body snippets for every `{% if is_incremental() %}` block."""
    return [" ".join(match.group(1).split()) for match in _INCREMENTAL_BLOCK_RE.finditer(text)]


def touches_incremental_logic(
    prod_workspace: Path, worktree_dir: Path, changed_files: list[str]
) -> bool:
    """Return True if any changed dbt model alters content inside `is_incremental()`."""
    for rel in changed_files:
        norm = rel.replace("\\", "/").lstrip("./")
        if not (norm.startswith("dbt_project/models/") and norm.endswith(".sql")):
            continue
        prod_file = prod_workspace / norm
        wt_file = worktree_dir / norm
        prod_blocks = (
            extract_incremental_blocks(prod_file.read_text(encoding="utf-8"))
            if prod_file.is_file()
            else []
        )
        wt_blocks = (
            extract_incremental_blocks(wt_file.read_text(encoding="utf-8"))
            if wt_file.is_file()
            else []
        )
        if prod_blocks != wt_blocks:
            return True
    return False


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
) -> tuple[list[dict[str, Any]], dict[str, Path], bool, bool, str, str, list[str]]:
    """Run affected Spark jobs then dbt models/tests against target silver and DuckDB paths."""
    failure_reasons: list[str] = []
    spark_runs: list[dict[str, Any]] = []
    silver_overrides: dict[str, Path] = {}

    target_silver_dir.mkdir(parents=True, exist_ok=True)
    target_duckdb_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Execute affected PySpark jobs into target_silver_dir using upstream shadow-over-prod read.
    for job_rel in order_spark_jobs(affected_spark_jobs):
        script_path = code_workspace / job_rel
        if not script_path.is_file():
            failure_reasons.append(f"Affected Spark script missing: {job_rel}")
            continue
        args = ["--raw-dir", str(raw_dir), "--output-dir", str(target_silver_dir)]
        if job_rel == "spark_jobs/build_silver_orders.py":
            silver_read = (
                target_silver_dir if "silver_users" in silver_overrides else prod_silver_dir
            )
            args.extend(["--silver-read-dir", str(silver_read)])

        res = spark_runner.run_job(script_path, args=args)
        spark_runs.append(res.model_dump(mode="json"))
        if res.returncode != 0 or res.timed_out:
            err_tail = (res.stderr or res.stdout).strip().splitlines()[-3:]
            failure_reasons.append(
                f"Spark job {job_rel} failed (rc={res.returncode}): {' | '.join(err_tail)}"
            )
            return spark_runs, silver_overrides, False, False, "", "", failure_reasons

        if table_name := SPARK_JOB_TO_TABLE.get(job_rel):
            written = target_silver_dir / table_name
            if written.is_dir():
                silver_overrides[table_name] = written.resolve()

    # 2. Execute affected dbt run & dbt test inside code_workspace/dbt_project.
    touches_dbt = bool(affected_dbt_models) or any(
        path.startswith("dbt_project/") for path in changed_files
    )
    if not touches_dbt and not spark_runs:
        failure_reasons.append("Patched files do not map to any Spark job or dbt asset.")
        return spark_runs, silver_overrides, False, False, "", "", failure_reasons

    dbt_paths = LakehousePaths.from_workspace(code_workspace).model_copy(
        update={
            "raw_dir": raw_dir,
            "prod_silver_dir": prod_silver_dir,
            "prod_duckdb_path": target_duckdb_path,
        }
    )

    requires_full_dbt = any(
        path.startswith(("dbt_project/macros/", "dbt_project/tests/", "spark_jobs/spark_config.py"))
        or path.endswith((".yml", "profiles.yml", "dbt_project.yml"))
        for path in changed_files
    )
    run_args = ["run"]
    if full_refresh:
        run_args.append("--full-refresh")
    if not requires_full_dbt and affected_dbt_models:
        run_args.extend(["--select", *affected_dbt_models])
    run_proc = dbt_executor(dbt_paths, silver_overrides, target_duckdb_path, run_args)
    dbt_run_ok = run_proc.returncode == 0
    if not dbt_run_ok:
        err_msg = (run_proc.stderr or run_proc.stdout).strip()[-800:]
        failure_reasons.append(f"dbt run failed (rc={run_proc.returncode}): {err_msg}")
        return (
            spark_runs,
            silver_overrides,
            False,
            False,
            run_proc.stdout,
            run_proc.stderr,
            failure_reasons,
        )

    test_proc = dbt_executor(dbt_paths, silver_overrides, target_duckdb_path, ["test"])
    dbt_test_ok = test_proc.returncode == 0
    combined_stdout = "\n".join(part for part in (run_proc.stdout, test_proc.stdout) if part)
    combined_stderr = "\n".join(part for part in (run_proc.stderr, test_proc.stderr) if part)
    if not dbt_test_ok:
        err_msg = (test_proc.stderr or test_proc.stdout).strip()[-800:]
        failure_reasons.append(f"dbt test failed (rc={test_proc.returncode}): {err_msg}")

    return (
        spark_runs,
        silver_overrides,
        dbt_run_ok,
        dbt_test_ok,
        combined_stdout,
        combined_stderr,
        failure_reasons,
    )


class ShadowOverlayVerifier:
    """Verify patched pipeline code in a per-Run Shadow Overlay without mutating production."""

    def __init__(
        self,
        prod_workspace: Path,
        runs_dir: Path,
        *,
        lineage_adapter: LineageAdapterBase | None = None,
        oracle: HiddenRegressionOracle | None = None,
        spark_runner: SparkJobRunner | None = None,
        dbt_executor: DbtExecutor | None = None,
    ) -> None:
        self.prod_workspace = prod_workspace.resolve()
        self.runs_dir = runs_dir.resolve()
        self.lineage_adapter = lineage_adapter or LineageAdapter.from_workspace(
            self.prod_workspace
        )
        self.oracle: HiddenRegressionOracle = oracle or DefaultIntegrityOracle()
        self._custom_spark_runner = spark_runner
        self._dbt_executor: DbtExecutor = dbt_executor or _default_dbt_executor

    def _compute_data_diffs(
        self, prod_duckdb_path: Path, shadow_duckdb_path: Path
    ) -> list[DataDiffSummary]:
        if not (prod_duckdb_path.is_file() and shadow_duckdb_path.is_file()):
            return []
        adapter = DuckDBAdapter(LakehousePaths.from_workspace(self.prod_workspace))
        diffs: list[DataDiffSummary] = []
        for table_name, key_col in (
            ("fct_daily_gmv", "order_date"),
            ("dim_user_ltv", "user_id"),
        ):
            try:
                diffs.append(
                    adapter.data_diff(
                        prod_db=prod_duckdb_path,
                        shadow_db=shadow_duckdb_path,
                        table_name=table_name,
                        key_column=key_col,
                    )
                )
            except (KeyError, duckdb.Error):
                continue
        return diffs

    def verify(self, run_id: str, worktree_dir: Path) -> ShadowVerificationResult:
        """Run affected Spark + dbt DAG in isolated shadow storage and validate via Oracle."""
        resolved_wt = worktree_dir.resolve()
        shadow_root = self.runs_dir / run_id / "shadow"
        shadow_silver_dir = shadow_root / "silver"
        shadow_mart_dir = shadow_root / "mart"
        shadow_duckdb_path = shadow_mart_dir / "warehouse.duckdb"
        shadow_events_dir = shadow_root / "spark_events"

        if shadow_root.exists():
            shutil.rmtree(shadow_root)
        shadow_silver_dir.mkdir(parents=True, exist_ok=True)
        shadow_mart_dir.mkdir(parents=True, exist_ok=True)
        shadow_events_dir.mkdir(parents=True, exist_ok=True)

        changed_files = RunWorktreeManager.changed_files(resolved_wt)
        if not changed_files:
            return ShadowVerificationResult(
                passed=False,
                run_id=run_id,
                worktree_path=str(resolved_wt),
                shadow_silver_dir=str(shadow_silver_dir),
                shadow_duckdb_path=str(shadow_duckdb_path),
                failure_reasons=["No modified or added files found in Run worktree."],
            )

        prod_paths = LakehousePaths.from_workspace(self.prod_workspace)
        if prod_paths.prod_duckdb_path.is_file():
            shutil.copy2(prod_paths.prod_duckdb_path, shadow_duckdb_path)

        affected = self.lineage_adapter.affected_downstream(changed_files)
        unified_diff = RunWorktreeManager.unified_diff(resolved_wt)
        spark_runner = self._custom_spark_runner or SparkJobRunner(
            event_log_dir=shadow_events_dir
        )

        (
            spark_runs,
            _silver_overrides,
            dbt_run_ok,
            dbt_test_ok,
            dbt_stdout,
            dbt_stderr,
            failure_reasons,
        ) = execute_pipeline_subset(
            code_workspace=resolved_wt,
            raw_dir=prod_paths.raw_dir,
            prod_silver_dir=prod_paths.prod_silver_dir,
            target_silver_dir=shadow_silver_dir,
            target_duckdb_path=shadow_duckdb_path,
            changed_files=changed_files,
            affected_spark_jobs=affected.spark_jobs,
            affected_dbt_models=affected.dbt_models,
            spark_runner=spark_runner,
            dbt_executor=self._dbt_executor,
            full_refresh=not touches_incremental_logic(
                self.prod_workspace, resolved_wt, changed_files
            ),
        )

        oracle_verdict: OracleVerdict | None = None
        data_diffs: list[DataDiffSummary] = []

        if not failure_reasons:
            data_diffs = self._compute_data_diffs(
                prod_paths.prod_duckdb_path, shadow_duckdb_path
            )
            oracle_verdict = self.oracle.verify(
                prod_workspace=self.prod_workspace,
                worktree_dir=resolved_wt,
                shadow_silver_dir=shadow_silver_dir,
                shadow_duckdb_path=shadow_duckdb_path,
                changed_files=changed_files,
                unified_diff=unified_diff,
            )
            if not oracle_verdict.passed:
                failure_reasons.extend(oracle_verdict.reasons)

        return ShadowVerificationResult(
            passed=not failure_reasons,
            run_id=run_id,
            worktree_path=str(resolved_wt),
            shadow_silver_dir=str(shadow_silver_dir),
            shadow_duckdb_path=str(shadow_duckdb_path),
            changed_files=changed_files,
            affected_spark_jobs=affected.spark_jobs,
            affected_dbt_models=affected.dbt_models,
            spark_runs=spark_runs,
            dbt_run_ok=dbt_run_ok,
            dbt_test_ok=dbt_test_ok,
            dbt_stdout=dbt_stdout,
            dbt_stderr=dbt_stderr,
            data_diffs=data_diffs,
            oracle_verdict=oracle_verdict,
            failure_reasons=failure_reasons,
        )

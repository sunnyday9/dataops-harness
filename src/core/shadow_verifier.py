from __future__ import annotations

import re
import shutil
import subprocess
from collections import Counter
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Protocol

import duckdb
import yaml
from pydantic import BaseModel

from src.adapters.base import DataDiffSummary, LakehousePaths, LineageAdapterBase, LineageNode
from src.adapters.duckdb_adapter import DuckDBAdapter
from src.adapters.lineage_adapter import LineageAdapter
from src.adapters.spark_adapter import SparkJobRunner
from src.core.project import ResolvedProject, _read_dbt_project_yml, discover_project
from src.core.rca import OracleVerdict, ShadowVerificationResult
from src.core.worktree import RunWorktreeManager

DbtExecutor = Callable[
    [LakehousePaths, dict[str, Path], Path, list[str]],
    subprocess.CompletedProcess[str],
]


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


def _resolve_dbt_subdir(root: Path, dbt_project_rel: str | None = None) -> tuple[str, Path]:
    if dbt_project_rel is not None:
        cleaned = dbt_project_rel.strip("/").replace("\\", "/")
        return cleaned, ((root / cleaned).resolve() if cleaned else root.resolve())
    if (root / "dbt_project.yml").is_file():
        return "", root.resolve()
    if root.is_dir():
        try:
            project = discover_project(root)
            if project.dbt_project_dir is not None:
                return project.dbt_project_rel, project.dbt_project_dir
        except FileNotFoundError:
            pass
    if (root / "dbt_project").is_dir():
        return "dbt_project", (root / "dbt_project").resolve()
    return "", root.resolve()


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


def _normalize_dbt_rel_paths(raw: Any, defaults: Sequence[str]) -> list[str]:
    entries = raw if isinstance(raw, list) else defaults
    normalized: list[str] = []
    seen: set[str] = set()
    for item in entries:
        cleaned = str(item).strip().replace("\\", "/")
        if not cleaned or cleaned.startswith(("/", "~")):
            continue
        while cleaned.startswith("./"):
            cleaned = cleaned[2:]
        cleaned = cleaned.strip("/")
        if not cleaned or any(seg == ".." for seg in cleaned.split("/")):
            continue
        if cleaned not in seen:
            normalized.append(cleaned)
            seen.add(cleaned)
    return normalized or list(defaults)


def _read_dbt_subpaths(dbt_dir: Path) -> dict[str, list[str]]:
    dbt_cfg = _read_dbt_project_yml(dbt_dir)
    raw_models = dbt_cfg.get("model-paths")
    if not isinstance(raw_models, list):
        raw_models = dbt_cfg.get("source-paths")
    return {
        "model-paths": _normalize_dbt_rel_paths(raw_models, ["models"]),
        "test-paths": _normalize_dbt_rel_paths(dbt_cfg.get("test-paths"), ["tests"]),
        "macro-paths": _normalize_dbt_rel_paths(dbt_cfg.get("macro-paths"), ["macros"]),
        "seed-paths": _normalize_dbt_rel_paths(dbt_cfg.get("seed-paths"), ["seeds"]),
        "snapshot-paths": _normalize_dbt_rel_paths(dbt_cfg.get("snapshot-paths"), ["snapshots"]),
    }


def _iter_schema_yaml_files(dbt_dir: Path) -> list[Path]:
    subpaths = _read_dbt_subpaths(dbt_dir)
    files: list[Path] = []
    seen: set[Path] = set()
    for rel_dir in (
        subpaths["model-paths"] + subpaths["seed-paths"] + subpaths["snapshot-paths"]
    ):
        cand_dir = dbt_dir / rel_dir
        if not cand_dir.is_dir():
            continue
        for yml_path in sorted({*cand_dir.rglob("*.yml"), *cand_dir.rglob("*.yaml")}):
            resolved = yml_path.resolve()
            if yml_path.is_file() and resolved not in seen:
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
                if str(w_cfg.get("severity") or "").lower() == "warn" and str(p_cfg.get("severity") or "").lower() != "warn":
                    reasons.append(f"Schema dbt test downgraded to warn on {key[0]} ({key[1]}).")
                if w_cfg.get("limit") == 0 and p_cfg.get("limit") != 0:
                    reasons.append(f"Schema dbt test neutralized with limit=0 on {key[0]} ({key[1]}).")
                if "where" in w_cfg and w_cfg.get("where") != p_cfg.get("where"):
                    reasons.append(f"Schema dbt test weakened with custom where filter on {key[0]} ({key[1]}).")

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


def order_spark_jobs(
    spark_jobs: Sequence[str],
    lineage_adapter: LineageAdapterBase | None = None,
) -> list[str]:
    """Sort affected Spark jobs in deterministic upstream-to-downstream lineage order."""
    if not spark_jobs:
        return []
    ordered_fn = getattr(lineage_adapter, "ordered_spark_jobs", None) if lineage_adapter else None
    if callable(ordered_fn):
        return list(ordered_fn(spark_jobs))

    all_nodes_fn = getattr(lineage_adapter, "all_nodes", None) if lineage_adapter else None
    if lineage_adapter is not None and callable(all_nodes_fn):
        nodes = all_nodes_fn()
        if isinstance(nodes, dict):
            path_to_asset = {
                node.file_path: name
                for name, node in nodes.items()
                if node.engine == "spark" and node.file_path
            }
            topo_assets = lineage_adapter.topological_order(
                [path_to_asset[j] for j in spark_jobs if j in path_to_asset]
            )
            ordered = [
                nodes[asset].file_path
                for asset in topo_assets
                if asset in nodes and nodes[asset].file_path in spark_jobs
            ]
            for job in sorted(spark_jobs):
                if job not in ordered:
                    ordered.append(job)
            return ordered

    return sorted(dict.fromkeys(spark_jobs))


_INCREMENTAL_BLOCK_RE = re.compile(
    r"\{%[-\s]*if\s+is_incremental\(\)\s*[-\s]*%\}(.*?)\{%[-\s]*endif\s*[-\s]*%\}",
    re.DOTALL | re.IGNORECASE,
)


def extract_incremental_blocks(text: str) -> list[str]:
    """Return normalized body snippets for every `{% if is_incremental() %}` block."""
    return [" ".join(match.group(1).split()) for match in _INCREMENTAL_BLOCK_RE.finditer(text)]


def _is_dbt_model_sql(
    norm_path: str,
    dbt_project_rel: str | None,
    model_paths: Sequence[str] | None = None,
) -> bool:
    if not norm_path.endswith(".sql"):
        return False
    if dbt_project_rel is not None:
        cleaned = dbt_project_rel.strip("/").replace("\\", "/")
        prefix = f"{cleaned}/" if cleaned else ""
        if model_paths is not None:
            return any(norm_path.startswith(f"{prefix}{mp}/") for mp in model_paths)
        if not norm_path.startswith(prefix):
            return False
        rel_in_dbt = norm_path[len(prefix) :]
        return not rel_in_dbt.startswith(("tests/", "macros/", "target/", "logs/"))
    return norm_path.startswith("models/") or "/models/" in norm_path


def touches_incremental_logic(
    prod_workspace: Path,
    worktree_dir: Path,
    changed_files: list[str],
    *,
    dbt_project_rel: str | None = None,
) -> bool:
    """Return True if any changed dbt model alters content inside `is_incremental()`."""
    effective_rel = dbt_project_rel
    model_paths: list[str] | None = None
    if effective_rel is not None:
        cleaned_rel = effective_rel.strip("/").replace("\\", "/")
        for base in (prod_workspace, worktree_dir):
            cand_dir = (base / cleaned_rel) if cleaned_rel else base
            if (cand_dir / "dbt_project.yml").is_file():
                model_paths = _read_dbt_subpaths(cand_dir)["model-paths"]
                break
    else:
        for base in (prod_workspace, worktree_dir):
            found_rel, found_dir = _resolve_dbt_subdir(base)
            if (found_dir / "dbt_project.yml").is_file():
                effective_rel = found_rel
                model_paths = _read_dbt_subpaths(found_dir)["model-paths"]
                break

    for rel in changed_files:
        norm = rel.replace("\\", "/").lstrip("./")
        if not _is_dbt_model_sql(norm, effective_rel, model_paths=model_paths):
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


def _infer_key_column(node: LineageNode) -> str | None:
    for test in node.tests:
        if test.test_type == "unique" and test.column_name:
            return test.column_name
    if node.partition_columns:
        return node.partition_columns[0]
    for col in node.columns:
        if col == "id" or col.endswith(("_id", "_date")):
            return col
    return node.columns[0] if node.columns else None


def _discover_mart_diff_targets(lineage: LineageAdapterBase | None) -> list[tuple[str, str]]:
    if lineage is None:
        return []
    all_nodes_fn = getattr(lineage, "all_nodes", None)
    if not callable(all_nodes_fn):
        return []
    nodes = all_nodes_fn()
    if not isinstance(nodes, dict):
        return []
    targets: list[tuple[str, str]] = []
    for name in sorted(nodes):
        node = nodes[name]
        if node.engine == "dbt" and node.layer == "mart":
            key_col = _infer_key_column(node)
            if key_col:
                targets.append((node.name, key_col))
    return targets


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
    failure_reasons: list[str] = []
    spark_runs: list[dict[str, Any]] = []
    silver_overrides: dict[str, Path] = {}

    target_silver_dir.mkdir(parents=True, exist_ok=True)
    target_duckdb_path.parent.mkdir(parents=True, exist_ok=True)

    lineage = lineage_adapter
    if lineage is None and code_workspace.is_dir():
        lineage = LineageAdapter.from_workspace(code_workspace)

    all_nodes_fn = getattr(lineage, "all_nodes", None) if lineage is not None else None
    all_nodes: dict[str, LineageNode] = all_nodes_fn() if callable(all_nodes_fn) else {}
    job_to_table = {
        node.file_path: name
        for name, node in all_nodes.items()
        if node.engine == "spark" and node.file_path
    }

    # 1. Execute affected PySpark jobs into target_silver_dir using upstream shadow-over-prod read.
    for job_rel in order_spark_jobs(affected_spark_jobs, lineage_adapter=lineage):
        script_path = code_workspace / job_rel
        if not script_path.is_file():
            failure_reasons.append(f"Affected Spark script missing: {job_rel}")
            continue
        args = ["--raw-dir", str(raw_dir), "--output-dir", str(target_silver_dir)]
        table_name = job_to_table.get(job_rel)
        node = all_nodes.get(table_name) if table_name else None
        upstream_spark_tables = [
            up
            for up in (node.upstream if node is not None else [])
            if up in all_nodes and all_nodes[up].engine == "spark"
        ]
        if upstream_spark_tables:
            has_shadow_upstream = any(up in silver_overrides for up in upstream_spark_tables)
            if has_shadow_upstream:
                for up in upstream_spark_tables:
                    if up in silver_overrides:
                        continue
                    prod_up = prod_silver_dir / up
                    tgt_up = target_silver_dir / up
                    if prod_up.exists() and not tgt_up.exists():
                        try:
                            tgt_up.symlink_to(prod_up.resolve())
                        except OSError:
                            if prod_up.is_dir():
                                shutil.copytree(prod_up, tgt_up, dirs_exist_ok=True)
                            else:
                                shutil.copy2(prod_up, tgt_up)
                silver_read = target_silver_dir
            else:
                silver_read = prod_silver_dir
            args.extend(["--silver-read-dir", str(silver_read)])

        res = spark_runner.run_job(script_path, args=args)
        spark_runs.append(res.model_dump(mode="json"))
        if res.returncode != 0 or res.timed_out:
            err_tail = (res.stderr or res.stdout).strip().splitlines()[-3:]
            failure_reasons.append(
                f"Spark job {job_rel} failed (rc={res.returncode}): {' | '.join(err_tail)}"
            )
            return spark_runs, silver_overrides, False, False, "", "", failure_reasons

        if table_name:
            written = target_silver_dir / table_name
            if written.is_dir():
                silver_overrides[table_name] = written.resolve()

    resolved_dbt_rel, resolved_dbt_dir = _resolve_dbt_subdir(code_workspace, dbt_project_rel)
    has_dbt_project = (resolved_dbt_dir / "dbt_project.yml").is_file()
    dbt_prefix = f"{resolved_dbt_rel}/" if resolved_dbt_rel else ""
    subpaths = _read_dbt_subpaths(resolved_dbt_dir)
    model_prefixes = tuple(f"{dbt_prefix}{p}/" for p in subpaths["model-paths"])
    test_prefixes = tuple(f"{dbt_prefix}{p}/" for p in subpaths["test-paths"])
    macro_prefixes = tuple(f"{dbt_prefix}{p}/" for p in subpaths["macro-paths"])
    seed_prefixes = tuple(f"{dbt_prefix}{p}/" for p in subpaths["seed-paths"])
    snapshot_prefixes = tuple(f"{dbt_prefix}{p}/" for p in subpaths["snapshot-paths"])
    all_dbt_prefixes = (
        model_prefixes + test_prefixes + macro_prefixes + seed_prefixes + snapshot_prefixes
    )
    dbt_config_files = (f"{dbt_prefix}dbt_project.yml", f"{dbt_prefix}profiles.yml")
    norm_changed = [path.replace("\\", "/").lstrip("./") for path in changed_files]

    # 2. Execute affected dbt seed/run/test inside code_workspace's dbt project.
    touches_dbt = bool(affected_dbt_models) or any(
        (path.startswith(dbt_prefix) if dbt_prefix else False)
        or path.startswith(all_dbt_prefixes)
        or path in dbt_config_files
        for path in norm_changed
    )
    if not has_dbt_project and not touches_dbt and dbt_executor is _default_dbt_executor:
        if spark_runs:
            return spark_runs, silver_overrides, True, True, "", "", failure_reasons
        failure_reasons.append("Patched files do not map to any Spark job or dbt asset.")
        return spark_runs, silver_overrides, False, False, "", "", failure_reasons
    if not touches_dbt and not spark_runs:
        failure_reasons.append("Patched files do not map to any Spark job or dbt asset.")
        return spark_runs, silver_overrides, False, False, "", "", failure_reasons

    dbt_paths = LakehousePaths.from_workspace(code_workspace).model_copy(
        update={
            "raw_dir": raw_dir,
            "prod_silver_dir": prod_silver_dir,
            "prod_duckdb_path": target_duckdb_path,
            "dbt_project_dir": resolved_dbt_dir,
        }
    )

    touches_seeds = any(path.startswith(seed_prefixes) for path in norm_changed)
    seed_stdout = ""
    seed_stderr = ""
    if touches_seeds:
        seed_args = ["seed", "--full-refresh"] if full_refresh else ["seed"]
        seed_proc = dbt_executor(dbt_paths, silver_overrides, target_duckdb_path, seed_args)
        seed_stdout = seed_proc.stdout
        seed_stderr = seed_proc.stderr
        if seed_proc.returncode != 0:
            err_msg = (seed_proc.stderr or seed_proc.stdout).strip()[-800:]
            failure_reasons.append(f"dbt seed failed (rc={seed_proc.returncode}): {err_msg}")
            return (
                spark_runs,
                silver_overrides,
                False,
                False,
                seed_stdout,
                seed_stderr,
                failure_reasons,
            )

    requires_full_dbt = any(
        path.startswith(macro_prefixes + test_prefixes)
        or path.endswith((".yml", ".yaml", "profiles.yml", "dbt_project.yml", "spark_config.py"))
        for path in norm_changed
    )
    inferred_models: list[str] = []
    if not affected_dbt_models:
        for path in norm_changed:
            if path.endswith(".sql") and path.startswith(model_prefixes):
                selector = f"{Path(path).stem}+"
                if selector not in inferred_models:
                    inferred_models.append(selector)

    select_targets = list(affected_dbt_models) or inferred_models
    run_args = ["run"]
    if full_refresh:
        run_args.append("--full-refresh")
    if not requires_full_dbt and select_targets:
        run_args.extend(["--select", *select_targets])
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
            "\n".join(part for part in (seed_stdout, run_proc.stdout) if part),
            "\n".join(part for part in (seed_stderr, run_proc.stderr) if part),
            failure_reasons,
        )

    test_proc = dbt_executor(dbt_paths, silver_overrides, target_duckdb_path, ["test"])
    dbt_test_ok = test_proc.returncode == 0
    combined_stdout = "\n".join(part for part in (seed_stdout, run_proc.stdout, test_proc.stdout) if part)
    combined_stderr = "\n".join(part for part in (seed_stderr, run_proc.stderr, test_proc.stderr) if part)
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
        project: ResolvedProject | None = None,
    ) -> None:
        self.prod_workspace = prod_workspace.resolve()
        self.runs_dir = runs_dir.resolve()
        self._project = project or discover_project(self.prod_workspace)
        self.lineage_adapter = lineage_adapter or LineageAdapter.from_project(self._project)
        self.oracle: HiddenRegressionOracle = oracle or DefaultIntegrityOracle()
        self._custom_spark_runner = spark_runner
        self._dbt_executor: DbtExecutor = dbt_executor or _default_dbt_executor

    def _compute_data_diffs(
        self, prod_duckdb_path: Path, shadow_duckdb_path: Path
    ) -> list[DataDiffSummary]:
        if not (prod_duckdb_path.is_file() and shadow_duckdb_path.is_file()):
            return []
        adapter = DuckDBAdapter(LakehousePaths.from_project(self._project))
        diffs: list[DataDiffSummary] = []
        for table_name, key_col in _discover_mart_diff_targets(self.lineage_adapter):
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
        prod_paths = LakehousePaths.from_project(self._project)
        shadow_root = self.runs_dir / run_id / "shadow"
        shadow_silver_dir = shadow_root / "silver"
        shadow_mart_dir = shadow_root / "mart"
        db_name = (
            prod_paths.prod_duckdb_path.name
            if prod_paths.prod_duckdb_path.name.endswith(".duckdb")
            else "warehouse.duckdb"
        )
        shadow_duckdb_path = shadow_mart_dir / db_name
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
                self.prod_workspace,
                resolved_wt,
                changed_files,
                dbt_project_rel=self._project.dbt_project_rel,
            ),
            lineage_adapter=self.lineage_adapter,
            dbt_project_rel=self._project.dbt_project_rel,
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

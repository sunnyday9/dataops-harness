from __future__ import annotations

import errno
import fcntl
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import duckdb
from pydantic import BaseModel, Field

from src.adapters.base import (
    AffectedSubgraph,
    DataDiffSummary,
    LakehousePaths,
    LineageAdapterBase,
    LineageNode,
)
from src.adapters.duckdb_adapter import DuckDBAdapter, compare_duckdb_relations
from src.adapters.spark_adapter import SparkJobRunner
from src.core.project import (
    ResolvedProject,
    _discover_profile_hints,
    _read_dbt_project_yml,
    discover_project,
    normalize_sql_dialect,
)
from src.core.rca import ShadowVerificationResult

DbtExecutor = Callable[
    [LakehousePaths, dict[str, Path], Path, list[str]],
    subprocess.CompletedProcess[str],
]

ShadowBackendFactory = Callable[..., "ShadowBackend"]

_INCREMENTAL_BLOCK_RE = re.compile(
    r"\{%[-\s]*if\s+is_incremental\(\)\s*[-\s]*%\}(.*?)\{%[-\s]*endif\s*[-\s]*%\}",
    re.DOTALL | re.IGNORECASE,
)


class ShadowContext(BaseModel):
    """Per-Run isolation state prepared by a `ShadowBackend` before subset execution."""

    run_id: str
    backend_name: str
    worktree_dir: Path
    shadow_root: Path
    shadow_silver_dir: Path | None = None
    shadow_duckdb_path: Path | None = None
    shadow_schema: str | None = None
    spark_events_dir: Path | None = None
    silver_overrides: dict[str, Path] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ShadowExecutionResult(BaseModel):
    """Unified execution outcome returned by a `ShadowBackend` or production replay."""

    spark_runs: list[dict[str, Any]] = Field(default_factory=list)
    silver_overrides: dict[str, Path] = Field(default_factory=dict)
    dbt_run_ok: bool = False
    dbt_test_ok: bool = False
    dbt_stdout: str = ""
    dbt_stderr: str = ""
    failure_reasons: list[str] = Field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.failure_reasons


class ProductionSnapshot(BaseModel):
    """Rollback snapshot captured before local in-place production replay."""

    run_id: str
    backend_name: str
    duckdb_backup_path: Path | None = None
    duckdb_wal_backup_path: Path | None = None
    duckdb_existed: bool = False
    silver_backup_dir: Path | None = None
    silver_existed: bool = False
    silver_backed_up: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


@runtime_checkable
class ShadowBackend(Protocol):
    """Pluggable isolation strategy preparing, executing, diffing, and cleaning a Shadow Overlay."""

    backend_name: str

    def prepare_shadow(self, run_id: str, worktree_dir: Path, runs_dir: Path) -> ShadowContext:
        """Prepare isolated shadow storage or schema state for `run_id`."""
        ...

    def execute_subset(
        self,
        ctx: ShadowContext,
        *,
        changed_files: list[str],
        affected: AffectedSubgraph,
        full_refresh: bool = True,
    ) -> ShadowExecutionResult:
        """Execute only impacted pipeline assets against the Run's Shadow Overlay."""
        ...

    def diff_against_prod(self, ctx: ShadowContext) -> list[DataDiffSummary]:
        """Compute key- and row-level data diffs between shadow outputs and production."""
        ...

    def teardown(self, ctx: ShadowContext) -> None:
        """Idempotently clean up temporary shadow storage or schemas for `ctx`."""
        ...


@runtime_checkable
class ProductionReplayBackend(Protocol):
    """Optional backend capability supporting local sandbox rollback-safe Promotion replay."""

    def snapshot_production(
        self,
        run_id: str,
        runs_dir: Path,
        *,
        affected: AffectedSubgraph,
    ) -> ProductionSnapshot:
        """Capture an atomic backup of local production artifacts before Promotion replay."""
        ...

    def replay_production(
        self,
        *,
        changed_files: list[str],
        affected: AffectedSubgraph,
        full_refresh: bool = True,
        run_id: str,
        runs_dir: Path,
    ) -> ShadowExecutionResult:
        """Replay affected assets against production storage after git Promotion commit."""
        ...

    def restore_production(self, snapshot: ProductionSnapshot) -> None:
        """Restore production artifacts atomically from `snapshot` after replay failure."""
        ...

    def cleanup_snapshot(self, snapshot: ProductionSnapshot) -> None:
        """Remove temporary backup files in `snapshot` once Promotion finishes or rolls back."""
        ...


@runtime_checkable
class ShadowVerifierProtocol(Protocol):
    """Verifier contract consumed by `StopGateHook` in Fix Mode."""

    runs_dir: Path

    def verify(self, run_id: str, worktree_dir: Path) -> ShadowVerificationResult:
        """Run Shadow Overlay verification on `worktree_dir` and return a structured verdict."""
        ...


def _quote_ident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _sql_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _snapshot_lock_path(db_path: Path) -> Path:
    resolved = db_path.resolve()
    digest = hashlib.sha256(str(resolved).encode("utf-8")).hexdigest()[:24]
    lock_dir = Path(tempfile.gettempdir()) / "dataops-harness-locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    return lock_dir / f"duckdb-{digest}.lock"


_LOCK_STATE = threading.local()


def _thread_locks() -> dict[Path, list[Any]]:
    held = getattr(_LOCK_STATE, "held", None)
    if held is None:
        held = {}
        _LOCK_STATE.held = held
    return held


def _acquire_flock(
    fd: int,
    db_path: Path,
    *,
    exclusive: bool,
    timeout_seconds: float,
) -> None:
    mode = (fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH) | fcntl.LOCK_NB
    deadline = time.monotonic() + max(0.0, timeout_seconds)
    while True:
        try:
            fcntl.flock(fd, mode)
            return
        except OSError as exc:
            if exc.errno not in (errno.EACCES, errno.EAGAIN, errno.EWOULDBLOCK):
                raise
            if time.monotonic() >= deadline:
                lock_kind = "exclusive" if exclusive else "shared"
                raise TimeoutError(
                    f"Timed out waiting for {lock_kind} DuckDB snapshot lock on "
                    f"{db_path} after {timeout_seconds:.2f}s"
                ) from exc
            time.sleep(0.02)


@contextmanager
def duckdb_snapshot_lock(
    db_path: Path,
    *,
    exclusive: bool = False,
    timeout_seconds: float = 30.0,
) -> Iterator[Path]:
    """Acquire a re-entrant POSIX `flock` coordinating DuckDB snapshot reads vs. writes."""
    lock_path = _snapshot_lock_path(db_path)
    held = _thread_locks()
    existing = held.get(lock_path)
    if existing is not None:
        fd, _depth, cur_exclusive = existing
        if exclusive and not cur_exclusive:
            _acquire_flock(fd, db_path, exclusive=True, timeout_seconds=timeout_seconds)
            existing[2] = True
        existing[1] += 1
        try:
            yield lock_path
        finally:
            existing[1] -= 1
        return

    fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR, 0o600)
    acquired = False
    try:
        _acquire_flock(fd, db_path, exclusive=exclusive, timeout_seconds=timeout_seconds)
        acquired = True
        held[lock_path] = [fd, 1, exclusive]
        yield lock_path
    finally:
        entry = held.get(lock_path)
        if entry is not None and entry[0] == fd:
            entry[1] -= 1
            if entry[1] <= 0:
                held.pop(lock_path, None)
            else:
                acquired = False
        if lock_path not in held:
            if acquired:
                try:
                    fcntl.flock(fd, fcntl.LOCK_UN)
                except OSError:
                    pass
            os.close(fd)


def _atomic_copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=dst.parent, prefix=f".{dst.name}.", suffix=".tmp")
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        shutil.copy2(src, tmp_path)
        os.replace(tmp_path, dst)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def copy_duckdb_snapshot(
    src_db: Path,
    dst_db: Path,
    *,
    timeout_seconds: float = 30.0,
) -> bool:
    """Atomically snapshot `src_db` (and its WAL if present) into `dst_db` under a shared lock."""
    resolved_src = src_db.resolve()
    resolved_dst = dst_db.resolve()
    with duckdb_snapshot_lock(resolved_src, exclusive=False, timeout_seconds=timeout_seconds):
        if not resolved_src.is_file():
            return False
        _atomic_copy_file(resolved_src, resolved_dst)
        src_wal = Path(f"{resolved_src}.wal")
        dst_wal = Path(f"{resolved_dst}.wal")
        if src_wal.is_file():
            _atomic_copy_file(src_wal, dst_wal)
        elif dst_wal.exists():
            dst_wal.unlink()
        return True


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


def _normalize_dbt_rel_paths(raw: Any, defaults: Sequence[str]) -> list[str]:
    entries = raw if isinstance(raw, list) else defaults
    normalized: list[str] = []
    seen: set[str] = set()
    for item in entries:
        raw_cleaned = str(item).strip().replace("\\", "/")
        if not raw_cleaned or raw_cleaned.startswith(("/", "~")):
            continue
        while raw_cleaned.startswith("./"):
            raw_cleaned = raw_cleaned[2:]
        if raw_cleaned in ("", "."):
            cleaned = ""
        else:
            cleaned = raw_cleaned.strip("/")
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
                if asset in nodes and nodes[asset].file_path in spark_jobs and nodes[asset].file_path
            ]
            for job in sorted(spark_jobs):
                if job not in ordered:
                    ordered.append(job)
            return ordered

    return sorted(dict.fromkeys(spark_jobs))


def extract_incremental_blocks(text: str) -> list[str]:
    """Return normalized body snippets for every `{% if is_incremental() %}` block."""
    return [" ".join(match.group(1).split()) for match in _INCREMENTAL_BLOCK_RE.finditer(text)]


def _is_dbt_model_sql(
    norm_path: str,
    dbt_project_rel: str | None,
    model_paths: Sequence[str] | None = None,
    excluded_subpaths: Sequence[str] = (),
) -> bool:
    if not norm_path.endswith(".sql"):
        return False
    if dbt_project_rel is not None:
        cleaned = dbt_project_rel.strip("/").replace("\\", "/")
        prefix = f"{cleaned}/" if cleaned else ""
        if model_paths is not None:
            if any(norm_path.startswith(f"{prefix}{mp}/") for mp in model_paths if mp):
                return True
            if "" in model_paths:
                if prefix and not norm_path.startswith(prefix):
                    return False
                rel_in_dbt = norm_path[len(prefix) :]
                if not rel_in_dbt:
                    return False
                blocked = (
                    "tests/",
                    "data_tests/",
                    "macros/",
                    "target/",
                    "logs/",
                    "seeds/",
                    "snapshots/",
                    "dbt_packages/",
                    *(f"{sub.strip('/')}/" for sub in excluded_subpaths if sub.strip("/")),
                )
                return not rel_in_dbt.startswith(blocked)
            return False
        if not norm_path.startswith(prefix):
            return False
        rel_in_dbt = norm_path[len(prefix) :]
        return not rel_in_dbt.startswith(("tests/", "macros/", "target/", "logs/"))
    return norm_path.startswith("models/") or "/models/" in norm_path


def _resolve_model_paths_for_workspaces(
    workspaces: Sequence[Path],
    dbt_project_rel: str | None,
) -> tuple[str | None, list[str] | None]:
    effective_rel = dbt_project_rel
    model_paths: list[str] | None = None
    if effective_rel is not None:
        cleaned_rel = effective_rel.strip("/").replace("\\", "/")
        for base in workspaces:
            cand_dir = (base / cleaned_rel) if cleaned_rel else base
            if (cand_dir / "dbt_project.yml").is_file():
                model_paths = _read_dbt_subpaths(cand_dir)["model-paths"]
                break
        return effective_rel, model_paths

    for base in workspaces:
        found_rel, found_dir = _resolve_dbt_subdir(base)
        if (found_dir / "dbt_project.yml").is_file():
            effective_rel = found_rel
            model_paths = _read_dbt_subpaths(found_dir)["model-paths"]
            break
    return effective_rel, model_paths


def capture_incremental_blocks(
    workspace_dir: Path,
    changed_files: Sequence[str],
    *,
    dbt_project_rel: str | None = None,
) -> dict[str, list[str]]:
    """Capture normalized `{% if is_incremental() %}` snippets for changed dbt model files."""
    effective_rel, model_paths = _resolve_model_paths_for_workspaces(
        [workspace_dir], dbt_project_rel
    )
    captured: dict[str, list[str]] = {}
    for rel in changed_files:
        norm = rel.replace("\\", "/").lstrip("./")
        if not _is_dbt_model_sql(norm, effective_rel, model_paths=model_paths):
            continue
        model_file = workspace_dir / norm
        captured[norm] = (
            extract_incremental_blocks(model_file.read_text(encoding="utf-8"))
            if model_file.is_file()
            else []
        )
    return captured


def touches_incremental_logic(
    prod_workspace: Path,
    worktree_dir: Path,
    changed_files: list[str],
    *,
    dbt_project_rel: str | None = None,
) -> bool:
    """Return True if any changed dbt model alters content inside `is_incremental()`."""
    effective_rel, model_paths = _resolve_model_paths_for_workspaces(
        [prod_workspace, worktree_dir], dbt_project_rel
    )
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


def _all_nodes_from(lineage: LineageAdapterBase | None) -> dict[str, LineageNode]:
    all_nodes_fn = getattr(lineage, "all_nodes", None) if lineage is not None else None
    nodes = all_nodes_fn() if callable(all_nodes_fn) else {}
    return nodes if isinstance(nodes, dict) else {}


def _lookup_node_ci(all_nodes: dict[str, LineageNode], name: str) -> LineageNode | None:
    if name in all_nodes:
        return all_nodes[name]
    lowered = name.lower()
    for node_name, node in all_nodes.items():
        if node_name.lower() == lowered or node.name.lower() == lowered:
            return node
    return None


def _run_dbt_subset(
    *,
    code_workspace: Path,
    prod_paths: LakehousePaths,
    silver_overrides: dict[str, Path],
    target_duckdb_path: Path,
    changed_files: Sequence[str],
    affected_dbt_models: Sequence[str],
    dbt_executor: DbtExecutor,
    full_refresh: bool,
    dbt_project_rel: str | None,
    spark_runs: list[dict[str, Any]],
    allow_spark_only_skip: bool = False,
) -> ShadowExecutionResult | None:
    """Execute dbt seed/run/test subset; return None if `allow_spark_only_skip` skipped dbt."""
    resolved_dbt_rel, resolved_dbt_dir = _resolve_dbt_subdir(code_workspace, dbt_project_rel)
    has_dbt_project = (resolved_dbt_dir / "dbt_project.yml").is_file()
    dbt_prefix = f"{resolved_dbt_rel}/" if resolved_dbt_rel else ""
    subpaths = _read_dbt_subpaths(resolved_dbt_dir)
    test_prefixes = tuple(f"{dbt_prefix}{p}/" for p in subpaths["test-paths"] if p)
    macro_prefixes = tuple(f"{dbt_prefix}{p}/" for p in subpaths["macro-paths"] if p)
    seed_prefixes = tuple(f"{dbt_prefix}{p}/" for p in subpaths["seed-paths"] if p)
    snapshot_prefixes = tuple(f"{dbt_prefix}{p}/" for p in subpaths["snapshot-paths"] if p)
    model_prefixes = tuple(f"{dbt_prefix}{p}/" for p in subpaths["model-paths"] if p)
    non_model_subpaths = (
        subpaths["test-paths"]
        + subpaths["macro-paths"]
        + subpaths["seed-paths"]
        + subpaths["snapshot-paths"]
    )
    all_dbt_prefixes = (
        model_prefixes + test_prefixes + macro_prefixes + seed_prefixes + snapshot_prefixes
    )
    dbt_config_files = (f"{dbt_prefix}dbt_project.yml", f"{dbt_prefix}profiles.yml")
    norm_changed = [path.replace("\\", "/").lstrip("./") for path in changed_files]

    def _is_model_path(norm_path: str) -> bool:
        return _is_dbt_model_sql(
            norm_path,
            resolved_dbt_rel,
            model_paths=subpaths["model-paths"],
            excluded_subpaths=non_model_subpaths,
        )

    touches_dbt = bool(affected_dbt_models) or any(
        (path.startswith(dbt_prefix) if dbt_prefix else False)
        or path.startswith(all_dbt_prefixes)
        or path in dbt_config_files
        or _is_model_path(path)
        for path in norm_changed
    )
    if not has_dbt_project and not touches_dbt and dbt_executor is _default_dbt_executor:
        if allow_spark_only_skip:
            return None
        return ShadowExecutionResult(
            spark_runs=spark_runs,
            silver_overrides=dict(silver_overrides),
            dbt_run_ok=bool(spark_runs),
            dbt_test_ok=bool(spark_runs),
            failure_reasons=(
                []
                if spark_runs
                else ["Patched files do not map to any Spark job or dbt asset."]
            ),
        )
    if not touches_dbt and not spark_runs:
        return ShadowExecutionResult(
            spark_runs=spark_runs,
            silver_overrides=dict(silver_overrides),
            failure_reasons=["Patched files do not map to any Spark job or dbt asset."],
        )

    dbt_paths = prod_paths.model_copy(
        update={
            "workspace_dir": code_workspace.resolve(),
            "prod_mart_dir": target_duckdb_path.parent,
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
            return ShadowExecutionResult(
                spark_runs=spark_runs,
                silver_overrides=dict(silver_overrides),
                dbt_stdout=seed_stdout,
                dbt_stderr=seed_stderr,
                failure_reasons=[f"dbt seed failed (rc={seed_proc.returncode}): {err_msg}"],
            )

    requires_full_dbt = any(
        path.startswith(macro_prefixes + test_prefixes)
        or path.endswith((".yml", ".yaml", "profiles.yml", "dbt_project.yml", "spark_config.py"))
        for path in norm_changed
    ) or ((bool(spark_runs) or touches_seeds) and not affected_dbt_models)
    inferred_models: list[str] = []
    affected_set = set(affected_dbt_models)
    for path in norm_changed:
        if _is_model_path(path):
            stem = Path(path).stem
            if stem not in affected_set:
                selector = f"{stem}+"
                if selector not in inferred_models:
                    inferred_models.append(selector)

    select_targets = list(dict.fromkeys([*affected_dbt_models, *inferred_models]))
    run_args = ["run"]
    if full_refresh:
        run_args.append("--full-refresh")
    if not requires_full_dbt and select_targets:
        run_args.extend(["--select", *select_targets])
    run_proc = dbt_executor(dbt_paths, silver_overrides, target_duckdb_path, run_args)
    dbt_run_ok = run_proc.returncode == 0
    if not dbt_run_ok:
        err_msg = (run_proc.stderr or run_proc.stdout).strip()[-800:]
        return ShadowExecutionResult(
            spark_runs=spark_runs,
            silver_overrides=dict(silver_overrides),
            dbt_stdout="\n".join(part for part in (seed_stdout, run_proc.stdout) if part),
            dbt_stderr="\n".join(part for part in (seed_stderr, run_proc.stderr) if part),
            failure_reasons=[f"dbt run failed (rc={run_proc.returncode}): {err_msg}"],
        )

    test_proc = dbt_executor(dbt_paths, silver_overrides, target_duckdb_path, ["test"])
    dbt_test_ok = test_proc.returncode == 0
    combined_stdout = "\n".join(part for part in (seed_stdout, run_proc.stdout, test_proc.stdout) if part)
    combined_stderr = "\n".join(part for part in (seed_stderr, run_proc.stderr, test_proc.stderr) if part)
    failure_reasons: list[str] = []
    if not dbt_test_ok:
        err_msg = (test_proc.stderr or test_proc.stdout).strip()[-800:]
        failure_reasons.append(f"dbt test failed (rc={test_proc.returncode}): {err_msg}")

    return ShadowExecutionResult(
        spark_runs=spark_runs,
        silver_overrides=dict(silver_overrides),
        dbt_run_ok=dbt_run_ok,
        dbt_test_ok=dbt_test_ok,
        dbt_stdout=combined_stdout,
        dbt_stderr=combined_stderr,
        failure_reasons=failure_reasons,
    )


def _delta_data_diff(
    prod_delta_dir: Path,
    shadow_delta_dir: Path,
    table_name: str,
    key_column: str | None = None,
) -> DataDiffSummary | None:
    """Best-effort row and key-level diff between two Delta table directories."""
    if not ((prod_delta_dir / "_delta_log").is_dir() and (shadow_delta_dir / "_delta_log").is_dir()):
        return None
    con = duckdb.connect(":memory:")
    try:
        try:
            con.execute("LOAD delta;")
        except duckdb.Error:
            con.execute("INSTALL delta; LOAD delta;")

        con.execute(
            f"CREATE TEMP VIEW prod_tbl AS SELECT * FROM delta_scan({_sql_literal(str(prod_delta_dir.resolve()))});"
        )
        con.execute(
            f"CREATE TEMP VIEW shadow_tbl AS SELECT * FROM delta_scan({_sql_literal(str(shadow_delta_dir.resolve()))});"
        )
        prod_cols = [row[0] for row in con.execute("DESCRIBE prod_tbl;").fetchall()]
        shadow_cols = [row[0] for row in con.execute("DESCRIBE shadow_tbl;").fetchall()]
        resolved_key = key_column
        if not resolved_key or resolved_key not in prod_cols or resolved_key not in shadow_cols:
            shared = [col for col in prod_cols if col in shadow_cols]
            if not shared:
                return None
            resolved_key = next(
                (col for col in shared if col == "id" or col.endswith(("_id", "_date"))),
                shared[0],
            )
        return compare_duckdb_relations(con, "prod_tbl", "shadow_tbl", table_name, resolved_key)
    except (duckdb.Error, KeyError, OSError):
        return None
    finally:
        con.close()


class DuckDBShadowBackend:
    """Isolated local DuckDB file-snapshot Shadow Backend with file-level coordination locking."""

    backend_name: str = "duckdb"

    def __init__(
        self,
        project: ResolvedProject,
        *,
        lineage_adapter: LineageAdapterBase | None = None,
        dbt_executor: DbtExecutor | None = None,
        lock_timeout_seconds: float = 30.0,
        preserve_artifacts: bool = True,
    ) -> None:
        self.project = project
        self.prod_paths = LakehousePaths.from_project(project)
        self.lineage_adapter = lineage_adapter
        self.dbt_executor: DbtExecutor = dbt_executor or _default_dbt_executor
        self.lock_timeout_seconds = lock_timeout_seconds
        self.preserve_artifacts = preserve_artifacts

    @classmethod
    def from_workspace(
        cls,
        prod_workspace: Path,
        *,
        lineage_adapter: LineageAdapterBase | None = None,
        dbt_executor: DbtExecutor | None = None,
        lock_timeout_seconds: float = 30.0,
        preserve_artifacts: bool = True,
    ) -> DuckDBShadowBackend:
        return cls(
            discover_project(prod_workspace),
            lineage_adapter=lineage_adapter,
            dbt_executor=dbt_executor,
            lock_timeout_seconds=lock_timeout_seconds,
            preserve_artifacts=preserve_artifacts,
        )

    def _effective_dbt_rel(self) -> str | None:
        return self.project.dbt_project_rel if self.project.dbt_project_dir is not None else None

    def _target_db_name(self) -> str:
        name = self.prod_paths.prod_duckdb_path.name
        return name if name.endswith(".duckdb") else "warehouse.duckdb"

    def prepare_shadow(self, run_id: str, worktree_dir: Path, runs_dir: Path) -> ShadowContext:
        shadow_root = runs_dir.resolve() / run_id / "shadow"
        if shadow_root.exists():
            shutil.rmtree(shadow_root)
        shadow_mart_dir = shadow_root / "mart"
        shadow_mart_dir.mkdir(parents=True, exist_ok=True)

        shadow_duckdb_path = shadow_mart_dir / self._target_db_name()
        prod_db = self.prod_paths.prod_duckdb_path
        copied = copy_duckdb_snapshot(
            prod_db, shadow_duckdb_path, timeout_seconds=self.lock_timeout_seconds
        )
        return ShadowContext(
            run_id=run_id,
            backend_name=self.backend_name,
            worktree_dir=worktree_dir.resolve(),
            shadow_root=shadow_root,
            shadow_silver_dir=None,
            shadow_duckdb_path=shadow_duckdb_path,
            metadata={
                "lock_path": str(_snapshot_lock_path(prod_db)),
                "prod_duckdb_path": str(prod_db),
                "snapshot_copied": copied,
            },
        )

    def execute_subset(
        self,
        ctx: ShadowContext,
        *,
        changed_files: list[str],
        affected: AffectedSubgraph,
        full_refresh: bool = True,
    ) -> ShadowExecutionResult:
        if ctx.shadow_duckdb_path is None:
            return ShadowExecutionResult(
                failure_reasons=["DuckDBShadowBackend context is missing shadow_duckdb_path."]
            )
        result = _run_dbt_subset(
            code_workspace=ctx.worktree_dir,
            prod_paths=self.prod_paths,
            silver_overrides=dict(ctx.silver_overrides),
            target_duckdb_path=ctx.shadow_duckdb_path,
            changed_files=changed_files,
            affected_dbt_models=affected.dbt_models,
            dbt_executor=self.dbt_executor,
            full_refresh=full_refresh,
            dbt_project_rel=self._effective_dbt_rel(),
            spark_runs=[],
        )
        assert result is not None
        return result

    def _resolve_materialized_mart_targets(
        self, prod_db: Path
    ) -> list[tuple[str, str]]:
        targets = _discover_mart_diff_targets(self.lineage_adapter)
        if not targets or self.lineage_adapter is None:
            return targets
        aliases_fn = getattr(self.lineage_adapter, "all_aliases", None)
        alias_map = aliases_fn() if callable(aliases_fn) else {}
        if not isinstance(alias_map, dict) or not alias_map:
            return targets

        con = duckdb.connect(str(prod_db.resolve()), read_only=True)
        try:
            prod_tables_ci = {
                row[0].lower(): row[0]
                for row in con.execute(
                    "SELECT table_name FROM information_schema.tables "
                    "WHERE table_schema = 'main'"
                ).fetchall()
            }
        finally:
            con.close()

        canonical_to_bare: dict[str, list[str]] = {}
        for alias_name, canonical in alias_map.items():
            if "." not in alias_name and alias_name.lower() != canonical.lower():
                canonical_to_bare.setdefault(canonical.lower(), []).append(alias_name)

        resolved: list[tuple[str, str]] = []
        for model_name, key_col in targets:
            if model_name.lower() in prod_tables_ci:
                resolved.append((prod_tables_ci[model_name.lower()], key_col))
                continue
            matched = next(
                (
                    prod_tables_ci[cand.lower()]
                    for cand in canonical_to_bare.get(model_name.lower(), [])
                    if cand.lower() in prod_tables_ci
                ),
                None,
            )
            resolved.append((matched or model_name, key_col))
        return resolved

    def diff_against_prod(self, ctx: ShadowContext) -> list[DataDiffSummary]:
        prod_db = self.prod_paths.prod_duckdb_path
        shadow_db = ctx.shadow_duckdb_path
        if shadow_db is None or not (prod_db.is_file() and shadow_db.is_file()):
            return []

        adapter = DuckDBAdapter(self.prod_paths)
        diffs: list[DataDiffSummary] = []
        with duckdb_snapshot_lock(prod_db, exclusive=False, timeout_seconds=self.lock_timeout_seconds):
            for table_name, key_col in self._resolve_materialized_mart_targets(prod_db):
                try:
                    diffs.append(
                        adapter.data_diff(
                            prod_db=prod_db,
                            shadow_db=shadow_db,
                            table_name=table_name,
                            key_column=key_col,
                        )
                    )
                except (KeyError, duckdb.Error):
                    continue
        return diffs

    def teardown(self, ctx: ShadowContext) -> None:
        if ctx.shadow_root.exists():
            shutil.rmtree(ctx.shadow_root, ignore_errors=True)

    @contextmanager
    def production_write_lock(self) -> Iterator[Path]:
        """Hold the exclusive DuckDB snapshot lock across production replay + rollback."""
        with duckdb_snapshot_lock(
            self.prod_paths.prod_duckdb_path,
            exclusive=True,
            timeout_seconds=self.lock_timeout_seconds,
        ) as lock_path:
            yield lock_path

    def snapshot_production(
        self,
        run_id: str,
        runs_dir: Path,
        *,
        affected: AffectedSubgraph,
    ) -> ProductionSnapshot:
        del affected
        prod_db = self.prod_paths.prod_duckdb_path.resolve()
        prod_wal = Path(f"{prod_db}.wal")
        run_root = runs_dir.resolve() / run_id
        backup_duckdb = run_root / ".prod_warehouse_backup.duckdb"
        backup_wal = run_root / ".prod_warehouse_backup.duckdb.wal"

        duckdb_existed = False
        wal_copied = False
        with duckdb_snapshot_lock(prod_db, exclusive=True, timeout_seconds=self.lock_timeout_seconds):
            for stale in (backup_duckdb, backup_wal):
                if stale.is_file():
                    stale.unlink()
            if prod_db.is_file():
                _atomic_copy_file(prod_db, backup_duckdb)
                duckdb_existed = True
            if duckdb_existed and prod_wal.is_file():
                _atomic_copy_file(prod_wal, backup_wal)
                wal_copied = True

        return ProductionSnapshot(
            run_id=run_id,
            backend_name=self.backend_name,
            duckdb_backup_path=backup_duckdb if duckdb_existed else None,
            duckdb_wal_backup_path=backup_wal if wal_copied else None,
            duckdb_existed=duckdb_existed,
        )

    def replay_production(
        self,
        *,
        changed_files: list[str],
        affected: AffectedSubgraph,
        full_refresh: bool = True,
        run_id: str,
        runs_dir: Path,
        silver_overrides: dict[str, Path] | None = None,
        spark_runs: list[dict[str, Any]] | None = None,
    ) -> ShadowExecutionResult:
        del run_id, runs_dir
        prod_db = self.prod_paths.prod_duckdb_path
        with duckdb_snapshot_lock(prod_db, exclusive=True, timeout_seconds=self.lock_timeout_seconds):
            result = _run_dbt_subset(
                code_workspace=self.project.repo_root,
                prod_paths=self.prod_paths,
                silver_overrides=dict(silver_overrides or {}),
                target_duckdb_path=prod_db,
                changed_files=changed_files,
                affected_dbt_models=affected.dbt_models,
                dbt_executor=self.dbt_executor,
                full_refresh=full_refresh,
                dbt_project_rel=self._effective_dbt_rel(),
                spark_runs=list(spark_runs or []),
            )
            assert result is not None
            return result

    def restore_production(self, snapshot: ProductionSnapshot) -> None:
        prod_db = self.prod_paths.prod_duckdb_path.resolve()
        prod_wal = Path(f"{prod_db}.wal")
        with duckdb_snapshot_lock(prod_db, exclusive=True, timeout_seconds=self.lock_timeout_seconds):
            if snapshot.duckdb_existed:
                if (
                    snapshot.duckdb_backup_path is not None
                    and snapshot.duckdb_backup_path.is_file()
                ):
                    _atomic_copy_file(snapshot.duckdb_backup_path, prod_db)
                if (
                    snapshot.duckdb_wal_backup_path is not None
                    and snapshot.duckdb_wal_backup_path.is_file()
                ):
                    _atomic_copy_file(snapshot.duckdb_wal_backup_path, prod_wal)
                elif prod_wal.exists():
                    prod_wal.unlink()
            else:
                if prod_db.exists():
                    prod_db.unlink()
                if prod_wal.exists():
                    prod_wal.unlink()

    def cleanup_snapshot(self, snapshot: ProductionSnapshot) -> None:
        for path in (snapshot.duckdb_backup_path, snapshot.duckdb_wal_backup_path):
            if path is not None and path.is_file():
                path.unlink()


def _find_dir_ci(
    parent: Path,
    name: str,
    *,
    exclude_symlinks: bool = False,
) -> Path | None:
    exact = parent / name
    if exact.is_dir() and not (exclude_symlinks and exact.is_symlink()):
        return exact.resolve()
    if not parent.is_dir():
        return None
    target_lower = name.lower()
    for entry in sorted(parent.iterdir()):
        if entry.name.lower() == target_lower and entry.is_dir():
            if exclude_symlinks and entry.is_symlink():
                continue
            return entry.resolve()
    return None


class SparkSubprocessBackend:
    """Isolated PySpark subprocess Shadow Backend driven by discovered lineage metadata."""

    backend_name: str = "spark"

    def __init__(
        self,
        project: ResolvedProject,
        *,
        lineage_adapter: LineageAdapterBase | None = None,
        spark_runner: SparkJobRunner | None = None,
        preserve_artifacts: bool = True,
    ) -> None:
        self.project = project
        self.prod_paths = LakehousePaths.from_project(project)
        self.lineage_adapter = lineage_adapter
        self.spark_runner = spark_runner
        self.preserve_artifacts = preserve_artifacts

    def _spark_jobs_rel(self) -> str | None:
        spark_dir = self.project.spark_jobs_dir
        if spark_dir is None:
            return None
        try:
            rel = spark_dir.resolve().relative_to(self.project.repo_root.resolve()).as_posix()
        except ValueError:
            return None
        return "" if rel == "." else rel.strip("/")

    @staticmethod
    def _matches_spark_dir(norm_path: str, spark_dir_rel: str) -> bool:
        if not norm_path.endswith(".py"):
            return False
        if spark_dir_rel == "":
            return "/" not in norm_path
        return norm_path.startswith(f"{spark_dir_rel}/")

    def _resolve_spark_jobs(
        self,
        *,
        changed_files: Sequence[str],
        affected: AffectedSubgraph,
    ) -> list[str]:
        spark_jobs = [p.replace("\\", "/").lstrip("./") for p in affected.spark_jobs]
        spark_dir_rel = self._spark_jobs_rel()
        if spark_dir_rel is not None:
            touches_config = False
            for rel in changed_files:
                norm = rel.replace("\\", "/").lstrip("./")
                if not self._matches_spark_dir(norm, spark_dir_rel):
                    continue
                fname = Path(norm).name
                if fname == "spark_config.py":
                    touches_config = True
                    continue
                if fname == "__init__.py":
                    continue
                if norm not in spark_jobs:
                    spark_jobs.append(norm)
            if touches_config and self.project.spark_jobs_dir is not None:
                for node in _all_nodes_from(self.lineage_adapter).values():
                    if node.engine == "spark" and node.file_path:
                        nf = node.file_path.replace("\\", "/").lstrip("./")
                        if nf not in spark_jobs:
                            spark_jobs.append(nf)
                spark_dir = self.project.spark_jobs_dir.resolve()
                candidates = (
                    spark_dir.iterdir() if spark_dir_rel == "" else spark_dir.rglob("*.py")
                )
                for cand in sorted(candidates):
                    if not cand.is_file() or cand.suffix != ".py":
                        continue
                    if cand.name in ("__init__.py", "spark_config.py"):
                        continue
                    cand_rel = cand.relative_to(self.project.repo_root.resolve()).as_posix()
                    if cand_rel not in spark_jobs:
                        spark_jobs.append(cand_rel)
        return order_spark_jobs(spark_jobs, lineage_adapter=self.lineage_adapter)

    def prepare_shadow(self, run_id: str, worktree_dir: Path, runs_dir: Path) -> ShadowContext:
        shadow_root = runs_dir.resolve() / run_id / "shadow"
        if shadow_root.exists():
            shutil.rmtree(shadow_root)
        shadow_silver_dir = shadow_root / "silver"
        shadow_events_dir = shadow_root / "spark_events"
        shadow_silver_dir.mkdir(parents=True, exist_ok=True)
        shadow_events_dir.mkdir(parents=True, exist_ok=True)
        return ShadowContext(
            run_id=run_id,
            backend_name=self.backend_name,
            worktree_dir=worktree_dir.resolve(),
            shadow_root=shadow_root,
            shadow_silver_dir=shadow_silver_dir,
            spark_events_dir=shadow_events_dir,
            metadata={"prod_silver_dir": str(self.prod_paths.prod_silver_dir)},
        )

    def _execute_in_dirs(
        self,
        *,
        code_workspace: Path,
        target_silver_dir: Path,
        events_dir: Path,
        changed_files: Sequence[str],
        affected: AffectedSubgraph,
        require_match: bool = True,
    ) -> ShadowExecutionResult:
        spark_jobs = self._resolve_spark_jobs(changed_files=changed_files, affected=affected)
        if not spark_jobs:
            if require_match:
                return ShadowExecutionResult(
                    failure_reasons=["Patched files do not map to any Spark job."]
                )
            return ShadowExecutionResult(dbt_run_ok=True, dbt_test_ok=True)

        target_silver_dir.mkdir(parents=True, exist_ok=True)
        events_dir.mkdir(parents=True, exist_ok=True)

        all_nodes = _all_nodes_from(self.lineage_adapter)
        job_to_tables: dict[str, list[str]] = {}
        for name, node in all_nodes.items():
            if node.engine == "spark" and node.file_path:
                norm_fp = node.file_path.replace("\\", "/").lstrip("./")
                job_to_tables.setdefault(norm_fp, []).append(name)

        runner = self.spark_runner or SparkJobRunner(event_log_dir=events_dir)
        spark_runs: list[dict[str, Any]] = []
        silver_overrides: dict[str, Path] = {}
        is_shadow_target = target_silver_dir.resolve() != self.prod_paths.prod_silver_dir.resolve()

        for job_rel in spark_jobs:
            norm_job = job_rel.replace("\\", "/").lstrip("./")
            script_path = code_workspace / norm_job
            if not script_path.is_file():
                return ShadowExecutionResult(
                    spark_runs=spark_runs,
                    silver_overrides=silver_overrides,
                    failure_reasons=[f"Affected Spark script missing: {norm_job}"],
                )

            args = ["--raw-dir", str(self.prod_paths.raw_dir), "--output-dir", str(target_silver_dir)]
            produced_tables = job_to_tables.get(norm_job, [])
            produced_lower = {tbl.lower() for tbl in produced_tables}
            upstream_spark_tables: list[str] = []
            seen_ups_lower: set[str] = set()
            for tbl in produced_tables:
                node = _lookup_node_ci(all_nodes, tbl)
                if node is None:
                    continue
                for up in node.upstream:
                    up_node = _lookup_node_ci(all_nodes, up)
                    if (
                        up_node is not None
                        and up_node.engine == "spark"
                        and up_node.name.lower() not in produced_lower
                        and up_node.name.lower() not in seen_ups_lower
                    ):
                        upstream_spark_tables.append(up_node.name)
                        seen_ups_lower.add(up_node.name.lower())

            if upstream_spark_tables:
                overrides_lower = {k.lower(): v for k, v in silver_overrides.items()}
                has_shadow_upstream = any(up.lower() in overrides_lower for up in upstream_spark_tables)
                if has_shadow_upstream:
                    for up in upstream_spark_tables:
                        if up.lower() in overrides_lower:
                            continue
                        prod_up = _find_dir_ci(self.prod_paths.prod_silver_dir, up)
                        if prod_up is not None and _find_dir_ci(target_silver_dir, up) is None:
                            tgt_up = target_silver_dir / prod_up.name
                            try:
                                tgt_up.symlink_to(prod_up)
                            except OSError:
                                shutil.copytree(prod_up, tgt_up, dirs_exist_ok=True)
                    silver_read = target_silver_dir
                else:
                    silver_read = self.prod_paths.prod_silver_dir
                args.extend(["--silver-read-dir", str(silver_read)])

            pre_existing_dirs = (
                {p.name for p in target_silver_dir.iterdir() if p.is_dir() and not p.is_symlink()}
                if is_shadow_target and not produced_tables and target_silver_dir.is_dir()
                else set()
            )
            res = runner.run_job(script_path, args=args)
            spark_runs.append(res.model_dump(mode="json"))
            if res.returncode != 0 or res.timed_out:
                err_tail = (res.stderr or res.stdout).strip().splitlines()[-3:]
                return ShadowExecutionResult(
                    spark_runs=spark_runs,
                    silver_overrides=silver_overrides,
                    failure_reasons=[
                        f"Spark job {norm_job} failed (rc={res.returncode}): {' | '.join(err_tail)}"
                    ],
                )

            if produced_tables:
                for table_name in produced_tables:
                    written = _find_dir_ci(target_silver_dir, table_name, exclude_symlinks=True)
                    if written is not None:
                        silver_overrides[table_name] = written
            elif is_shadow_target and target_silver_dir.is_dir():
                for entry in sorted(target_silver_dir.iterdir()):
                    if (
                        entry.is_dir()
                        and not entry.is_symlink()
                        and entry.name not in pre_existing_dirs
                    ):
                        silver_overrides[entry.name] = entry.resolve()

        return ShadowExecutionResult(
            spark_runs=spark_runs,
            silver_overrides=silver_overrides,
            dbt_run_ok=True,
            dbt_test_ok=True,
        )

    def execute_subset(
        self,
        ctx: ShadowContext,
        *,
        changed_files: list[str],
        affected: AffectedSubgraph,
        full_refresh: bool = True,
    ) -> ShadowExecutionResult:
        del full_refresh
        if ctx.shadow_silver_dir is None:
            return ShadowExecutionResult(
                failure_reasons=["SparkSubprocessBackend context is missing shadow_silver_dir."]
            )
        events_dir = ctx.spark_events_dir or (ctx.shadow_root / "spark_events")
        res = self._execute_in_dirs(
            code_workspace=ctx.worktree_dir,
            target_silver_dir=ctx.shadow_silver_dir,
            events_dir=events_dir,
            changed_files=changed_files,
            affected=affected,
            require_match=True,
        )
        ctx.silver_overrides = dict(res.silver_overrides)
        return res

    def diff_against_prod(self, ctx: ShadowContext) -> list[DataDiffSummary]:
        if not ctx.silver_overrides:
            return []
        all_nodes = _all_nodes_from(self.lineage_adapter)
        diffs: list[DataDiffSummary] = []
        for table_name in sorted(ctx.silver_overrides):
            shadow_dir = ctx.silver_overrides[table_name]
            prod_dir = _find_dir_ci(self.prod_paths.prod_silver_dir, table_name) or (
                self.prod_paths.prod_silver_dir / table_name
            )
            node = _lookup_node_ci(all_nodes, table_name)
            key_col = _infer_key_column(node) if node is not None else None
            summary = _delta_data_diff(prod_dir, shadow_dir, table_name, key_col)
            if summary is not None:
                diffs.append(summary)
        return diffs

    def teardown(self, ctx: ShadowContext) -> None:
        if ctx.shadow_root.exists():
            shutil.rmtree(ctx.shadow_root, ignore_errors=True)

    def _will_run_spark(self, affected: AffectedSubgraph) -> bool:
        if affected.spark_jobs:
            return True
        if affected.changed_paths:
            return bool(
                self._resolve_spark_jobs(
                    changed_files=affected.changed_paths,
                    affected=affected,
                )
            )
        return self.project.spark_jobs_dir is not None

    def snapshot_production(
        self,
        run_id: str,
        runs_dir: Path,
        *,
        affected: AffectedSubgraph,
    ) -> ProductionSnapshot:
        prod_silver = self.prod_paths.prod_silver_dir
        backup_silver_dir = runs_dir.resolve() / run_id / ".prod_silver_backup"
        silver_existed = prod_silver.exists()
        silver_backed_up = False
        if self._will_run_spark(affected):
            if backup_silver_dir.exists():
                shutil.rmtree(backup_silver_dir)
            if silver_existed:
                backup_silver_dir.parent.mkdir(parents=True, exist_ok=True)
                shutil.copytree(prod_silver, backup_silver_dir)
            silver_backed_up = True

        return ProductionSnapshot(
            run_id=run_id,
            backend_name=self.backend_name,
            silver_backup_dir=backup_silver_dir if silver_backed_up else None,
            silver_existed=silver_existed,
            silver_backed_up=silver_backed_up,
        )

    def replay_production(
        self,
        *,
        changed_files: list[str],
        affected: AffectedSubgraph,
        full_refresh: bool = True,
        run_id: str,
        runs_dir: Path,
    ) -> ShadowExecutionResult:
        del full_refresh
        events_dir = runs_dir.resolve() / run_id / "prod_spark_events"
        return self._execute_in_dirs(
            code_workspace=self.project.repo_root,
            target_silver_dir=self.prod_paths.prod_silver_dir,
            events_dir=events_dir,
            changed_files=changed_files,
            affected=affected,
            require_match=True,
        )

    def restore_production(self, snapshot: ProductionSnapshot) -> None:
        if not snapshot.silver_backed_up:
            return
        prod_silver = self.prod_paths.prod_silver_dir
        if prod_silver.exists():
            shutil.rmtree(prod_silver)
        if (
            snapshot.silver_existed
            and snapshot.silver_backup_dir is not None
            and snapshot.silver_backup_dir.is_dir()
        ):
            prod_silver.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(snapshot.silver_backup_dir, prod_silver)

    def cleanup_snapshot(self, snapshot: ProductionSnapshot) -> None:
        if snapshot.silver_backup_dir is not None and snapshot.silver_backup_dir.exists():
            shutil.rmtree(snapshot.silver_backup_dir)


class CompositeShadowBackend:
    """Shadow Backend composing PySpark subprocess silver execution with DuckDB dbt marts."""

    backend_name: str = "duckdb+spark"

    def __init__(
        self,
        spark_backend: SparkSubprocessBackend,
        warehouse_backend: DuckDBShadowBackend,
        *,
        preserve_artifacts: bool | None = None,
    ) -> None:
        self.spark_backend = spark_backend
        self.warehouse_backend = warehouse_backend
        self.preserve_artifacts = (
            (spark_backend.preserve_artifacts and warehouse_backend.preserve_artifacts)
            if preserve_artifacts is None
            else preserve_artifacts
        )

    def prepare_shadow(self, run_id: str, worktree_dir: Path, runs_dir: Path) -> ShadowContext:
        shadow_root = runs_dir.resolve() / run_id / "shadow"
        if shadow_root.exists():
            shutil.rmtree(shadow_root)
        shadow_silver_dir = shadow_root / "silver"
        shadow_events_dir = shadow_root / "spark_events"
        shadow_mart_dir = shadow_root / "mart"
        shadow_silver_dir.mkdir(parents=True, exist_ok=True)
        shadow_events_dir.mkdir(parents=True, exist_ok=True)
        shadow_mart_dir.mkdir(parents=True, exist_ok=True)

        prod_db = self.warehouse_backend.prod_paths.prod_duckdb_path
        db_name = prod_db.name if prod_db.name.endswith(".duckdb") else "warehouse.duckdb"
        shadow_duckdb_path = shadow_mart_dir / db_name
        copied = copy_duckdb_snapshot(
            prod_db,
            shadow_duckdb_path,
            timeout_seconds=self.warehouse_backend.lock_timeout_seconds,
        )
        return ShadowContext(
            run_id=run_id,
            backend_name=self.backend_name,
            worktree_dir=worktree_dir.resolve(),
            shadow_root=shadow_root,
            shadow_silver_dir=shadow_silver_dir,
            shadow_duckdb_path=shadow_duckdb_path,
            spark_events_dir=shadow_events_dir,
            metadata={
                "lock_path": str(_snapshot_lock_path(prod_db)),
                "prod_duckdb_path": str(prod_db),
                "snapshot_copied": copied,
            },
        )

    def execute_subset(
        self,
        ctx: ShadowContext,
        *,
        changed_files: list[str],
        affected: AffectedSubgraph,
        full_refresh: bool = True,
    ) -> ShadowExecutionResult:
        if ctx.shadow_silver_dir is None or ctx.shadow_duckdb_path is None:
            return ShadowExecutionResult(
                failure_reasons=["CompositeShadowBackend context is incomplete."]
            )
        events_dir = ctx.spark_events_dir or (ctx.shadow_root / "spark_events")
        spark_res = self.spark_backend._execute_in_dirs(
            code_workspace=ctx.worktree_dir,
            target_silver_dir=ctx.shadow_silver_dir,
            events_dir=events_dir,
            changed_files=changed_files,
            affected=affected,
            require_match=False,
        )
        ctx.silver_overrides = dict(spark_res.silver_overrides)
        if not spark_res.passed:
            return spark_res

        dbt_res = _run_dbt_subset(
            code_workspace=ctx.worktree_dir,
            prod_paths=self.warehouse_backend.prod_paths,
            silver_overrides=dict(ctx.silver_overrides),
            target_duckdb_path=ctx.shadow_duckdb_path,
            changed_files=changed_files,
            affected_dbt_models=affected.dbt_models,
            dbt_executor=self.warehouse_backend.dbt_executor,
            full_refresh=full_refresh,
            dbt_project_rel=self.warehouse_backend._effective_dbt_rel(),
            spark_runs=spark_res.spark_runs,
        )
        assert dbt_res is not None
        return dbt_res

    def diff_against_prod(self, ctx: ShadowContext) -> list[DataDiffSummary]:
        diffs = self.warehouse_backend.diff_against_prod(ctx)
        if not diffs and ctx.silver_overrides:
            diffs = self.spark_backend.diff_against_prod(ctx)
        return diffs

    def teardown(self, ctx: ShadowContext) -> None:
        if ctx.shadow_root.exists():
            shutil.rmtree(ctx.shadow_root, ignore_errors=True)

    @contextmanager
    def production_write_lock(self) -> Iterator[Path]:
        """Coordinate concurrent DuckDB reads across Composite promotion replay and rollback."""
        with self.warehouse_backend.production_write_lock() as lock_path:
            yield lock_path

    def snapshot_production(
        self,
        run_id: str,
        runs_dir: Path,
        *,
        affected: AffectedSubgraph,
    ) -> ProductionSnapshot:
        db_snap = self.warehouse_backend.snapshot_production(run_id, runs_dir, affected=affected)
        try:
            sp_snap = (
                self.spark_backend.snapshot_production(run_id, runs_dir, affected=affected)
                if self.spark_backend._will_run_spark(affected)
                else ProductionSnapshot(run_id=run_id, backend_name=self.backend_name)
            )
        except Exception:
            self.warehouse_backend.cleanup_snapshot(db_snap)
            raise
        return ProductionSnapshot(
            run_id=run_id,
            backend_name=self.backend_name,
            duckdb_backup_path=db_snap.duckdb_backup_path,
            duckdb_wal_backup_path=db_snap.duckdb_wal_backup_path,
            duckdb_existed=db_snap.duckdb_existed,
            silver_backup_dir=sp_snap.silver_backup_dir,
            silver_existed=sp_snap.silver_existed,
            silver_backed_up=sp_snap.silver_backed_up,
        )

    def replay_production(
        self,
        *,
        changed_files: list[str],
        affected: AffectedSubgraph,
        full_refresh: bool = True,
        run_id: str,
        runs_dir: Path,
    ) -> ShadowExecutionResult:
        events_dir = runs_dir.resolve() / run_id / "prod_spark_events"
        spark_res = self.spark_backend._execute_in_dirs(
            code_workspace=self.spark_backend.project.repo_root,
            target_silver_dir=self.spark_backend.prod_paths.prod_silver_dir,
            events_dir=events_dir,
            changed_files=changed_files,
            affected=affected,
            require_match=False,
        )
        if not spark_res.passed:
            return spark_res
        return self.warehouse_backend.replay_production(
            changed_files=changed_files,
            affected=affected,
            full_refresh=full_refresh,
            run_id=run_id,
            runs_dir=runs_dir,
            silver_overrides=spark_res.silver_overrides,
            spark_runs=spark_res.spark_runs,
        )

    def restore_production(self, snapshot: ProductionSnapshot) -> None:
        warehouse_done = False
        try:
            self.warehouse_backend.restore_production(snapshot)
            warehouse_done = True
        finally:
            if not warehouse_done:
                self.spark_backend.restore_production(snapshot)
        self.spark_backend.restore_production(snapshot)

    def cleanup_snapshot(self, snapshot: ProductionSnapshot) -> None:
        warehouse_done = False
        try:
            self.warehouse_backend.cleanup_snapshot(snapshot)
            warehouse_done = True
        finally:
            if not warehouse_done:
                self.spark_backend.cleanup_snapshot(snapshot)
        self.spark_backend.cleanup_snapshot(snapshot)


class NoOpShadowBackend:
    """Explicit no-op backend when a Project Manifest declares `shadow.backend: none`."""

    backend_name: str = "none"
    preserve_artifacts: bool = False

    def prepare_shadow(self, run_id: str, worktree_dir: Path, runs_dir: Path) -> ShadowContext:
        shadow_root = runs_dir.resolve() / run_id / "shadow"
        return ShadowContext(
            run_id=run_id,
            backend_name=self.backend_name,
            worktree_dir=worktree_dir.resolve(),
            shadow_root=shadow_root,
        )

    def execute_subset(
        self,
        ctx: ShadowContext,
        *,
        changed_files: list[str],
        affected: AffectedSubgraph,
        full_refresh: bool = True,
    ) -> ShadowExecutionResult:
        del ctx, changed_files, affected, full_refresh
        return ShadowExecutionResult(
            failure_reasons=[
                "Shadow verification is disabled because Project Manifest sets shadow.backend='none'."
            ]
        )

    def diff_against_prod(self, ctx: ShadowContext) -> list[DataDiffSummary]:
        del ctx
        return []

    def teardown(self, ctx: ShadowContext) -> None:
        if ctx.shadow_root.exists():
            shutil.rmtree(ctx.shadow_root, ignore_errors=True)


SHADOW_BACKEND_REGISTRY: dict[str, ShadowBackendFactory] = {}


def register_shadow_backend(name: str, factory: ShadowBackendFactory) -> None:
    """Register a custom `ShadowBackend` factory under a normalized backend name."""
    cleaned = name.strip().lower()
    if not cleaned:
        raise ValueError("Shadow backend name must not be empty.")
    if cleaned == "none":
        raise ValueError("Shadow backend 'none' is reserved and cannot be overridden.")
    SHADOW_BACKEND_REGISTRY[cleaned] = factory


def _dbt_declares_non_duckdb(project: ResolvedProject) -> bool:
    if project.dbt_manifest_path is not None and project.dbt_manifest_path.is_file():
        try:
            raw_manifest = json.loads(project.dbt_manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            raw_manifest = {}
        adapter_type = (raw_manifest.get("metadata") or {}).get("adapter_type")
        if isinstance(adapter_type, str) and adapter_type.strip():
            return normalize_sql_dialect(adapter_type) != "duckdb"
    if project.dbt_project_dir is not None:
        dbt_cfg = _read_dbt_project_yml(project.dbt_project_dir, project.repo_root)
        profile_dialect, _ = _discover_profile_hints(
            project.repo_root, project.dbt_project_dir, dbt_cfg
        )
        if profile_dialect is not None:
            return profile_dialect != "duckdb"
    return False


def create_shadow_backend(
    project: ResolvedProject,
    *,
    lineage_adapter: LineageAdapterBase | None = None,
    spark_runner: SparkJobRunner | None = None,
    dbt_executor: DbtExecutor | None = None,
    lock_timeout_seconds: float = 30.0,
) -> ShadowBackend:
    """Select or construct a `ShadowBackend` from `project` manifest and discovered layout."""
    requested = project.manifest.shadow.backend
    all_nodes = _all_nodes_from(lineage_adapter)
    has_spark = (
        project.spark_jobs_dir is not None
        or project.spark_lineage_path is not None
        or spark_runner is not None
        or any(node.engine == "spark" for node in all_nodes.values())
    )
    has_dbt = (
        project.dbt_project_dir is not None
        or project.dbt_manifest_path is not None
        or (dbt_executor is not None and dbt_executor is not _default_dbt_executor)
        or (project.repo_root / "dbt_project").is_dir()
    )

    if requested is not None:
        key = requested.strip().lower()
        if key == "none":
            return NoOpShadowBackend()
        if key in SHADOW_BACKEND_REGISTRY:
            return SHADOW_BACKEND_REGISTRY[key](
                project,
                lineage_adapter=lineage_adapter,
                spark_runner=spark_runner,
                dbt_executor=dbt_executor,
                lock_timeout_seconds=lock_timeout_seconds,
            )
        if key == "spark":
            return SparkSubprocessBackend(
                project,
                lineage_adapter=lineage_adapter,
                spark_runner=spark_runner,
            )
        if key == "duckdb":
            return DuckDBShadowBackend(
                project,
                lineage_adapter=lineage_adapter,
                dbt_executor=dbt_executor,
                lock_timeout_seconds=lock_timeout_seconds,
            )
        raise NotImplementedError(
            f"Shadow backend {requested!r} is not implemented or registered."
        )

    dialect = (project.sql_dialect or "").strip().lower()
    if (
        has_dbt
        and dialect not in ("", "duckdb")
        and not (has_spark and dialect == "spark" and not _dbt_declares_non_duckdb(project))
    ):
        if dialect in SHADOW_BACKEND_REGISTRY:
            return SHADOW_BACKEND_REGISTRY[dialect](
                project,
                lineage_adapter=lineage_adapter,
                spark_runner=spark_runner,
                dbt_executor=dbt_executor,
                lock_timeout_seconds=lock_timeout_seconds,
            )
        raise NotImplementedError(
            f"No ShadowBackend registered for discovered SQL dialect {dialect!r}. "
            "Configure shadow.backend or register a backend."
        )

    if has_spark and has_dbt:
        return CompositeShadowBackend(
            SparkSubprocessBackend(
                project,
                lineage_adapter=lineage_adapter,
                spark_runner=spark_runner,
            ),
            DuckDBShadowBackend(
                project,
                lineage_adapter=lineage_adapter,
                dbt_executor=dbt_executor,
                lock_timeout_seconds=lock_timeout_seconds,
            ),
        )
    if has_spark and not has_dbt:
        return SparkSubprocessBackend(
            project,
            lineage_adapter=lineage_adapter,
            spark_runner=spark_runner,
        )
    return DuckDBShadowBackend(
        project,
        lineage_adapter=lineage_adapter,
        dbt_executor=dbt_executor,
        lock_timeout_seconds=lock_timeout_seconds,
    )


__all__ = [
    "SHADOW_BACKEND_REGISTRY",
    "CompositeShadowBackend",
    "DbtExecutor",
    "DuckDBShadowBackend",
    "NoOpShadowBackend",
    "ProductionReplayBackend",
    "ProductionSnapshot",
    "ShadowBackend",
    "ShadowBackendFactory",
    "ShadowContext",
    "ShadowExecutionResult",
    "ShadowVerifierProtocol",
    "SparkSubprocessBackend",
    "_default_dbt_executor",
    "_discover_mart_diff_targets",
    "_infer_key_column",
    "_is_dbt_model_sql",
    "_normalize_dbt_rel_paths",
    "_read_dbt_subpaths",
    "_resolve_dbt_subdir",
    "_run_dbt_subset",
    "capture_incremental_blocks",
    "copy_duckdb_snapshot",
    "create_shadow_backend",
    "duckdb_snapshot_lock",
    "extract_incremental_blocks",
    "order_spark_jobs",
    "register_shadow_backend",
    "touches_incremental_logic",
]

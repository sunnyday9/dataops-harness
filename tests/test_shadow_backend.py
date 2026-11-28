from __future__ import annotations

import json
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import duckdb
import pytest

from src.adapters.base import (
    AffectedSubgraph,
    DataDiffSummary,
    LineageAdapterBase,
    SparkRunResult,
)
from src.adapters.lineage_adapter import LineageAdapter
from src.adapters.model_adapter import FakeReplayModelAdapter, TurnResponse
from src.adapters.shadow_backend import (
    SHADOW_BACKEND_REGISTRY,
    CompositeShadowBackend,
    DuckDBShadowBackend,
    NoOpShadowBackend,
    ShadowBackend,
    ShadowContext,
    ShadowExecutionResult,
    ShadowVerifierProtocol,
    SparkSubprocessBackend,
    create_shadow_backend,
    register_shadow_backend,
)
from src.core.context import CheckpointStore, IncidentInput, UserContext
from src.core.loop import HarnessLoop
from src.core.project import discover_project
from src.core.promotion import PromotionService
from src.core.rca import ApprovalPacket, OracleVerdict, ShadowVerificationResult
from src.core.shadow_verifier import ShadowOverlayVerifier
from src.core.worktree import RunWorktreeManager
from src.hooks.base import HookPipeline
from src.hooks.stop_gate import StopGateHook
from src.tools.registry import ToolRegistry

REPO_ROOT = Path(__file__).resolve().parent.parent


def _dbt_exe() -> str:
    venv_dbt = Path(sys.executable).parent / "dbt"
    return str(venv_dbt) if venv_dbt.exists() else (shutil.which("dbt") or "dbt")


def _init_git_repo(repo_dir: Path, message: str = "baseline") -> str:
    subprocess.run(["git", "init", "-b", "main"], cwd=repo_dir, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "test"], cwd=repo_dir, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "t@e.com"], cwd=repo_dir, check=True, capture_output=True)
    subprocess.run(["git", "add", "."], cwd=repo_dir, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", message], cwd=repo_dir, check=True, capture_output=True)
    return subprocess.run(
        ["git", "-C", str(repo_dir), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _write_delta_table(table_dir: Path, select_sql: str, schema_fields: list[tuple[str, object]]) -> None:
    """Create a real minimal Delta table via DuckDB Parquet + `_delta_log/000...000.json`."""
    table_dir.mkdir(parents=True, exist_ok=True)
    log_dir = table_dir / "_delta_log"
    log_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = table_dir / "part-000.snappy.parquet"

    con = duckdb.connect(":memory:")
    try:
        escaped_path = str(parquet_path.resolve()).replace("'", "''")
        con.execute(f"COPY ({select_sql}) TO '{escaped_path}' (FORMAT PARQUET);")
    finally:
        con.close()

    schema_json = json.dumps(
        {
            "type": "struct",
            "fields": [
                {"name": col_name, "type": col_type, "nullable": True, "metadata": {}}
                for col_name, col_type in schema_fields
            ],
        }
    )
    entries = [
        {"protocol": {"minReaderVersion": 1, "minWriterVersion": 2}},
        {
            "metaData": {
                "id": f"delta-{table_dir.name}",
                "format": {"provider": "parquet", "options": {}},
                "schemaString": schema_json,
                "partitionColumns": [],
                "configuration": {},
                "createdTime": 1732750000000,
            }
        },
        {
            "add": {
                "path": parquet_path.name,
                "partitionValues": {},
                "size": parquet_path.stat().st_size,
                "modificationTime": 1732750000000,
                "dataChange": True,
            }
        },
    ]
    (log_dir / "00000000000000000000.json").write_text(
        "\n".join(json.dumps(item) for item in entries) + "\n",
        encoding="utf-8",
    )


def _create_external_dbt_repo(
    repo_dir: Path,
    *,
    model_subdir: str = "models",
    inject_2x_bug: bool = True,
) -> None:
    models_dir = repo_dir / model_subdir / "marts"
    models_dir.mkdir(parents=True, exist_ok=True)
    (repo_dir / "tests").mkdir(parents=True, exist_ok=True)

    (repo_dir / "dataops-harness.yaml").write_text(
        "version: 1\nname: ext_mrr_app\nshadow:\n  backend: duckdb\n",
        encoding="utf-8",
    )
    (repo_dir / "dbt_project.yml").write_text(
        "name: ext_mrr_app\n"
        'version: "1.0.0"\n'
        "config-version: 2\n"
        "profile: ext_mrr_app\n"
        f'model-paths: ["{model_subdir}"]\n'
        'test-paths: ["tests"]\n'
        'target-path: "target"\n'
        "models:\n"
        "  ext_mrr_app:\n"
        "    marts:\n"
        "      +materialized: table\n",
        encoding="utf-8",
    )
    (repo_dir / "profiles.yml").write_text(
        "ext_mrr_app:\n"
        "  target: dev\n"
        "  outputs:\n"
        "    dev:\n"
        "      type: duckdb\n"
        "      path: 'warehouse.duckdb'\n",
        encoding="utf-8",
    )
    factor = " * 2.0" if inject_2x_bug else ""
    (models_dir / "fct_mrr.sql").write_text(
        "SELECT '2026-11-01'::DATE AS billing_date, "
        f"ROUND(100.0{factor}, 2) AS total_mrr_usd\n",
        encoding="utf-8",
    )
    (models_dir / "schema.yml").write_text(
        "version: 2\n"
        "models:\n"
        "  - name: fct_mrr\n"
        "    meta:\n"
        "      layer: mart\n"
        "    columns:\n"
        "      - name: billing_date\n"
        "        tests: [unique, not_null]\n",
        encoding="utf-8",
    )
    (repo_dir / "tests" / "assert_mrr_exact.sql").write_text(
        "SELECT * FROM {{ ref('fct_mrr') }} WHERE total_mrr_usd > 150.0\n",
        encoding="utf-8",
    )
    (repo_dir / ".gitignore").write_text("target/\nlogs/\nwarehouse.duckdb*\n", encoding="utf-8")

    dbt = _dbt_exe()
    run_proc = subprocess.run(
        [dbt, "--no-version-check", "run", "--profiles-dir", "."],
        cwd=repo_dir,
        capture_output=True,
        text=True,
        check=False,
    )
    assert run_proc.returncode == 0, run_proc.stderr or run_proc.stdout
    _init_git_repo(repo_dir)


def test_custom_shadow_backend_protocol_and_stop_gate_decoupling(tmp_path: Path) -> None:
    """Custom in-memory ShadowBackend and ShadowVerifierProtocol decouple Verifier & StopGateHook."""
    repo = tmp_path / "ext_repo"
    (repo / "models").mkdir(parents=True)
    (repo / "dbt_project.yml").write_text('name: mem_app\nmodel-paths: ["models"]\n', encoding="utf-8")
    (repo / "models" / "m.sql").write_text("SELECT 1 AS x\n", encoding="utf-8")
    _init_git_repo(repo)

    class _MemorySchemaBackend:
        backend_name = "memory_schema"
        preserve_artifacts = False

        def __init__(self, mode: str = "pass") -> None:
            self.mode = mode
            self.events: list[str] = []

        def prepare_shadow(self, run_id: str, worktree_dir: Path, runs_dir: Path) -> ShadowContext:
            self.events.append("prepare_shadow")
            return ShadowContext(
                run_id=run_id,
                backend_name=self.backend_name,
                worktree_dir=worktree_dir.resolve(),
                shadow_root=runs_dir.resolve() / run_id / "shadow_mem",
                shadow_schema=f"shadow_{run_id.replace('-', '_')}",
                metadata={"mode": self.mode},
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
            self.events.append("execute_subset")
            if self.mode == "raise":
                raise RuntimeError("boom during execute_subset")
            if self.mode == "fail":
                return ShadowExecutionResult(failure_reasons=["query error"])
            return ShadowExecutionResult(dbt_run_ok=True, dbt_test_ok=True)

        def diff_against_prod(self, ctx: ShadowContext) -> list[DataDiffSummary]:
            self.events.append("diff_against_prod")
            return [
                DataDiffSummary(
                    table_name="m",
                    key_column="x",
                    prod_row_count=1,
                    shadow_row_count=1,
                    changed_keys=["1"],
                    is_identical=False,
                )
            ]

        def teardown(self, ctx: ShadowContext) -> None:
            del ctx
            self.events.append("teardown")

    runs_dir = tmp_path / "runs"
    wt = RunWorktreeManager.create(repo, runs_dir, "run-mem-ok")
    (wt / "models" / "m.sql").write_text("SELECT 2 AS x\n", encoding="utf-8")

    # 1. Green path verifies protocol conformance, schema metadata, and automatic teardown.
    ok_backend = _MemorySchemaBackend(mode="pass")
    assert isinstance(ok_backend, ShadowBackend)
    verifier = ShadowOverlayVerifier(repo, runs_dir, backend=ok_backend)
    assert isinstance(verifier, ShadowVerifierProtocol)
    res = verifier.verify("run-mem-ok", wt)
    assert res.passed is True
    assert res.backend_name == "memory_schema"
    assert res.shadow_schema == "shadow_run_mem_ok"
    assert res.shadow_silver_dir == ""
    assert res.shadow_duckdb_path == ""
    assert res.shadow_metadata == {"mode": "pass"}
    assert len(res.data_diffs) == 1
    assert ok_backend.events == ["prepare_shadow", "execute_subset", "diff_against_prod", "teardown"]

    # 2. Failed execute_subset skips diff but still invokes teardown in finally.
    fail_backend = _MemorySchemaBackend(mode="fail")
    fail_res = ShadowOverlayVerifier(repo, runs_dir, backend=fail_backend).verify("run-mem-fail", wt)
    assert fail_res.passed is False
    assert fail_res.failure_reasons == ["query error"]
    assert fail_backend.events == ["prepare_shadow", "execute_subset", "teardown"]

    # 3. Exception inside execute_subset still guarantees teardown in finally.
    err_backend = _MemorySchemaBackend(mode="raise")
    with pytest.raises(RuntimeError, match="boom during execute_subset"):
        ShadowOverlayVerifier(repo, runs_dir, backend=err_backend).verify("run-mem-raise", wt)
    assert err_backend.events == ["prepare_shadow", "execute_subset", "teardown"]

    # 4. StopGateHook accepts any ShadowVerifierProtocol without ShadowOverlayVerifier subclassing.
    class _ProtocolOnlyVerifier:
        def __init__(self, target_runs: Path) -> None:
            self.runs_dir = target_runs
            self.calls: list[str] = []

        def verify(self, run_id: str, worktree_dir: Path) -> ShadowVerificationResult:
            self.calls.append(run_id)
            return ShadowVerificationResult(
                passed=True,
                run_id=run_id,
                worktree_path=str(worktree_dir),
                backend_name="remote_stub",
                shadow_schema="shadow_remote_1",
                changed_files=["models/m.sql"],
                dbt_run_ok=True,
                dbt_test_ok=True,
                oracle_verdict=OracleVerdict(passed=True, oracle_name="stub"),
            )

    proto_verifier = _ProtocolOnlyVerifier(runs_dir)
    assert isinstance(proto_verifier, ShadowVerifierProtocol)
    assert not isinstance(proto_verifier, ShadowOverlayVerifier)

    rca_payload = json.dumps(
        {
            "incident_id": "INC-PROTO",
            "root_cause_summary": "Verified via custom remote ShadowVerifierProtocol.",
            "affected_assets": ["m"],
            "evidence": ["shadow_remote_1 query parity passed"],
            "engine": "dbt",
            "proposed_fix_summary": "Fixed SQL expression in models/m.sql.",
            "patched_files": ["models/m.sql"],
            "confidence": "high",
        }
    )
    loop = HarnessLoop(
        model_adapter=FakeReplayModelAdapter(
            [TurnResponse(content="not json yet"), TurnResponse(content=rca_payload)]
        ),
        tool_registry=ToolRegistry(wt),
        hooks=HookPipeline(stop_hooks=[StopGateHook(verifier=proto_verifier, worktree_dir=wt)]),
        runs_dir=runs_dir,
    )
    state = loop.run(
        IncidentInput(incident_id="INC-PROTO", title="Proto check", summary="Check StopGateHook"),
        UserContext(role="de", resolution_mode="fix"),
        run_id="run-proto-stop",
    )
    assert state.status == "AWAITING_APPROVAL"
    assert state.turn == 2
    assert proto_verifier.calls == ["run-proto-stop"]
    packet = ApprovalPacket.model_validate_json(
        (runs_dir / "run-proto-stop" / "approval_packet.json").read_text(encoding="utf-8")
    )
    assert packet.shadow_verification.backend_name == "remote_stub"
    assert packet.shadow_verification.shadow_schema == "shadow_remote_1"


def test_duckdb_shadow_backend_lifecycle_teardown_and_snapshot_locking(tmp_path: Path) -> None:
    """DuckDBShadowBackend isolates writes, diffs, tears down cleanly, and coordinates flock locks."""
    repo = tmp_path / "ext_dbt_repo"
    _create_external_dbt_repo(repo, inject_2x_bug=True)
    runs_dir = tmp_path / "runs"

    # Populate lineage with compiled manifest so diff_against_prod discovers fct_mrr + unique key.
    dbt = _dbt_exe()
    compile_proc = subprocess.run(
        [dbt, "--no-version-check", "compile", "--profiles-dir", "."],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    )
    assert compile_proc.returncode == 0, compile_proc.stderr or compile_proc.stdout
    project = discover_project(repo)
    lineage = LineageAdapter.from_project(project)
    backend = DuckDBShadowBackend(project, lineage_adapter=lineage)

    wt = RunWorktreeManager.create(repo, runs_dir, "run-duck-life")
    (wt / "models" / "marts" / "fct_mrr.sql").write_text(
        "SELECT '2026-11-01'::DATE AS billing_date, ROUND(100.0, 2) AS total_mrr_usd\n",
        encoding="utf-8",
    )
    changed = RunWorktreeManager.changed_files(wt)
    affected = lineage.affected_downstream(changed)

    ctx = backend.prepare_shadow("run-duck-life", wt, runs_dir)
    assert ctx.shadow_duckdb_path is not None and ctx.shadow_duckdb_path.is_file()
    exec_res = backend.execute_subset(ctx, changed_files=changed, affected=affected, full_refresh=True)
    assert exec_res.passed is True, exec_res.failure_reasons
    assert exec_res.dbt_run_ok is True and exec_res.dbt_test_ok is True

    diffs = backend.diff_against_prod(ctx)
    assert any(d.table_name == "fct_mrr" and not d.is_identical for d in diffs)

    # Prod DuckDB stays untouched at buggy 200.0 while shadow holds 100.0.
    prod_db = repo / "warehouse.duckdb"
    con_p = duckdb.connect(str(prod_db), read_only=True)
    try:
        assert float(con_p.execute("SELECT total_mrr_usd FROM fct_mrr").fetchone()[0]) == 200.0
    finally:
        con_p.close()

    backend.teardown(ctx)
    assert not ctx.shadow_root.exists()
    backend.teardown(ctx)  # Idempotent repeat teardown.

    # Real cross-process flock coordination: prepare_shadow waits for exclusive writer commit.
    wait_script = (
        "import sys, time, duckdb\n"
        "from pathlib import Path\n"
        "from src.adapters.shadow_backend import duckdb_snapshot_lock\n"
        "db = Path(sys.argv[1])\n"
        "with duckdb_snapshot_lock(db, exclusive=True):\n"
        "    print('LOCKED', flush=True)\n"
        "    time.sleep(0.35)\n"
        "    con = duckdb.connect(str(db))\n"
        "    con.execute('CREATE OR REPLACE TABLE lock_sentinel AS SELECT 42 AS val')\n"
        "    con.close()\n"
    )
    writer_proc = subprocess.Popen(
        [sys.executable, "-c", wait_script, str(prod_db)],
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert writer_proc.stdout is not None
    assert writer_proc.stdout.readline().strip() == "LOCKED"
    waited_ctx = backend.prepare_shadow("run-duck-wait", wt, runs_dir)
    writer_out, writer_err = writer_proc.communicate(timeout=15)
    assert writer_proc.returncode == 0, writer_err or writer_out

    assert waited_ctx.shadow_duckdb_path is not None
    con_s = duckdb.connect(str(waited_ctx.shadow_duckdb_path), read_only=True)
    try:
        assert int(con_s.execute("SELECT val FROM lock_sentinel").fetchone()[0]) == 42
    finally:
        con_s.close()

    # Exclusive lock held past lock_timeout_seconds triggers deterministic TimeoutError.
    hold_script = (
        "import sys, time\n"
        "from pathlib import Path\n"
        "from src.adapters.shadow_backend import duckdb_snapshot_lock\n"
        "with duckdb_snapshot_lock(Path(sys.argv[1]), exclusive=True):\n"
        "    print('LOCKED', flush=True)\n"
        "    time.sleep(1.0)\n"
    )
    holder = subprocess.Popen(
        [sys.executable, "-c", hold_script, str(prod_db)],
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        assert holder.stdout is not None
        assert holder.stdout.readline().strip() == "LOCKED"
        short_backend = DuckDBShadowBackend(
            project, lineage_adapter=lineage, lock_timeout_seconds=0.1
        )
        with pytest.raises(TimeoutError, match="DuckDB snapshot lock"):
            short_backend.prepare_shadow("run-duck-timeout", wt, runs_dir)
    finally:
        holder.wait(timeout=15)
    assert holder.returncode == 0

    # Shared locks allow concurrent prepare_shadow calls across threads without corruption.
    def _concurrent_prepare(idx: int) -> Path:
        c = backend.prepare_shadow(f"run-conc-{idx}", wt, runs_dir)
        assert c.shadow_duckdb_path is not None and c.shadow_duckdb_path.is_file()
        return c.shadow_duckdb_path

    with ThreadPoolExecutor(max_workers=4) as pool:
        snap_paths = list(pool.map(_concurrent_prepare, range(4)))
    for snap in snap_paths:
        con_c = duckdb.connect(str(snap), read_only=True)
        try:
            assert int(con_c.execute("SELECT val FROM lock_sentinel").fetchone()[0]) == 42
        finally:
            con_c.close()

    # External repository checkout remains completely free of lock files or untracked artifacts.
    git_status = subprocess.run(
        ["git", "-C", str(repo), "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert git_status == ""


def test_spark_subprocess_backend_discovered_metadata_without_static_tables(tmp_path: Path) -> None:
    """SparkSubprocessBackend discovers jobs via manifest + OpenLineage with zero static job names."""
    repo = tmp_path / "ext_spark_repo"
    (repo / "etl_spark").mkdir(parents=True)
    (repo / "openlineage").mkdir(parents=True)

    (repo / "dataops-harness.yaml").write_text(
        "version: 1\nspark_jobs_dir: etl_spark\nlakehouse_dir: storage\n",
        encoding="utf-8",
    )
    (repo / "etl_spark" / "build_sessions.py").write_text("# v1\n", encoding="utf-8")
    (repo / "etl_spark" / "build_revenue_summary.py").write_text("# v1\n", encoding="utf-8")

    # Baseline prod Delta table for silver_revenue_summary under custom lakehouse_dir (`storage`).
    prod_rev = repo / "storage" / "prod" / "silver" / "silver_revenue_summary"
    _write_delta_table(
        prod_rev,
        "SELECT 's1' AS session_id, 50.0::DOUBLE AS revenue_usd",
        [("session_id", "string"), ("revenue_usd", "double")],
    )
    (repo / ".gitignore").write_text("storage/\n", encoding="utf-8")

    ol_events = [
        {
            "eventType": "COMPLETE",
            "eventTime": "2026-11-28T00:00:00Z",
            "producer": "spark://custom-etl",
            "job": {
                "namespace": "spark://prod",
                "name": "build_sessions",
                "facets": {
                    "sourceCodeLocation": {"path": "etl_spark/build_sessions.py"},
                },
            },
            "inputs": [{"namespace": "s3://raw", "name": "raw_clicks"}],
            "outputs": [
                {
                    "namespace": "spark://prod",
                    "name": "silver_sessions",
                    "facets": {
                        "schema": {
                            "fields": [
                                {"name": "session_id", "type": "string"},
                                {"name": "amount", "type": "double"},
                            ]
                        }
                    },
                }
            ],
        },
        {
            "eventType": "COMPLETE",
            "eventTime": "2026-11-28T00:01:00Z",
            "producer": "spark://custom-etl",
            "job": {
                "namespace": "spark://prod",
                "name": "build_revenue_summary",
                "facets": {
                    "sourceCodeLocation": {"path": "etl_spark/build_revenue_summary.py"},
                },
            },
            "inputs": [{"namespace": "spark://prod", "name": "silver_sessions"}],
            "outputs": [
                {
                    "namespace": "spark://prod",
                    "name": "silver_revenue_summary",
                    "facets": {
                        "schema": {
                            "fields": [
                                {"name": "session_id", "type": "string"},
                                {"name": "revenue_usd", "type": "double"},
                            ]
                        }
                    },
                }
            ],
        },
    ]
    (repo / "openlineage" / "lineage.jsonl").write_text(
        "\n".join(json.dumps(ev) for ev in ol_events) + "\n",
        encoding="utf-8",
    )
    _init_git_repo(repo)

    project = discover_project(repo)
    lineage = LineageAdapter.from_project(project)
    assert project.dbt_project_dir is None

    call_order: list[str] = []

    class _RecordingSpark:
        def __init__(self, event_log_dir: Path) -> None:
            self.event_log_dir = event_log_dir

        def run_job(self, script_path: Path, args: list[str] | None = None, **_kwargs) -> SparkRunResult:
            rel_name = script_path.name
            call_order.append(f"etl_spark/{rel_name}")
            arg_list = list(args or [])
            out_dir = Path(arg_list[arg_list.index("--output-dir") + 1])
            if rel_name == "build_sessions.py":
                _write_delta_table(
                    out_dir / "silver_sessions",
                    "SELECT 's1' AS session_id, 100.0::DOUBLE AS amount "
                    "UNION ALL SELECT 's2' AS session_id, 25.0::DOUBLE AS amount",
                    [("session_id", "string"), ("amount", "double")],
                )
            elif rel_name == "build_revenue_summary.py":
                assert "--silver-read-dir" in arg_list
                read_dir = Path(arg_list[arg_list.index("--silver-read-dir") + 1])
                assert (read_dir / "silver_sessions" / "_delta_log").is_dir()
                _write_delta_table(
                    out_dir / "silver_revenue_summary",
                    "SELECT 's1' AS session_id, 100.0::DOUBLE AS revenue_usd "
                    "UNION ALL SELECT 's2' AS session_id, 25.0::DOUBLE AS revenue_usd",
                    [("session_id", "string"), ("revenue_usd", "double")],
                )
            return SparkRunResult(
                job_name=script_path.stem,
                script_path=script_path,
                pid=101,
                returncode=0,
                duration_ms=2.0,
                event_log_path=None,
                stdout="ok",
                stderr="",
            )

    runs_dir = tmp_path / "runs_spark"
    spark_runner = _RecordingSpark(event_log_dir=runs_dir / "ev")
    selected = create_shadow_backend(
        project,
        lineage_adapter=lineage,
        spark_runner=spark_runner,  # type: ignore[arg-type]
    )
    assert isinstance(selected, SparkSubprocessBackend)

    wt = RunWorktreeManager.create(repo, runs_dir, "run-sp-dyn")
    (wt / "etl_spark" / "build_sessions.py").write_text("# v2 fixed\n", encoding="utf-8")

    verifier = ShadowOverlayVerifier(
        repo,
        runs_dir,
        project=project,
        lineage_adapter=lineage,
        spark_runner=spark_runner,  # type: ignore[arg-type]
        backend=selected,
    )
    result = verifier.verify("run-sp-dyn", wt)
    assert result.passed is True, result.failure_reasons
    assert result.backend_name == "spark"
    assert call_order == ["etl_spark/build_sessions.py", "etl_spark/build_revenue_summary.py"]
    assert len(result.data_diffs) == 1
    diff = result.data_diffs[0]
    assert diff.table_name == "silver_revenue_summary"
    assert diff.added_keys == ["s2"]
    assert diff.changed_keys == ["s1"]

    verifier.teardown(
        ShadowContext(
            run_id="run-sp-dyn",
            backend_name="spark",
            worktree_dir=wt,
            shadow_root=runs_dir / "run-sp-dyn" / "shadow",
        )
    )
    assert not (runs_dir / "run-sp-dyn" / "shadow").exists()


def test_shadow_backend_registry_factory_and_promotion_decoupling(tmp_path: Path) -> None:
    """Factory matrix selects right backends and PromotionService works on external custom dbt repos."""
    # 1. Backend selection matrix.
    none_repo = tmp_path / "none_repo"
    none_repo.mkdir()
    (none_repo / "dataops-harness.yaml").write_text("version: 1\nshadow:\n  backend: none\n", encoding="utf-8")
    assert isinstance(create_shadow_backend(discover_project(none_repo)), NoOpShadowBackend)

    pg_repo = tmp_path / "pg_repo"
    pg_repo.mkdir()
    (pg_repo / "dataops-harness.yaml").write_text("version: 1\nshadow:\n  backend: postgres\n", encoding="utf-8")
    with pytest.raises(NotImplementedError, match="postgres"):
        create_shadow_backend(discover_project(pg_repo))

    dual_repo = tmp_path / "dual_repo"
    (dual_repo / "spark_jobs").mkdir(parents=True)
    (dual_repo / "dbt_project").mkdir(parents=True)
    (dual_repo / "dbt_project" / "dbt_project.yml").write_text("name: dual\n", encoding="utf-8")
    dual_backend = create_shadow_backend(discover_project(dual_repo))
    assert isinstance(dual_backend, CompositeShadowBackend)

    class _CustomPostgresStub:
        backend_name = "postgres"

        def prepare_shadow(self, run_id: str, worktree_dir: Path, runs_dir: Path) -> ShadowContext:
            return ShadowContext(
                run_id=run_id,
                backend_name="postgres",
                worktree_dir=worktree_dir,
                shadow_root=runs_dir / run_id / "shadow",
                shadow_schema=f"shadow_{run_id}",
            )

        def execute_subset(self, ctx: ShadowContext, *, changed_files: list[str], affected: AffectedSubgraph, full_refresh: bool = True) -> ShadowExecutionResult:
            del ctx, changed_files, affected, full_refresh
            return ShadowExecutionResult(dbt_run_ok=True, dbt_test_ok=True)

        def diff_against_prod(self, ctx: ShadowContext) -> list[DataDiffSummary]:
            del ctx
            return []

        def teardown(self, ctx: ShadowContext) -> None:
            del ctx

    register_shadow_backend("postgres", lambda *_a, **_kw: _CustomPostgresStub())
    try:
        assert isinstance(create_shadow_backend(discover_project(pg_repo)), _CustomPostgresStub)
    finally:
        SHADOW_BACKEND_REGISTRY.pop("postgres", None)

    # 2. PromotionService promotes cleanly on external root-level repo with custom model-paths.
    ext_repo = tmp_path / "ext_promo"
    _create_external_dbt_repo(ext_repo, model_subdir="transformations", inject_2x_bug=True)
    runs_dir = tmp_path / "runs_promo"
    wt = RunWorktreeManager.create(ext_repo, runs_dir, "run-ext-promo")
    (wt / "transformations" / "marts" / "fct_mrr.sql").write_text(
        "SELECT '2026-11-01'::DATE AS billing_date, ROUND(100.0, 2) AS total_mrr_usd\n",
        encoding="utf-8",
    )

    verifier = ShadowOverlayVerifier(ext_repo, runs_dir)
    assert isinstance(verifier.backend, DuckDBShadowBackend)
    stop_hook = StopGateHook(verifier=verifier, worktree_dir=wt, runs_dir=runs_dir)

    rca_json = json.dumps(
        {
            "incident_id": "INC-EXT-19",
            "root_cause_summary": "Removed duplicate 2.0 factor in transformations/marts/fct_mrr.sql.",
            "affected_assets": ["fct_mrr"],
            "evidence": ["assert_mrr_exact failed on 200.0"],
            "engine": "dbt",
            "proposed_fix_summary": "Drop * 2.0 multiplier.",
            "patched_files": ["transformations/marts/fct_mrr.sql"],
            "confidence": "high",
        }
    )
    loop = HarnessLoop(
        model_adapter=FakeReplayModelAdapter([TurnResponse(content=rca_json)]),
        tool_registry=ToolRegistry(wt),
        hooks=HookPipeline(stop_hooks=[stop_hook]),
        runs_dir=runs_dir,
    )
    state = loop.run(
        IncidentInput(incident_id="INC-EXT-19", title="Ext MRR 2x", summary="fct_mrr doubled"),
        UserContext(role="de", resolution_mode="fix"),
        run_id="run-ext-promo",
    )
    assert state.status == "AWAITING_APPROVAL"

    # Reject non-replayable backend without mutating git HEAD.
    head_before = subprocess.run(
        ["git", "-C", str(ext_repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    no_replay_svc = PromotionService(ext_repo, runs_dir, shadow_backend=NoOpShadowBackend())
    with pytest.raises(ValueError, match="does not support local in-place replay Promotion"):
        no_replay_svc.approve("run-ext-promo")
    head_after_refusal = subprocess.run(
        ["git", "-C", str(ext_repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert head_after_refusal == head_before

    # Failed replay rolls back git HEAD and restores warehouse.duckdb atomically.
    prod_db = ext_repo / "warehouse.duckdb"
    orig_db_bytes = prod_db.read_bytes()

    def _corrupt_and_fail(dbt_paths, silver_overrides, mart_duckdb_path, args):  # type: ignore[no-untyped-def]
        del dbt_paths, silver_overrides, args
        mart_duckdb_path.write_bytes(b"CORRUPT")
        return subprocess.CompletedProcess(["dbt"], 1, stdout="", stderr="failed replay")

    failing_svc = PromotionService(ext_repo, runs_dir, dbt_executor=_corrupt_and_fail)
    with pytest.raises(RuntimeError, match="Production post-commit verification failed"):
        failing_svc.approve("run-ext-promo", cleanup_worktree=False)
    assert (
        subprocess.run(
            ["git", "-C", str(ext_repo), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        == head_before
    )
    assert prod_db.read_bytes() == orig_db_bytes
    assert CheckpointStore(runs_dir, "run-ext-promo").load_latest().status == "AWAITING_APPROVAL"

    # Real PromotionService.approve() commits fix and heals prod warehouse.duckdb.
    promo = PromotionService(ext_repo, runs_dir).approve("run-ext-promo")
    assert promo.status == "RESOLVED"
    assert promo.dbt_run_ok is True and promo.dbt_test_ok is True
    assert not wt.exists()
    con_p = duckdb.connect(str(prod_db), read_only=True)
    try:
        assert float(con_p.execute("SELECT total_mrr_usd FROM fct_mrr").fetchone()[0]) == 100.0
    finally:
        con_p.close()
    assert isinstance(LineageAdapter.from_project(discover_project(ext_repo)), LineageAdapterBase)

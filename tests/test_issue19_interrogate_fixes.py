from __future__ import annotations

import shutil
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import duckdb
import pytest

from src.adapters.base import AffectedSubgraph, LakehousePaths, SparkRunResult
from src.adapters.shadow_backend import (
    CompositeShadowBackend,
    DuckDBShadowBackend,
    SparkSubprocessBackend,
    duckdb_snapshot_lock,
)
from src.core.context import CheckpointStore, IncidentInput, RunState, UserContext
from src.core.project import discover_project
from src.core.promotion import PromotionService
from src.core.rca import ApprovalPacket, RcaReport, ShadowVerificationResult
from src.core.shadow_verifier import ShadowOverlayVerifier
from src.core.worktree import RunWorktreeManager


def _dbt_exe() -> str:
    import sys

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


def _valid_rca(incident_id: str, patched_files: list[str]) -> RcaReport:
    return RcaReport(
        incident_id=incident_id,
        root_cause_summary="Reproduced and patched pipeline defect cleanly.",
        affected_assets=["asset_a"],
        evidence=["shadow run passed"],
        engine="cross_engine",
        proposed_fix_summary="Patched failing transformation logic.",
        patched_files=patched_files,
        confidence="high",
    )


def _write_minimal_dbt_repo(repo: Path, *, dbt_subdir: str = "dbt_project") -> None:
    base = (repo / dbt_subdir) if dbt_subdir else repo
    (base / "models" / "marts").mkdir(parents=True, exist_ok=True)
    (base / "tests").mkdir(parents=True, exist_ok=True)
    (base / "dbt_project.yml").write_text(
        'name: mini_app\nversion: "1.0.0"\nconfig-version: 2\nprofile: mini_app\n'
        'model-paths: ["models"]\ntest-paths: ["tests"]\ntarget-path: "target"\n',
        encoding="utf-8",
    )
    (base / "profiles.yml").write_text(
        "mini_app:\n  target: dev\n  outputs:\n    dev:\n      type: duckdb\n"
        "      path: '../lakehouse/prod/mart/warehouse.duckdb'\n"
        if dbt_subdir
        else "mini_app:\n  target: dev\n  outputs:\n    dev:\n      type: duckdb\n      path: 'warehouse.duckdb'\n",
        encoding="utf-8",
    )
    (base / "models" / "marts" / "fct_mrr.sql").write_text("SELECT 100 AS val\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Batch 1: Atomic Rollback, WAL Safety & Cross-Process Locking (F1, F3, F4, F5)
# ---------------------------------------------------------------------------


def test_batch1_f1_composite_snapshot_backs_up_silver_when_affected_spark_jobs_empty(
    tmp_path: Path,
) -> None:
    """F1: CompositeShadowBackend must snapshot Silver even when affected.spark_jobs is empty."""
    for silver_initially_present in (True, False):
        tag = "present" if silver_initially_present else "absent"
        repo = tmp_path / f"repo_{tag}"
        runs_dir = tmp_path / f"runs_{tag}"
        _write_minimal_dbt_repo(repo, dbt_subdir="dbt_project")
        (repo / "spark_jobs").mkdir(parents=True)
        (repo / "spark_jobs" / "custom_etl.py").write_text("# v1\n", encoding="utf-8")
        (repo / ".gitignore").write_text(
            "lakehouse/\ndbt_project/target/\ndbt_project/logs/\n", encoding="utf-8"
        )
        _init_git_repo(repo)

        paths = LakehousePaths.from_workspace(repo)
        paths.prod_mart_dir.mkdir(parents=True, exist_ok=True)
        con_db = duckdb.connect(str(paths.prod_duckdb_path))
        con_db.execute("CREATE TABLE baseline (id INT);")
        con_db.close()
        silver_marker = paths.prod_silver_dir / "silver_custom" / "part-000.parquet"
        if silver_initially_present:
            silver_marker.parent.mkdir(parents=True, exist_ok=True)
            silver_marker.write_bytes(b"ORIG-SILVER")

        class _NoOpOkSpark:
            event_log_dir = runs_dir / "ev_ok"

            def run_job(self, script_path: Path, args=None, **_kwargs):  # type: ignore[no-untyped-def]
                return SparkRunResult(
                    job_name=script_path.stem,
                    script_path=script_path,
                    pid=1,
                    returncode=0,
                    duration_ms=1.0,
                    event_log_path=None,
                    stdout="ok",
                    stderr="",
                )

        class _MutatingPromoteSpark:
            event_log_dir = runs_dir / "ev_mut"

            def run_job(self, script_path: Path, args=None, **_kwargs):  # type: ignore[no-untyped-def]
                arg_list = list(args or [])
                out_dir = Path(arg_list[arg_list.index("--output-dir") + 1])
                target = out_dir / "silver_custom"
                target.mkdir(parents=True, exist_ok=True)
                (target / "part-000.parquet").write_bytes(b"CORRUPTED-BY-FAILED-PROMOTE")
                return SparkRunResult(
                    job_name=script_path.stem,
                    script_path=script_path,
                    pid=2,
                    returncode=0,
                    duration_ms=1.0,
                    event_log_path=None,
                    stdout="ok",
                    stderr="",
                )

        def _ok_dbt(dbt_paths, silver_overrides, mart_duckdb_path, args):  # type: ignore[no-untyped-def]
            return subprocess.CompletedProcess(["dbt", *args], 0, stdout="ok", stderr="")

        def _failing_dbt(dbt_paths, silver_overrides, mart_duckdb_path, args):  # type: ignore[no-untyped-def]
            return subprocess.CompletedProcess(["dbt", *args], 1, stdout="", stderr="dbt boom")

        run_id = f"run-f1-{tag}"
        wt = RunWorktreeManager.create(repo, runs_dir, run_id)
        (wt / "spark_jobs" / "custom_etl.py").write_text("# v2\n", encoding="utf-8")

        verifier = ShadowOverlayVerifier(
            repo,
            runs_dir,
            spark_runner=_NoOpOkSpark(),  # type: ignore[arg-type]
            dbt_executor=_ok_dbt,
        )
        shadow = verifier.verify(run_id, wt)
        assert shadow.passed is True
        assert shadow.affected_spark_jobs == []

        packet = ApprovalPacket(
            run_id=run_id,
            incident_id=f"INC-F1-{tag}",
            rca_report=_valid_rca(f"INC-F1-{tag}", ["spark_jobs/custom_etl.py"]),
            unified_diff=RunWorktreeManager.unified_diff(wt),
            shadow_verification=shadow,
        )
        state = RunState(
            run_id=run_id,
            incident=IncidentInput(incident_id=f"INC-F1-{tag}", title="F1", summary="F1"),
            user_context=UserContext(role="de", resolution_mode="fix"),
            status="AWAITING_APPROVAL",
            metadata={"approval_packet": packet.model_dump(mode="json")},
        )
        CheckpointStore(runs_dir, run_id).append("awaiting", state)

        svc = PromotionService(
            repo,
            runs_dir,
            spark_runner=_MutatingPromoteSpark(),  # type: ignore[arg-type]
            dbt_executor=_failing_dbt,
        )
        with pytest.raises(RuntimeError, match="Production post-commit verification failed"):
            svc.approve(run_id, cleanup_worktree=False)

        if silver_initially_present:
            assert silver_marker.read_bytes() == b"ORIG-SILVER"
        else:
            assert not paths.prod_silver_dir.exists()
        RunWorktreeManager.cleanup(repo, wt)


def test_batch1_f3_duckdb_snapshot_purges_stale_wal_and_resolves_symlink_wal(
    tmp_path: Path,
) -> None:
    """F3: snapshot_production purges stale WAL backups and resolves symlinked DuckDB WAL paths."""
    # Part A: Stale .wal backup from an older checkpoint iteration must not brick prod_db.
    repo = tmp_path / "repo_wal"
    runs_dir = tmp_path / "runs_wal"
    _write_minimal_dbt_repo(repo, dbt_subdir="")
    (repo / ".gitignore").write_text("target/\nlogs/\nwarehouse.duckdb*\n", encoding="utf-8")
    _init_git_repo(repo)

    prod_db = repo / "warehouse.duckdb"
    con = duckdb.connect(str(prod_db))
    try:
        con.execute("CREATE TABLE t (id INT);")
        con.execute("INSERT INTO t VALUES (1);")
        con.execute("CHECKPOINT;")
    finally:
        con.close()

    # Advance checkpoint iteration past iter 1, recording an obsolete iteration-1 WAL as stale backup.
    stale_wal = runs_dir / "run-wal" / ".prod_warehouse_backup.duckdb.wal"
    stale_wal.parent.mkdir(parents=True, exist_ok=True)
    tmp_old = tmp_path / "old.duckdb"
    shutil.copy2(prod_db, tmp_old)
    con_old = duckdb.connect(str(tmp_old))
    con_old.execute("INSERT INTO t VALUES (2);")
    # Save uncheckpointed WAL file if created, or advance main DB to higher checkpoint iteration.
    old_wal_cand = tmp_path / "old.duckdb.wal"
    if old_wal_cand.is_file():
        shutil.copy2(old_wal_cand, stale_wal)
    else:
        stale_wal.write_bytes(b"STALE-WAL-BYTES")
    con_old.close()

    for step in range(2, 5):
        c2 = duckdb.connect(str(prod_db))
        c2.execute(f"INSERT INTO t VALUES ({step});")
        c2.execute("CHECKPOINT;")
        c2.close()

    prod_wal = Path(f"{prod_db}.wal")
    if prod_wal.exists():
        prod_wal.unlink()

    wt = RunWorktreeManager.create(repo, runs_dir, "run-wal")
    (wt / "models" / "marts" / "fct_mrr.sql").write_text("SELECT 101 AS val\n", encoding="utf-8")
    shadow = ShadowVerificationResult(
        passed=True,
        run_id="run-wal",
        worktree_path=str(wt),
        shadow_duckdb_path=str(runs_dir / "run-wal" / "shadow" / "mart" / "warehouse.duckdb"),
        changed_files=["models/marts/fct_mrr.sql"],
    )
    packet = ApprovalPacket(
        run_id="run-wal",
        incident_id="INC-WAL",
        rca_report=_valid_rca("INC-WAL", ["models/marts/fct_mrr.sql"]),
        unified_diff=RunWorktreeManager.unified_diff(wt),
        shadow_verification=shadow,
    )
    state = RunState(
        run_id="run-wal",
        incident=IncidentInput(incident_id="INC-WAL", title="WAL", summary="WAL"),
        user_context=UserContext(role="de", resolution_mode="fix"),
        status="AWAITING_APPROVAL",
        metadata={"approval_packet": packet.model_dump(mode="json")},
    )
    CheckpointStore(runs_dir, "run-wal").append("awaiting", state)

    def _fail_dbt(dbt_paths, silver_overrides, mart_duckdb_path, args):  # type: ignore[no-untyped-def]
        return subprocess.CompletedProcess(["dbt", *args], 1, stdout="", stderr="fail")

    with pytest.raises(RuntimeError, match="Production post-commit verification failed"):
        PromotionService(repo, runs_dir, dbt_executor=_fail_dbt).approve(
            "run-wal", cleanup_worktree=False
        )

    assert not prod_wal.exists()
    reopen = duckdb.connect(str(prod_db), read_only=True)
    try:
        assert [r[0] for r in reopen.execute("SELECT id FROM t ORDER BY 1").fetchall()] == [
            1,
            2,
            3,
            4,
        ]
    finally:
        reopen.close()
    RunWorktreeManager.cleanup(repo, wt)

    # Part B: Non-existent prod_db with pre-existing stale WAL backup must not materialize prod_wal.
    no_db_repo = tmp_path / "repo_nodb"
    no_db_runs = tmp_path / "runs_nodb"
    _write_minimal_dbt_repo(no_db_repo, dbt_subdir="dbt_project")
    no_db_repo_db = LakehousePaths.from_workspace(no_db_repo).prod_duckdb_path
    assert not no_db_repo_db.exists()
    no_db_stale_wal = no_db_runs / "run-nodb" / ".prod_warehouse_backup.duckdb.wal"
    no_db_stale_wal.parent.mkdir(parents=True, exist_ok=True)
    no_db_stale_wal.write_bytes(b"ORPHAN-WAL")
    snap_nodb = DuckDBShadowBackend.from_workspace(no_db_repo).snapshot_production(
        "run-nodb", no_db_runs, affected=AffectedSubgraph(changed_paths=[], spark_jobs=[], dbt_models=[], affected_assets=[])
    )
    assert snap_nodb.duckdb_existed is False
    assert snap_nodb.duckdb_wal_backup_path is None
    assert not no_db_stale_wal.exists()

    # Part C: Symlinked prod_duckdb_path captures and restores <resolved>.wal.
    sym_repo = tmp_path / "repo_sym"
    sym_runs = tmp_path / "runs_sym"
    _write_minimal_dbt_repo(sym_repo, dbt_subdir="dbt_project")
    real_target = tmp_path / "external_real_db"
    real_target.mkdir()
    real_db = real_target / "actual.duckdb"
    real_wal = Path(f"{real_db}.wal")
    real_db.write_bytes(b"REAL-DB-BYTES")
    real_wal.write_bytes(b"REAL-WAL-BYTES")
    sym_db = LakehousePaths.from_workspace(sym_repo).prod_duckdb_path
    sym_db.parent.mkdir(parents=True, exist_ok=True)
    sym_db.symlink_to(real_db)

    sym_backend = DuckDBShadowBackend.from_workspace(sym_repo)
    snap_sym = sym_backend.snapshot_production(
        "run-sym",
        sym_runs,
        affected=AffectedSubgraph(changed_paths=[], spark_jobs=[], dbt_models=[], affected_assets=[]),
    )
    assert snap_sym.duckdb_existed is True
    assert snap_sym.duckdb_wal_backup_path is not None and snap_sym.duckdb_wal_backup_path.is_file()
    real_db.write_bytes(b"CORRUPT-DB")
    real_wal.write_bytes(b"CORRUPT-WAL")
    sym_backend.restore_production(snap_sym)
    assert real_db.read_bytes() == b"REAL-DB-BYTES"
    assert real_wal.read_bytes() == b"REAL-WAL-BYTES"
    sym_backend.cleanup_snapshot(snap_sym)


def test_batch1_f4_reentrant_lock_promotion_lock_window_and_composite_partial_restore(
    tmp_path: Path,
) -> None:
    """F4: Re-entrant flock works, concurrent prepare_shadow never sees mutated DB, Composite restores siblings."""
    db_file = tmp_path / "reentrant.duckdb"
    db_file.write_bytes(b"db")

    with (
        duckdb_snapshot_lock(db_file, exclusive=True, timeout_seconds=0.2),
        duckdb_snapshot_lock(db_file, exclusive=False, timeout_seconds=0.1),
        duckdb_snapshot_lock(db_file, exclusive=True, timeout_seconds=0.1),
    ):
        pass
    with (
        duckdb_snapshot_lock(db_file, exclusive=False, timeout_seconds=0.2),
        duckdb_snapshot_lock(db_file, exclusive=False, timeout_seconds=0.1),
        duckdb_snapshot_lock(db_file, exclusive=True, timeout_seconds=0.1),
    ):
        pass

    # Concurrent prepare_shadow during a failing PromotionService.approve() must never read -999.
    repo = tmp_path / "repo_race"
    runs_dir = tmp_path / "runs_race"
    _write_minimal_dbt_repo(repo, dbt_subdir="")
    (repo / ".gitignore").write_text("target/\nlogs/\nwarehouse.duckdb*\n", encoding="utf-8")
    _init_git_repo(repo)
    prod_db = repo / "warehouse.duckdb"
    c_init = duckdb.connect(str(prod_db))
    c_init.execute("CREATE TABLE mrr AS SELECT 100 AS val;")
    c_init.close()

    wt = RunWorktreeManager.create(repo, runs_dir, "run-race")
    (wt / "models" / "marts" / "fct_mrr.sql").write_text("SELECT 200 AS val\n", encoding="utf-8")
    shadow = ShadowVerificationResult(
        passed=True,
        run_id="run-race",
        worktree_path=str(wt),
        shadow_duckdb_path=str(runs_dir / "run-race" / "shadow" / "mart" / "warehouse.duckdb"),
        changed_files=["models/marts/fct_mrr.sql"],
    )
    packet = ApprovalPacket(
        run_id="run-race",
        incident_id="INC-RACE",
        rca_report=_valid_rca("INC-RACE", ["models/marts/fct_mrr.sql"]),
        unified_diff=RunWorktreeManager.unified_diff(wt),
        shadow_verification=shadow,
    )
    state = RunState(
        run_id="run-race",
        incident=IncidentInput(incident_id="INC-RACE", title="Race", summary="Race"),
        user_context=UserContext(role="de", resolution_mode="fix"),
        status="AWAITING_APPROVAL",
        metadata={"approval_packet": packet.model_dump(mode="json")},
    )
    CheckpointStore(runs_dir, "run-race").append("awaiting", state)

    mutated_event = threading.Event()

    def _corrupt_and_pause_dbt(dbt_paths, silver_overrides, mart_duckdb_path, args):  # type: ignore[no-untyped-def]
        c_mut = duckdb.connect(str(mart_duckdb_path))
        c_mut.execute("CREATE OR REPLACE TABLE mrr AS SELECT -999 AS val;")
        c_mut.close()
        mutated_event.set()
        time.sleep(0.25)
        return subprocess.CompletedProcess(["dbt", *args], 1, stdout="", stderr="fail")

    project = discover_project(repo)
    reader_backend = DuckDBShadowBackend(project, lock_timeout_seconds=10.0)
    wt_read = RunWorktreeManager.create(repo, runs_dir, "run-reader")

    def _concurrent_reader() -> int:
        assert mutated_event.wait(timeout=5.0)
        ctx_r = reader_backend.prepare_shadow("run-reader", wt_read, runs_dir)
        assert ctx_r.shadow_duckdb_path is not None
        con_r = duckdb.connect(str(ctx_r.shadow_duckdb_path), read_only=True)
        try:
            return int(con_r.execute("SELECT val FROM mrr").fetchone()[0])
        finally:
            con_r.close()

    with ThreadPoolExecutor(max_workers=1) as pool:
        fut = pool.submit(_concurrent_reader)
        with pytest.raises(RuntimeError, match="Production post-commit verification failed"):
            PromotionService(repo, runs_dir, dbt_executor=_corrupt_and_pause_dbt).approve(
                "run-race", cleanup_worktree=False
            )
        observed_val = fut.result(timeout=10.0)
    assert observed_val == 100
    RunWorktreeManager.cleanup(repo, wt)
    RunWorktreeManager.cleanup(repo, wt_read)

    # CompositeShadowBackend partial failure guarantees sibling restore and snapshot cleanup.
    comp_repo = tmp_path / "repo_comp_partial"
    comp_runs = tmp_path / "runs_comp_partial"
    _write_minimal_dbt_repo(comp_repo, dbt_subdir="dbt_project")
    (comp_repo / "spark_jobs").mkdir(parents=True)
    comp_project = discover_project(comp_repo)
    comp_paths = LakehousePaths.from_project(comp_project)
    comp_paths.prod_mart_dir.mkdir(parents=True, exist_ok=True)
    comp_paths.prod_duckdb_path.write_bytes(b"ORIG-DB")
    silver_f = comp_paths.prod_silver_dir / "silver_a" / "part.parquet"
    silver_f.parent.mkdir(parents=True, exist_ok=True)
    silver_f.write_bytes(b"ORIG-SILVER")

    comp_backend = CompositeShadowBackend(
        SparkSubprocessBackend(comp_project),
        DuckDBShadowBackend(comp_project),
    )
    affected_sp = AffectedSubgraph(
        changed_paths=["spark_jobs/a.py"],
        spark_jobs=["spark_jobs/a.py"],
        dbt_models=[],
        affected_assets=["silver_a"],
    )
    snap = comp_backend.snapshot_production("run-partial-1", comp_runs, affected=affected_sp)
    silver_f.write_bytes(b"CORRUPT-SILVER")

    def _boom_restore(_snap):  # type: ignore[no-untyped-def]
        raise TimeoutError("duckdb lock timeout")

    comp_backend.warehouse_backend.restore_production = _boom_restore  # type: ignore[method-assign]
    with pytest.raises(TimeoutError, match="duckdb lock timeout"):
        comp_backend.restore_production(snap)
    assert silver_f.read_bytes() == b"ORIG-SILVER"
    comp_backend.cleanup_snapshot(snap)

    # Snapshot partial failure cleans up DB backup.
    def _boom_sp_snap(*_a, **_kw):  # type: ignore[no-untyped-def]
        raise RuntimeError("spark snapshot failed")

    comp_backend2 = CompositeShadowBackend(
        SparkSubprocessBackend(comp_project),
        DuckDBShadowBackend(comp_project),
    )
    comp_backend2.spark_backend.snapshot_production = _boom_sp_snap  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="spark snapshot failed"):
        comp_backend2.snapshot_production("run-partial-2", comp_runs, affected=affected_sp)
    assert not (comp_runs / "run-partial-2" / ".prod_warehouse_backup.duckdb").exists()


def test_batch1_f5_promotion_empty_changed_files_new_file_leak_and_broad_rollback(
    tmp_path: Path,
) -> None:
    """F5: Empty changed_files preserves untracked files, failed git add cleans new files, and DuckDB errors roll back."""
    repo = tmp_path / "repo_f5"
    runs_dir = tmp_path / "runs_f5"
    _write_minimal_dbt_repo(repo, dbt_subdir="dbt_project")
    (repo / ".gitignore").write_text("lakehouse/\ndbt_project/target/\n", encoding="utf-8")
    pre_head = _init_git_repo(repo)
    paths = LakehousePaths.from_workspace(repo)
    paths.prod_mart_dir.mkdir(parents=True, exist_ok=True)
    paths.prod_duckdb_path.write_bytes(b"ORIG-DB-F5")

    keep_note = repo / "notes_keep.txt"
    keep_note.write_text("user local note\n", encoding="utf-8")

    # Part A: Empty ShadowVerificationResult.changed_files + failure preserves keep_note.
    wt_a = RunWorktreeManager.create(repo, runs_dir, "run-f5a")
    (wt_a / "dbt_project" / "models" / "marts" / "fct_mrr.sql").write_text(
        "SELECT 200 AS val\n", encoding="utf-8"
    )
    diff_a = RunWorktreeManager.unified_diff(wt_a)
    shadow_a = ShadowVerificationResult(
        passed=True,
        run_id="run-f5a",
        worktree_path=str(wt_a),
        changed_files=[],
    )
    pkt_a = ApprovalPacket(
        run_id="run-f5a",
        incident_id="INC-F5A",
        rca_report=_valid_rca("INC-F5A", ["dbt_project/models/marts/fct_mrr.sql"]),
        unified_diff=diff_a,
        shadow_verification=shadow_a,
    )
    st_a = RunState(
        run_id="run-f5a",
        incident=IncidentInput(incident_id="INC-F5A", title="F5A", summary="F5A"),
        user_context=UserContext(role="de", resolution_mode="fix"),
        status="AWAITING_APPROVAL",
        metadata={"approval_packet": pkt_a.model_dump(mode="json")},
    )
    CheckpointStore(runs_dir, "run-f5a").append("awaiting", st_a)

    def _corrupt_and_io_fail(dbt_paths, silver_overrides, mart_duckdb_path, args):  # type: ignore[no-untyped-def]
        mart_duckdb_path.write_bytes(b"CORRUPT-F5")
        raise duckdb.IOException("simulated duckdb I/O crash")

    with pytest.raises(duckdb.IOException, match="simulated duckdb I/O crash"):
        PromotionService(repo, runs_dir, dbt_executor=_corrupt_and_io_fail).approve(
            "run-f5a", cleanup_worktree=False
        )

    assert keep_note.read_text(encoding="utf-8") == "user local note\n"
    assert paths.prod_duckdb_path.read_bytes() == b"ORIG-DB-F5"
    cur_head = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
    assert cur_head == pre_head
    RunWorktreeManager.cleanup(repo, wt_a)

    # Part B: Newly applied file is deleted on rollback even when git add fails mid-apply.
    wt_b = RunWorktreeManager.create(repo, runs_dir, "run-f5b")
    (wt_b / "dbt_project" / "models" / "marts" / "new_leaked.sql").write_text(
        "SELECT 1 AS x\n", encoding="utf-8"
    )
    diff_b = RunWorktreeManager.unified_diff(wt_b)
    shadow_b = ShadowVerificationResult(
        passed=True,
        run_id="run-f5b",
        worktree_path=str(wt_b),
        changed_files=["dbt_project/models/marts/new_leaked.sql", "dbt_project/models/marts/missing.sql"],
    )
    pkt_b = ApprovalPacket(
        run_id="run-f5b",
        incident_id="INC-F5B",
        rca_report=_valid_rca("INC-F5B", ["dbt_project/models/marts/new_leaked.sql"]),
        unified_diff=diff_b,
        shadow_verification=shadow_b,
    )
    st_b = RunState(
        run_id="run-f5b",
        incident=IncidentInput(incident_id="INC-F5B", title="F5B", summary="F5B"),
        user_context=UserContext(role="de", resolution_mode="fix"),
        status="AWAITING_APPROVAL",
        metadata={"approval_packet": pkt_b.model_dump(mode="json")},
    )
    CheckpointStore(runs_dir, "run-f5b").append("awaiting", st_b)

    with pytest.raises(subprocess.CalledProcessError):
        PromotionService(repo, runs_dir).approve("run-f5b", cleanup_worktree=False)

    assert not (repo / "dbt_project" / "models" / "marts" / "new_leaked.sql").exists()
    assert keep_note.read_text(encoding="utf-8") == "user local note\n"
    RunWorktreeManager.cleanup(repo, wt_b)


# ---------------------------------------------------------------------------
# Batch 2: Protocol Decoupling, Dialect Safety & Registry Symmetry (F2, F6-F8)
# ---------------------------------------------------------------------------


def test_batch2_f2_non_duckdb_dialect_never_falls_back_to_duckdb_shadow(
    tmp_path: Path,
) -> None:
    """F2: Non-DuckDB dbt projects without manifest.shadow.backend must not fall back to DuckDBShadowBackend."""
    from src.adapters.shadow_backend import (
        SHADOW_BACKEND_REGISTRY,
        create_shadow_backend,
        register_shadow_backend,
    )

    repo = tmp_path / "pg_repo"
    (repo / "models" / "marts").mkdir(parents=True)
    (repo / "dbt_project.yml").write_text(
        'name: pg_app\nprofile: pg_app\nmodel-paths: ["models"]\n',
        encoding="utf-8",
    )
    (repo / "profiles.yml").write_text(
        "pg_app:\n"
        "  target: prod\n"
        "  outputs:\n"
        "    prod:\n"
        "      type: postgres\n"
        "      host: prod.db.internal\n"
        "      user: analytics\n"
        "      dbname: analytics_prod\n"
        "      schema: public\n",
        encoding="utf-8",
    )
    (repo / "models" / "marts" / "fct_orders.sql").write_text(
        "SELECT 1 AS order_id\n", encoding="utf-8"
    )
    _init_git_repo(repo)

    project = discover_project(repo)
    assert project.sql_dialect == "postgres"
    assert project.manifest.shadow.backend is None

    with pytest.raises(NotImplementedError, match="postgres"):
        create_shadow_backend(project)

    class _PgStub:
        backend_name = "postgres"
        preserve_artifacts = False

        def prepare_shadow(self, run_id: str, worktree_dir: Path, runs_dir: Path):  # type: ignore[no-untyped-def]
            from src.adapters.shadow_backend import ShadowContext

            return ShadowContext(
                run_id=run_id,
                backend_name="postgres",
                worktree_dir=worktree_dir,
                shadow_root=runs_dir / run_id / "shadow",
                shadow_schema=f"shadow_{run_id}",
            )

        def execute_subset(self, ctx, *, changed_files, affected, full_refresh=True):  # type: ignore[no-untyped-def]
            from src.adapters.shadow_backend import ShadowExecutionResult

            return ShadowExecutionResult(dbt_run_ok=True, dbt_test_ok=True)

        def diff_against_prod(self, ctx):  # type: ignore[no-untyped-def]
            return []

        def teardown(self, ctx):  # type: ignore[no-untyped-def]
            pass

    register_shadow_backend("postgres", lambda *_a, **_kw: _PgStub())
    try:
        selected = create_shadow_backend(project)
        assert isinstance(selected, _PgStub)
        verifier = ShadowOverlayVerifier(repo, tmp_path / "runs")
        assert isinstance(verifier.backend, _PgStub)
    finally:
        SHADOW_BACKEND_REGISTRY.pop("postgres", None)


def test_batch2_f6_promotion_accepts_non_dbt_replay_backend_with_passed_true(
    tmp_path: Path,
) -> None:
    """F6: PromotionService.approve relies on exec_res.passed rather than requiring dbt_run_ok/dbt_test_ok."""
    from src.adapters.shadow_backend import ProductionSnapshot, ShadowContext, ShadowExecutionResult

    repo = tmp_path / "non_dbt_promo"
    runs_dir = tmp_path / "runs_non_dbt_promo"
    (repo / "jobs").mkdir(parents=True)
    (repo / "jobs" / "pipeline.py").write_text("x = 1\n", encoding="utf-8")
    _init_git_repo(repo)

    class _CustomReplayOnlyBackend:
        backend_name = "custom_spark_remote"
        preserve_artifacts = False

        def prepare_shadow(self, run_id: str, worktree_dir: Path, runs_dir: Path) -> ShadowContext:
            return ShadowContext(
                run_id=run_id,
                backend_name=self.backend_name,
                worktree_dir=worktree_dir,
                shadow_root=runs_dir / run_id / "shadow",
            )

        def execute_subset(self, ctx, *, changed_files, affected, full_refresh=True):  # type: ignore[no-untyped-def]
            return ShadowExecutionResult(spark_runs=[{"rc": 0}], dbt_run_ok=False, dbt_test_ok=False)

        def diff_against_prod(self, ctx):  # type: ignore[no-untyped-def]
            return []

        def teardown(self, ctx):  # type: ignore[no-untyped-def]
            pass

        def snapshot_production(self, run_id, runs_dir, *, affected):  # type: ignore[no-untyped-def]
            return ProductionSnapshot(run_id=run_id, backend_name=self.backend_name)

        def replay_production(self, *, changed_files, affected, full_refresh=True, run_id, runs_dir):  # type: ignore[no-untyped-def]
            return ShadowExecutionResult(spark_runs=[{"rc": 0}], dbt_run_ok=False, dbt_test_ok=False)

        def restore_production(self, snapshot):  # type: ignore[no-untyped-def]
            pass

        def cleanup_snapshot(self, snapshot):  # type: ignore[no-untyped-def]
            pass

    backend = _CustomReplayOnlyBackend()
    wt = RunWorktreeManager.create(repo, runs_dir, "run-f6")
    (wt / "jobs" / "pipeline.py").write_text("x = 2\n", encoding="utf-8")
    shadow = ShadowOverlayVerifier(repo, runs_dir, backend=backend).verify("run-f6", wt)
    assert shadow.passed is True

    pkt = ApprovalPacket(
        run_id="run-f6",
        incident_id="INC-F6",
        rca_report=_valid_rca("INC-F6", ["jobs/pipeline.py"]),
        unified_diff=RunWorktreeManager.unified_diff(wt),
        shadow_verification=shadow,
    )
    st = RunState(
        run_id="run-f6",
        incident=IncidentInput(incident_id="INC-F6", title="F6", summary="F6"),
        user_context=UserContext(role="de", resolution_mode="fix"),
        status="AWAITING_APPROVAL",
        metadata={"approval_packet": pkt.model_dump(mode="json")},
    )
    CheckpointStore(runs_dir, "run-f6").append("awaiting", st)

    promo = PromotionService(repo, runs_dir, shadow_backend=backend).approve("run-f6")
    assert promo.status == "RESOLVED"


def test_batch2_f7_non_duckdb_verifier_never_fabricates_local_duckdb_or_silver_paths(
    tmp_path: Path,
) -> None:
    """F7: Non-DuckDB backends do not fabricate shadow/mart/warehouse.duckdb or trigger stale DuckDB Oracle checks."""
    from src.adapters.shadow_backend import ShadowContext, ShadowExecutionResult

    repo = tmp_path / "pg_ver_repo"
    runs_dir = tmp_path / "runs_pg_ver"
    (repo / "models").mkdir(parents=True)
    (repo / "dbt_project.yml").write_text('name: pg_ver\nmodel-paths: ["models"]\n', encoding="utf-8")
    (repo / "models" / "m.sql").write_text("SELECT 1 AS id\n", encoding="utf-8")
    _init_git_repo(repo)

    class _RemotePgBackend:
        backend_name = "postgres"
        preserve_artifacts = False

        def prepare_shadow(self, run_id: str, worktree_dir: Path, runs_dir: Path) -> ShadowContext:
            return ShadowContext(
                run_id=run_id,
                backend_name="postgres",
                worktree_dir=worktree_dir,
                shadow_root=runs_dir / run_id / "shadow",
                shadow_duckdb_path=None,
                shadow_silver_dir=None,
                shadow_schema=f"shadow_{run_id.replace('-', '_')}",
            )

        def execute_subset(self, ctx, *, changed_files, affected, full_refresh=True):  # type: ignore[no-untyped-def]
            return ShadowExecutionResult(dbt_run_ok=True, dbt_test_ok=True)

        def diff_against_prod(self, ctx):  # type: ignore[no-untyped-def]
            return []

        def teardown(self, ctx):  # type: ignore[no-untyped-def]
            pass

    # Plant a drifting warehouse.duckdb inside shadow/mart/ — non-DuckDB backend must ignore it!
    stale_db = runs_dir / "run-f7-ok" / "shadow" / "mart" / "warehouse.duckdb"
    stale_db.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(stale_db))
    con.execute("CREATE TABLE fct_daily_gmv AS SELECT 100.0 AS net_gmv_usd;")
    con.execute("CREATE TABLE dim_user_ltv AS SELECT 999.0 AS lifetime_gmv_usd;")
    con.close()

    wt = RunWorktreeManager.create(repo, runs_dir, "run-f7-ok")
    (wt / "models" / "m.sql").write_text("SELECT 2 AS id\n", encoding="utf-8")
    backend = _RemotePgBackend()
    res = ShadowOverlayVerifier(repo, runs_dir, backend=backend).verify("run-f7-ok", wt)
    assert res.passed is True, res.failure_reasons
    assert res.shadow_duckdb_path == ""
    assert res.shadow_silver_dir == ""

    # Empty changed_files fast-return also avoids fabricating local silver/duckdb paths.
    wt_empty = RunWorktreeManager.create(repo, runs_dir, "run-f7-empty")
    empty_res = ShadowOverlayVerifier(repo, runs_dir, backend=backend).verify("run-f7-empty", wt_empty)
    assert empty_res.passed is False
    assert empty_res.shadow_silver_dir == ""
    assert empty_res.shadow_duckdb_path == ""
    RunWorktreeManager.cleanup(repo, wt)
    RunWorktreeManager.cleanup(repo, wt_empty)


def test_batch2_f8_explicit_duckdb_backend_symmetry_and_reserved_none_registry(
    tmp_path: Path,
) -> None:
    """F8: Explicit shadow.backend='duckdb' returns DuckDBShadowBackend even with spark_jobs/, and 'none' is reserved."""
    from src.adapters.shadow_backend import create_shadow_backend, register_shadow_backend

    dual_explicit = tmp_path / "dual_explicit_duck"
    (dual_explicit / "spark_jobs").mkdir(parents=True)
    (dual_explicit / "dbt_project").mkdir(parents=True)
    (dual_explicit / "dbt_project" / "dbt_project.yml").write_text("name: dual_ex\n", encoding="utf-8")
    (dual_explicit / "dataops-harness.yaml").write_text(
        "version: 1\nshadow:\n  backend: duckdb\n", encoding="utf-8"
    )
    selected = create_shadow_backend(discover_project(dual_explicit))
    assert isinstance(selected, DuckDBShadowBackend)

    with pytest.raises(ValueError, match="reserved"):
        register_shadow_backend("none", lambda *_a, **_kw: selected)


# ---------------------------------------------------------------------------
# Batch 3: Spark Discovery, Multi-Output Overrides, Data Diff & dbt Subset (F9-F12)
# ---------------------------------------------------------------------------


def test_batch3_f9_spark_discovery_mixed_untracked_config_and_root_dir(
    tmp_path: Path,
) -> None:
    """F9: Resolve mixed tracked+untracked Spark jobs, spark_config.py edits w/o lineage, and spark_jobs_dir='.'."""
    import json

    from src.adapters.lineage_adapter import LineageAdapter

    # 9a + 9b: Mixed OpenLineage-tracked + unlogged Spark scripts & spark_config.py fanout without lineage.
    repo = tmp_path / "repo_f9"
    (repo / "etl_spark").mkdir(parents=True)
    (repo / "openlineage").mkdir(parents=True)
    (repo / "dataops-harness.yaml").write_text("version: 1\nspark_jobs_dir: etl_spark\n", encoding="utf-8")
    (repo / "etl_spark" / "known.py").write_text("# v1\n", encoding="utf-8")
    (repo / "etl_spark" / "new_job.py").write_text("# v1\n", encoding="utf-8")
    (repo / "etl_spark" / "spark_config.py").write_text("# cfg\n", encoding="utf-8")
    (repo / "openlineage" / "ev.jsonl").write_text(
        json.dumps(
            {
                "eventType": "COMPLETE",
                "eventTime": "2026-11-28T00:00:00Z",
                "producer": "spark://etl",
                "job": {
                    "namespace": "spark://prod",
                    "name": "known",
                    "facets": {"sourceCodeLocation": {"path": "etl_spark/known.py"}},
                },
                "inputs": [],
                "outputs": [{"namespace": "spark://prod", "name": "silver_known"}],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    project = discover_project(repo)
    lineage = LineageAdapter.from_project(project)
    backend = SparkSubprocessBackend(project, lineage_adapter=lineage)

    mixed_changed = ["etl_spark/known.py", "etl_spark/new_job.py"]
    mixed_resolved = backend._resolve_spark_jobs(
        changed_files=mixed_changed,
        affected=lineage.affected_downstream(mixed_changed),
    )
    assert mixed_resolved == ["etl_spark/known.py", "etl_spark/new_job.py"]

    # Manifestless spark_config.py change fans out to all runnable Spark jobs in etl_spark/.
    repo_nolineage = tmp_path / "repo_f9_nolin"
    (repo_nolineage / "etl_spark").mkdir(parents=True)
    (repo_nolineage / "dataops-harness.yaml").write_text(
        "version: 1\nspark_jobs_dir: etl_spark\n", encoding="utf-8"
    )
    (repo_nolineage / "etl_spark" / "job_a.py").write_text("# a\n", encoding="utf-8")
    (repo_nolineage / "etl_spark" / "spark_config.py").write_text("# cfg\n", encoding="utf-8")
    proj_nolin = discover_project(repo_nolineage)
    lin_nolin = LineageAdapter.from_project(proj_nolin)
    bk_nolin = SparkSubprocessBackend(proj_nolin, lineage_adapter=lin_nolin)
    cfg_changed = ["etl_spark/spark_config.py"]
    assert bk_nolin._resolve_spark_jobs(
        changed_files=cfg_changed,
        affected=lin_nolin.affected_downstream(cfg_changed),
    ) == ["etl_spark/job_a.py"]

    # 9c: Root spark_jobs_dir='.' resolves top-level .py scripts.
    repo_root_sp = tmp_path / "repo_f9_root"
    repo_root_sp.mkdir()
    (repo_root_sp / "dataops-harness.yaml").write_text("version: 1\nspark_jobs_dir: .\n", encoding="utf-8")
    (repo_root_sp / "job_root.py").write_text("# root\n", encoding="utf-8")
    proj_root = discover_project(repo_root_sp)
    lin_root = LineageAdapter.from_project(proj_root)
    bk_root = SparkSubprocessBackend(proj_root, lineage_adapter=lin_root)
    assert bk_root._resolve_spark_jobs(
        changed_files=["job_root.py"],
        affected=lin_root.affected_downstream(["job_root.py"]),
    ) == ["job_root.py"]


def test_batch3_f10_spark_multi_output_and_case_insensitive_overrides(tmp_path: Path) -> None:
    """F10: Multi-output Spark jobs, case-insensitive upstream matching, and mixed-case OL dir lookup."""
    import json

    from src.adapters.lineage_adapter import LineageAdapter

    repo = tmp_path / "repo_f10"
    runs_dir = tmp_path / "runs_f10"
    (repo / "etl_spark").mkdir(parents=True)
    (repo / "openlineage").mkdir(parents=True)
    (repo / "dataops-harness.yaml").write_text("version: 1\nspark_jobs_dir: etl_spark\n", encoding="utf-8")
    (repo / "etl_spark" / "multi.py").write_text("# multi\n", encoding="utf-8")
    (repo / "etl_spark" / "downstream.py").write_text("# down\n", encoding="utf-8")
    (repo / ".gitignore").write_text("lakehouse/\n", encoding="utf-8")

    ol_events = [
        {
            "eventType": "COMPLETE",
            "eventTime": "2026-11-28T00:00:00Z",
            "producer": "spark://etl",
            "job": {
                "namespace": "spark://prod",
                "name": "multi",
                "facets": {"sourceCodeLocation": {"path": "etl_spark/multi.py"}},
            },
            "inputs": [{"namespace": "s3://raw", "name": "raw_src"}],
            "outputs": [
                {"namespace": "spark://prod", "name": "Silver_Orders"},
                {"namespace": "spark://prod", "name": "silver_order_items"},
            ],
        },
        {
            "eventType": "COMPLETE",
            "eventTime": "2026-11-28T00:01:00Z",
            "producer": "spark://etl",
            "job": {
                "namespace": "spark://prod",
                "name": "downstream",
                "facets": {"sourceCodeLocation": {"path": "etl_spark/downstream.py"}},
            },
            "inputs": [{"namespace": "spark://prod", "name": "SILVER_ORDERS"}],
            "outputs": [{"namespace": "spark://prod", "name": "silver_metrics"}],
        },
    ]
    (repo / "openlineage" / "ev.jsonl").write_text(
        "\n".join(json.dumps(e) for e in ol_events) + "\n", encoding="utf-8"
    )
    _init_git_repo(repo)

    seen_read_dirs: dict[str, str] = {}

    class _MultiRecordingSpark:
        event_log_dir = runs_dir / "ev"

        def run_job(self, script_path: Path, args=None, **_kwargs):  # type: ignore[no-untyped-def]
            arg_list = list(args or [])
            out_dir = Path(arg_list[arg_list.index("--output-dir") + 1])
            if script_path.name == "multi.py":
                (out_dir / "silver_orders").mkdir(parents=True, exist_ok=True)
                (out_dir / "silver_order_items").mkdir(parents=True, exist_ok=True)
            elif script_path.name == "downstream.py":
                if "--silver-read-dir" in arg_list:
                    seen_read_dirs["downstream"] = arg_list[arg_list.index("--silver-read-dir") + 1]
                (out_dir / "silver_metrics").mkdir(parents=True, exist_ok=True)
            return SparkRunResult(
                job_name=script_path.stem,
                script_path=script_path,
                pid=1,
                returncode=0,
                duration_ms=1.0,
                event_log_path=None,
                stdout="ok",
                stderr="",
            )

    project = discover_project(repo)
    lineage = LineageAdapter.from_project(project)
    backend = SparkSubprocessBackend(
        project,
        lineage_adapter=lineage,
        spark_runner=_MultiRecordingSpark(),  # type: ignore[arg-type]
    )
    wt = RunWorktreeManager.create(repo, runs_dir, "run-f10")
    (wt / "etl_spark" / "multi.py").write_text("# multi fixed\n", encoding="utf-8")
    changed = RunWorktreeManager.changed_files(wt)
    ctx = backend.prepare_shadow("run-f10", wt, runs_dir)
    res = backend.execute_subset(
        ctx,
        changed_files=changed,
        affected=lineage.affected_downstream(changed),
    )
    assert res.passed is True
    assert {k.lower() for k in res.silver_overrides} >= {
        "silver_orders",
        "silver_order_items",
        "silver_metrics",
    }
    assert ctx.shadow_silver_dir is not None
    assert Path(seen_read_dirs["downstream"]).resolve() == ctx.shadow_silver_dir.resolve()
    RunWorktreeManager.cleanup(repo, wt)


def test_batch3_f11_data_diff_null_dup_nonunique_type_change_and_aliased_marts(
    tmp_path: Path,
) -> None:
    """F11: _delta_data_diff & DuckDBAdapter.data_diff handle NULL/dup/non-unique/types; aliased marts diff."""
    from src.adapters.duckdb_adapter import DuckDBAdapter
    from src.adapters.lineage_adapter import LineageAdapter
    from src.adapters.shadow_backend import _delta_data_diff
    from tests.test_shadow_backend import _write_delta_table

    # Delta tables covering NULL keys, fanout duplicates, non-unique identical rows, and type changes.
    d_null_p = tmp_path / "d_null_p"
    d_null_s = tmp_path / "d_null_s"
    for d in (d_null_p, d_null_s):
        _write_delta_table(
            d,
            "SELECT 'a' AS id, 1 AS v UNION ALL SELECT NULL::VARCHAR AS id, 2 AS v",
            [("id", "string"), ("v", "integer")],
        )
    diff_null = _delta_data_diff(d_null_p, d_null_s, "t_null", "id")
    assert diff_null is not None
    assert diff_null.added_keys == [] and diff_null.removed_keys == []
    assert diff_null.is_identical is True

    d_dup_p = tmp_path / "d_dup_p"
    d_dup_s = tmp_path / "d_dup_s"
    _write_delta_table(d_dup_p, "SELECT 'a' AS id, 1 AS v", [("id", "string"), ("v", "integer")])
    _write_delta_table(
        d_dup_s,
        "SELECT 'a' AS id, 1 AS v UNION ALL SELECT 'a' AS id, 1 AS v",
        [("id", "string"), ("v", "integer")],
    )
    diff_dup = _delta_data_diff(d_dup_p, d_dup_s, "t_dup", "id")
    assert diff_dup is not None
    assert diff_dup.is_identical is False and diff_dup.changed_keys == ["a"]

    d_nonuniq_p = tmp_path / "d_nu_p"
    d_nonuniq_s = tmp_path / "d_nu_s"
    for d in (d_nonuniq_p, d_nonuniq_s):
        _write_delta_table(
            d,
            "SELECT '2026-01-01' AS order_date, 10 AS v UNION ALL SELECT '2026-01-01' AS order_date, 20 AS v",
            [("order_date", "string"), ("v", "integer")],
        )
    diff_nu = _delta_data_diff(d_nonuniq_p, d_nonuniq_s, "t_nu", "order_date")
    assert diff_nu is not None
    assert diff_nu.changed_keys == [] and diff_nu.is_identical is True

    d_type_p = tmp_path / "d_tp_p"
    d_type_s = tmp_path / "d_tp_s"
    _write_delta_table(d_type_p, "SELECT 'k1' AS id, TRUE AS val", [("id", "string"), ("val", "boolean")])
    _write_delta_table(
        d_type_s,
        "SELECT 'k1' AS id, {'a': 1} AS val",
        [
            ("id", "string"),
            (
                "val",
                {
                    "type": "struct",
                    "fields": [{"name": "a", "type": "integer", "nullable": True, "metadata": {}}],
                },
            ),
        ],
    )
    diff_type = _delta_data_diff(d_type_p, d_type_s, "t_type", "id")
    assert diff_type is not None
    assert diff_type.is_identical is False and diff_type.changed_keys == ["k1"]

    # Matching DuckDBAdapter.data_diff edge cases.
    prod_db = tmp_path / "prod.duckdb"
    shad_db = tmp_path / "shad.duckdb"
    cp = duckdb.connect(str(prod_db))
    cs = duckdb.connect(str(shad_db))
    try:
        for c in (cp, cs):
            c.execute("CREATE TABLE t_null AS SELECT 'a' AS id, 1 AS v UNION ALL SELECT NULL::VARCHAR, 2;")
            c.execute(
                "CREATE TABLE t_nonuniq AS SELECT '2026-01-01' AS order_date, 10 AS v "
                "UNION ALL SELECT '2026-01-01', 20;"
            )
        cp.execute("CREATE TABLE t_dup AS SELECT 'a' AS id, 1 AS v;")
        cs.execute("CREATE TABLE t_dup AS SELECT 'a' AS id, 1 AS v UNION ALL SELECT 'a', 1;")
        cp.execute("CREATE TABLE t_type AS SELECT 'k1' AS id, TRUE AS val;")
        cs.execute("CREATE TABLE t_type AS SELECT 'k1' AS id, {'a': 1} AS val;")
    finally:
        cp.close()
        cs.close()

    adapter = DuckDBAdapter(LakehousePaths.from_workspace(tmp_path))
    assert adapter.data_diff(prod_db, shad_db, "t_null", "id").is_identical is True
    assert adapter.data_diff(prod_db, shad_db, "t_nonuniq", "order_date").is_identical is True
    assert adapter.data_diff(prod_db, shad_db, "t_dup", "id").is_identical is False
    assert adapter.data_diff(prod_db, shad_db, "t_type", "id").changed_keys == ["k1"]

    # Aliased dbt mart table (model `mrr_calc` materialized as `fct_mrr_aliased`).
    alias_repo = tmp_path / "repo_alias"
    (alias_repo / "models" / "marts").mkdir(parents=True)
    (alias_repo / "target").mkdir(parents=True)
    (alias_repo / "dbt_project.yml").write_text('name: alias_app\nmodel-paths: ["models"]\n', encoding="utf-8")
    (alias_repo / "profiles.yml").write_text(
        "alias_app:\n  target: dev\n  outputs:\n    dev:\n      type: duckdb\n      path: 'warehouse.duckdb'\n",
        encoding="utf-8",
    )
    import json

    (alias_repo / "target" / "manifest.json").write_text(
        json.dumps(
            {
                "metadata": {"adapter_type": "duckdb"},
                "nodes": {
                    "model.alias_app.mrr_calc": {
                        "resource_type": "model",
                        "name": "mrr_calc",
                        "alias": "fct_mrr_aliased",
                        "schema": "main",
                        "original_file_path": "models/marts/mrr_calc.sql",
                        "config": {"meta": {"layer": "mart"}},
                        "columns": {"billing_date": {"name": "billing_date"}, "total_mrr_usd": {"name": "total_mrr_usd"}},
                        "depends_on": {"nodes": [], "macros": []},
                    }
                },
                "sources": {},
                "macros": {},
            }
        ),
        encoding="utf-8",
    )
    ca_p = duckdb.connect(str(alias_repo / "warehouse.duckdb"))
    ca_p.execute("CREATE TABLE fct_mrr_aliased AS SELECT '2026-11-01' AS billing_date, 200.0 AS total_mrr_usd;")
    ca_p.close()

    runs_alias = tmp_path / "runs_alias"
    ctx_alias_db = runs_alias / "run-al" / "shadow" / "mart" / "warehouse.duckdb"
    ctx_alias_db.parent.mkdir(parents=True, exist_ok=True)
    ca_s = duckdb.connect(str(ctx_alias_db))
    ca_s.execute("CREATE TABLE fct_mrr_aliased AS SELECT '2026-11-01' AS billing_date, 100.0 AS total_mrr_usd;")
    ca_s.close()

    from src.adapters.shadow_backend import ShadowContext

    proj_alias = discover_project(alias_repo)
    lin_alias = LineageAdapter.from_project(proj_alias)
    bk_alias = DuckDBShadowBackend(proj_alias, lineage_adapter=lin_alias)
    alias_diffs = bk_alias.diff_against_prod(
        ShadowContext(
            run_id="run-al",
            backend_name="duckdb",
            worktree_dir=alias_repo,
            shadow_root=runs_alias / "run-al" / "shadow",
            shadow_duckdb_path=ctx_alias_db,
        )
    )
    assert len(alias_diffs) == 1
    assert alias_diffs[0].table_name == "fct_mrr_aliased"
    assert alias_diffs[0].changed_keys == ["2026-11-01"]


def test_batch3_f12_dbt_root_model_path_and_mixed_subset_selection(tmp_path: Path) -> None:
    """F12: Support model-paths=['.'], new+existing models, and Spark/seed + manifestless dbt."""
    from src.adapters.shadow_backend import _run_dbt_subset, touches_incremental_logic

    # 12a: model-paths=['.'] detects is_incremental() and maps root-relative .sql in _run_dbt_subset.
    prod = tmp_path / "root_mp_prod"
    wt = tmp_path / "root_mp_wt"
    for r in (prod, wt):
        r.mkdir(parents=True)
        (r / "dbt_project.yml").write_text(
            'name: root_mp\nmodel-paths: ["."]\ntest-paths: ["data_tests"]\n',
            encoding="utf-8",
        )
        (r / "data_tests").mkdir()
        (r / "fct_x.sql").write_text(
            "SELECT 1 AS id {% if is_incremental() %} WHERE dt > '2026-01-01' {% endif %}\n",
            encoding="utf-8",
        )
    (wt / "fct_x.sql").write_text(
        "SELECT 1 AS id {% if is_incremental() %} WHERE dt > '2026-11-01' {% endif %}\n",
        encoding="utf-8",
    )
    assert touches_incremental_logic(prod, wt, ["fct_x.sql"], dbt_project_rel="") is True

    cmds_root: list[list[str]] = []

    def _rec_dbt(_paths, _ov, _db, args):  # type: ignore[no-untyped-def]
        cmds_root.append(list(args))
        return subprocess.CompletedProcess(args, 0, stdout="ok", stderr="")

    res_root = _run_dbt_subset(
        code_workspace=wt,
        prod_paths=LakehousePaths.from_workspace(prod),
        silver_overrides={},
        target_duckdb_path=tmp_path / "root.duckdb",
        changed_files=["fct_x.sql"],
        affected_dbt_models=[],
        dbt_executor=_rec_dbt,  # type: ignore[arg-type]
        full_refresh=False,
        dbt_project_rel="",
        spark_runs=[],
    )
    assert res_root is not None and res_root.passed is True
    assert any("run" in c and "fct_x+" in c for c in cmds_root)

    # 12b: Existing manifest model + newly added uncompiled model -> both included in --select.
    ext = tmp_path / "mixed_models"
    (ext / "models" / "marts").mkdir(parents=True)
    (ext / "dbt_project.yml").write_text('name: mm\nmodel-paths: ["models"]\n', encoding="utf-8")
    (ext / "models" / "marts" / "fct_existing.sql").write_text("SELECT 1\n", encoding="utf-8")
    (ext / "models" / "marts" / "fct_new.sql").write_text("SELECT 2\n", encoding="utf-8")
    cmds_existing_new: list[list[str]] = []

    def _rec_existing_new(_paths, _ov, _db, args):  # type: ignore[no-untyped-def]
        cmds_existing_new.append(list(args))
        return subprocess.CompletedProcess(args, 0, stdout="ok", stderr="")

    res_en = _run_dbt_subset(
        code_workspace=ext,
        prod_paths=LakehousePaths.from_workspace(ext),
        silver_overrides={},
        target_duckdb_path=tmp_path / "en.duckdb",
        changed_files=["models/marts/fct_existing.sql", "models/marts/fct_new.sql"],
        affected_dbt_models=["fct_existing"],
        dbt_executor=_rec_existing_new,  # type: ignore[arg-type]
        full_refresh=True,
        dbt_project_rel="",
        spark_runs=[],
    )
    assert res_en is not None and res_en.passed is True
    run_cmd_en = next(c for c in cmds_existing_new if c[0] == "run")
    assert "fct_existing" in run_cmd_en and "fct_new+" in run_cmd_en

    # 12c: Spark + manifestless dbt (and seed + manifestless dbt) runs full dbt rather than narrowing --select.
    cmds_spark_dbt: list[list[str]] = []
    _run_dbt_subset(
        code_workspace=ext,
        prod_paths=LakehousePaths.from_workspace(ext),
        silver_overrides={"silver_x": tmp_path},
        target_duckdb_path=tmp_path / "sd.duckdb",
        changed_files=["spark_jobs/x.py", "models/marts/fct_unrelated.sql"],
        affected_dbt_models=[],
        dbt_executor=lambda _p, _o, _d, a: (cmds_spark_dbt.append(list(a)) or subprocess.CompletedProcess(a, 0, "", "")),
        full_refresh=True,
        dbt_project_rel="",
        spark_runs=[{"job_name": "x", "returncode": 0}],
    )
    run_cmd_sd = next(c for c in cmds_spark_dbt if c[0] == "run")
    assert "--select" not in run_cmd_sd

    (ext / "seeds").mkdir()
    (ext / "seeds" / "rates.csv").write_text("c,r\nUSD,1\n", encoding="utf-8")
    cmds_seed_dbt: list[list[str]] = []
    _run_dbt_subset(
        code_workspace=ext,
        prod_paths=LakehousePaths.from_workspace(ext),
        silver_overrides={},
        target_duckdb_path=tmp_path / "seed.duckdb",
        changed_files=["seeds/rates.csv", "models/marts/fct_other.sql"],
        affected_dbt_models=[],
        dbt_executor=lambda _p, _o, _d, a: (cmds_seed_dbt.append(list(a)) or subprocess.CompletedProcess(a, 0, "", "")),
        full_refresh=True,
        dbt_project_rel="",
        spark_runs=[],
    )
    run_cmd_seed = next(c for c in cmds_seed_dbt if c[0] == "run")
    assert "--select" not in run_cmd_seed

    # Hybrid Spark OpenLineage + manifestless dbt (without profiles.yml) still selects CompositeShadowBackend.
    import json

    from src.adapters.lineage_adapter import LineageAdapter
    from src.adapters.shadow_backend import create_shadow_backend

    ol_hybrid = tmp_path / "ol_hybrid"
    (ol_hybrid / "etl_spark").mkdir(parents=True)
    (ol_hybrid / "openlineage").mkdir(parents=True)
    (ol_hybrid / "models").mkdir(parents=True)
    (ol_hybrid / "dataops-harness.yaml").write_text(
        "version: 1\nspark_jobs_dir: etl_spark\n", encoding="utf-8"
    )
    (ol_hybrid / "dbt_project.yml").write_text('name: hyb\nmodel-paths: ["models"]\n', encoding="utf-8")
    (ol_hybrid / "etl_spark" / "build_silver.py").write_text("# s\n", encoding="utf-8")
    (ol_hybrid / "models" / "fct_u.sql").write_text("SELECT 1\n", encoding="utf-8")
    (ol_hybrid / "openlineage" / "ol.json").write_text(
        json.dumps(
            {
                "eventType": "COMPLETE",
                "eventTime": "2026-11-28T00:00:00Z",
                "producer": "spark://etl",
                "job": {
                    "namespace": "spark://prod",
                    "name": "build_silver",
                    "facets": {"sourceCodeLocation": {"path": "etl_spark/build_silver.py"}},
                },
                "inputs": [],
                "outputs": [{"namespace": "spark://prod", "name": "silver_orders"}],
            }
        ),
        encoding="utf-8",
    )
    hyb_proj = discover_project(ol_hybrid)
    hyb_lin = LineageAdapter.from_project(hyb_proj)
    assert isinstance(create_shadow_backend(hyb_proj, lineage_adapter=hyb_lin), CompositeShadowBackend)

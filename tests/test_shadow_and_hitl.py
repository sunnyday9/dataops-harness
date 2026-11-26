from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import duckdb
import pytest

from src.adapters.base import LakehousePaths, SparkRunResult
from src.adapters.duckdb_adapter import DuckDBAdapter
from src.adapters.lineage_adapter import LineageAdapter
from src.adapters.model_adapter import FakeReplayModelAdapter, TurnResponse
from src.cli import main as cli_main
from src.core.context import CheckpointStore, IncidentInput, ToolCall, UserContext
from src.core.loop import HarnessLoop
from src.core.promotion import PromotionService
from src.core.rca import ApprovalPacket, parse_rca_report
from src.core.shadow_verifier import (
    DefaultIntegrityOracle,
    ShadowOverlayVerifier,
    touches_incremental_logic,
)
from src.core.worktree import RunWorktreeManager
from src.hooks.base import HookPipeline
from src.hooks.stop_gate import StopGateHook
from src.tools.registry import ToolRegistry

REPO_ROOT = Path(__file__).resolve().parent.parent
LIVE_WORKSPACE = REPO_ROOT / "workspace"
SKILLS_DIR = REPO_ROOT / "skills"


def _create_isolated_workspace(tmp_path: Path, *, inject_fct_bug: bool = False) -> Path:
    """Copy the sandbox lakehouse + git repo into `tmp_path/workspace` in ~20ms."""
    ws = tmp_path / "workspace"
    shutil.copytree(
        LIVE_WORKSPACE,
        ws,
        ignore=shutil.ignore_patterns(".git", ".scratch", "logs", ".user.yml", "__pycache__"),
    )

    if inject_fct_bug:
        fct_path = ws / "dbt_project" / "models" / "marts" / "fct_daily_gmv.sql"
        orig = fct_path.read_text(encoding="utf-8")
        buggy = orig.replace("ROUND(SUM(amount_usd), 2)", "ROUND(SUM(amount_usd) * 2.0, 2)")
        assert buggy != orig
        fct_path.write_text(buggy, encoding="utf-8")
        # Materialize the broken baseline inside this isolated workspace's DuckDB.
        paths = LakehousePaths.from_workspace(ws)
        proc = DuckDBAdapter(paths).run_dbt(["run", "--full-refresh", "--select", "fct_daily_gmv"])
        assert proc.returncode == 0, proc.stderr or proc.stdout

    subprocess.run(["git", "init", "-b", "main"], cwd=ws, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "test"], cwd=ws, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "t@e.com"], cwd=ws, check=True, capture_output=True)
    subprocess.run(["git", "add", "."], cwd=ws, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "baseline"], cwd=ws, check=True, capture_output=True)
    return ws


def _rca_json(
    incident_id: str,
    *,
    patched_files: list[str] | None = None,
    engine: str = "dbt",
) -> str:
    return json.dumps(
        {
            "incident_id": incident_id,
            "root_cause_summary": "Duplicate multiplier in fct_daily_gmv inflated daily GMV 2x.",
            "affected_assets": ["fct_daily_gmv"],
            "evidence": ["assert_gmv_reconciliation failed; fct_daily_gmv was 2x dim_user_ltv."],
            "engine": engine,
            "proposed_fix_summary": "Removed * 2.0 factor in fct_daily_gmv.sql and re-ran dbt.",
            "patched_files": patched_files or ["dbt_project/models/marts/fct_daily_gmv.sql"],
            "confidence": "high",
        }
    )


def _fix_fct_patch() -> str:
    return (
        "--- a/dbt_project/models/marts/fct_daily_gmv.sql\n"
        "+++ b/dbt_project/models/marts/fct_daily_gmv.sql\n"
        "@@ -9,6 +9,6 @@\n"
        " SELECT\n"
        "     order_date,\n"
        "     COUNT(*) AS order_count,\n"
        "-    ROUND(SUM(amount_usd) * 2.0, 2) AS net_gmv_usd\n"
        "+    ROUND(SUM(amount_usd), 2) AS net_gmv_usd\n"
        " FROM {{ ref('int_orders_usd') }}\n"
        " {% if is_incremental() %}\n"
    )


def _mart_totals(duckdb_path: Path) -> tuple[float, float]:
    con = duckdb.connect(str(duckdb_path), read_only=True)
    try:
        fct_total = float(con.execute("SELECT SUM(net_gmv_usd) FROM fct_daily_gmv").fetchone()[0])
        dim_total = float(
            con.execute("SELECT SUM(lifetime_gmv_usd) FROM dim_user_ltv").fetchone()[0]
        )
        return round(fct_total, 2), round(dim_total, 2)
    finally:
        con.close()


def test_worktree_isolation_real_dbt_shadow_overlay_and_oracle(tmp_path: Path) -> None:
    """Patches stay isolated in worktree, real dbt passes in shadow, and Oracle blocks test deletion."""
    prod_ws = _create_isolated_workspace(tmp_path, inject_fct_bug=True)
    runs_dir = tmp_path / "runs"
    prod_paths = LakehousePaths.from_workspace(prod_ws)
    prod_before_fct, prod_before_dim = _mart_totals(prod_paths.prod_duckdb_path)
    assert abs(prod_before_fct - (2.0 * prod_before_dim)) <= 0.05

    # 1. Detached worktree fixes fct_daily_gmv without touching prod files or prod DuckDB.
    wt = RunWorktreeManager.create(prod_ws, runs_dir, "run-wt-1")
    fct_wt = wt / "dbt_project" / "models" / "marts" / "fct_daily_gmv.sql"
    fct_wt.write_text(
        fct_wt.read_text(encoding="utf-8").replace(
            "ROUND(SUM(amount_usd) * 2.0, 2)", "ROUND(SUM(amount_usd), 2)"
        ),
        encoding="utf-8",
    )

    verifier = ShadowOverlayVerifier(prod_ws, runs_dir)
    result = verifier.verify("run-wt-1", wt)
    assert result.passed is True, result.failure_reasons
    assert result.dbt_run_ok is True and result.dbt_test_ok is True
    assert result.changed_files == ["dbt_project/models/marts/fct_daily_gmv.sql"]
    assert result.oracle_verdict is not None and result.oracle_verdict.passed is True
    assert any(
        d.table_name == "fct_daily_gmv" and not d.is_identical for d in result.data_diffs
    )

    # Production workspace SQL and DuckDB remain untouched before approval.
    assert "* 2.0" in (prod_ws / "dbt_project/models/marts/fct_daily_gmv.sql").read_text(
        encoding="utf-8"
    )
    assert _mart_totals(prod_paths.prod_duckdb_path) == (prod_before_fct, prod_before_dim)

    # Shadow DuckDB has healed, reconciled totals.
    shadow_fct, shadow_dim = _mart_totals(Path(result.shadow_duckdb_path))
    assert abs(shadow_fct - shadow_dim) <= 0.01

    # 2. Oracle rejects deleting singular or schema tests.
    wt_cheat = RunWorktreeManager.create(prod_ws, runs_dir, "run-wt-cheat")
    (wt_cheat / "dbt_project" / "tests" / "assert_gmv_reconciliation.sql").unlink()
    oracle_verdict = DefaultIntegrityOracle().verify(
        prod_workspace=prod_ws,
        worktree_dir=wt_cheat,
        shadow_silver_dir=runs_dir / "run-wt-1" / "shadow" / "silver",
        shadow_duckdb_path=Path(result.shadow_duckdb_path),
        changed_files=["dbt_project/tests/assert_gmv_reconciliation.sql"],
        unified_diff="",
    )
    assert oracle_verdict.passed is False
    assert any("assert_gmv_reconciliation.sql" in r for r in oracle_verdict.reasons)

    RunWorktreeManager.cleanup(prod_ws, wt)
    RunWorktreeManager.cleanup(prod_ws, wt_cheat)
    assert not wt.exists()


def test_shadow_overlay_env_routing_uses_shadow_silver_over_prod_fallback(
    tmp_path: Path,
) -> None:
    """Patched build_silver_orders.py reads prod silver_users and overrides only shadow silver_orders."""
    prod_ws = _create_isolated_workspace(tmp_path, inject_fct_bug=False)
    runs_dir = tmp_path / "runs"
    wt = RunWorktreeManager.create(prod_ws, runs_dir, "run-silver-shadow")
    spark_file = wt / "spark_jobs" / "build_silver_orders.py"
    spark_file.write_text(
        spark_file.read_text(encoding="utf-8") + "\n# verified idempotent overwrite\n",
        encoding="utf-8",
    )

    recorded_spark_calls: list[tuple[Path, list[str]]] = []
    recorded_dbt_overlays: list[tuple[dict[str, Path], list[str]]] = []

    class FakeShadowSparkRunner:
        def run_job(self, script_path: Path, args: list[str] | None = None, **_kwargs) -> SparkRunResult:
            resolved_args = list(args or [])
            recorded_spark_calls.append((script_path, resolved_args))
            out_dir = Path(resolved_args[resolved_args.index("--output-dir") + 1])
            src_orders = LakehousePaths.from_workspace(prod_ws).prod_silver_dir / "silver_orders"
            shutil.copytree(src_orders, out_dir / "silver_orders", dirs_exist_ok=True)
            return SparkRunResult(
                job_name=script_path.stem,
                script_path=script_path,
                pid=99999,
                returncode=0,
                duration_ms=5.0,
                event_log_path=None,
                stdout="ok",
                stderr="",
            )

    def spying_dbt_executor(
        dbt_paths: LakehousePaths,
        silver_overrides: dict[str, Path],
        mart_duckdb_path: Path,
        args: list[str],
    ):
        recorded_dbt_overlays.append((dict(silver_overrides), list(args)))
        return DuckDBAdapter(
            dbt_paths,
            silver_overrides=silver_overrides,
            mart_duckdb_path=mart_duckdb_path,
        ).run_dbt(args)

    verifier = ShadowOverlayVerifier(
        prod_ws,
        runs_dir,
        spark_runner=FakeShadowSparkRunner(),  # type: ignore[arg-type]
        dbt_executor=spying_dbt_executor,
    )
    res = verifier.verify("run-silver-shadow", wt)
    assert res.passed is True, res.failure_reasons
    assert len(recorded_spark_calls) == 1
    script_called, args_called = recorded_spark_calls[0]
    assert script_called == (wt / "spark_jobs" / "build_silver_orders.py").resolve()
    prod_silver = str(LakehousePaths.from_workspace(prod_ws).prod_silver_dir)
    assert args_called[args_called.index("--silver-read-dir") + 1] == prod_silver

    # Shadow Overlay overrode ONLY silver_orders while silver_users/silver_exchange_rates used prod.
    expected_shadow_orders = (runs_dir / "run-silver-shadow" / "shadow" / "silver" / "silver_orders").resolve()
    assert recorded_dbt_overlays[0][0] == {"silver_orders": expected_shadow_orders}


def test_stop_gate_hook_diagnosis_mode_and_fix_mode_reject_then_pass(tmp_path: Path) -> None:
    """StopGateHook validates RCA JSON in diagnosis mode and enforces Shadow Overlay in fix mode."""
    prod_ws = _create_isolated_workspace(tmp_path, inject_fct_bug=True)
    runs_dir = tmp_path / "runs"

    # 1. Diagnosis-only mode: non-JSON rejected, valid RCA transitions to DIAGNOSED
    da_model = FakeReplayModelAdapter(
        [
            TurnResponse(content="The GMV looks doubled."),
            TurnResponse(content=f"```json\n{_rca_json('INC-DA', patched_files=[])}\n```"),
        ]
    )
    da_loop = HarnessLoop(
        model_adapter=da_model,
        tool_registry=ToolRegistry(workspace_dir=prod_ws, skills_dir=SKILLS_DIR),
        hooks=HookPipeline(stop_hooks=[StopGateHook()]),
        runs_dir=runs_dir,
    )
    da_state = da_loop.run(
        IncidentInput(incident_id="INC-DA", title="GMV 2x", summary="fct_daily_gmv mismatch"),
        UserContext(role="da"),
        run_id="run-da-1",
    )
    assert da_state.status == "DIAGNOSED"
    assert da_state.turn == 2
    assert parse_rca_report(da_state.final_output or "").incident_id == "INC-DA"

    # 2. Fix mode: failed shadow rejected by StopGateHook, green shadow emits ApprovalPacket
    wt = RunWorktreeManager.create(prod_ws, runs_dir, "run-fix-stop")
    verifier = ShadowOverlayVerifier(prod_ws, runs_dir)
    stop_hook = StopGateHook(verifier=verifier, worktree_dir=wt, runs_dir=runs_dir)

    de_model = FakeReplayModelAdapter(
        [
            # Turn 1: tries to stop without patching anything -> rejected by Shadow Overlay gate
            TurnResponse(content=_rca_json("INC-FIX")),
            # Turn 2: applies real fix in worktree
            TurnResponse(
                tool_calls=[
                    ToolCall(id="patch_1", name="apply_patch", arguments={"patch": _fix_fct_patch()})
                ]
            ),
            # Turn 3: stops with valid RCA JSON -> shadow passes -> AWAITING_APPROVAL
            TurnResponse(content=_rca_json("INC-FIX")),
        ]
    )
    de_loop = HarnessLoop(
        model_adapter=de_model,
        tool_registry=ToolRegistry(
            workspace_dir=wt,
            skills_dir=SKILLS_DIR,
            lineage_adapter=LineageAdapter.from_workspace(prod_ws),
        ),
        hooks=HookPipeline(stop_hooks=[stop_hook]),
        runs_dir=runs_dir,
    )
    de_state = de_loop.run(
        IncidentInput(incident_id="INC-FIX", title="GMV 2x", summary="fct_daily_gmv mismatch"),
        UserContext(role="de"),
        run_id="run-fix-stop",
    )
    assert de_state.status == "AWAITING_APPROVAL"
    assert de_state.turn == 3
    packet_path = runs_dir / "run-fix-stop" / "approval_packet.json"
    assert packet_path.is_file()
    packet = ApprovalPacket.model_validate_json(packet_path.read_text(encoding="utf-8"))
    assert packet.shadow_verification.passed is True
    assert "-    ROUND(SUM(amount_usd) * 2.0, 2)" in packet.unified_diff


def test_full_hitl_reject_extension_approve_promotion_and_cli(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Verify resume --reject (+8 turns, max 2) and resume --approve Promotion via CLI."""
    prod_ws = _create_isolated_workspace(tmp_path, inject_fct_bug=True)
    runs_dir = tmp_path / "runs"
    run_id = "run-hitl-full"
    wt = RunWorktreeManager.create(prod_ws, runs_dir, run_id)
    verifier = ShadowOverlayVerifier(prod_ws, runs_dir)

    initial_model = FakeReplayModelAdapter(
        [
            TurnResponse(
                tool_calls=[
                    ToolCall(id="p1", name="apply_patch", arguments={"patch": _fix_fct_patch()})
                ]
            ),
            TurnResponse(content=_rca_json("INC-HITL")),
        ]
    )
    loop = HarnessLoop(
        model_adapter=initial_model,
        tool_registry=ToolRegistry(workspace_dir=wt, skills_dir=SKILLS_DIR),
        hooks=HookPipeline(stop_hooks=[StopGateHook(verifier=verifier, worktree_dir=wt)]),
        runs_dir=runs_dir,
    )
    state = loop.run(
        IncidentInput(incident_id="INC-HITL", title="GMV 2x", summary="fct_daily_gmv mismatch"),
        UserContext(role="de"),
        run_id=run_id,
    )
    assert state.status == "AWAITING_APPROVAL"
    assert state.max_turns == 20 and state.rejection_count == 0

    # 1. CLI status reports AWAITING_APPROVAL
    assert cli_main(["status", run_id, "--runs-dir", str(runs_dir)]) == 0
    status_out = json.loads(capsys.readouterr().out)
    assert status_out["status"] == "AWAITING_APPROVAL"
    assert status_out["has_approval_packet"] is True

    # 2. First CLI resume --reject adds +8 turns (20 -> 28) and re-runs loop
    def make_resumed_loop(ws_path: Path, r_dir: Path) -> HarnessLoop:
        return HarnessLoop(
            model_adapter=FakeReplayModelAdapter(
                [TurnResponse(content=_rca_json("INC-HITL"))]
            ),
            tool_registry=ToolRegistry(workspace_dir=wt, skills_dir=SKILLS_DIR),
            hooks=HookPipeline(stop_hooks=[StopGateHook(verifier=verifier, worktree_dir=wt)]),
            runs_dir=r_dir,
        )

    assert (
        cli_main(
            [
                "resume",
                run_id,
                "--reject",
                "--reason",
                "Clarify root cause impact on dim_user_ltv parity.",
                "--workspace",
                str(prod_ws),
                "--runs-dir",
                str(runs_dir),
            ],
            loop_factory=make_resumed_loop,
        )
        == 0
    )
    rej1_out = json.loads(capsys.readouterr().out)
    assert rej1_out["status"] == "AWAITING_APPROVAL"
    assert rej1_out["max_turns"] == 28
    assert rej1_out["rejection_count"] == 1

    # 3. Second reject adds +8 turns (28 -> 36)
    resumed_loop_2 = make_resumed_loop(prod_ws, runs_dir)
    service = PromotionService(prod_ws, runs_dir)
    state2 = service.reject(run_id, "Double check incremental safety.", loop=resumed_loop_2)
    assert state2.status == "AWAITING_APPROVAL"
    assert state2.max_turns == 36
    assert state2.rejection_count == 2

    # 4. Third reject exceeds max_rejections (2) -> CLI exits cleanly with code 1
    assert (
        cli_main(
            [
                "resume",
                run_id,
                "--reject",
                "--reason",
                "One too many rejections",
                "--workspace",
                str(prod_ws),
                "--runs-dir",
                str(runs_dir),
            ]
        )
        == 1
    )
    _ = capsys.readouterr()

    # 5. CLI resume --approve promotes patch to prod sandbox git + production lakehouse -> RESOLVED
    assert (
        cli_main(
            [
                "resume",
                run_id,
                "--approve",
                "--workspace",
                str(prod_ws),
                "--runs-dir",
                str(runs_dir),
            ]
        )
        == 0
    )
    approve_out = json.loads(capsys.readouterr().out)
    assert approve_out["status"] == "RESOLVED"
    assert approve_out["dbt_run_ok"] is True
    assert approve_out["dbt_test_ok"] is True
    assert not wt.exists()

    # Verify production SQL commit + healed production lakehouse numbers
    assert "* 2.0" not in (prod_ws / "dbt_project/models/marts/fct_daily_gmv.sql").read_text(
        encoding="utf-8"
    )
    prod_git_msg = subprocess.run(
        ["git", "-C", str(prod_ws), "log", "-n", "1", "--pretty=%s"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert "INC-HITL" in prod_git_msg
    prod_fct, prod_dim = _mart_totals(LakehousePaths.from_workspace(prod_ws).prod_duckdb_path)
    assert abs(prod_fct - prod_dim) <= 0.01

    # Re-approving an already RESOLVED run is rejected cleanly
    assert CheckpointStore(runs_dir, run_id).load_latest().status == "RESOLVED"
    with pytest.raises(ValueError, match="RESOLVED"):
        service.approve(run_id)


def test_incremental_block_detection_controls_full_refresh(tmp_path: Path) -> None:
    """Editing inside {% if is_incremental() %} disables full-refresh; outside keeps full-refresh."""
    prod_ws = tmp_path / "prod"
    wt_ws = tmp_path / "wt"
    rel = "dbt_project/models/marts/fct_daily_gmv.sql"
    (prod_ws / "dbt_project" / "models" / "marts").mkdir(parents=True)
    (wt_ws / "dbt_project" / "models" / "marts").mkdir(parents=True)

    prod_sql = (
        "SELECT order_date, ROUND(SUM(amount_usd) * 2.0, 2) AS net_gmv_usd\n"
        "FROM {{ ref('int_orders_usd') }}\n"
        "{% if is_incremental() %}\n"
        "WHERE order_date > (SELECT MAX(order_date) FROM {{ this }})\n"
        "{% endif %}\n"
    )
    (prod_ws / rel).write_text(prod_sql, encoding="utf-8")

    # 1. Fix strictly inside {% if is_incremental() %} (`>` -> `>=`): incremental mode enabled
    (wt_ws / rel).write_text(
        prod_sql.replace("WHERE order_date >", "WHERE order_date >="),
        encoding="utf-8",
    )
    assert touches_incremental_logic(prod_ws, wt_ws, [rel]) is True

    # 2. Fix outside {% if is_incremental() %} (`* 2.0` -> 1x): full-refresh enabled
    (wt_ws / rel).write_text(
        prod_sql.replace("SUM(amount_usd) * 2.0", "SUM(amount_usd)"),
        encoding="utf-8",
    )
    assert touches_incremental_logic(prod_ws, wt_ws, [rel]) is False


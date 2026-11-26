from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

import duckdb

from benchmarks.scenarios.base import (
    FaultScenario,
    InjectionOutcome,
    build_unified_diff,
    commit_sandbox_fault,
    read_mart_metrics,
    run_sandbox_dbt,
)
from src.adapters.base import LakehousePaths
from src.core.aggregator import RawAlert
from src.core.context import IncidentInput
from src.core.rca import OracleVerdict
from src.core.shadow_verifier import DefaultIntegrityOracle, HiddenRegressionOracle

_NOT_IN_PATTERN = re.compile(r"\bNOT\s+IN\s*\(", re.IGNORECASE)
_INCREMENTAL_MAT_RE = re.compile(r"materialized\s*=\s*['\"]incremental['\"]", re.IGNORECASE)


def _sql_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _append_duplicate_fx_partition_commit(silver_rates_dir: Path, rate_date: str) -> None:
    """Append duplicate `rate_date` FX rows to Delta `silver_exchange_rates` via DuckDB + log."""
    dup_file = silver_rates_dir / "part-00001-dup-fx.snappy.parquet"
    con = duckdb.connect(":memory:")
    try:
        try:
            con.execute("LOAD delta;")
        except duckdb.Error:
            con.execute("INSTALL delta; LOAD delta;")
        con.execute(
            f"COPY (SELECT rate_date, currency, rate_to_usd "
            f"FROM delta_scan({_sql_literal(str(silver_rates_dir.resolve()))}) "
            f"WHERE rate_date = DATE '{rate_date}') "
            f"TO {_sql_literal(str(dup_file.resolve()))} (FORMAT PARQUET);"
        )
    finally:
        con.close()

    now_ms = int(time.time() * 1000)
    delta_log = silver_rates_dir / "_delta_log" / "00000000000000000001.json"
    delta_log.write_text(
        json.dumps(
            {
                "commitInfo": {
                    "timestamp": now_ms,
                    "operation": "WRITE",
                    "operationParameters": {"mode": "Append"},
                }
            }
        )
        + "\n"
        + json.dumps(
            {
                "add": {
                    "path": dup_file.name,
                    "partitionValues": {},
                    "size": dup_file.stat().st_size,
                    "modificationTime": now_ms,
                    "dataChange": True,
                }
            }
        )
        + "\n",
        encoding="utf-8",
    )


def _append_late_orders_partition_commit(
    silver_orders_dir: Path, *, order_date: str, count: int, amount_usd: float
) -> None:
    """Append late completed USD orders into Delta `silver_orders` for `order_date`."""
    part_dir = silver_orders_dir / f"order_date={order_date}"
    part_dir.mkdir(parents=True, exist_ok=True)
    late_file = part_dir / "part-00001-late-orders.snappy.parquet"

    con = duckdb.connect(":memory:")
    try:
        con.execute(
            f"COPY ("
            f"SELECT 'ORD-LATE-' || LPAD(i::VARCHAR, 4, '0') AS order_id, "
            f"'U0001' AS user_id, 'USD' AS currency, {amount_usd}::DOUBLE AS amount_local, "
            f"'completed' AS status, 'US' AS user_country "
            f"FROM range(1, {count + 1}) t(i)"
            f") TO {_sql_literal(str(late_file.resolve()))} (FORMAT PARQUET);"
        )
    finally:
        con.close()

    now_ms = int(time.time() * 1000)
    delta_log = silver_orders_dir / "_delta_log" / "00000000000000000001.json"
    delta_log.write_text(
        json.dumps(
            {
                "commitInfo": {
                    "timestamp": now_ms,
                    "operation": "WRITE",
                    "operationParameters": {"mode": "Append"},
                }
            }
        )
        + "\n"
        + json.dumps(
            {
                "add": {
                    "path": f"order_date={order_date}/{late_file.name}",
                    "partitionValues": {"order_date": order_date},
                    "size": late_file.stat().st_size,
                    "modificationTime": now_ms,
                    "dataChange": True,
                }
            }
        )
        + "\n",
        encoding="utf-8",
    )


class _ParityDbtOracle:
    """Shared dbt Oracle base enforcing integrity, GMV conservation, and optional extra rules."""

    oracle_name = "dbt_parity_oracle"

    def __init__(
        self,
        *,
        expected_gmv_total: float,
        expected_order_count: int,
        expected_daily_rows: int = 7,
    ) -> None:
        self.expected_gmv_total = expected_gmv_total
        self.expected_order_count = expected_order_count
        self.expected_daily_rows = expected_daily_rows
        self._integrity = DefaultIntegrityOracle()

    def _extra_checks(self, worktree_dir: Path, reasons: list[str]) -> None:
        return

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
        base = self._integrity.verify(
            prod_workspace=prod_workspace,
            worktree_dir=worktree_dir,
            shadow_silver_dir=shadow_silver_dir,
            shadow_duckdb_path=shadow_duckdb_path,
            changed_files=changed_files,
            unified_diff=unified_diff,
        )
        reasons = list(base.reasons)
        metrics: dict[str, Any] = dict(base.metrics)

        if not shadow_duckdb_path.is_file():
            reasons.append("Shadow DuckDB warehouse was not generated.")
        else:
            mart = read_mart_metrics(shadow_duckdb_path)
            metrics.update(mart)
            if abs(float(mart["fct_daily_gmv_total"]) - self.expected_gmv_total) > 0.01:
                reasons.append(
                    f"Expected conserved GMV ${self.expected_gmv_total:.2f}, "
                    f"got ${float(mart['fct_daily_gmv_total']):.2f}."
                )
            if int(mart["fct_order_count"]) != self.expected_order_count:
                reasons.append(
                    f"Expected {self.expected_order_count} completed orders, "
                    f"got {int(mart['fct_order_count'])}."
                )
            if int(mart["daily_gmv_rows"]) != self.expected_daily_rows:
                reasons.append(
                    f"Expected {self.expected_daily_rows} daily mart rows, "
                    f"got {int(mart['daily_gmv_rows'])}."
                )

        self._extra_checks(worktree_dir, reasons)
        return OracleVerdict(
            passed=not reasons,
            oracle_name=self.oracle_name,
            reasons=reasons,
            metrics=metrics,
        )


class DbtFxFanoutOracle(_ParityDbtOracle):
    oracle_name = "dbt_fx_fanout_oracle"


class DbtFxFanoutScenario(FaultScenario):
    """Scenario 5: Upstream duplicate-key FX rows causing many-to-many join fan-out."""

    scenario_id = "dbt_fx_fanout"
    title = "Upstream duplicate-key exchange-rate fan-out in int_orders_usd"
    category = "dbt_cross_engine"

    def inject(
        self, workspace_dir: Path, *, event_log_dir: Path | None = None
    ) -> InjectionOutcome:
        paths = LakehousePaths.from_workspace(workspace_dir)
        baseline_mart = read_mart_metrics(paths.prod_duckdb_path)

        _append_duplicate_fx_partition_commit(
            paths.prod_silver_dir / "silver_exchange_rates", "2026-11-07"
        )
        dbt_run = run_sandbox_dbt(workspace_dir, ["run", "--full-refresh"])
        if dbt_run.returncode != 0:
            raise RuntimeError(f"dbt run failed in FX fan-out inject: {dbt_run.stderr or dbt_run.stdout}")
        run_sandbox_dbt(workspace_dir, ["test"])
        inflated = read_mart_metrics(paths.prod_duckdb_path)

        if int(inflated["fct_order_count"]) <= int(baseline_mart["fct_order_count"]):
            raise RuntimeError("FX duplicate-key fan-out failed to increase order count.")

        stg_rel = "dbt_project/models/staging/stg_exchange_rates.sql"
        stg_text = (workspace_dir / stg_rel).read_text(encoding="utf-8")
        incident = IncidentInput(
            incident_id=f"INC-{self.scenario_id.upper()}",
            title="Daily GMV and order count spiked on 2026-11-07 due to FX fan-out",
            summary=(
                f"Mart order count inflated from {baseline_mart['fct_order_count']} to "
                f"{inflated['fct_order_count']} and GMV surged from "
                f"${baseline_mart['fct_daily_gmv_total']:.2f} to ${inflated['fct_daily_gmv_total']:.2f}."
            ),
            affected_assets=["stg_exchange_rates", "int_orders_usd", "fct_daily_gmv", "dim_user_ltv"],
            shared_ancestors=["silver_exchange_rates", "stg_exchange_rates"],
        )
        return InjectionOutcome(
            scenario_id=self.scenario_id,
            category=self.category,
            workspace_dir=workspace_dir,
            incident=incident,
            raw_alerts=[
                RawAlert(
                    alert_id=f"alt-{self.scenario_id}-1",
                    asset="fct_daily_gmv",
                    timestamp="2026-11-26T00:00:00Z",
                    severity="high",
                    alert_type="gmv_fanout",
                    summary=incident.summary,
                ),
                RawAlert(
                    alert_id=f"alt-{self.scenario_id}-2",
                    asset="dim_user_ltv",
                    timestamp="2026-11-26T00:01:00Z",
                    severity="high",
                    alert_type="ltv_fanout",
                    summary=incident.summary,
                ),
            ],
            baseline_gmv_total=float(baseline_mart["fct_daily_gmv_total"]),
            baseline_order_count=int(baseline_mart["fct_order_count"]),
            fault_reproduced=True,
            reproduction_summary=(
                f"orders {baseline_mart['fct_order_count']}->{inflated['fct_order_count']}, "
                f"gmv ${baseline_mart['fct_daily_gmv_total']:.2f}->${inflated['fct_daily_gmv_total']:.2f}"
            ),
            metadata={
                "orig_stg_fx": stg_text,
                "inflated_order_count": int(inflated["fct_order_count"]),
                "inflated_gmv_total": float(inflated["fct_daily_gmv_total"]),
            },
        )

    def build_oracle(self, outcome: InjectionOutcome) -> HiddenRegressionOracle:
        return DbtFxFanoutOracle(
            expected_gmv_total=outcome.baseline_gmv_total,
            expected_order_count=outcome.baseline_order_count,
        )

    def reference_fix_patch(self, outcome: InjectionOutcome) -> str:
        rel = "dbt_project/models/staging/stg_exchange_rates.sql"
        old_text = str(outcome.metadata["orig_stg_fx"])
        new_text = (
            old_text.rstrip()
            + "\nQUALIFY ROW_NUMBER() OVER (PARTITION BY rate_date, currency ORDER BY rate_to_usd) = 1\n"
        )
        return build_unified_diff(rel, old_text, new_text)


class DbtBrokenMacroOracle(_ParityDbtOracle):
    oracle_name = "dbt_broken_macro_oracle"

    def _extra_checks(self, worktree_dir: Path, reasons: list[str]) -> None:
        macro_text = (
            worktree_dir / "dbt_project" / "macros" / "currency_utils.sql"
        ).read_text(encoding="utf-8")
        if "/" in macro_text or "*" not in macro_text:
            reasons.append("currency_utils.sql macro was not restored to multiply amount * rate.")


class DbtBrokenMacroGitScenario(FaultScenario):
    """Scenario 6: Broken shared Jinja macro introduced in a recent sandbox Git commit."""

    scenario_id = "dbt_broken_macro_git"
    title = "Broken shared Jinja macro (`to_usd` division bug) in recent sandbox Git commit"
    category = "dbt_cross_engine"

    def inject(
        self, workspace_dir: Path, *, event_log_dir: Path | None = None
    ) -> InjectionOutcome:
        paths = LakehousePaths.from_workspace(workspace_dir)
        baseline_mart = read_mart_metrics(paths.prod_duckdb_path)

        macro_rel = "dbt_project/macros/currency_utils.sql"
        macro_path = workspace_dir / macro_rel
        orig_text = macro_path.read_text(encoding="utf-8")
        faulty_text = orig_text.replace(
            "* CAST({{ rate_col }} AS DOUBLE)",
            "/ NULLIF(CAST({{ rate_col }} AS DOUBLE), 0)",
        )
        if faulty_text == orig_text:
            raise RuntimeError("Unexpected currency_utils.sql macro contents.")
        macro_path.write_text(faulty_text, encoding="utf-8")
        bad_sha = commit_sandbox_fault(
            workspace_dir,
            "refactor(macros): invert FX normalization in to_usd",
            [macro_rel],
        )

        dbt_run = run_sandbox_dbt(workspace_dir, ["run", "--full-refresh"])
        if dbt_run.returncode != 0:
            raise RuntimeError(f"dbt run failed in broken macro inject: {dbt_run.stderr or dbt_run.stdout}")
        distorted = read_mart_metrics(paths.prod_duckdb_path)
        if float(distorted["fct_daily_gmv_total"]) <= baseline_mart["fct_daily_gmv_total"] * 2.0:
            raise RuntimeError("Broken to_usd macro did not distort GMV as expected.")

        incident = IncidentInput(
            incident_id=f"INC-{self.scenario_id.upper()}",
            title="Mart GMV exploded >100x after recent macro commit",
            summary=(
                f"Both fct_daily_gmv (${distorted['fct_daily_gmv_total']:.2f}) and dim_user_ltv "
                f"diverged sharply from baseline (${baseline_mart['fct_daily_gmv_total']:.2f}) "
                f"right after commit {bad_sha}."
            ),
            affected_assets=["fct_daily_gmv", "dim_user_ltv", "int_orders_usd"],
            shared_ancestors=["int_orders_usd"],
        )
        return InjectionOutcome(
            scenario_id=self.scenario_id,
            category=self.category,
            workspace_dir=workspace_dir,
            incident=incident,
            raw_alerts=[
                RawAlert(
                    alert_id=f"alt-{self.scenario_id}-1",
                    asset="fct_daily_gmv",
                    timestamp="2026-11-26T00:00:00Z",
                    severity="critical",
                    alert_type="gmv_distortion",
                    summary=incident.summary,
                ),
            ],
            baseline_gmv_total=float(baseline_mart["fct_daily_gmv_total"]),
            baseline_order_count=int(baseline_mart["fct_order_count"]),
            fault_reproduced=True,
            reproduction_summary=f"commit={bad_sha}, gmv=${distorted['fct_daily_gmv_total']:.2f}",
            metadata={
                "fault_commit_sha": bad_sha,
                "faulty_macro": faulty_text,
                "clean_macro": orig_text,
                "distorted_gmv_total": float(distorted["fct_daily_gmv_total"]),
            },
        )

    def build_oracle(self, outcome: InjectionOutcome) -> HiddenRegressionOracle:
        return DbtBrokenMacroOracle(
            expected_gmv_total=outcome.baseline_gmv_total,
            expected_order_count=outcome.baseline_order_count,
        )

    def reference_fix_patch(self, outcome: InjectionOutcome) -> str:
        return build_unified_diff(
            "dbt_project/macros/currency_utils.sql",
            str(outcome.metadata["faulty_macro"]),
            str(outcome.metadata["clean_macro"]),
        )


class DbtNotInNullTrapOracle(_ParityDbtOracle):
    oracle_name = "dbt_not_in_null_trap_oracle"

    def _extra_checks(self, worktree_dir: Path, reasons: list[str]) -> None:
        sql_text = (
            worktree_dir / "dbt_project" / "models" / "intermediate" / "int_orders_usd.sql"
        ).read_text(encoding="utf-8")
        if _NOT_IN_PATTERN.search(sql_text):
            reasons.append("int_orders_usd.sql still uses unsafe `NOT IN (SELECT ...)`.")


class DbtNotInNullTrapScenario(FaultScenario):
    """Scenario 7: SQL `NOT IN` NULL trap wiping out downstream orders to 0 rows."""

    scenario_id = "dbt_not_in_null_trap"
    title = "SQL `NOT IN` NULL trap silently returning 0 rows in int_orders_usd"
    category = "dbt_cross_engine"

    def inject(
        self, workspace_dir: Path, *, event_log_dir: Path | None = None
    ) -> InjectionOutcome:
        paths = LakehousePaths.from_workspace(workspace_dir)
        baseline_mart = read_mart_metrics(paths.prod_duckdb_path)

        model_rel = "dbt_project/models/intermediate/int_orders_usd.sql"
        model_path = workspace_dir / model_rel
        orig_text = model_path.read_text(encoding="utf-8")
        trap_clause = (
            "\n  AND o.user_id NOT IN (\n"
            "      SELECT CASE WHEN country = 'JP' THEN NULL ELSE 'BLOCKED_TEST_USER' END\n"
            "      FROM {{ ref('stg_users') }}\n"
            "  )\n"
        )
        faulty_text = orig_text.rstrip() + trap_clause
        model_path.write_text(faulty_text, encoding="utf-8")
        commit_sandbox_fault(
            workspace_dir,
            "feat(dbt): exclude blocked test users from int_orders_usd",
            [model_rel],
        )

        dbt_run = run_sandbox_dbt(workspace_dir, ["run", "--full-refresh"])
        if dbt_run.returncode != 0:
            raise RuntimeError(f"dbt run failed in NOT IN trap inject: {dbt_run.stderr or dbt_run.stdout}")
        run_sandbox_dbt(workspace_dir, ["test"])
        empty_mart = read_mart_metrics(paths.prod_duckdb_path)
        if int(empty_mart["fct_order_count"]) != 0 or float(empty_mart["fct_daily_gmv_total"]) != 0.0:
            raise RuntimeError(f"NOT IN NULL trap failed to zero out mart rows: {empty_mart}")

        incident = IncidentInput(
            incident_id=f"INC-{self.scenario_id.upper()}",
            title="fct_daily_gmv dropped to 0 rows ($0.00) unexpectedly",
            summary=(
                f"Mart revenue plunged from ${baseline_mart['fct_daily_gmv_total']:.2f} "
                f"({baseline_mart['fct_order_count']} orders) to $0.00 (0 rows) after adding "
                "excluded-user filter in int_orders_usd."
            ),
            affected_assets=["int_orders_usd", "fct_daily_gmv", "dim_user_ltv"],
            shared_ancestors=["int_orders_usd"],
        )
        return InjectionOutcome(
            scenario_id=self.scenario_id,
            category=self.category,
            workspace_dir=workspace_dir,
            incident=incident,
            raw_alerts=[
                RawAlert(
                    alert_id=f"alt-{self.scenario_id}-1",
                    asset="fct_daily_gmv",
                    timestamp="2026-11-26T00:00:00Z",
                    severity="critical",
                    alert_type="empty_mart",
                    summary=incident.summary,
                ),
            ],
            baseline_gmv_total=float(baseline_mart["fct_daily_gmv_total"]),
            baseline_order_count=int(baseline_mart["fct_order_count"]),
            fault_reproduced=True,
            reproduction_summary="fct_order_count=0, fct_daily_gmv_total=0.00",
            metadata={"faulty_sql": faulty_text, "orig_sql": orig_text},
        )

    def build_oracle(self, outcome: InjectionOutcome) -> HiddenRegressionOracle:
        return DbtNotInNullTrapOracle(
            expected_gmv_total=outcome.baseline_gmv_total,
            expected_order_count=outcome.baseline_order_count,
        )

    def reference_fix_patch(self, outcome: InjectionOutcome) -> str:
        rel = "dbt_project/models/intermediate/int_orders_usd.sql"
        old_text = str(outcome.metadata["faulty_sql"])
        safe_clause = (
            "\n  AND NOT EXISTS (\n"
            "      SELECT 1\n"
            "      FROM (\n"
            "          SELECT CASE WHEN country = 'JP' THEN NULL ELSE 'BLOCKED_TEST_USER' END AS blocked_id\n"
            "          FROM {{ ref('stg_users') }}\n"
            "      ) AS b\n"
            "      WHERE b.blocked_id IS NOT NULL AND b.blocked_id = o.user_id\n"
            "  )\n"
        )
        new_text = str(outcome.metadata["orig_sql"]).rstrip() + safe_clause
        return build_unified_diff(rel, old_text, new_text)


class DbtIncrementalWatermarkOracle(_ParityDbtOracle):
    oracle_name = "dbt_incremental_watermark_oracle"

    def _extra_checks(self, worktree_dir: Path, reasons: list[str]) -> None:
        fct_text = (
            worktree_dir / "dbt_project" / "models" / "marts" / "fct_daily_gmv.sql"
        ).read_text(encoding="utf-8")
        if not _INCREMENTAL_MAT_RE.search(fct_text):
            reasons.append("Cheat rejected: fct_daily_gmv.sql must stay materialized='incremental'.")


class DbtIncrementalWatermarkScenario(FaultScenario):
    """Scenario 8: Strict `>` incremental watermark defect skipping late boundary-date orders."""

    scenario_id = "dbt_incremental_watermark"
    title = "dbt `is_incremental()` strict `>` watermark bug missing boundary orders"
    category = "dbt_cross_engine"

    def inject(
        self, workspace_dir: Path, *, event_log_dir: Path | None = None
    ) -> InjectionOutcome:
        paths = LakehousePaths.from_workspace(workspace_dir)
        baseline_mart = read_mart_metrics(paths.prod_duckdb_path)

        fct_rel = "dbt_project/models/marts/fct_daily_gmv.sql"
        fct_path = workspace_dir / fct_rel
        orig_text = fct_path.read_text(encoding="utf-8")
        faulty_text = orig_text.replace("WHERE order_date >=(", "WHERE order_date >(").replace(
            "WHERE order_date >= (", "WHERE order_date > ("
        )
        if faulty_text == orig_text:
            raise RuntimeError("Unexpected fct_daily_gmv.sql incremental predicate.")
        fct_path.write_text(faulty_text, encoding="utf-8")
        commit_sandbox_fault(
            workspace_dir,
            "perf(dbt): advance incremental watermark strictly past max order_date",
            [fct_rel],
        )

        # Append 10 late completed USD $100 orders (+$1,000.00) onto boundary date 2026-11-07.
        _append_late_orders_partition_commit(
            paths.prod_silver_dir / "silver_orders",
            order_date="2026-11-07",
            count=10,
            amount_usd=100.0,
        )
        dbt_run = run_sandbox_dbt(workspace_dir, ["run"])
        if dbt_run.returncode != 0:
            raise RuntimeError(f"Incremental dbt run failed: {dbt_run.stderr or dbt_run.stdout}")
        dbt_test = run_sandbox_dbt(workspace_dir, ["test"])
        after_mart = read_mart_metrics(paths.prod_duckdb_path)

        expected_gmv = round(float(baseline_mart["fct_daily_gmv_total"]) + 1000.0, 2)
        expected_orders = int(baseline_mart["fct_order_count"]) + 10
        if (
            abs(float(after_mart["dim_user_ltv_total"]) - expected_gmv) > 0.01
            or abs(float(after_mart["fct_daily_gmv_total"]) - float(baseline_mart["fct_daily_gmv_total"])) > 0.01
            or dbt_test.returncode == 0
        ):
            raise RuntimeError(
                f"Watermark boundary drift failed to reproduce: mart={after_mart}, "
                f"test_rc={dbt_test.returncode}"
            )

        incident = IncidentInput(
            incident_id=f"INC-{self.scenario_id.upper()}",
            title="assert_gmv_reconciliation failed: dim_user_ltv exceeds fct_daily_gmv by $1,000",
            summary=(
                f"After late-arriving orders on 2026-11-07, dim_user_ltv reached "
                f"${after_mart['dim_user_ltv_total']:.2f} while incremental fct_daily_gmv stayed at "
                f"${after_mart['fct_daily_gmv_total']:.2f}."
            ),
            affected_assets=["fct_daily_gmv", "dim_user_ltv"],
            shared_ancestors=["int_orders_usd"],
        )
        return InjectionOutcome(
            scenario_id=self.scenario_id,
            category=self.category,
            workspace_dir=workspace_dir,
            incident=incident,
            raw_alerts=[
                RawAlert(
                    alert_id=f"alt-{self.scenario_id}-1",
                    asset="fct_daily_gmv",
                    timestamp="2026-11-26T00:00:00Z",
                    severity="high",
                    alert_type="reconciliation_drift",
                    summary=incident.summary,
                ),
            ],
            baseline_gmv_total=float(baseline_mart["fct_daily_gmv_total"]),
            baseline_order_count=int(baseline_mart["fct_order_count"]),
            fault_reproduced=True,
            reproduction_summary=(
                f"fct_gmv=${after_mart['fct_daily_gmv_total']:.2f} vs "
                f"dim_ltv=${after_mart['dim_user_ltv_total']:.2f}"
            ),
            metadata={
                "faulty_fct": faulty_text,
                "clean_fct": orig_text,
                "expected_gmv_total": expected_gmv,
                "expected_order_count": expected_orders,
            },
        )

    def build_oracle(self, outcome: InjectionOutcome) -> HiddenRegressionOracle:
        return DbtIncrementalWatermarkOracle(
            expected_gmv_total=float(outcome.metadata["expected_gmv_total"]),
            expected_order_count=int(outcome.metadata["expected_order_count"]),
        )

    def reference_fix_patch(self, outcome: InjectionOutcome) -> str:
        return build_unified_diff(
            "dbt_project/models/marts/fct_daily_gmv.sql",
            str(outcome.metadata["faulty_fct"]),
            str(outcome.metadata["clean_fct"]),
        )


DBT_SCENARIOS: dict[str, FaultScenario] = {
    s.scenario_id: s
    for s in (
        DbtFxFanoutScenario(),
        DbtBrokenMacroGitScenario(),
        DbtNotInNullTrapScenario(),
        DbtIncrementalWatermarkScenario(),
    )
}

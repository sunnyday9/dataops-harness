from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from benchmarks.scenarios.base import (
    FaultScenario,
    InjectionOutcome,
    build_unified_diff,
    commit_sandbox_fault,
    find_latest_spark_event_summary,
    read_mart_metrics,
    read_silver_order_stats,
    run_sandbox_dbt,
    write_spark_metrics_artifact,
)
from src.adapters.base import LakehousePaths
from src.adapters.spark_adapter import SparkEventSummary, SparkJobRunner
from src.core.aggregator import RawAlert
from src.core.context import IncidentInput
from src.core.rca import OracleVerdict
from src.core.shadow_verifier import DefaultIntegrityOracle, HiddenRegressionOracle

_BASELINE_ORDERS_SELECT_OLD = """        enriched = (
            orders_raw.join(users_country, on="user_id", how="left")
            .select(
                F.col("order_id").cast(StringType()).alias("order_id"),
                F.col("user_id").cast(StringType()).alias("user_id"),
                F.col("order_date").cast(DateType()).alias("order_date"),
                F.upper(F.trim(F.col("currency"))).cast(StringType()).alias("currency"),
                F.col("amount_local").cast(DoubleType()).alias("amount_local"),
                F.lower(F.trim(F.col("status"))).cast(StringType()).alias("status"),
                F.coalesce(F.col("user_country"), F.lit("UNKNOWN")).alias("user_country"),
            )
            .filter(F.col("order_id").isNotNull() & F.col("order_date").isNotNull())
        )"""


def _spark_run_args(paths: LakehousePaths) -> list[str]:
    return [
        "--raw-dir",
        str(paths.raw_dir),
        "--output-dir",
        str(paths.prod_silver_dir),
        "--silver-read-dir",
        str(paths.prod_silver_dir),
    ]


def _etl_max_skew_ratio(summary: SparkEventSummary) -> float:
    """Compute max skew ratio across multi-task ETL stages, excluding Delta log commits."""
    etl_stages = [
        st
        for st in summary.stages
        if st.task_count >= 2
        and "recordDeltaOperationInternal" not in st.stage_name
        and "DatabricksLogging" not in st.stage_name
    ]
    if etl_stages:
        return max(st.skew_ratio for st in etl_stages)
    return summary.max_skew_ratio


class SparkJoinSkewOracle:
    """Rejects dropped hot keys or residual >10x task skew after Spark shadow execution."""

    def __init__(
        self,
        *,
        expected_order_count: int,
        expected_gmv_total: float,
    ) -> None:
        self.expected_order_count = expected_order_count
        self.expected_gmv_total = expected_gmv_total
        self._integrity = DefaultIntegrityOracle()

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

        silver_orders = shadow_silver_dir / "silver_orders"
        if not silver_orders.is_dir():
            reasons.append("Shadow silver_orders was not rebuilt by fixed Spark job.")
        else:
            stats = read_silver_order_stats(silver_orders)
            metrics.update(stats)
            if stats["total_rows"] != self.expected_order_count:
                reasons.append(
                    f"silver_orders row count mismatch: expected {self.expected_order_count}, "
                    f"got {stats['total_rows']} (hot-key rows may have been filtered out)."
                )

        if shadow_duckdb_path.is_file():
            mart = read_mart_metrics(shadow_duckdb_path)
            metrics.update(mart)
            if abs(float(mart["fct_daily_gmv_total"]) - self.expected_gmv_total) > 0.05:
                reasons.append(
                    f"Expected conserved GMV ~${self.expected_gmv_total:.2f}, "
                    f"got ${float(mart['fct_daily_gmv_total']):.2f}."
                )

        job_text = (worktree_dir / "spark_jobs" / "build_silver_orders.py").read_text(
            encoding="utf-8"
        )
        if "aqe_enabled=False" in job_text or (
            '"-1"' in job_text and "autoBroadcast" in job_text
        ):
            reasons.append("build_silver_orders.py still disables AQE / broadcast join.")

        summary = find_latest_spark_event_summary(shadow_silver_dir.parent / "spark_events")
        if summary is None:
            reasons.append("Missing shadow Spark event log for skew verification.")
        else:
            etl_skew = _etl_max_skew_ratio(summary)
            metrics["shadow_max_skew_ratio"] = summary.max_skew_ratio
            metrics["shadow_etl_skew_ratio"] = etl_skew
            if etl_skew >= 10.0:
                reasons.append(
                    f"Shadow Spark run still exhibits >10x skew (etl_ratio={etl_skew})."
                )

        return OracleVerdict(
            passed=not reasons,
            oracle_name="spark_join_skew_oracle",
            reasons=reasons,
            metrics=metrics,
        )


class SparkJoinSkewScenario(FaultScenario):
    """Scenario 1: >10x Join Skew triggered with AQE & broadcast disabled and a hot user_id."""

    scenario_id = "spark_join_skew"
    title = "PySpark >10x Join Skew on silver_orders (AQE & broadcast disabled)"
    category = "spark"

    def inject(
        self, workspace_dir: Path, *, event_log_dir: Path | None = None
    ) -> InjectionOutcome:
        paths = LakehousePaths.from_workspace(workspace_dir)
        baseline_mart = read_mart_metrics(paths.prod_duckdb_path)

        # Spread 28,000 hot-key (`U0001`) orders evenly across all 7 raw date partitions.
        day_dirs = sorted(
            p for p in (paths.raw_dir / "orders").iterdir() if p.is_dir() and p.name.startswith("dt=")
        )
        added_orders = 0
        for d_idx, day_dir in enumerate(day_dirs):
            day_str = day_dir.name.removeprefix("dt=")
            orders_file = day_dir / "orders.json"
            with orders_file.open("a", encoding="utf-8") as handle:
                for i in range(4000):
                    added_orders += 1
                    rec = {
                        "order_id": f"ORD-SKEW-{d_idx}-{i:04d}",
                        "user_id": "U0001",
                        "order_date": day_str,
                        "currency": "USD",
                        "amount_local": "10.00",
                        "status": "completed",
                        "note": "flash sale hot key",
                    }
                    handle.write(json.dumps(rec) + "\n")

        job_rel = "spark_jobs/build_silver_orders.py"
        job_path = workspace_dir / job_rel
        orig_text = job_path.read_text(encoding="utf-8")
        faulty_text = orig_text.replace(
            'spark = build_spark_session("build_silver_orders")',
            'spark = build_spark_session("build_silver_orders", shuffle_partitions=8, aqe_enabled=False)\n'
            '    spark.conf.set("spark.sql.autoBroadcastJoinThreshold", "-1")',
        )
        faulty_select = """        enriched = (
            orders_raw.join(users_country, on="user_id", how="left")
            .withColumn(
                "_skew_digest",
                F.aggregate(
                    F.sequence(F.lit(1), F.lit(450)),
                    F.concat_ws(
                        "|",
                        F.col("order_id"),
                        F.coalesce(F.col("user_country"), F.lit("UNKNOWN")),
                    ),
                    lambda acc, _: F.sha2(acc, 256),
                ),
            )
            .filter(F.length(F.col("_skew_digest")) == 64)
            .select(
                F.col("order_id").cast(StringType()).alias("order_id"),
                F.col("user_id").cast(StringType()).alias("user_id"),
                F.col("order_date").cast(DateType()).alias("order_date"),
                F.upper(F.trim(F.col("currency"))).cast(StringType()).alias("currency"),
                F.col("amount_local").cast(DoubleType()).alias("amount_local"),
                F.lower(F.trim(F.col("status"))).cast(StringType()).alias("status"),
                F.coalesce(F.col("user_country"), F.lit("UNKNOWN")).alias("user_country"),
            )
            .filter(F.col("order_id").isNotNull() & F.col("order_date").isNotNull())
        )"""
        if _BASELINE_ORDERS_SELECT_OLD not in faulty_text:
            raise RuntimeError("Unexpected build_silver_orders.py structure for skew injection.")
        faulty_text = faulty_text.replace(_BASELINE_ORDERS_SELECT_OLD, faulty_select)
        job_path.write_text(faulty_text, encoding="utf-8")
        commit_sandbox_fault(
            workspace_dir,
            "perf(spark): tune shuffle partitions and disable AQE for orders job",
            [job_rel],
        )

        resolved_events = event_log_dir or (workspace_dir / "incidents" / self.scenario_id / "events")
        spark_res = SparkJobRunner(event_log_dir=resolved_events).run_job(
            job_path, args=_spark_run_args(paths)
        )
        if spark_res.returncode != 0:
            raise RuntimeError(f"Skewed Spark job failed unexpectedly: {spark_res.stderr}")
        metrics_path = write_spark_metrics_artifact(workspace_dir, self.scenario_id, spark_res)

        skew_ratio = _etl_max_skew_ratio(spark_res.event_summary) if spark_res.event_summary else 1.0
        if skew_ratio <= 10.0:
            raise RuntimeError(
                f"Expected >10x skew ratio in event log, observed {skew_ratio}."
            )

        dbt_run = run_sandbox_dbt(workspace_dir, ["run", "--full-refresh"])
        dbt_test = run_sandbox_dbt(workspace_dir, ["test"])
        if dbt_run.returncode != 0 or dbt_test.returncode != 0:
            raise RuntimeError(
                f"Baseline dbt update failed in join skew scenario: {dbt_test.stderr or dbt_test.stdout}"
            )
        updated_mart = read_mart_metrics(paths.prod_duckdb_path)
        silver_stats = read_silver_order_stats(paths.prod_silver_dir / "silver_orders")

        incident = IncidentInput(
            incident_id=f"INC-{self.scenario_id.upper()}",
            title="Severe stage straggler skew in build_silver_orders",
            summary=(
                f"build_silver_orders completed with max_skew_ratio={skew_ratio} (>10x threshold). "
                f"See {metrics_path.relative_to(workspace_dir).as_posix()}."
            ),
            affected_assets=["silver_orders", "fct_daily_gmv"],
            shared_ancestors=["silver_orders"],
        )
        return InjectionOutcome(
            scenario_id=self.scenario_id,
            category=self.category,
            workspace_dir=workspace_dir,
            incident=incident,
            raw_alerts=[
                RawAlert(
                    alert_id=f"alt-{self.scenario_id}-1",
                    asset="silver_orders",
                    timestamp="2026-11-26T00:00:00Z",
                    severity="high",
                    alert_type="spark_stage_skew",
                    summary=incident.summary,
                )
            ],
            baseline_gmv_total=float(baseline_mart["fct_daily_gmv_total"]),
            baseline_order_count=int(silver_stats["total_rows"]),
            spark_metrics_path=metrics_path,
            fault_reproduced=True,
            reproduction_summary=f"max_skew_ratio={skew_ratio}, added_hot_orders={added_orders}",
            metadata={
                "expected_order_count": int(silver_stats["total_rows"]),
                "expected_gmv_total": float(updated_mart["fct_daily_gmv_total"]),
                "max_skew_ratio": skew_ratio,
                "faulty_job_text": faulty_text,
            },
        )

    def build_oracle(self, outcome: InjectionOutcome) -> HiddenRegressionOracle:
        return SparkJoinSkewOracle(
            expected_order_count=int(outcome.metadata["expected_order_count"]),
            expected_gmv_total=float(outcome.metadata["expected_gmv_total"]),
        )

    def reference_fix_patch(self, outcome: InjectionOutcome) -> str:
        rel = "spark_jobs/build_silver_orders.py"
        old_text = str(outcome.metadata["faulty_job_text"])
        new_text = old_text.replace(
            'spark = build_spark_session("build_silver_orders", shuffle_partitions=8, aqe_enabled=False)\n'
            '    spark.conf.set("spark.sql.autoBroadcastJoinThreshold", "-1")',
            'spark = build_spark_session("build_silver_orders", shuffle_partitions=8, aqe_enabled=True)',
        ).replace(
            'orders_raw.join(users_country, on="user_id", how="left")',
            'orders_raw.join(F.broadcast(users_country), on="user_id", how="left")',
        )
        return build_unified_diff(rel, old_text, new_text)


class SparkDriverOomOracle:
    """Rejects driver-collection anti-patterns and verifies 384m OOM-free execution."""

    def __init__(self, *, expected_order_count: int, expected_gmv_total: float) -> None:
        self.expected_order_count = expected_order_count
        self.expected_gmv_total = expected_gmv_total
        self._integrity = DefaultIntegrityOracle()

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

        job_text = (worktree_dir / "spark_jobs" / "build_silver_orders.py").read_text(
            encoding="utf-8"
        )
        if ".collect(" in job_text or ".toPandas(" in job_text:
            reasons.append("build_silver_orders.py still collects rows into Spark driver memory.")

        silver_orders = shadow_silver_dir / "silver_orders"
        if not silver_orders.is_dir():
            reasons.append("Shadow silver_orders was not produced by fixed Spark job.")
        else:
            stats = read_silver_order_stats(silver_orders)
            metrics.update(stats)
            if stats["total_rows"] != self.expected_order_count:
                reasons.append(
                    f"silver_orders count mismatch: expected {self.expected_order_count}, "
                    f"got {stats['total_rows']}."
                )

        if shadow_duckdb_path.is_file():
            mart = read_mart_metrics(shadow_duckdb_path)
            metrics.update(mart)
            if abs(float(mart["fct_daily_gmv_total"]) - self.expected_gmv_total) > 0.05:
                reasons.append(
                    f"GMV mismatch after OOM fix: expected ${self.expected_gmv_total:.2f}, "
                    f"got ${float(mart['fct_daily_gmv_total']):.2f}."
                )

        summary = find_latest_spark_event_summary(shadow_silver_dir.parent / "spark_events")
        if summary is not None and summary.oom_detected:
            reasons.append(f"Shadow Spark run still hit OOM: {summary.oom_reasons}")

        return OracleVerdict(
            passed=not reasons,
            oracle_name="spark_driver_oom_oracle",
            reasons=reasons,
            metrics=metrics,
        )


class SparkDriverOomScenario(FaultScenario):
    """Scenario 2: Authentic isolated 384m Driver OOM caused by .collect() payload build-up."""

    scenario_id = "spark_driver_oom"
    title = "PySpark 384m Driver OutOfMemoryError from .collect() on build_silver_orders"
    category = "spark"

    def inject(
        self, workspace_dir: Path, *, event_log_dir: Path | None = None
    ) -> InjectionOutcome:
        paths = LakehousePaths.from_workspace(workspace_dir)
        baseline_mart = read_mart_metrics(paths.prod_duckdb_path)
        baseline_silver = read_silver_order_stats(paths.prod_silver_dir / "silver_orders")

        job_rel = "spark_jobs/build_silver_orders.py"
        job_path = workspace_dir / job_rel
        orig_text = job_path.read_text(encoding="utf-8")
        anchor = "        target = str(output_dir / \"silver_orders\")"
        if anchor not in orig_text:
            raise RuntimeError("Unexpected build_silver_orders.py layout for OOM injection.")

        oom_block = (
            "        # Anti-pattern: pulling inflated wide rows into driver memory.\n"
            '        _driver_cache = enriched.withColumn("_pad", F.repeat(F.lit("X"), 131072)).collect()\n'
            "        _ = len(_driver_cache)\n" + anchor
        )
        faulty_text = orig_text.replace(anchor, oom_block)
        job_path.write_text(faulty_text, encoding="utf-8")
        commit_sandbox_fault(
            workspace_dir,
            "feat(spark): add driver-side cache audit to build_silver_orders",
            [job_rel],
        )

        resolved_events = event_log_dir or (workspace_dir / "incidents" / self.scenario_id / "events")
        runner = SparkJobRunner(
            event_log_dir=resolved_events,
            default_driver_memory="384m",
            default_timeout_seconds=120,
        )
        spark_res = runner.run_job(
            job_path,
            args=_spark_run_args(paths),
            env_overrides={"SPARK_DRIVER_EXTRA_JAVA_OPTIONS": "-Dspark.testing.memory=471859200"},
        )
        metrics_path = write_spark_metrics_artifact(workspace_dir, self.scenario_id, spark_res)
        oom_ok = bool(spark_res.event_summary and spark_res.event_summary.oom_detected)
        if spark_res.returncode == 0 or not oom_ok:
            raise RuntimeError(
                f"Expected 384m Driver OOM subprocess failure, got rc={spark_res.returncode}, "
                f"oom_detected={oom_ok}."
            )

        incident = IncidentInput(
            incident_id=f"INC-{self.scenario_id.upper()}",
            title="build_silver_orders crashed with Driver OutOfMemoryError at 384m heap",
            summary=(
                f"Child Spark process failed (rc={spark_res.returncode}) with JVM heap OOM "
                f"at 384m driver memory. Inspect {metrics_path.relative_to(workspace_dir).as_posix()}."
            ),
            affected_assets=["silver_orders", "fct_daily_gmv", "dim_user_ltv"],
            shared_ancestors=["silver_orders"],
        )
        return InjectionOutcome(
            scenario_id=self.scenario_id,
            category=self.category,
            workspace_dir=workspace_dir,
            incident=incident,
            raw_alerts=[
                RawAlert(
                    alert_id=f"alt-{self.scenario_id}-1",
                    asset="silver_orders",
                    timestamp="2026-11-26T00:00:00Z",
                    severity="critical",
                    alert_type="spark_oom",
                    summary=incident.summary,
                )
            ],
            baseline_gmv_total=float(baseline_mart["fct_daily_gmv_total"]),
            baseline_order_count=int(baseline_silver["total_rows"]),
            spark_metrics_path=metrics_path,
            fault_reproduced=True,
            reproduction_summary=f"rc={spark_res.returncode}, oom_reasons={spark_res.event_summary.oom_reasons if spark_res.event_summary else []}",
            metadata={
                "faulty_job_text": faulty_text,
                "clean_job_text": orig_text,
                "oom_reasons": spark_res.event_summary.oom_reasons if spark_res.event_summary else [],
            },
        )

    def build_oracle(self, outcome: InjectionOutcome) -> HiddenRegressionOracle:
        return SparkDriverOomOracle(
            expected_order_count=outcome.baseline_order_count,
            expected_gmv_total=outcome.baseline_gmv_total,
        )

    def reference_fix_patch(self, outcome: InjectionOutcome) -> str:
        return build_unified_diff(
            "spark_jobs/build_silver_orders.py",
            str(outcome.metadata["faulty_job_text"]),
            str(outcome.metadata["clean_job_text"]),
        )


class SparkAppendGmvOracle:
    """Rejects downstream DISTINCT masking; requires Spark overwrite + exact 1x silver rows."""

    def __init__(self, *, expected_order_count: int, expected_gmv_total: float) -> None:
        self.expected_order_count = expected_order_count
        self.expected_gmv_total = expected_gmv_total
        self._integrity = DefaultIntegrityOracle()

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

        job_text = (worktree_dir / "spark_jobs" / "build_silver_orders.py").read_text(
            encoding="utf-8"
        )
        if re.search(r'\.mode\(\s*["\']append["\']\s*\)', job_text):
            reasons.append(
                "Cheat rejected: build_silver_orders.py still writes with .mode('append')."
            )

        silver_orders = shadow_silver_dir / "silver_orders"
        if not silver_orders.is_dir():
            reasons.append(
                "Cheat rejected: upstream Spark job must execute cleanly in Shadow Overlay."
            )
        else:
            stats = read_silver_order_stats(silver_orders)
            metrics.update(stats)
            if (
                stats["total_rows"] != self.expected_order_count
                or stats["distinct_order_ids"] != self.expected_order_count
            ):
                reasons.append(
                    f"silver_orders must contain exact deduplicated 1x rows "
                    f"(expected={self.expected_order_count}, total={stats['total_rows']}, "
                    f"distinct={stats['distinct_order_ids']})."
                )

        if shadow_duckdb_path.is_file():
            mart = read_mart_metrics(shadow_duckdb_path)
            metrics.update(mart)
            if abs(float(mart["fct_daily_gmv_total"]) - self.expected_gmv_total) > 0.01:
                reasons.append(
                    f"Mart GMV must equal exact baseline ${self.expected_gmv_total:.2f}, "
                    f"got ${float(mart['fct_daily_gmv_total']):.2f}."
                )

        return OracleVerdict(
            passed=not reasons,
            oracle_name="spark_append_gmv_oracle",
            reasons=reasons,
            metrics=metrics,
        )


class SparkAppendGmvDoublingScenario(FaultScenario):
    """Scenario 3: .mode('append') duplicate-partition GMV doubling (Delta Time-Travel)."""

    scenario_id = "spark_append_gmv_doubling"
    title = "PySpark .mode('append') duplicate-partition write doubling GMV"
    category = "spark"

    def inject(
        self, workspace_dir: Path, *, event_log_dir: Path | None = None
    ) -> InjectionOutcome:
        paths = LakehousePaths.from_workspace(workspace_dir)
        baseline_mart = read_mart_metrics(paths.prod_duckdb_path)
        baseline_silver = read_silver_order_stats(paths.prod_silver_dir / "silver_orders")

        job_rel = "spark_jobs/build_silver_orders.py"
        job_path = workspace_dir / job_rel
        orig_text = job_path.read_text(encoding="utf-8")
        if '.mode("overwrite")' not in orig_text:
            raise RuntimeError("Expected .mode('overwrite') in build_silver_orders.py.")
        faulty_text = orig_text.replace('.mode("overwrite")', '.mode("append")')
        job_path.write_text(faulty_text, encoding="utf-8")
        commit_sandbox_fault(
            workspace_dir,
            "chore(spark): switch build_silver_orders write mode to append",
            [job_rel],
        )

        resolved_events = event_log_dir or (workspace_dir / "incidents" / self.scenario_id / "events")
        spark_res = SparkJobRunner(event_log_dir=resolved_events).run_job(
            job_path, args=_spark_run_args(paths)
        )
        if spark_res.returncode != 0:
            raise RuntimeError(f"Spark append job failed: {spark_res.stderr}")
        metrics_path = write_spark_metrics_artifact(workspace_dir, self.scenario_id, spark_res)

        dbt_run = run_sandbox_dbt(workspace_dir, ["run", "--full-refresh"])
        if dbt_run.returncode != 0:
            raise RuntimeError(f"dbt run failed: {dbt_run.stderr or dbt_run.stdout}")
        dbt_test = run_sandbox_dbt(workspace_dir, ["test"])
        doubled_silver = read_silver_order_stats(paths.prod_silver_dir / "silver_orders")
        doubled_mart = read_mart_metrics(paths.prod_duckdb_path)

        if doubled_silver["total_rows"] != 2 * baseline_silver["total_rows"] or dbt_test.returncode == 0:
            raise RuntimeError(
                f"Append doubling did not reproduce cleanly: rows={doubled_silver['total_rows']} "
                f"vs baseline={baseline_silver['total_rows']}, dbt_test_rc={dbt_test.returncode}"
            )

        incident = IncidentInput(
            incident_id=f"INC-{self.scenario_id.upper()}",
            title="GMV doubled after build_silver_orders write & dbt unique_stg_orders_order_id failed",
            summary=(
                f"silver_orders rows doubled from {baseline_silver['total_rows']} to "
                f"{doubled_silver['total_rows']}, inflating fct_daily_gmv from "
                f"${baseline_mart['fct_daily_gmv_total']:.2f} to "
                f"${doubled_mart['fct_daily_gmv_total']:.2f}."
            ),
            affected_assets=["silver_orders", "stg_orders", "fct_daily_gmv", "dim_user_ltv"],
            shared_ancestors=["silver_orders"],
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
                    alert_type="gmv_anomaly",
                    summary=incident.summary,
                ),
                RawAlert(
                    alert_id=f"alt-{self.scenario_id}-2",
                    asset="dim_user_ltv",
                    timestamp="2026-11-26T00:01:00Z",
                    severity="critical",
                    alert_type="ltv_anomaly",
                    summary=incident.summary,
                ),
            ],
            baseline_gmv_total=float(baseline_mart["fct_daily_gmv_total"]),
            baseline_order_count=int(baseline_silver["total_rows"]),
            spark_metrics_path=metrics_path,
            fault_reproduced=True,
            reproduction_summary=(
                f"rows={doubled_silver['total_rows']} (2x), "
                f"gmv=${doubled_mart['fct_daily_gmv_total']:.2f}"
            ),
            metadata={
                "faulty_job_text": faulty_text,
                "clean_job_text": orig_text,
                "doubled_gmv_total": float(doubled_mart["fct_daily_gmv_total"]),
            },
        )

    def build_oracle(self, outcome: InjectionOutcome) -> HiddenRegressionOracle:
        return SparkAppendGmvOracle(
            expected_order_count=outcome.baseline_order_count,
            expected_gmv_total=outcome.baseline_gmv_total,
        )

    def reference_fix_patch(self, outcome: InjectionOutcome) -> str:
        return build_unified_diff(
            "spark_jobs/build_silver_orders.py",
            str(outcome.metadata["faulty_job_text"]),
            str(outcome.metadata["clean_job_text"]),
        )


class SparkJsonSchemaDriftOracle:
    """Rejects test deletion or silently dropping valid currency-formatted JSON strings."""

    def __init__(self, *, expected_order_count: int, expected_gmv_total: float) -> None:
        self.expected_order_count = expected_order_count
        self.expected_gmv_total = expected_gmv_total
        self._integrity = DefaultIntegrityOracle()

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

        silver_orders = shadow_silver_dir / "silver_orders"
        if not silver_orders.is_dir():
            reasons.append("Shadow silver_orders was not rebuilt by fixed Spark job.")
        else:
            stats = read_silver_order_stats(silver_orders)
            metrics.update(stats)
            if stats["null_amount_rows"] != 0:
                reasons.append(
                    f"silver_orders still contains {stats['null_amount_rows']} NULL amount rows."
                )
            if stats["total_rows"] != self.expected_order_count:
                reasons.append(
                    f"Expected {self.expected_order_count} valid rows after cleaning drifted JSON "
                    f"(20 valid formatted recovered, 5 corrupt filtered), got {stats['total_rows']}."
                )

        if shadow_duckdb_path.is_file():
            mart = read_mart_metrics(shadow_duckdb_path)
            metrics.update(mart)
            if abs(float(mart["fct_daily_gmv_total"]) - self.expected_gmv_total) > 0.02:
                reasons.append(
                    f"Expected conserved GMV ${self.expected_gmv_total:.2f} (including recovered "
                    f"drifted rows), got ${float(mart['fct_daily_gmv_total']):.2f}."
                )

        return OracleVerdict(
            passed=not reasons,
            oracle_name="spark_json_schema_drift_oracle",
            reasons=reasons,
            metrics=metrics,
        )


class SparkJsonSchemaDriftScenario(FaultScenario):
    """Scenario 4: Dirty/currency-formatted JSON amount_local + corrupt rows breaking not_null."""

    scenario_id = "spark_json_schema_drift"
    title = "PySpark raw JSON dirty-type/schema drift causing NULL amount_local"
    category = "spark"

    def inject(
        self, workspace_dir: Path, *, event_log_dir: Path | None = None
    ) -> InjectionOutcome:
        paths = LakehousePaths.from_workspace(workspace_dir)
        baseline_mart = read_mart_metrics(paths.prod_duckdb_path)
        baseline_silver = read_silver_order_stats(paths.prod_silver_dir / "silver_orders")

        orders_file = paths.raw_dir / "orders" / "dt=2026-11-07" / "orders.json"
        valid_drifted = [
            ("$100.00", 100.0),
            (" 250.50 USD ", 250.5),
            ("$1,200.00", 1200.0),
            ("€75.25", 75.25),
        ]
        expected_added_gmv = 0.0
        added_valid_rows = 0
        with orders_file.open("a", encoding="utf-8") as handle:
            for batch_idx in range(5):
                for fmt_idx, (raw_amt, numeric_val) in enumerate(valid_drifted):
                    added_valid_rows += 1
                    expected_added_gmv += numeric_val
                    handle.write(
                        json.dumps(
                            {
                                "order_id": f"ORD-DRIFT-OK-{batch_idx}-{fmt_idx}",
                                "user_id": f"U{1 + batch_idx:04d}",
                                "order_date": "2026-11-07",
                                "currency": "USD",
                                "amount_local": raw_amt,
                                "status": "completed",
                                "note": "upstream POS formatted currency string",
                            }
                        )
                        + "\n"
                    )
            for bad_idx in range(5):
                handle.write(
                    json.dumps(
                        {
                            "order_id": f"ORD-DRIFT-BAD-{bad_idx}",
                            "user_id": "U0001",
                            "order_date": "2026-11-07",
                            "currency": "USD",
                            "amount_local": "N/A" if bad_idx % 2 == 0 else "CORRUPT",
                            "status": "completed",
                            "note": "genuinely malformed row",
                        }
                    )
                    + "\n"
                )

        job_rel = "spark_jobs/build_silver_orders.py"
        job_path = workspace_dir / job_rel
        orig_text = job_path.read_text(encoding="utf-8")

        resolved_events = event_log_dir or (workspace_dir / "incidents" / self.scenario_id / "events")
        spark_res = SparkJobRunner(event_log_dir=resolved_events).run_job(
            job_path, args=_spark_run_args(paths)
        )
        if spark_res.returncode != 0:
            raise RuntimeError(f"Spark schema-drift reproduction failed: {spark_res.stderr}")
        metrics_path = write_spark_metrics_artifact(workspace_dir, self.scenario_id, spark_res)

        dbt_run = run_sandbox_dbt(workspace_dir, ["run", "--full-refresh"])
        if dbt_run.returncode != 0:
            raise RuntimeError(f"dbt run failed: {dbt_run.stderr or dbt_run.stdout}")
        dbt_test = run_sandbox_dbt(workspace_dir, ["test"])
        drifted_silver = read_silver_order_stats(paths.prod_silver_dir / "silver_orders")

        if drifted_silver["null_amount_rows"] != 25 or dbt_test.returncode == 0:
            raise RuntimeError(
                f"Expected 25 NULL amount_local rows & dbt test failure, got "
                f"nulls={drifted_silver['null_amount_rows']}, test_rc={dbt_test.returncode}"
            )

        expected_total_gmv = round(float(baseline_mart["fct_daily_gmv_total"]) + expected_added_gmv, 2)
        expected_total_rows = int(baseline_silver["total_rows"]) + added_valid_rows

        incident = IncidentInput(
            incident_id=f"INC-{self.scenario_id.upper()}",
            title="not_null_stg_orders_amount_local failed due to raw JSON dirty currency strings",
            summary=(
                f"{drifted_silver['null_amount_rows']} rows in silver_orders have NULL amount_local "
                f"after ingesting formatted/corrupt JSON strings in lakehouse/raw/orders."
            ),
            affected_assets=["silver_orders", "stg_orders", "fct_daily_gmv"],
            shared_ancestors=["silver_orders"],
        )
        return InjectionOutcome(
            scenario_id=self.scenario_id,
            category=self.category,
            workspace_dir=workspace_dir,
            incident=incident,
            raw_alerts=[
                RawAlert(
                    alert_id=f"alt-{self.scenario_id}-1",
                    asset="stg_orders",
                    timestamp="2026-11-26T00:00:00Z",
                    severity="high",
                    alert_type="dbt_test_failure",
                    summary=incident.summary,
                )
            ],
            baseline_gmv_total=float(baseline_mart["fct_daily_gmv_total"]),
            baseline_order_count=int(baseline_silver["total_rows"]),
            spark_metrics_path=metrics_path,
            fault_reproduced=True,
            reproduction_summary=f"null_amount_rows={drifted_silver['null_amount_rows']}",
            metadata={
                "expected_order_count": expected_total_rows,
                "expected_gmv_total": expected_total_gmv,
                "faulty_job_text": orig_text,
            },
        )

    def build_oracle(self, outcome: InjectionOutcome) -> HiddenRegressionOracle:
        return SparkJsonSchemaDriftOracle(
            expected_order_count=int(outcome.metadata["expected_order_count"]),
            expected_gmv_total=float(outcome.metadata["expected_gmv_total"]),
        )

    def reference_fix_patch(self, outcome: InjectionOutcome) -> str:
        rel = "spark_jobs/build_silver_orders.py"
        old_text = str(outcome.metadata["faulty_job_text"])
        replacement = """        cleaned_amount = F.regexp_replace(F.trim(F.col("amount_local")), r"[^0-9.\\-]", "")
        parsed_amount = F.when(
            cleaned_amount.rlike(r"^-?\\d+(?:\\.\\d+)?$"), cleaned_amount
        ).cast(DoubleType())

        enriched = (
            orders_raw.join(users_country, on="user_id", how="left")
            .select(
                F.col("order_id").cast(StringType()).alias("order_id"),
                F.col("user_id").cast(StringType()).alias("user_id"),
                F.col("order_date").cast(DateType()).alias("order_date"),
                F.upper(F.trim(F.col("currency"))).cast(StringType()).alias("currency"),
                parsed_amount.alias("amount_local"),
                F.lower(F.trim(F.col("status"))).cast(StringType()).alias("status"),
                F.coalesce(F.col("user_country"), F.lit("UNKNOWN")).alias("user_country"),
            )
            .filter(
                F.col("order_id").isNotNull()
                & F.col("order_date").isNotNull()
                & F.col("amount_local").isNotNull()
            )
        )"""
        if _BASELINE_ORDERS_SELECT_OLD not in old_text:
            raise RuntimeError("Unexpected build_silver_orders.py structure for reference fix.")
        new_text = old_text.replace(_BASELINE_ORDERS_SELECT_OLD, replacement)
        return build_unified_diff(rel, old_text, new_text)


SPARK_SCENARIOS: dict[str, FaultScenario] = {
    s.scenario_id: s
    for s in (
        SparkJoinSkewScenario(),
        SparkDriverOomScenario(),
        SparkAppendGmvDoublingScenario(),
        SparkJsonSchemaDriftScenario(),
    )
}

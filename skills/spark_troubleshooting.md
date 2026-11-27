# PySpark & Delta Lake Troubleshooting Playbook

Load this playbook on demand when `query_lineage`, `incidents/<id>/spark_metrics.json`, or `git_inspect` indicates a fault inside `spark_jobs/*.py` or Delta silver tables (`silver_users`, `silver_exchange_rates`, `silver_orders`).

## 1. Join Skew (`>10x` Task Latency Ratio)
- **Signals**: `spark_metrics.json` reports `max_skew_ratio > 10.0` or `skewed_stages` non-empty while overall code still finishes slowly or stalls at shuffle stages.
- **Verify**: Inspect the join key distribution with `run_sql_readonly` (e.g. `GROUP BY user_id` / top counts on `silver_orders` or `EXPLAIN FORMATTED` via `engine="spark"`). Check whether AQE (`spark.sql.adaptive.enabled`) or skew optimization was disabled, or whether hot keys (e.g. dominant `user_id` or `NULL` join keys) are un-salted or non-broadcasted.
- **Fix pattern**: Filter out invalid keys early, re-enable adaptive execution / broadcast small dimension tables (`silver_users`, `silver_exchange_rates` via `F.broadcast(...)`), or salt skewed keys rather than disabling downstream checks.

## 2. Driver Out-of-Memory (`384m` Heap / `OutOfMemoryError`)
- **Signals**: Spark subprocess fails with `java.lang.OutOfMemoryError: Java heap space` or `oom_detected: true` in `spark_metrics.json`.
- **Verify**: Look inside `spark_jobs/*.py` for driver-collecting calls (`.collect()`, `.toPandas()`, `.take(n)` on large datasets, or large Python loops pulling partitions to the driver).
- **Fix pattern**: Push aggregations, filtering, and deduplication strictly into Spark DataFrame transformations (`groupBy`, `agg`, `filter`, `dropDuplicates`) so data stays distributed on executors and only written directly to Delta.

## 3. Write-Mode / Duplicate-Partition GMV Doubling
- **Signals**: Downstream `fct_daily_gmv` and `dim_user_ltv` show exact 2x / N-x inflation across valid dbt runs while both marts agree with each other or fail uniqueness tests on `order_id`.
- **Verify**: Inspect Delta table history with `run_sql_readonly(sql="DESCRIBE HISTORY silver_orders", engine="spark")` to check commit operations (`WRITE` with `mode: Append` vs `Overwrite`) and compare partition counts by `order_date`. Inspect `spark_jobs/build_silver_orders.py` `.write.format("delta").mode(...)`.
- **Fix pattern**: Restore idempotent partition/table writes (`.mode("overwrite")` or `replaceWhere` for targeted partitions) at the Spark writer. Never hide duplicate upstream rows downstream via `SELECT DISTINCT` in staging/mart SQL. After patching `spark_jobs/build_silver_orders.py`, do NOT run `RESTORE TABLE`, `CREATE OR REPLACE TABLE`, or DDL in `run_sql_readonly` — output your structured RCA JSON so the Harness StopGate automatically re-runs the patched Spark + dbt pipeline in the Shadow Overlay.

## 4. JSON Dirty-Type & Schema Drift
- **Signals**: Spark job fails parsing `lakehouse/raw/orders`, or silently coerces malformed rows into `NULL` values breaking `not_null` / `accepted_values` tests downstream.
- **Verify**: Read `spark_jobs/*.py` schema definitions (`RAW_ORDER_SCHEMA`) and run bounded SQL checks on malformed/NULL counts in `silver_orders`.
- **Fix pattern**: Apply explicit schema enforcement, safe casting/cleaning (trim, normalize casing/currencies, handle corrupt/extra string values cleanly), and quarantine or filter genuinely malformed corrupt records while conserving valid GMV rows.

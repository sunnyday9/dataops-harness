# dbt-duckdb & Cross-Engine SQL Troubleshooting Playbook

Load this playbook on demand when `query_lineage` or `dbt test` failures point to SQL/Jinja regressions inside `dbt_project/models/` or `dbt_project/macros/`.

## 1. Upstream Duplicate-Key Fan-Out
- **Signals**: `fct_daily_gmv` or `dim_user_ltv` GMV totals exceed expected revenue, or uniqueness/relationship tests fail on intermediate/mart joins (`int_orders_usd` joining `stg_exchange_rates`).
- **Verify**: Query duplicate keys on staging/intermediate models via `run_sql_readonly` (e.g., `SELECT rate_date, currency, COUNT(*) FROM stg_exchange_rates GROUP BY 1, 2 HAVING COUNT(*) > 1`). Check grain before and after every join.
- **Fix pattern**: Enforce 1:1 / unique grain at the source staging model (`ROW_NUMBER()` / deterministic deduplication per business key) or fix upstream fan-out cleanly so downstream `fct_daily_gmv` and `dim_user_ltv` reconcile to the cent.

## 2. Shared Jinja Macro Regression
- **Signals**: Multiple downstream models break together right after a commit even though model SQL files look unchanged.
- **Verify**: Run `git_inspect(subcommand="log", args=["-n", "5", "--oneline"])` and `git_inspect(subcommand="diff", args=["HEAD~1"])` or inspect `dbt_project/macros/*.sql` with `read_file`.
- **Fix pattern**: Restore exact arithmetic/currency-conversion or helper logic inside `dbt_project/macros/*.sql` and verify all dependent models conserve GMV/row counts.

## 3. SQL `NOT IN` NULL Trap
- **Signals**: A model filtering with `col NOT IN (SELECT ...)` unexpectedly returns 0 rows or drops valid records, breaking row-count or GMV reconciliation tests.
- **Verify**: Inspect SQL predicates via `grep_search(pattern="NOT IN", path="dbt_project")` and check for `NULL` values in the subquery column via `run_sql_readonly`. Remember SQL three-valued logic evaluates `x NOT IN (..., NULL)` to `UNKNOWN` (empty set!).
- **Fix pattern**: Replace `NOT IN (subquery)` with `NOT EXISTS (...)`, `WHERE ... IS NOT NULL`, or an explicit anti-join (`LEFT JOIN ... WHERE right_key IS NULL`).

## 4. `is_incremental()` Watermark Boundary Defect
- **Signals**: Incremental models miss boundary/late-arriving dates (`>` instead of `>=`) or duplicate rows across runs (missing `unique_key` / delete+insert merge consistency).
- **Verify**: Read the incremental filter block (`{% if is_incremental() %}`) in the model SQL and compare max watermark counts vs source staging counts by `order_date`.
- **Fix pattern**: Align `unique_key` / `incremental_strategy` and inclusive `>=` (plus safe lookback window if needed) so repeated `dbt run` executions remain 100% idempotent.

## 5. Reconciliation & Test Preservation
- Total GMV in `fct_daily_gmv` and total LTV in `dim_user_ltv` must remain positive and reconcile within `$0.01`.
- Never delete or edit `schema.yml` tests or custom SQL assertions to force a green `dbt test`.

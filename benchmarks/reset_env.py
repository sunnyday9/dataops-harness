from __future__ import annotations

import argparse
import csv
import json
import os
import random
import shutil
import subprocess
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import duckdb
from rich.console import Console
from rich.table import Table

from src.adapters.base import LakehousePaths, ResetSummary, SparkRunResult
from src.adapters.spark_adapter import SparkJobRunner

DEFAULT_EVENT_LOG_DIR = Path("/tmp/spark-events")
COUNTRIES = ("US", "GB", "DE", "JP")
BASE_RATES = {
    "USD": 1.0,
    "EUR": 1.08,
    "GBP": 1.27,
    "JPY": 0.0067,
}
CURRENCIES = tuple(BASE_RATES.keys())


def _run_git(workspace_dir: Path, *args: str) -> str:
    proc = subprocess.run(
        ["git", *args],
        cwd=str(workspace_dir),
        text=True,
        capture_output=True,
        check=True,
    )
    return proc.stdout.strip()


def _init_sandbox_git(workspace_dir: Path) -> int:
    """Initialize workspace/ as an independent git repo with realistic baseline commits."""
    gitignore = (
        "lakehouse/\n"
        ".scratch/\n"
        "incidents/\n"
        "dbt_project/target/\n"
        "dbt_project/logs/\n"
        "dbt_project/dbt_packages/\n"
        "__pycache__/\n"
        "*.pyc\n"
    )
    (workspace_dir / ".gitignore").write_text(gitignore, encoding="utf-8")

    _run_git(workspace_dir, "init", "-b", "main")
    _run_git(workspace_dir, "config", "user.name", "dataops-bot")
    _run_git(workspace_dir, "config", "user.email", "dataops-bot@example.com")

    _run_git(workspace_dir, "add", ".gitignore", "spark_jobs")
    _run_git(workspace_dir, "commit", "-m", "feat: add upstream PySpark silver ETL jobs")

    _run_git(
        workspace_dir,
        "add",
        "dbt_project/dbt_project.yml",
        "dbt_project/profiles.yml",
        "dbt_project/macros",
        "dbt_project/models",
    )
    _run_git(workspace_dir, "commit", "-m", "feat: add dbt-duckdb staging and mart models")

    _run_git(workspace_dir, "add", "dbt_project/tests")
    _run_git(workspace_dir, "commit", "-m", "test: add GMV reconciliation singular test")

    return int(_run_git(workspace_dir, "rev-list", "--count", "HEAD"))


def _generate_synthetic_raw_data(
    raw_dir: Path,
    seed: int,
    order_count: int,
    user_count: int,
) -> tuple[int, int, int]:
    """Generate deterministic raw users CSV, exchange rates CSV, and partitioned orders JSON."""
    rng = random.Random(seed)
    dates = [date(2026, 11, 1) + timedelta(days=offset) for offset in range(7)]

    users_dir = raw_dir / "users"
    users_dir.mkdir(parents=True, exist_ok=True)
    with (users_dir / "users.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["user_id", "signup_date", "country", "email", "phone", "national_id"])
        for idx in range(1, user_count + 1):
            uid = f"U{idx:04d}"
            signup = date(2026, 1, 1) + timedelta(days=rng.randint(0, 290))
            country = rng.choice(COUNTRIES)
            email = f"user.{idx:04d}@synthetic-shop.test"
            phone = f"+1-555-{100 + (idx // 100):03d}-{idx % 10000:04d}"
            national_id = f"NID-{rng.randint(100000, 999999)}-{idx % 100:02d}"
            writer.writerow([uid, signup.isoformat(), country, email, phone, national_id])

    rates_dir = raw_dir / "exchange_rates"
    rates_dir.mkdir(parents=True, exist_ok=True)
    rate_rows = 0
    with (rates_dir / "exchange_rates.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["rate_date", "currency", "rate_to_usd"])
        for day_idx, d in enumerate(dates):
            for currency, base in BASE_RATES.items():
                jitter = 1.0 if currency == "USD" else (1.0 + ((day_idx - 3) * 0.005) + rng.uniform(-0.003, 0.003))
                writer.writerow([d.isoformat(), currency, round(base * jitter, 6)])
                rate_rows += 1

    orders_by_date: dict[str, list[dict[str, str]]] = {d.isoformat(): [] for d in dates}
    for idx in range(1, order_count + 1):
        d = rng.choice(dates).isoformat()
        uid = f"U{rng.randint(1, user_count):04d}"
        currency = rng.choice(CURRENCIES)
        base_amt = rng.uniform(15.0, 250.0)
        amount_local = round(base_amt * 150.0, 2) if currency == "JPY" else round(base_amt, 2)
        status = "cancelled" if rng.random() < 0.15 else "completed"
        orders_by_date[d].append(
            {
                "order_id": f"ORD-{idx:06d}",
                "user_id": uid,
                "order_date": d,
                "currency": currency,
                "amount_local": f"{amount_local:.2f}",
                "status": status,
                "note": f"synthetic order {idx}",
            }
        )

    for day_str, records in orders_by_date.items():
        part_dir = raw_dir / "orders" / f"dt={day_str}"
        part_dir.mkdir(parents=True, exist_ok=True)
        with (part_dir / "orders.json").open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record) + "\n")

    return order_count, user_count, rate_rows


def _run_dbt_command(
    args: list[str],
    paths: LakehousePaths,
    timeout_seconds: int = 120,
) -> subprocess.CompletedProcess[str]:
    """Invoke dbt inside the active virtual environment against the sandbox lakehouse."""
    venv_dbt = Path(sys.executable).parent / "dbt"
    dbt_exe = str(venv_dbt) if venv_dbt.exists() else (shutil.which("dbt") or "dbt")
    env = os.environ.copy()
    env["DATAOPS_DUCKDB_PATH"] = str(paths.prod_duckdb_path)
    env["DATAOPS_PROD_SILVER_DIR"] = str(paths.prod_silver_dir)
    return subprocess.run(
        [dbt_exe, *args, "--profiles-dir", "."],
        cwd=str(paths.dbt_project_dir),
        env=env,
        text=True,
        capture_output=True,
        timeout=timeout_seconds,
        check=False,
    )


def _read_mart_totals(duckdb_path: Path) -> tuple[float, float, int, int]:
    con = duckdb.connect(str(duckdb_path), read_only=True)
    try:
        fct_total, daily_rows = con.execute(
            "SELECT ROUND(COALESCE(SUM(net_gmv_usd), 0), 2), COUNT(*) FROM fct_daily_gmv"
        ).fetchone()
        dim_total, user_rows = con.execute(
            "SELECT ROUND(COALESCE(SUM(lifetime_gmv_usd), 0), 2), COUNT(*) FROM dim_user_ltv"
        ).fetchone()
    finally:
        con.close()
    return float(fct_total), float(dim_total), int(daily_rows), int(user_rows)


def reset_environment(
    repo_root: Path | None = None,
    seed: int = 42,
    order_count: int = 6000,
    user_count: int = 1000,
    event_log_dir: Path | None = None,
) -> ResetSummary:
    """Idempotently rebuild workspace/ from template, populate synthetic data, and run Spark + dbt."""
    start = time.monotonic()
    root = (repo_root or Path(__file__).resolve().parent.parent).resolve()
    template_dir = root / "benchmarks" / "workspace_template"
    workspace_dir = root / "workspace"
    resolved_event_dir = (event_log_dir or DEFAULT_EVENT_LOG_DIR).resolve()

    if not template_dir.is_dir():
        raise FileNotFoundError(f"Workspace template missing at {template_dir}")

    staging_dir = root / f".workspace_staging_{os.getpid()}"
    if staging_dir.exists():
        shutil.rmtree(staging_dir)

    try:
        shutil.copytree(template_dir, staging_dir)
        for rel in (
            "lakehouse/raw",
            "lakehouse/prod/silver",
            "lakehouse/prod/mart",
            "lakehouse/shadow",
            "incidents",
            ".scratch",
        ):
            (staging_dir / rel).mkdir(parents=True, exist_ok=True)

        old_dir = root / f".workspace_old_{os.getpid()}"
        if old_dir.exists():
            shutil.rmtree(old_dir)
        if workspace_dir.exists():
            workspace_dir.rename(old_dir)
        staging_dir.rename(workspace_dir)
        if old_dir.exists():
            shutil.rmtree(old_dir, ignore_errors=True)
    except Exception:
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise

    paths = LakehousePaths.from_workspace(workspace_dir)
    git_commits = _init_sandbox_git(workspace_dir)
    raw_orders, raw_users, raw_rates = _generate_synthetic_raw_data(
        paths.raw_dir, seed=seed, order_count=order_count, user_count=user_count
    )

    runner = SparkJobRunner(event_log_dir=resolved_event_dir)
    spark_runs: list[SparkRunResult] = []
    job_specs = [
        ("build_silver_users.py", ["--raw-dir", str(paths.raw_dir), "--output-dir", str(paths.prod_silver_dir)]),
        (
            "build_silver_exchange_rates.py",
            ["--raw-dir", str(paths.raw_dir), "--output-dir", str(paths.prod_silver_dir)],
        ),
        (
            "build_silver_orders.py",
            [
                "--raw-dir",
                str(paths.raw_dir),
                "--output-dir",
                str(paths.prod_silver_dir),
                "--silver-read-dir",
                str(paths.prod_silver_dir),
            ],
        ),
    ]
    for script_name, args in job_specs:
        res = runner.run_job(paths.spark_jobs_dir / script_name, args=args)
        spark_runs.append(res)
        if res.returncode != 0:
            raise RuntimeError(
                f"Spark job {script_name} failed (rc={res.returncode}):\n{res.stderr}"
            )

    dbt_run = _run_dbt_command(["run"], paths)
    if dbt_run.returncode != 0:
        raise RuntimeError(f"dbt run failed:\n{dbt_run.stdout}\n{dbt_run.stderr}")

    dbt_test = _run_dbt_command(["test"], paths)
    if dbt_test.returncode != 0:
        raise RuntimeError(f"dbt test failed:\n{dbt_test.stdout}\n{dbt_test.stderr}")

    fct_total, dim_total, daily_rows, user_rows = _read_mart_totals(paths.prod_duckdb_path)
    elapsed = round(time.monotonic() - start, 2)
    return ResetSummary(
        elapsed_seconds=elapsed,
        git_commits=git_commits,
        raw_orders=raw_orders,
        raw_users=raw_users,
        raw_exchange_rates=raw_rates,
        spark_runs=spark_runs,
        dbt_run_ok=True,
        dbt_test_ok=True,
        fct_daily_gmv_total=fct_total,
        dim_user_ltv_total=dim_total,
        daily_gmv_rows=daily_rows,
        user_ltv_rows=user_rows,
    )


def _print_summary(summary: ResetSummary) -> None:
    console = Console()
    table = Table(title="DataOps-Harness Synthetic Lakehouse Reset Summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Elapsed (s)", f"{summary.elapsed_seconds:.2f}")
    table.add_row("Sandbox Git Commits", str(summary.git_commits))
    table.add_row("Raw Orders / Users / FX", f"{summary.raw_orders} / {summary.raw_users} / {summary.raw_exchange_rates}")
    table.add_row("fct_daily_gmv (rows / USD)", f"{summary.daily_gmv_rows} / {summary.fct_daily_gmv_total:.2f}")
    table.add_row("dim_user_ltv (rows / USD)", f"{summary.user_ltv_rows} / {summary.dim_user_ltv_total:.2f}")
    table.add_row("dbt run / dbt test", f"{summary.dbt_run_ok} / {summary.dbt_test_ok}")
    for run in summary.spark_runs:
        log_name = run.event_log_path.name if run.event_log_path else "missing"
        table.add_row(f"Spark {run.job_name}", f"pid={run.pid} rc={run.returncode} {run.duration_ms:.0f}ms log={log_name}")
    console.print(table)


def main() -> None:
    parser = argparse.ArgumentParser(description="Reset synthetic e-commerce lakehouse workspace.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--orders", type=int, default=6000)
    parser.add_argument("--users", type=int, default=1000)
    args = parser.parse_args()
    summary = reset_environment(seed=args.seed, order_count=args.orders, user_count=args.users)
    _print_summary(summary)


if __name__ == "__main__":
    main()

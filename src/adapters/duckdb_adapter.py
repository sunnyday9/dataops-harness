from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import duckdb

from src.adapters.base import (
    DataDiffSummary,
    LakehousePaths,
    SqlQueryResult,
    StorageAdapter,
)


def _quote_ident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _sql_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


class DuckDBAdapter(StorageAdapter):
    """Read-only lakehouse SQL probing, dbt runner, and table data diff via DuckDB."""

    def __init__(
        self,
        paths: LakehousePaths,
        silver_overrides: dict[str, Path] | None = None,
        mart_duckdb_path: Path | None = None,
    ) -> None:
        self.paths = paths
        self.silver_overrides = {
            k: v.resolve() for k, v in (silver_overrides or {}).items()
        }
        self.mart_duckdb_path = (
            mart_duckdb_path.resolve() if mart_duckdb_path else paths.prod_duckdb_path
        )

    def _discover_silver_tables(self) -> dict[str, Path]:
        tables: dict[str, Path] = {}
        if self.paths.prod_silver_dir.is_dir():
            for entry in sorted(self.paths.prod_silver_dir.iterdir()):
                if entry.is_dir() and (entry / "_delta_log").is_dir():
                    tables[entry.name] = entry.resolve()
        for name, override_path in self.silver_overrides.items():
            tables[name] = override_path.resolve()
        return tables

    def _open_readonly_connection(self) -> duckdb.DuckDBPyConnection:
        con = duckdb.connect(":memory:")
        try:
            con.execute("LOAD delta;")
        except duckdb.Error:
            con.execute("INSTALL delta; LOAD delta;")

        if self.mart_duckdb_path.is_file():
            con.execute(
                f"ATTACH {_sql_literal(str(self.mart_duckdb_path))} AS mart (READ_ONLY);"
            )
            mart_tables = con.execute(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_catalog = 'mart' AND table_schema = 'main'"
            ).fetchall()
            for (table_name,) in mart_tables:
                q_name = _quote_ident(table_name)
                con.execute(
                    f"CREATE OR REPLACE TEMP VIEW {q_name} AS "
                    f"SELECT * FROM mart.main.{q_name};"
                )

        for silver_name, silver_path in self._discover_silver_tables().items():
            q_name = _quote_ident(silver_name)
            con.execute(
                f"CREATE OR REPLACE TEMP VIEW {q_name} AS "
                f"SELECT * FROM delta_scan({_sql_literal(str(silver_path))});"
            )
        return con

    def execute_readonly(self, sql: str) -> SqlQueryResult:
        """Execute an exploratory read-only SQL query against Delta silver and mart views."""
        start_ns = time.monotonic_ns()
        con = self._open_readonly_connection()
        try:
            cursor = con.execute(sql)
            if cursor.description is None:
                columns: list[str] = []
                rows: list[list[Any]] = []
            else:
                columns = [desc[0] for desc in cursor.description]
                rows = [list(row) for row in cursor.fetchall()]
        finally:
            con.close()

        duration_ms = round((time.monotonic_ns() - start_ns) / 1_000_000.0, 2)
        return SqlQueryResult(
            sql=sql,
            columns=columns,
            rows=rows,
            row_count=len(rows),
            duration_ms=duration_ms,
        )

    def run_dbt(
        self,
        args: list[str],
        env_overrides: dict[str, str] | None = None,
        timeout_seconds: int = 120,
    ) -> subprocess.CompletedProcess[str]:
        """Run a dbt CLI command against the workspace dbt_project with Shadow Overlay env."""
        venv_dbt = Path(sys.executable).parent / "dbt"
        dbt_exe = str(venv_dbt) if venv_dbt.exists() else (shutil.which("dbt") or "dbt")

        env = os.environ.copy()
        env["DATAOPS_DUCKDB_PATH"] = str(self.mart_duckdb_path)
        env["DATAOPS_PROD_SILVER_DIR"] = str(self.paths.prod_silver_dir)
        for table_name, override_path in self.silver_overrides.items():
            env[f"DATAOPS_SILVER_{table_name.upper()}_PATH"] = str(override_path)
        if env_overrides:
            env.update(env_overrides)

        return subprocess.run(
            [dbt_exe, *args, "--profiles-dir", "."],
            cwd=str(self.paths.dbt_project_dir),
            env=env,
            text=True,
            capture_output=True,
            timeout=timeout_seconds,
            check=False,
        )

    def data_diff(
        self,
        prod_db: Path,
        shadow_db: Path,
        table_name: str,
        key_column: str,
    ) -> DataDiffSummary:
        """Compare prod vs. shadow rows for `table_name` keyed by `key_column`."""
        q_table = _quote_ident(table_name)
        q_key = _quote_ident(key_column)
        con = duckdb.connect(":memory:")
        try:
            con.execute(
                f"ATTACH {_sql_literal(str(prod_db.resolve()))} AS prod_db (READ_ONLY);"
            )
            con.execute(
                f"ATTACH {_sql_literal(str(shadow_db.resolve()))} AS shadow_db (READ_ONLY);"
            )

            prod_cols = [
                row[0]
                for row in con.execute(f"DESCRIBE prod_db.main.{q_table};").fetchall()
            ]
            shadow_cols = [
                row[0]
                for row in con.execute(f"DESCRIBE shadow_db.main.{q_table};").fetchall()
            ]
            if key_column not in prod_cols or key_column not in shadow_cols:
                raise KeyError(
                    f"Key column {key_column!r} missing from table {table_name!r}"
                )

            prod_count = int(
                con.execute(f"SELECT COUNT(*) FROM prod_db.main.{q_table};").fetchone()[0]
            )
            shadow_count = int(
                con.execute(f"SELECT COUNT(*) FROM shadow_db.main.{q_table};").fetchone()[0]
            )

            added_keys = [
                str(r[0])
                for r in con.execute(
                    f"SELECT CAST(s.{q_key} AS VARCHAR) "
                    f"FROM shadow_db.main.{q_table} s "
                    f"LEFT JOIN prod_db.main.{q_table} p ON s.{q_key} = p.{q_key} "
                    f"WHERE p.{q_key} IS NULL ORDER BY 1;"
                ).fetchall()
            ]
            removed_keys = [
                str(r[0])
                for r in con.execute(
                    f"SELECT CAST(p.{q_key} AS VARCHAR) "
                    f"FROM prod_db.main.{q_table} p "
                    f"LEFT JOIN shadow_db.main.{q_table} s ON p.{q_key} = s.{q_key} "
                    f"WHERE s.{q_key} IS NULL ORDER BY 1;"
                ).fetchall()
            ]

            shared_non_keys = [
                col for col in prod_cols if col in shadow_cols and col != key_column
            ]
            changed_keys: list[str] = []
            if set(prod_cols) != set(shadow_cols):
                changed_keys = [
                    str(r[0])
                    for r in con.execute(
                        f"SELECT CAST(s.{q_key} AS VARCHAR) "
                        f"FROM shadow_db.main.{q_table} s "
                        f"INNER JOIN prod_db.main.{q_table} p ON s.{q_key} = p.{q_key} "
                        f"ORDER BY 1;"
                    ).fetchall()
                ]
            elif shared_non_keys:
                predicates = " OR ".join(
                    f"p.{_quote_ident(c)} IS DISTINCT FROM s.{_quote_ident(c)}"
                    for c in shared_non_keys
                )
                changed_keys = [
                    str(r[0])
                    for r in con.execute(
                        f"SELECT CAST(s.{q_key} AS VARCHAR) "
                        f"FROM shadow_db.main.{q_table} s "
                        f"INNER JOIN prod_db.main.{q_table} p ON s.{q_key} = p.{q_key} "
                        f"WHERE {predicates} ORDER BY 1;"
                    ).fetchall()
                ]

            is_identical = not (added_keys or removed_keys or changed_keys)
            return DataDiffSummary(
                table_name=table_name,
                key_column=key_column,
                prod_row_count=prod_count,
                shadow_row_count=shadow_count,
                added_keys=added_keys,
                removed_keys=removed_keys,
                changed_keys=changed_keys,
                is_identical=is_identical,
            )
        finally:
            con.close()

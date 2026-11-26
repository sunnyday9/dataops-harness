from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Literal

import duckdb

from src.adapters.base import LakehousePaths, LineageAdapterBase, StorageAdapter
from src.adapters.model_adapter import ToolDefinition
from src.adapters.spark_adapter import discover_java_home
from src.core.context import ToolCall, ToolResult

ALLOWED_GIT_SUBCOMMANDS = frozenset({"log", "diff", "blame", "show"})
BLOCKED_GIT_FLAGS = ("--exec", "--upload-pack", "--receive-pack", "--ext-diff")
MAX_READ_LINES = 200
MAX_GREP_MATCHES = 200
SKIP_GREP_DIRS = frozenset(
    {".git", "lakehouse", "target", "logs", "dbt_packages", "__pycache__", ".venv"}
)


class ToolRegistry:
    """Registry exposing strictly the six orthogonal atomic tools (Q2 Option A)."""

    def __init__(
        self,
        workspace_dir: Path,
        skills_dir: Path | None = None,
        lineage_adapter: LineageAdapterBase | None = None,
        duckdb_adapter: StorageAdapter | None = None,
    ) -> None:
        self.workspace_dir = workspace_dir.resolve()
        self.skills_dir = skills_dir.resolve() if skills_dir else None
        self.lineage_adapter = lineage_adapter
        self.duckdb_adapter = duckdb_adapter

    def definitions(self) -> list[ToolDefinition]:
        """Return the fixed 6-tool JSON-schema list kept identical across turns."""
        return [
            ToolDefinition(
                name="read_file",
                description=(
                    "Read a line range (max 200 lines per call) from a file inside "
                    "workspace/ or read-only skills/*.md playbooks."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "start_line": {"type": "integer", "minimum": 1, "default": 1},
                        "end_line": {"type": "integer", "minimum": 1, "default": 200},
                    },
                    "required": ["path"],
                },
            ),
            ToolDefinition(
                name="grep_search",
                description="Regex search inside workspace/ files (including .scratch/ and incidents/).",
                input_schema={
                    "type": "object",
                    "properties": {
                        "pattern": {"type": "string"},
                        "path": {"type": "string", "default": "."},
                    },
                    "required": ["pattern"],
                },
            ),
            ToolDefinition(
                name="query_lineage",
                description=(
                    "Query N-hop upstream/downstream cross-engine lineage, columns, "
                    "and dbt tests for a table or model."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "table_name": {"type": "string"},
                        "direction": {
                            "type": "string",
                            "enum": ["upstream", "downstream", "both"],
                            "default": "both",
                        },
                        "depth": {"type": "integer", "minimum": 1, "maximum": 10, "default": 3},
                    },
                    "required": ["table_name"],
                },
            ),
            ToolDefinition(
                name="run_sql_readonly",
                description=(
                    "Execute read-only SQL against the lakehouse (duckdb by default; "
                    "spark for EXPLAIN FORMATTED or DESCRIBE HISTORY)."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "sql": {"type": "string"},
                        "engine": {
                            "type": "string",
                            "enum": ["duckdb", "spark"],
                            "default": "duckdb",
                        },
                    },
                    "required": ["sql"],
                },
            ),
            ToolDefinition(
                name="git_inspect",
                description="Inspect sandbox git history with read-only subcommands (log, diff, blame, show).",
                input_schema={
                    "type": "object",
                    "properties": {
                        "subcommand": {
                            "type": "string",
                            "enum": ["log", "diff", "blame", "show"],
                        },
                        "args": {
                            "type": "array",
                            "items": {"type": "string"},
                            "default": [],
                        },
                    },
                    "required": ["subcommand"],
                },
            ),
            ToolDefinition(
                name="apply_patch",
                description="Apply a unified diff patch strictly inside workspace/.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "patch": {"type": "string"},
                    },
                    "required": ["patch"],
                },
            ),
        ]

    def _resolve_jailed_path(self, raw_path: str, *, allow_skills: bool) -> Path:
        cleaned = raw_path.strip()
        if not cleaned:
            raise PermissionError("Empty path is not permitted.")
        candidate = Path(cleaned)

        if allow_skills and self.skills_dir is not None and not candidate.is_absolute():
            normalized = cleaned.replace("\\", "/").lstrip("./")
            if normalized == "skills" or normalized.startswith("skills/"):
                skills_root = self.skills_dir
                rel_after_skills = (
                    normalized[len("skills/") :] if normalized.startswith("skills/") else ""
                )
                target = (skills_root / rel_after_skills).resolve()
                if not target.is_relative_to(skills_root):
                    raise PermissionError(f"Path escapes skills/: {raw_path}")
                return target

        resolved = (
            candidate.resolve()
            if candidate.is_absolute()
            else (self.workspace_dir / candidate).resolve()
        )
        if not resolved.is_relative_to(self.workspace_dir):
            raise PermissionError(f"Path escapes workspace/: {raw_path}")
        return resolved

    def _tool_read_file(self, args: dict[str, Any]) -> ToolResult:
        raw_path = str(args.get("path") or "")
        target = self._resolve_jailed_path(raw_path, allow_skills=True)
        if not target.is_file():
            return ToolResult(
                tool_call_id="",
                name="read_file",
                output=f"File not found: {raw_path}",
                is_error=True,
            )
        start_line = max(int(args.get("start_line") or 1), 1)
        requested_end = int(args.get("end_line") or (start_line + MAX_READ_LINES - 1))
        end_line = min(max(requested_end, start_line), start_line + MAX_READ_LINES - 1)

        lines = target.read_text(encoding="utf-8", errors="replace").splitlines()
        total = len(lines)
        slice_lines = lines[start_line - 1 : end_line]
        formatted = [
            f"{idx}: {line}"
            for idx, line in enumerate(slice_lines, start=start_line)
        ]
        header = f"# {raw_path} (lines {start_line}-{min(end_line, total)} of {total})"
        return ToolResult(
            tool_call_id="",
            name="read_file",
            output="\n".join([header, *formatted]),
        )

    def _tool_grep_search(self, args: dict[str, Any]) -> ToolResult:
        pattern = str(args.get("pattern") or "")
        raw_root = str(args.get("path") or ".")
        if not pattern:
            return ToolResult(
                tool_call_id="",
                name="grep_search",
                output="Pattern must not be empty.",
                is_error=True,
            )
        regex = re.compile(pattern)
        root = self._resolve_jailed_path(raw_root, allow_skills=False)
        files = [root] if root.is_file() else sorted(root.rglob("*"))

        matches: list[str] = []
        for file_path in files:
            if not file_path.is_file():
                continue
            rel = file_path.relative_to(self.workspace_dir)
            if any(part in SKIP_GREP_DIRS for part in rel.parts):
                continue
            try:
                text = file_path.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue
            for line_no, line in enumerate(text.splitlines(), start=1):
                if regex.search(line):
                    matches.append(f"{rel.as_posix()}:{line_no}: {line}")
                    if len(matches) >= MAX_GREP_MATCHES:
                        matches.append("... (truncated at 200 matches)")
                        return ToolResult(
                            tool_call_id="",
                            name="grep_search",
                            output="\n".join(matches),
                        )

        return ToolResult(
            tool_call_id="",
            name="grep_search",
            output="\n".join(matches) if matches else "No matches found.",
        )

    def _tool_query_lineage(self, args: dict[str, Any]) -> ToolResult:
        if self.lineage_adapter is None:
            return ToolResult(
                tool_call_id="",
                name="query_lineage",
                output="LineageAdapter is not configured.",
                is_error=True,
            )
        table_name = str(args.get("table_name") or "")
        direction: Literal["upstream", "downstream", "both"] = args.get(
            "direction", "both"
        )
        depth = int(args.get("depth") or 3)
        subgraph = self.lineage_adapter.query_lineage(
            table_name=table_name, direction=direction, depth=depth
        )
        return ToolResult(
            tool_call_id="",
            name="query_lineage",
            output=subgraph.model_dump_json(indent=2),
        )

    def _tool_run_sql_readonly(self, args: dict[str, Any]) -> ToolResult:
        sql = str(args.get("sql") or "")
        engine = str(args.get("engine") or "duckdb")
        if engine == "duckdb":
            if self.duckdb_adapter is None:
                return ToolResult(
                    tool_call_id="",
                    name="run_sql_readonly",
                    output="DuckDBAdapter is not configured.",
                    is_error=True,
                )
            res = self.duckdb_adapter.execute_readonly(sql)
            return ToolResult(
                tool_call_id="",
                name="run_sql_readonly",
                output=res.model_dump_json(indent=2),
            )
        if engine == "spark":
            return self._run_spark_readonly_sql(sql)
        return ToolResult(
            tool_call_id="",
            name="run_sql_readonly",
            output=f"Unsupported SQL engine: {engine}",
            is_error=True,
        )

    def _run_spark_readonly_sql(self, sql: str) -> ToolResult:
        paths = LakehousePaths.from_workspace(self.workspace_dir)
        java_home = discover_java_home()
        helper_code = (
            "import json, sys\n"
            "from pathlib import Path\n"
            "sys.path.insert(0, str(Path(sys.argv[1])))\n"
            "from spark_config import build_spark_session\n"
            "silver_dir = Path(sys.argv[2])\n"
            "sql_text = sys.argv[3]\n"
            "spark = build_spark_session('readonly_probe')\n"
            "try:\n"
            "    for p in sorted(silver_dir.iterdir()):\n"
            "        if p.is_dir() and (p / '_delta_log').is_dir():\n"
            "            spark.read.format('delta').load(str(p)).createOrReplaceTempView(p.name)\n"
            "    df = spark.sql(sql_text)\n"
            "    rows = [list(r) for r in df.limit(50).collect()]\n"
            "    print(json.dumps({'columns': df.columns, 'rows': rows}, default=str))\n"
            "finally:\n"
            "    spark.stop()\n"
        )
        env = os.environ.copy()
        env["JAVA_HOME"] = str(java_home)
        env["PATH"] = f"{java_home / 'bin'}:{env.get('PATH', '')}"
        env["PYSPARK_PYTHON"] = sys.executable
        env["PYSPARK_DRIVER_PYTHON"] = sys.executable
        proc = subprocess.run(
            [
                sys.executable,
                "-c",
                helper_code,
                str(paths.spark_jobs_dir),
                str(paths.prod_silver_dir),
                sql,
            ],
            cwd=str(paths.spark_jobs_dir),
            env=env,
            text=True,
            capture_output=True,
            timeout=120,
            check=False,
        )
        if proc.returncode != 0:
            return ToolResult(
                tool_call_id="",
                name="run_sql_readonly",
                output=proc.stderr.strip() or proc.stdout.strip(),
                is_error=True,
            )
        return ToolResult(
            tool_call_id="",
            name="run_sql_readonly",
            output=proc.stdout.strip(),
        )

    def _tool_git_inspect(self, args: dict[str, Any]) -> ToolResult:
        subcommand = str(args.get("subcommand") or "")
        if subcommand not in ALLOWED_GIT_SUBCOMMANDS:
            return ToolResult(
                tool_call_id="",
                name="git_inspect",
                output=f"Disallowed git subcommand: {subcommand!r}",
                is_error=True,
                blocked=True,
            )
        extra_args = [str(a) for a in (args.get("args") or [])]
        for flag in extra_args:
            if flag == "-c" or any(flag.startswith(bad) for bad in BLOCKED_GIT_FLAGS):
                return ToolResult(
                    tool_call_id="",
                    name="git_inspect",
                    output=f"Disallowed git flag: {flag!r}",
                    is_error=True,
                    blocked=True,
                )

        proc = subprocess.run(
            ["git", subcommand, *extra_args],
            cwd=str(self.workspace_dir),
            text=True,
            capture_output=True,
            check=False,
        )
        output = proc.stdout if proc.returncode == 0 else (proc.stderr or proc.stdout)
        return ToolResult(
            tool_call_id="",
            name="git_inspect",
            output=output.strip() or "(empty git output)",
            is_error=proc.returncode != 0,
        )

    def _extract_patch_targets(self, patch_text: str) -> list[str]:
        targets: list[str] = []
        for line in patch_text.splitlines():
            if not line.startswith(("--- ", "+++ ")):
                continue
            raw_target = line[4:].split("\t", 1)[0].strip()
            if raw_target == "/dev/null":
                continue
            if raw_target.startswith(("a/", "b/")):
                raw_target = raw_target[2:]
            resolved = self._resolve_jailed_path(raw_target, allow_skills=False)
            rel = resolved.relative_to(self.workspace_dir).as_posix()
            if rel == ".git" or rel.startswith(".git/"):
                raise PermissionError(f"Patching .git/ is forbidden: {raw_target}")
            if rel not in targets:
                targets.append(rel)
        if not targets:
            raise ValueError("Patch contains no valid target files inside workspace/.")
        return targets

    def _tool_apply_patch(self, args: dict[str, Any]) -> ToolResult:
        patch_text = str(args.get("patch") or "")
        if not patch_text.strip():
            return ToolResult(
                tool_call_id="",
                name="apply_patch",
                output="Patch text must not be empty.",
                is_error=True,
            )
        targets = self._extract_patch_targets(patch_text)
        normalized_patch = patch_text if patch_text.endswith("\n") else patch_text + "\n"

        check_proc = subprocess.run(
            ["git", "apply", "--check", "-"],
            cwd=str(self.workspace_dir),
            input=normalized_patch,
            text=True,
            capture_output=True,
            check=False,
        )
        if check_proc.returncode != 0:
            return ToolResult(
                tool_call_id="",
                name="apply_patch",
                output=f"git apply --check failed:\n{check_proc.stderr.strip()}",
                is_error=True,
            )

        apply_proc = subprocess.run(
            ["git", "apply", "-"],
            cwd=str(self.workspace_dir),
            input=normalized_patch,
            text=True,
            capture_output=True,
            check=False,
        )
        if apply_proc.returncode != 0:
            return ToolResult(
                tool_call_id="",
                name="apply_patch",
                output=f"git apply failed:\n{apply_proc.stderr.strip()}",
                is_error=True,
            )

        return ToolResult(
            tool_call_id="",
            name="apply_patch",
            output=f"Applied patch to: {', '.join(targets)}",
            metadata={"patched_files": targets},
        )

    def execute(self, tool_call: ToolCall) -> ToolResult:
        """Dispatch one of the six registered orthogonal tools."""
        handlers = {
            "read_file": self._tool_read_file,
            "grep_search": self._tool_grep_search,
            "query_lineage": self._tool_query_lineage,
            "run_sql_readonly": self._tool_run_sql_readonly,
            "git_inspect": self._tool_git_inspect,
            "apply_patch": self._tool_apply_patch,
        }
        handler = handlers.get(tool_call.name)
        if handler is None:
            return ToolResult(
                tool_call_id=tool_call.id,
                name=tool_call.name,
                output=f"Unknown tool: {tool_call.name!r}",
                is_error=True,
            )
        try:
            result = handler(tool_call.arguments)
            return result.model_copy(update={"tool_call_id": tool_call.id})
        except PermissionError as exc:
            return ToolResult(
                tool_call_id=tool_call.id,
                name=tool_call.name,
                output=f"Permission denied: {exc}",
                is_error=True,
                blocked=True,
            )
        except (
            ValueError,
            KeyError,
            OSError,
            re.error,
            subprocess.SubprocessError,
            duckdb.Error,
            RuntimeError,
        ) as exc:
            return ToolResult(
                tool_call_id=tool_call.id,
                name=tool_call.name,
                output=f"{type(exc).__name__}: {exc}",
                is_error=True,
            )

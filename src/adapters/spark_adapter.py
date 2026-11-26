from __future__ import annotations

import json
import os
import shutil
import signal
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from src.adapters.base import (
    ComputeAdapter,
    SparkEventSummary,
    SparkRunResult,
    SparkStageMetrics,
)


def discover_java_home() -> Path:
    """Locate a valid OpenJDK installation directory."""
    candidates: list[Path] = []
    env_home = os.environ.get("JAVA_HOME")
    if env_home:
        candidates.append(Path(env_home))
    candidates.append(Path.home() / ".local" / "opt" / "jdk-17")

    java_bin = shutil.which("java")
    if java_bin:
        candidates.append(Path(java_bin).resolve().parent.parent)

    for candidate in candidates:
        java_exe = candidate / "bin" / "java"
        if java_exe.is_file() and os.access(java_exe, os.X_OK):
            return candidate.resolve()

    raise RuntimeError(
        "OpenJDK not found. Set JAVA_HOME or install JDK 17 into ~/.local/opt/jdk-17."
    )


class SparkEventLogCapture:
    """Snapshot helper that identifies the event log written by a single Spark run."""

    @staticmethod
    def snapshot(event_log_dir: Path) -> dict[Path, int]:
        if not event_log_dir.exists():
            return {}
        return {
            p: p.stat().st_mtime_ns
            for p in event_log_dir.iterdir()
            if p.is_file()
        }

    @staticmethod
    def find_new_log(
        event_log_dir: Path, before: dict[Path, int]
    ) -> Path | None:
        if not event_log_dir.exists():
            return None
        changed: list[tuple[int, int, Path]] = []
        for entry in event_log_dir.iterdir():
            if not entry.is_file():
                continue
            mtime_ns = entry.stat().st_mtime_ns
            prev_mtime = before.get(entry)
            if prev_mtime is None or mtime_ns > prev_mtime:
                is_final = 0 if entry.name.endswith(".inprogress") else 1
                changed.append((is_final, mtime_ns, entry))
        if not changed:
            return None
        changed.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return changed[0][2]


class SparkEventLogParser:
    """Deterministic parser extracting per-stage task skew and OOM signals from Spark logs."""

    SKEW_THRESHOLD: float = 10.0
    OOM_PATTERNS: tuple[str, ...] = (
        "outofmemoryerror",
        "java heap space",
        "gc overhead limit exceeded",
        "direct buffer memory",
        "memorystore",
    )

    @classmethod
    def _contains_oom(cls, text: str) -> bool:
        lowered = text.lower()
        return any(pattern in lowered for pattern in cls.OOM_PATTERNS)

    @classmethod
    def _extract_task_duration_ms(cls, record: dict[str, Any]) -> float:
        task_info = record.get("Task Info") or {}
        launch = task_info.get("Launch Time")
        finish = task_info.get("Finish Time")
        if (
            isinstance(launch, (int, float))
            and isinstance(finish, (int, float))
            and launch > 0
            and finish >= launch
        ):
            return float(finish - launch)
        metrics = record.get("Task Metrics") or {}
        exec_run = metrics.get("Executor Run Time")
        if isinstance(exec_run, (int, float)) and exec_run >= 0:
            return float(exec_run)
        return 0.0

    @classmethod
    def parse_file(
        cls,
        event_log_path: Path | None,
        stderr: str = "",
        skew_threshold: float = SKEW_THRESHOLD,
    ) -> SparkEventSummary:
        """Parse a Spark JSON-lines event log and optional stderr into a diagnostic summary."""
        app_id: str | None = None
        app_name: str | None = None
        stage_names: dict[int, str] = {}
        stage_durations: dict[int, list[float]] = defaultdict(list)
        stage_failures: dict[int, int] = defaultdict(int)
        total_tasks = 0
        failed_tasks = 0
        oom_reasons: list[str] = []

        def record_oom(reason: str) -> None:
            cleaned = reason.strip()
            if cleaned and cls._contains_oom(cleaned) and cleaned not in oom_reasons:
                oom_reasons.append(cleaned)

        if event_log_path is not None and event_log_path.is_file():
            with event_log_path.open("r", encoding="utf-8", errors="replace") as handle:
                for raw_line in handle:
                    line = raw_line.strip()
                    if not line:
                        continue
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        # Crashed JVMs can leave an incomplete trailing JSON event line.
                        continue

                    etype = event.get("Event")
                    if etype == "SparkListenerApplicationStart":
                        app_id = event.get("App ID") or app_id
                        app_name = event.get("App Name") or app_name
                    elif etype == "SparkListenerStageSubmitted":
                        info = event.get("Stage Info") or {}
                        sid = info.get("Stage ID")
                        if isinstance(sid, int):
                            stage_names[sid] = str(info.get("Stage Name") or "")
                    elif etype == "SparkListenerStageCompleted":
                        info = event.get("Stage Info") or {}
                        failure = info.get("Failure Reason")
                        if isinstance(failure, str):
                            record_oom(failure)
                    elif etype == "SparkListenerJobEnd":
                        result = event.get("Job Result") or {}
                        exc = result.get("Exception")
                        if isinstance(exc, dict):
                            msg = str(exc.get("Message") or "")
                            record_oom(msg)
                    elif etype == "SparkListenerTaskEnd":
                        sid = event.get("Stage ID")
                        if not isinstance(sid, int):
                            continue
                        total_tasks += 1
                        duration_ms = cls._extract_task_duration_ms(event)
                        stage_durations[sid].append(duration_ms)

                        task_info = event.get("Task Info") or {}
                        end_reason = event.get("Task End Reason") or {}
                        reason_name = str(end_reason.get("Reason") or "")
                        is_failed = bool(task_info.get("Failed")) or (
                            reason_name not in ("", "Success")
                        )
                        if is_failed:
                            failed_tasks += 1
                            stage_failures[sid] += 1
                            end_text = json.dumps(end_reason, ensure_ascii=False)
                            if cls._contains_oom(end_text):
                                desc = (
                                    end_reason.get("Description")
                                    or end_reason.get("Class Name")
                                    or end_text
                                )
                                record_oom(str(desc))

        if stderr and cls._contains_oom(stderr):
            for line in stderr.splitlines():
                if cls._contains_oom(line):
                    record_oom(line)
            if not oom_reasons:
                oom_reasons.append("OutOfMemoryError detected in subprocess stderr")

        stages: list[SparkStageMetrics] = []
        skewed_stages: list[int] = []
        max_skew = 1.0
        for sid in sorted(stage_durations):
            durations = stage_durations[sid]
            min_ms = round(min(durations), 2)
            median_ms = round(float(statistics.median(durations)), 2)
            max_ms = round(max(durations), 2)
            skew_ratio = round(max_ms / max(median_ms, 1.0), 2)
            max_skew = max(max_skew, skew_ratio)
            if len(durations) >= 2 and skew_ratio >= skew_threshold:
                skewed_stages.append(sid)
            stages.append(
                SparkStageMetrics(
                    stage_id=sid,
                    stage_name=stage_names.get(sid, ""),
                    task_count=len(durations),
                    failed_tasks=stage_failures[sid],
                    min_ms=min_ms,
                    median_ms=median_ms,
                    max_ms=max_ms,
                    skew_ratio=skew_ratio,
                )
            )

        return SparkEventSummary(
            app_id=app_id,
            app_name=app_name,
            event_log_path=event_log_path,
            total_tasks=total_tasks,
            failed_tasks=failed_tasks,
            max_skew_ratio=max_skew,
            skewed_stages=skewed_stages,
            stages=stages,
            oom_detected=bool(oom_reasons),
            oom_reasons=oom_reasons,
        )


class SparkJobRunner(ComputeAdapter):
    """Execute PySpark scripts inside isolated child subprocesses."""

    def __init__(
        self,
        event_log_dir: Path = Path("/tmp/spark-events"),
        default_driver_memory: str = "512m",
        default_timeout_seconds: int = 240,
    ) -> None:
        self.event_log_dir = event_log_dir
        self.default_driver_memory = default_driver_memory
        self.default_timeout_seconds = default_timeout_seconds

    def run_job(
        self,
        script_path: Path,
        args: list[str] | None = None,
        driver_memory: str | None = None,
        timeout_seconds: int | None = None,
        env_overrides: dict[str, str] | None = None,
    ) -> SparkRunResult:
        """Run a PySpark script in a fresh subprocess with memory, timeout, and event logging."""
        resolved_script = script_path.resolve()
        self.event_log_dir.mkdir(parents=True, exist_ok=True)
        java_home = discover_java_home()

        env = os.environ.copy()
        env["JAVA_HOME"] = str(java_home)
        env["PATH"] = f"{java_home / 'bin'}:{env.get('PATH', '')}"
        env["PYSPARK_PYTHON"] = sys.executable
        env["PYSPARK_DRIVER_PYTHON"] = sys.executable
        env["SPARK_DRIVER_MEMORY"] = driver_memory or self.default_driver_memory
        env["SPARK_EVENT_LOG_DIR"] = str(self.event_log_dir.resolve())
        if env_overrides:
            env.update(env_overrides)

        before_snapshot = SparkEventLogCapture.snapshot(self.event_log_dir)
        cmd = [sys.executable, str(resolved_script), *(args or [])]
        timeout = timeout_seconds or self.default_timeout_seconds
        start_ns = time.monotonic_ns()
        timed_out = False

        proc = subprocess.Popen(
            cmd,
            cwd=str(resolved_script.parent),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
        try:
            stdout, stderr = proc.communicate(timeout=timeout)
            returncode = proc.returncode
        except subprocess.TimeoutExpired:
            timed_out = True
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            stdout, stderr = proc.communicate()
            returncode = -9

        duration_ms = (time.monotonic_ns() - start_ns) / 1_000_000.0
        event_log_path = SparkEventLogCapture.find_new_log(
            self.event_log_dir, before_snapshot
        )
        event_summary = SparkEventLogParser.parse_file(event_log_path, stderr=stderr)
        return SparkRunResult(
            job_name=resolved_script.stem,
            script_path=resolved_script,
            pid=proc.pid,
            returncode=returncode,
            duration_ms=round(duration_ms, 2),
            event_log_path=event_log_path,
            stdout=stdout,
            stderr=stderr,
            timed_out=timed_out,
            event_summary=event_summary,
        )

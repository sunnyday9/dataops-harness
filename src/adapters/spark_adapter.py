from __future__ import annotations

import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

from src.adapters.base import SparkRunResult


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


class SparkJobRunner:
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
        )

from __future__ import annotations

import os
from pathlib import Path

from delta import configure_spark_with_delta_pip
from pyspark.sql import SparkSession


def build_spark_session(
    app_name: str,
    driver_memory: str | None = None,
    event_log_dir: str | None = None,
    shuffle_partitions: int = 4,
    aqe_enabled: bool = True,
) -> SparkSession:
    """Build a lightweight local PySpark session with Delta Lake and event logging."""
    resolved_memory = driver_memory or os.environ.get("SPARK_DRIVER_MEMORY", "512m")
    resolved_event_dir = event_log_dir or os.environ.get(
        "SPARK_EVENT_LOG_DIR", "/tmp/spark-events"
    )
    Path(resolved_event_dir).mkdir(parents=True, exist_ok=True)
    event_log_uri = Path(resolved_event_dir).resolve().as_uri()

    builder = (
        SparkSession.builder.appName(app_name)
        .master("local[2]")
        .config("spark.driver.memory", resolved_memory)
        .config("spark.sql.shuffle.partitions", str(shuffle_partitions))
        .config("spark.default.parallelism", str(shuffle_partitions))
        .config("spark.ui.enabled", "false")
        .config("spark.eventLog.enabled", "true")
        .config("spark.eventLog.dir", event_log_uri)
        .config("spark.sql.adaptive.enabled", str(aqe_enabled).lower())
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config(
            "spark.sql.catalog.spark_catalog",
            "org.apache.spark.sql.delta.catalog.DeltaCatalog",
        )
        .config("spark.sql.sources.partitionOverwriteMode", "dynamic")
        .config("spark.databricks.delta.snapshotPartitions", "2")
    )
    extra_java_opts = os.environ.get("SPARK_DRIVER_EXTRA_JAVA_OPTIONS")
    if extra_java_opts:
        builder = builder.config("spark.driver.extraJavaOptions", extra_java_opts)
    return configure_spark_with_delta_pip(builder).getOrCreate()

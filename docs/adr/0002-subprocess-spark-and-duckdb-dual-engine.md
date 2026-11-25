---
status: accepted
---

# Subprocess-Isolated PySpark Paired With In-Process DuckDB

Real PySpark ETL execution, physical execution plans (`EXPLAIN FORMATTED`), and shadow Spark verification run in isolated child subprocesses with explicit memory and timeout ceilings, while high-frequency read-only exploration SQL runs in-process via DuckDB against local Parquet/Delta files. Fault scenarios intentionally trigger real JVM heap OOMs and severe task skew, so running Spark inside the harness process would crash the controller, whereas invoking a fresh JVM for every exploratory SQL query would make multi-turn diagnosis prohibitively slow.

## Considered Options

- **In-process persistent SparkSession**: Faster repeated queries, but an intentional driver OOM kills the harness process and breaks crash-safe checkpointing.
- **Subprocess Spark for everything**: Safe isolation, but 3–8 second JVM cold starts on every read-only exploratory SQL turn degrade latency and benchmark runtime.
- **DuckDB-only simulation**: Fast, but cannot reproduce genuine Spark physical execution failures (join skew task ratios, driver heap OOM, Spark event logs).
- **Subprocess PySpark + in-process DuckDB (chosen)**: Isolates real JVM crashes and preserves genuine Spark diagnostics while keeping interactive SQL exploration sub-second.

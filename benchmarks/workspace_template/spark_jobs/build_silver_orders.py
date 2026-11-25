from __future__ import annotations

import argparse
from pathlib import Path

from pyspark.sql import functions as F
from pyspark.sql.types import DateType, DoubleType, StringType, StructField, StructType
from spark_config import build_spark_session

RAW_ORDER_SCHEMA = StructType(
    [
        StructField("order_id", StringType(), False),
        StructField("user_id", StringType(), True),
        StructField("order_date", StringType(), False),
        StructField("currency", StringType(), False),
        StructField("amount_local", StringType(), False),
        StructField("status", StringType(), False),
        StructField("note", StringType(), True),
    ]
)


def run_job(raw_dir: Path, output_dir: Path, silver_read_dir: Path) -> None:
    """Enrich raw orders JSON with silver_users and write partitioned Delta silver_orders."""
    spark = build_spark_session("build_silver_orders")
    try:
        raw_orders_path = str(raw_dir / "orders")
        orders_raw = spark.read.schema(RAW_ORDER_SCHEMA).json(raw_orders_path)

        users_df = spark.read.format("delta").load(str(silver_read_dir / "silver_users"))
        users_country = users_df.select(
            F.col("user_id"),
            F.col("country").alias("user_country"),
        )

        enriched = (
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
        )

        target = str(output_dir / "silver_orders")
        (
            enriched.write.format("delta")
            .mode("overwrite")
            .partitionBy("order_date")
            .save(target)
        )
    finally:
        spark.stop()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=Path("../lakehouse/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("../lakehouse/prod/silver"))
    parser.add_argument("--silver-read-dir", type=Path, default=None)
    args = parser.parse_args()
    silver_read = (args.silver_read_dir or args.output_dir).resolve()
    run_job(args.raw_dir.resolve(), args.output_dir.resolve(), silver_read)


if __name__ == "__main__":
    main()

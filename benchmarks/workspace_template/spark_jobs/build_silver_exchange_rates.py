from __future__ import annotations

import argparse
from pathlib import Path

from pyspark.sql import functions as F
from pyspark.sql.types import DateType, DoubleType, StringType
from spark_config import build_spark_session


def run_job(raw_dir: Path, output_dir: Path) -> None:
    """Clean raw exchange rates CSV and write the Delta silver_exchange_rates table."""
    spark = build_spark_session("build_silver_exchange_rates")
    try:
        raw_path = str(raw_dir / "exchange_rates" / "exchange_rates.csv")
        df = spark.read.option("header", "true").csv(raw_path)
        cleaned = df.select(
            F.col("rate_date").cast(DateType()).alias("rate_date"),
            F.upper(F.trim(F.col("currency"))).cast(StringType()).alias("currency"),
            F.col("rate_to_usd").cast(DoubleType()).alias("rate_to_usd"),
        ).filter(F.col("rate_date").isNotNull() & F.col("currency").isNotNull())

        target = str(output_dir / "silver_exchange_rates")
        cleaned.write.format("delta").mode("overwrite").option(
            "overwriteSchema", "true"
        ).save(target)
    finally:
        spark.stop()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=Path("../lakehouse/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("../lakehouse/prod/silver"))
    args = parser.parse_args()
    run_job(args.raw_dir.resolve(), args.output_dir.resolve())


if __name__ == "__main__":
    main()

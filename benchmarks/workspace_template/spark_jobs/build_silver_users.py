from __future__ import annotations

import argparse
from pathlib import Path

from pyspark.sql import functions as F
from pyspark.sql.types import DateType, StringType
from spark_config import build_spark_session


def run_job(raw_dir: Path, output_dir: Path) -> None:
    """Clean raw users CSV and write the Delta silver_users table."""
    spark = build_spark_session("build_silver_users")
    try:
        raw_path = str(raw_dir / "users" / "users.csv")
        df = spark.read.option("header", "true").csv(raw_path)
        cleaned = (
            df.select(
                F.col("user_id").cast(StringType()).alias("user_id"),
                F.col("signup_date").cast(DateType()).alias("signup_date"),
                F.upper(F.trim(F.col("country"))).cast(StringType()).alias("country"),
                F.lower(F.trim(F.col("email"))).cast(StringType()).alias("email"),
                F.trim(F.col("phone")).cast(StringType()).alias("phone"),
                F.trim(F.col("national_id")).cast(StringType()).alias("national_id"),
            )
            .filter(F.col("user_id").isNotNull())
            .dropDuplicates(["user_id"])
        )
        target = str(output_dir / "silver_users")
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

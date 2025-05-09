from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_date, year, month, substring

# ── 1. Start Spark ──────────────────────────────────────────────────────────
spark = (SparkSession.builder
         .appName("CrimeDataIngestion")
         .config("spark.sql.shuffle.partitions", "8")
         .getOrCreate())

raw_base = "hdfs://localhost:9000/user/yashwanthreddy/crime/raw/"

# ── 2. Read each file type with nested month directories ────────────────────
street_df   = spark.read.option("header", True).csv(f"{raw_base}*/*-street.csv")
outcomes_df = spark.read.option("header", True).csv(f"{raw_base}*/*-outcomes.csv")
stop_df     = spark.read.option("header", True).csv(f"{raw_base}*/*-stop-and-search.csv")

# ── 3. Normalize column names ───────────────────────────────────────────────
def normalize(df):
    for c in df.columns:
        df = df.withColumnRenamed(c, c.strip()
            .lower()
            .replace(" ", "_")
            .replace("-", "_"))
    return df

street_df, outcomes_df, stop_df = map(normalize, [street_df, outcomes_df, stop_df])

# ── 4. Parse dates & add year/month ─────────────────────────────────────────
street_df = (
    street_df
    .withColumn("date",  to_date(col("month"), "yyyy-MM"))
    .withColumn("year",  year(col("date")))
    .withColumn("month", month(col("date")))
)

outcomes_df = (
    outcomes_df
    .withColumn("date",  to_date(col("month"), "yyyy-MM"))
    .withColumn("year",  year(col("date")))
    .withColumn("month", month(col("date")))
)

# For stop-and-search: strip time+offset, then parse
stop_df = (
    stop_df
    .withColumn("date_only", substring(col("date"), 1, 10))
    .withColumn("date", to_date(col("date_only"), "yyyy-MM-dd"))
    .withColumn("year", year(col("date")))
    .withColumn("month", month(col("date")))
    .drop("date_only")
)

# ── 4.5. Quick sanity checks ─────────────────────────────────────────────────
print("→ street_df schema:")
street_df.printSchema()
print(f"→ street_df count: {street_df.count()}")
print(f"→ outcomes_df count: {outcomes_df.count()}")
print(f"→ stop_df count: {stop_df.count()}")

# ── 5. Write partitioned Parquet ─────────────────────────────────────────────
street_df.write.mode("overwrite")\
    .partitionBy("year", "month")\
    .parquet("hdfs://localhost:9000/user/yashwanthreddy/crime/processed/street")

outcomes_df.write.mode("overwrite")\
    .partitionBy("year", "month")\
    .parquet("hdfs://localhost:9000/user/yashwanthreddy/crime/processed/outcomes")

stop_df.write.mode("overwrite")\
    .partitionBy("year", "month")\
    .parquet("hdfs://localhost:9000/user/yashwanthreddy/crime/processed/stop_and_search")

spark.stop()

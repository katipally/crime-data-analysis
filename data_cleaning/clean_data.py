from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lower, trim, regexp_replace

# ── 1. Initialize Spark ──────────────────────────────────────────────────────
spark = (SparkSession.builder
         .appName("CrimeDataCleaning")
         .config("spark.sql.shuffle.partitions", "8")
         .getOrCreate())

# ── 2. Define HDFS paths ─────────────────────────────────────────────────────
processed_base = "hdfs://localhost:9000/user/yashwanthreddy/crime/processed"
clean_base     = "hdfs://localhost:9000/user/yashwanthreddy/crime/clean"

# ── 3. Read partitioned Parquet tables ──────────────────────────────────────
street_df   = spark.read.parquet(f"{processed_base}/street")
outcomes_df = spark.read.parquet(f"{processed_base}/outcomes")
stop_df     = spark.read.parquet(f"{processed_base}/stop_and_search")

# ── 4. Drop duplicates & filter invalid coordinates ───────────────────────────
def clean_geo(df):
    return (df.dropDuplicates()
              .filter(col("latitude").isNotNull())
              .filter(col("longitude").isNotNull())
              .filter((col("latitude") != 0) & (col("longitude") != 0)))

street_df   = clean_geo(street_df)
outcomes_df = clean_geo(outcomes_df)
stop_df     = clean_geo(stop_df)

# ── 5. Normalize text columns (remove non-ASCII, trim, lowercase) ────────────
def normalize_text(df, cols):
    for c in cols:
        df = df.withColumn(
            c,
            lower(trim(regexp_replace(col(c), "[^\\x00-\\x7F]", "")))
        )
    return df

street_text_cols = [
    "reported_by", "falls_within", "location",
    "lsoa_code", "lsoa_name", "crime_type",
    "last_outcome_category", "context"
]
outcomes_text_cols = [
    "reported_by", "falls_within", "location",
    "lsoa_code", "lsoa_name", "outcome_type"
]
stop_text_cols = [
    "type", "part_of_a_policing_operation", "policing_operation",
    "gender", "age_range", "self_defined_ethnicity",
    "officer_defined_ethnicity", "legislation", "object_of_search",
    "outcome", "outcome_linked_to_object_of_search",
    "removal_of_more_than_just_outer_clothing"
]

street_df   = normalize_text(street_df, street_text_cols)
outcomes_df = normalize_text(outcomes_df, outcomes_text_cols)
stop_df     = normalize_text(stop_df, stop_text_cols)

# ── 6. Write cleaned Parquet (partitioned by year & month) ──────────────────
street_df.write.mode("overwrite")\
    .partitionBy("year", "month")\
    .parquet(f"{clean_base}/street")

outcomes_df.write.mode("overwrite")\
    .partitionBy("year", "month")\
    .parquet(f"{clean_base}/outcomes")

stop_df.write.mode("overwrite")\
    .partitionBy("year", "month")\
    .parquet(f"{clean_base}/stop_and_search")

# ── 7. Stop Spark ───────────────────────────────────────────────────────────
spark.stop()

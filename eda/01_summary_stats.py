from pyspark.sql import SparkSession
from pyspark.sql.functions import col, hour, dayofweek, month, year, min as _min, max as _max, avg, desc

# ── Initialize Spark ─────────────────────────────────────────────────────────
spark = (SparkSession.builder
         .appName("CrimeEDAStats")
         .config("spark.sql.shuffle.partitions", "50")  # adjust partitions
         .getOrCreate())

# ── Define HDFS paths ─────────────────────────────────────────────────────────
clean_base = "hdfs://localhost:9000/user/yashwanthreddy/crime/clean"
eda_base   = "hdfs://localhost:9000/user/yashwanthreddy/crime/eda_output"

# ── Read cleaned Parquet tables ───────────────────────────────────────────────
street_df   = spark.read.parquet(f"{clean_base}/street")
outcomes_df = spark.read.parquet(f"{clean_base}/outcomes")
stop_df     = spark.read.parquet(f"{clean_base}/stop_and_search")

# ── Join street-level crimes with outcomes ─────────────────────────────────────
street_with_outcomes = street_df.join(
    outcomes_df.select("crime_id", "outcome_type"),
    on="crime_id", how="left"
)

# ── 1. Counts by crime type ───────────────────────────────────────────────────
crime_counts = street_with_outcomes.groupBy("crime_type").count()
crime_counts.coalesce(1).write.mode("overwrite") \
    .csv(f"{eda_base}/crime_type_counts", header=True)

# ── 2. Counts by outcome type ─────────────────────────────────────────────────
outcome_counts = street_with_outcomes.groupBy("outcome_type").count()
outcome_counts.coalesce(1).write.mode("overwrite") \
    .csv(f"{eda_base}/outcome_type_counts", header=True)

# ── 3. Time distributions ────────────────────────────────────────────────────
# By hour of day
time_by_hour = street_df.withColumn("hour", hour("date")).groupBy("hour").count()
time_by_hour.coalesce(1).write.mode("overwrite") \
    .csv(f"{eda_base}/counts_by_hour", header=True)

# By day of week
time_by_dow = street_df.withColumn("dow", dayofweek("date")).groupBy("dow").count()
time_by_dow.coalesce(1).write.mode("overwrite") \
    .csv(f"{eda_base}/counts_by_dow", header=True)

# By month within a year
monthly_trend = street_df.withColumn("month", month("date")).groupBy("month").count()
monthly_trend.coalesce(1).write.mode("overwrite") \
    .csv(f"{eda_base}/counts_by_month", header=True)

# Combined year-month trend
ym_trend = street_df.groupBy("year", "month").count().orderBy("year", "month")
ym_trend.coalesce(1).write.mode("overwrite") \
    .csv(f"{eda_base}/year_month_trend", header=True)

# ── 4. Distinct counts and top categories ─────────────────────────────────────
# Number of distinct crime types
distinct_crime = street_df.select("crime_type").distinct().count()
spark.createDataFrame([(distinct_crime,)], ["distinct_crime_types"]) \
    .coalesce(1).write.mode("overwrite") \
    .csv(f"{eda_base}/distinct_crime_types", header=True)

# Top 10 LSOAs by incident count
lsoa_counts = street_df.groupBy("lsoa_code", "lsoa_name").count().orderBy(desc("count")).limit(10)
lsoa_counts.coalesce(1).write.mode("overwrite") \
    .csv(f"{eda_base}/top10_lsoas", header=True)

# Geographic coordinate stats
coord_stats = street_df.select(
    _min("latitude").alias("min_latitude"),
    _max("latitude").alias("max_latitude"),
    _min("longitude").alias("min_longitude"),
    _max("longitude").alias("max_longitude"),
    avg("latitude").alias("avg_latitude"),
    avg("longitude").alias("avg_longitude")
)
coord_stats.coalesce(1).write.mode("overwrite") \
    .csv(f"{eda_base}/coordinate_stats", header=True)

# ── 5. Stop-and-Search analysis ───────────────────────────────────────────────
# Counts by object_of_search
stop_obj_counts = stop_df.groupBy("object_of_search").count()
stop_obj_counts.coalesce(1).write.mode("overwrite") \
    .csv(f"{eda_base}/stop_object_counts", header=True)

# Outcome linked to search
stop_outcome_counts = stop_df.groupBy("outcome_linked_to_object_of_search").count()
stop_outcome_counts.coalesce(1).write.mode("overwrite") \
    .csv(f"{eda_base}/stop_outcome_linked_counts", header=True)

spark.stop()

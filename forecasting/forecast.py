#!/usr/bin/env python3

import os
import pandas as pd
import pickle
from pyspark.sql import SparkSession
from pyspark.sql.functions import to_date, year, month, col
from statsmodels.tsa.arima.model import ARIMA

def main():
    # ── 1. Spark Session ────────────────────────────────────────────────────────
    spark = (
        SparkSession.builder
        .appName("CrimeForecast")
        .config("spark.sql.shuffle.partitions", "50")
        .getOrCreate()
    )
    
    # ── 2. Read cleaned street-level Parquet from HDFS ─────────────────────────
    hdfs_path = "hdfs://localhost:9000/user/yashwanthreddy/crime/clean/street"
    df = spark.read.parquet(hdfs_path)
    
    # ── 3. Ensure `date` column is DateType ────────────────────────────────────
    df = df.withColumn("date", to_date(col("date")))

    # ── 4. Aggregate to (year, month) counts ───────────────────────────────────
    monthly = (
        df.groupBy(year("date").alias("year"), month("date").alias("month"))
          .count()
          .orderBy("year", "month")
    )

    # ── 5. Convert to Pandas time series ────────────────────────────────────────
    pdf = monthly.toPandas()
    # Create a proper datetime index on the first of each month
    pdf["date"] = pd.to_datetime(dict(year=pdf["year"], month=pdf["month"], day=1))
    pdf.set_index("date", inplace=True)
    ts = pdf["count"].sort_index()

    # ── 6. Train/Test split ─────────────────────────────────────────────────────
    train, test = ts[:-6], ts[-6:]

    # ── 7. Fit ARIMA model ─────────────────────────────────────────────────────
    model = ARIMA(train, order=(1,1,1)).fit()

    # ── 8. Forecast next 6 months ──────────────────────────────────────────────
    steps = len(test)
    fc = model.get_forecast(steps=steps)
    fc_df = fc.summary_frame()  # columns: mean, mean_ci_lower, mean_ci_upper

    # ── 9. Write forecast CSV ──────────────────────────────────────────────────
    os.makedirs("forecasting/output", exist_ok=True)
    fc_df.to_csv("forecasting/output/forecast.csv")

    # ── 10. Save fitted model ──────────────────────────────────────────────────
    with open("forecasting/output/arima_model.pkl", "wb") as f:
        pickle.dump(model, f)

    # ── 11. Clean up ───────────────────────────────────────────────────────────
    spark.stop()
    print("Forecasting complete. Outputs in forecasting/output/")

if __name__ == "__main__":
    main()

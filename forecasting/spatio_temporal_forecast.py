import pandas as pd
import numpy as np
import pickle
from pyspark.sql import SparkSession
from pyspark.sql.functions import to_date, year, month, col
import pmdarima as pm  # for auto_arima
import folium
from folium.plugins import HeatMap

# ── 1. Start Spark and read clustered Parquet ────────────────────────────────
spark = SparkSession.builder.appName("SpatioTemporalForecast").getOrCreate()
clusters = spark.read.parquet("hdfs://localhost:9000/user/yashwanthreddy/crime/clusters") \
            .withColumn("date", to_date(col("date")))

# ── 2. Compute monthly counts per cluster ────────────────────────────────────
agg = (clusters.groupBy("prediction", year("date").alias("year"), month("date").alias("month"))
              .count()
              .orderBy("prediction","year","month")
              .toPandas())
agg["date"] = pd.to_datetime(dict(year=agg["year"], month=agg["month"], day=1))

# ── 3. Get centroids ─────────────────────────────────────────────────────────
centroids = (clusters
    .groupBy("prediction")
    .agg({"latitude":"avg","longitude":"avg"})
    .toPandas()
    .rename(columns={"avg(latitude)":"centroid_lat","avg(longitude)":"centroid_lon"}))

# ── 4. Forecast per cluster ──────────────────────────────────────────────────
forecasts = []
models = {}
horizon = 6

for cid, grp in agg.groupby("prediction"):
    ts = grp.set_index("date")["count"].asfreq("MS")
    # Fit auto_arima for best params
    model = pm.auto_arima(ts, seasonal=True, m=12,
                          error_action="ignore", suppress_warnings=True)
    fc, conf_int = model.predict(n_periods=horizon, return_conf_int=True)
    future_idx = pd.date_range(start=ts.index[-1] + pd.offsets.MonthBegin(),
                               periods=horizon, freq="MS")
    df_fc = pd.DataFrame({
        "prediction": cid,
        "date": future_idx,
        "forecast": fc,
        "lower": conf_int[:,0],
        "upper": conf_int[:,1]
    })
    forecasts.append(df_fc)
    models[cid] = model  # store if you want diagnostics

forecast_df = pd.concat(forecasts, ignore_index=True)

# ── 5. Join with centroids for spatial plotting ──────────────────────────────
vis_df = forecast_df.merge(centroids, on="prediction")

# ── 6. Create a Folium map of forecasted hotspots ────────────────────────────
m = folium.Map(location=[vis_df.centroid_lat.mean(), vis_df.centroid_lon.mean()], zoom_start=10)
for _, row in vis_df.iterrows():
    folium.CircleMarker(
        location=(row.centroid_lat, row.centroid_lon),
        radius=5 + row.forecast * 0.1,  # size scaled by forecast
        color=None,
        fill=True,
        fill_color=plt.cm.Reds(row.forecast / vis_df.forecast.max()),
        fill_opacity=0.7,
        popup=f"Cluster {int(row.prediction)}<br>{row.date.date()}: {int(row.forecast)}"
    ).add_to(m)
m.save("spatio_temporal_forecast_map.html")

# ── 7. Save combined forecast table ──────────────────────────────────────────
forecast_df.to_csv("forecasting/output/spatio_forecast.csv", index=False)
with open("forecasting/output/spatio_models.pkl","wb") as f:
    pickle.dump(models, f)

spark.stop()
print("Spatio-temporal forecasting complete.")

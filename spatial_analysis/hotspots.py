from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.clustering import KMeans
from pyspark.sql.functions import col

spark = SparkSession.builder.appName("HotspotClustering").getOrCreate()
df = spark.read.parquet("hdfs://localhost:9000/user/yashwanthreddy/crime/clean/street").withColumn("latitude",  col("latitude").cast("double")).withColumn("longitude", col("longitude").cast("double"))

df = df.select("latitude", "longitude").na.drop()
assembler = VectorAssembler(inputCols=["latitude","longitude"], outputCol="features")
data = assembler.transform(df)

kmeans = KMeans().setK(8).setSeed(42).setFeaturesCol("features")
model = kmeans.fit(data)
clusters = model.transform(data)

clusters.write.mode("overwrite") \
    .parquet("hdfs://localhost:9000/user/yashwanthreddy/crime/clusters")
spark.stop()

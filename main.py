# main.py
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, avg, round, count, hour, rank, desc
from pyspark.sql.window import Window
import os

spark = SparkSession.builder.appName("MusicAnalysis").getOrCreate()
spark.sparkContext.setLogLevel("ERROR")
os.makedirs("outputs", exist_ok=True)

# Load datasets
logs = spark.read.format("csv").option("header", "true").option("inferSchema", "true").load("input/listening_logs.csv")
songs = spark.read.format("csv").option("header", "true").option("inferSchema", "true").load("input/songs_metadata.csv")


# Task 1: User Favorite Genres
joined = logs.join(songs, "song_id")
genre_counts = joined.groupBy("user_id", "genre").agg(count("*").alias("listen_count"))
window = Window.partitionBy("user_id").orderBy(desc("listen_count"))
favorite_genres = genre_counts.withColumn("rank", rank().over(window)).filter(col("rank") == 1).drop("rank")
favorite_genres.show(10)
favorite_genres.toPandas().to_csv(r"C:\Users\Ameer\Hands-on-Spark-API\outputs\favorite_genres.csv", index=False)
print("✅ Task 1 complete: favorite_genres.csv saved")

# Task 2: Average Listen Time
avg_listen = logs.groupBy("user_id").agg(round(avg("duration_sec"), 2).alias("avg_duration_sec")).orderBy("user_id")
avg_listen.show(10)
avg_listen.toPandas().to_csv(r"C:\Users\Ameer\Hands-on-Spark-API\outputs\avg_listen_time.csv", index=False)
print("✅ Task 2 complete: avg_listen_time.csv saved")


# Task 3: Create your own Genre Loyalty Scores and rank them and list out top 10
total_listens = logs.groupBy("user_id").agg(count("*").alias("total_listens"))
top_genre_counts = genre_counts.withColumn("rank", rank().over(window)).filter(col("rank") == 1).drop("rank")
loyalty = top_genre_counts.join(total_listens, "user_id")
loyalty = loyalty.withColumn("loyalty_score", round((col("listen_count") / col("total_listens")) * 100, 2))
top10_loyalty = loyalty.orderBy(desc("loyalty_score")).limit(10)
top10_loyalty.show()
top10_loyalty.toPandas().to_csv(r"C:\Users\Ameer\Hands-on-Spark-API\outputs\genre_loyalty_scores.csv", index=False)
print("✅ Task 3 complete: genre_loyalty_scores.csv saved")

# Task 4: Identify users who listen between 12 AM and 5 AM
logs_with_hour = logs.withColumn("hour", hour(col("timestamp")))
night_owls = logs_with_hour.filter((col("hour") >= 0) & (col("hour") < 5)).select("user_id").distinct().orderBy("user_id")
night_owls.show()
night_owls.toPandas().to_csv(r"C:\Users\Ameer\Hands-on-Spark-API\outputs\night_owl_users.csv", index=False)
print("✅ Task 4 complete: night_owl_users.csv saved")

spark.stop()




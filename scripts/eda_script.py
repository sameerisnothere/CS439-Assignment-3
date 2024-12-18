from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, when, isnan, year, month, to_date, lit, avg

# Initialize Spark Session
spark = SparkSession.builder.appName("AdvancedNetflixEDA").getOrCreate()

# Load the dataset
df = spark.read.csv("/data/netflix_titles.csv", header=True, inferSchema=True)

# 1. Print schema and data sample
print("Schema of the dataset:")
df.printSchema()
print("Sample records:")
df.show(5)

# 2. Summary Statistics
print("Summary statistics of numeric columns:")
df.describe().show()

# 3. Handle Missing Values
# Count missing values in each column
print("Count of missing/null values per column:")
df.select([count(when(col(c).isNull() | isnan(c), c)).alias(c) for c in df.columns]).show()

# Drop rows with significant missing data or handle them based on analysis
df_cleaned = df.dropna(subset=["title", "type", "release_year"])

# 4. Analyze Content Type Distribution
print("Distribution of TV Shows and Movies:")
df_cleaned.groupBy("type").count().show()

# 5. Analyze Release Year Trends
# Convert release_year to integer and analyze
df_cleaned = df_cleaned.withColumn("release_year", col("release_year").cast("int"))
print("Number of TV Shows and Movies by Release Year:")
df_cleaned.groupBy("release_year").count().orderBy("release_year", ascending=False).show(10)

# 6. Derive New Features
# Convert `date_added` to proper date format and extract year and month
df_cleaned = df_cleaned.withColumn("date_added", to_date(col("date_added"), "MMMM d, yyyy"))
df_cleaned = df_cleaned.withColumn("year_added", year(col("date_added")))
df_cleaned = df_cleaned.withColumn("month_added", month(col("date_added")))

print("Content count by year added:")
df_cleaned.groupBy("year_added").count().orderBy("year_added").show()

# 7. Analyze Movie Durations
if "duration" in df_cleaned.columns:
    # Extract duration in minutes for movies
    df_cleaned = df_cleaned.withColumn("duration_minutes", 
                                       when(col("type") == "Movie", col("duration").substr(1, 3).cast("int"))
                                       .otherwise(lit(None)))
    print("Average duration of Movies:")
    df_cleaned.filter(col("type") == "Movie").agg(avg("duration_minutes").alias("avg_duration")).show()

# 8. Most Frequent Directors
if "director" in df_cleaned.columns:
    print("Top 5 most frequent directors:")
    df_cleaned.groupBy("director").count().orderBy(col("count").desc()).show(5)

# 9. Genre Analysis (using `listed_in` column)
if "listed_in" in df_cleaned.columns:
    print("Top 10 genres by count:")
    df_cleaned.select("listed_in").rdd.flatMap(lambda x: x[0].split(", ") if x[0] else []) \
        .map(lambda genre: (genre, 1)).reduceByKey(lambda a, b: a + b) \
        .toDF(["genre", "count"]).orderBy(col("count").desc()).show(10)

# 10. Global Availability Analysis (using `country` column)
if "country" in df_cleaned.columns:
    print("Top 10 countries by content count:")
    df_cleaned.groupBy("country").count().orderBy(col("count").desc()).show(10)

# Stop Spark Session
spark.stop()

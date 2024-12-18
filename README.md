# CS439 Assignment 3
## Muhammad Sameer Shahzad
## 2021451

## Objective
Perform an advanced exploratory data analysis (EDA) on the Netflix TV shows and movies dataset using Apache Spark and Docker.

---

## Prerequisites

1. **Docker**: Install Docker Desktop ([Download Here](https://www.docker.com/products/docker-desktop/)).
2. **Dataset**: Download the Netflix dataset from Kaggle ([Netflix Movies and TV Shows Dataset](https://www.kaggle.com/shivamb/netflix-shows)). Place it in a `data` folder at the root of the repository.
3. **Python Environment**: Ensure that your PySpark script is placed in the `scripts` folder and named appropriately (e.g., `eda_script.py`).

---

## Folder Structure

```
|-- netflix-eda-assignment
    |-- data
        |-- netflix_titles.csv
    |-- scripts
        |-- eda_script.py
    |-- Dockerfile
    |-- docker-compose.yml
    |-- README.md
```

---

## Step 1: Setting Up Docker

### 1.1 Dockerfile
Create a `Dockerfile` to set up a containerized PySpark environment:
```dockerfile
FROM bitnami/spark:latest

# Set working directory
WORKDIR /scripts

# Copy dataset and script
COPY data /data
COPY scripts /scripts

# Set environment variables
ENV HOME=/root
ENV IVY_HOME=/tmp/.ivy2
RUN mkdir -p $IVY_HOME
```

### 1.2 Docker Compose File
Create a `docker-compose.yml` to orchestrate the container:
```yaml
version: '3.9'
services:
  spark:
    build: .
    container_name: spark-eda
    networks:
      - spark-network
    stdin_open: true
    tty: true
networks:
  spark-network:
    driver: bridge
```

---

## Step 2: Writing the EDA Script

Save the following Python script in `scripts/eda_script.py` for advanced EDA:

```python
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, when, isnan, year, month, to_date, lit, avg

# Initialize Spark Session
spark = SparkSession.builder.appName("AdvancedNetflixEDA").getOrCreate()

# Load the dataset
df = spark.read.csv("/data/netflix_titles.csv", header=True, inferSchema=True)

# Print schema and sample records
print("Schema of the dataset:")
df.printSchema()
print("Sample records:")
df.show(5)

# Summary statistics
print("Summary statistics of numeric columns:")
df.describe().show()

# Missing values analysis
print("Count of missing/null values per column:")
df.select([count(when(col(c).isNull() | isnan(c), c)).alias(c) for c in df.columns]).show()

# Drop rows with significant missing data
df_cleaned = df.dropna(subset=["title", "type", "release_year"])

# Distribution of TV Shows and Movies
df_cleaned.groupBy("type").count().show()

# Release year trends
df_cleaned = df_cleaned.withColumn("release_year", col("release_year").cast("int"))
df_cleaned.groupBy("release_year").count().orderBy("release_year", ascending=False).show(10)

# Derive new features
df_cleaned = df_cleaned.withColumn("date_added", to_date(col("date_added"), "MMMM d, yyyy"))
df_cleaned = df_cleaned.withColumn("year_added", year(col("date_added")))
df_cleaned = df_cleaned.withColumn("month_added", month(col("date_added")))

df_cleaned.groupBy("year_added").count().orderBy("year_added").show()

# Analyze movie durations
if "duration" in df_cleaned.columns:
    df_cleaned = df_cleaned.withColumn("duration_minutes", 
                                       when(col("type") == "Movie", col("duration").substr(1, 3).cast("int"))
                                       .otherwise(lit(None)))
    df_cleaned.filter(col("type") == "Movie").agg(avg("duration_minutes").alias("avg_duration")).show()

# Top directors
if "director" in df_cleaned.columns:
    df_cleaned.groupBy("director").count().orderBy(col("count").desc()).show(5)

# Genre analysis
if "listed_in" in df_cleaned.columns:
    df_cleaned.select("listed_in").rdd.flatMap(lambda x: x[0].split(", ") if x[0] else []) \
        .map(lambda genre: (genre, 1)).reduceByKey(lambda a, b: a + b) \
        .toDF(["genre", "count"]).orderBy(col("count").desc()).show(10)

# Global availability analysis
if "country" in df_cleaned.columns:
    df_cleaned.groupBy("country").count().orderBy(col("count").desc()).show(10)

# Stop Spark Session
spark.stop()
```

---

## Step 3: Running the Container

### 3.1 Build the Docker Image
Run the following command to build the Docker image:
```bash
docker-compose build
```

### 3.2 Start the Container
Start the container:
```bash
docker-compose up
```

### 3.3 Run the EDA Script
Once the container is running, execute the script:
```bash
docker exec -it spark-eda spark-submit /scripts/eda_script.py
```

---

## Step 4: Output and Analysis

- The script will display the following key insights:
  - Schema and sample data
  - Summary statistics
  - Distribution of TV Shows vs. Movies
  - Trends in release years
  - Content added by year and month
  - Average duration of movies
  - Top directors and genres
  - Country-wise content distribution



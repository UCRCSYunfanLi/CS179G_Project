from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when
import time
from config import SPARK_CONFIG, PATHS

def initialize_spark():
    """Initialize and return Spark session with configured settings"""
    return SparkSession.builder \
        .appName(SPARK_CONFIG["app_name"]) \
        .config("spark.driver.memory", SPARK_CONFIG["driver_memory"]) \
        .config("spark.executor.memory", SPARK_CONFIG["executor_memory"]) \
        .config("spark.sql.shuffle.partitions", SPARK_CONFIG["shuffle_partitions"]) \
        .getOrCreate()

def load_and_preprocess_data(spark):
    """Load and preprocess the dataset"""
    print("Loading data...")
    start_time = time.time()
    
    df = spark.read.csv(PATHS["input"], 
                      header=True,
                      inferSchema=True,
                      quote='"',
                      escape='"') \
             .select("polarity", "title") \
             .na.drop()
    
    # Convert polarity labels (1→0[negative], 2→1[positive])
    df = df.withColumn("label", when(col("polarity") == 2, 1).otherwise(0))
    
    print(f"Data loaded. Time taken: {time.time()-start_time:.2f} seconds")
    print(f"Total records: {df.count():,}")
    
    return df

def check_class_distribution(df):
    """Print class distribution statistics"""
    class_dist = df.groupBy("label").count().collect()
    print(f"\nClass distribution:\nPositive(1): {class_dist[1][1]:,}\nNegative(0): {class_dist[0][1]:,}")

def split_data(df):
    """Split data into training and test sets"""
    print("\nSplitting dataset...")
    return df.randomSplit([0.8, 0.2], seed=42)
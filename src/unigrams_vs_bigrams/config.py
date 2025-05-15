# Spark configuration settings
SPARK_CONFIG = {
    "app_name": "BigramUnigramComparison_LargeData",
    "driver_memory": "4g",
    "executor_memory": "4g",
    "shuffle_partitions": "200"
}

# Feature engineering parameters
FEATURE_PARAMS = {
    "unigram_features": 5000,
    "bigram_features": 10000,
    "max_iter": 100,
    "reg_param": 0.01
}

# File paths
PATHS = {
    "input": "preprocessed_reviews.csv",
    "unigram_model": "unigram_model",
    "bigram_model": "bigram_model"
}
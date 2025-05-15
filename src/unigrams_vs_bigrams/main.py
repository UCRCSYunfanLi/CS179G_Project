from data_loading import initialize_spark, load_and_preprocess_data, check_class_distribution, split_data
from feature_engineering import build_unigram_pipeline, build_bigram_pipeline
from model_training import train_model, evaluate_model, save_model, compare_models, show_sample_predictions

def main():
    # Initialize Spark
    spark = initialize_spark()
    
    try:
        # Data loading and preprocessing
        df = load_and_preprocess_data(spark)
        check_class_distribution(df)
        train_df, test_df = split_data(df)
        
        # Build feature pipelines
        unigram_pipeline = build_unigram_pipeline()
        bigram_pipeline = build_bigram_pipeline()
        
        # Model training
        uni_model = train_model(unigram_pipeline, train_df, "Unigram")
        bi_model = train_model(bigram_pipeline, train_df, "Bigram")
        
        # Model evaluation
        uni_predictions = evaluate_model(uni_model, test_df, "Unigram")
        bi_predictions = evaluate_model(bi_model, test_df, "Bigram")
        
        # Results comparison
        compare_models(uni_predictions, bi_predictions, test_df)
        
        # Save models
        save_model(uni_model, "unigram")
        save_model(bi_model, "bigram")
        
        # Show samples
        show_sample_predictions(bi_predictions)
        
    finally:
        spark.stop()
        print("All tasks completed!")

if __name__ == "__main__":
    main()
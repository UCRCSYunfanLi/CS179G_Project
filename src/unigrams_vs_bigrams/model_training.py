from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.sql.functions import col
import time
from config import PATHS

def train_model(pipeline, train_data, model_name):
    """Train a model and return the trained model"""
    print(f"\nTraining {model_name} model...")
    start_time = time.time()
    model = pipeline.fit(train_data)
    print(f"{model_name} training completed. Time taken: {time.time()-start_time:.2f} seconds")
    return model

def evaluate_model(model, test_data, model_name):
    """Evaluate model performance and return metrics"""
    print(f"\nEvaluating {model_name} model...")
    start_time = time.time()
    evaluator = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="rawPrediction")
    predictions = model.transform(test_data)
    
    # Calculate metrics
    accuracy = predictions.filter(col("label") == col("prediction")).count() / test_data.count()
    auc = evaluator.evaluate(predictions, {evaluator.metricName: "areaUnderROC"})
    
    # Calculate precision, recall, F1
    true_positives = predictions.filter((col("prediction") == 1) & (col("label") == 1)).count()
    predicted_positives = predictions.filter(col("prediction") == 1).count()
    actual_positives = predictions.filter(col("label") == 1).count()
    
    precision = true_positives / predicted_positives if predicted_positives > 0 else 0
    recall = true_positives / actual_positives if actual_positives > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"{model_name} evaluation completed. Time taken: {time.time()-start_time:.2f} seconds")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"AUC: {auc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    return predictions

def save_model(model, model_name):
    """Save trained model to disk"""
    print(f"\nSaving {model_name} model...")
    model.write().overwrite().save(PATHS[f"{model_name.lower()}_model"])

def compare_models(uni_predictions, bi_predictions, test_df):
    """Compare performance between unigram and bigram models"""
    print("\nModel comparison:")
    improvement = (bi_predictions.filter(col("label") == col("prediction")).count() - 
                 uni_predictions.filter(col("label") == col("prediction")).count())
    improvement_percent = improvement / test_df.count() * 100
    print(f"Bigram correctly predicted {improvement} more samples than Unigram ({improvement_percent:.2f}%)")

def show_sample_predictions(predictions):
    """Display sample predictions"""
    print("\nSample predictions (Bigram model):")
    predictions.select("title", "polarity", "label", "prediction", "probability") \
             .filter(col("title").isNotNull()) \
             .limit(10).show(truncate=False)
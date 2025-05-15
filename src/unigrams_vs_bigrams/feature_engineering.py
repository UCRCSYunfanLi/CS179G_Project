from pyspark.ml.feature import Tokenizer, NGram, HashingTF, IDF, StopWordsRemover
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
from config import FEATURE_PARAMS

def build_unigram_pipeline():
    """Build pipeline for unigram features"""
    tokenizer = Tokenizer(inputCol="title", outputCol="words")
    remover = StopWordsRemover(inputCol="words", outputCol="filtered_words")
    hashingTF = HashingTF(inputCol="filtered_words", 
                         outputCol="unigrams_tf", 
                         numFeatures=FEATURE_PARAMS["unigram_features"])
    idf = IDF(inputCol="unigrams_tf", outputCol="unigrams_features")
    lr = LogisticRegression(featuresCol="unigrams_features", 
                          labelCol="label",
                          maxIter=FEATURE_PARAMS["max_iter"],
                          regParam=FEATURE_PARAMS["reg_param"])
    
    return Pipeline(stages=[tokenizer, remover, hashingTF, idf, lr])

def build_bigram_pipeline():
    """Build pipeline for bigram features"""
    tokenizer = Tokenizer(inputCol="title", outputCol="words")
    remover = StopWordsRemover(inputCol="words", outputCol="filtered_words")
    ngram = NGram(n=2, inputCol="filtered_words", outputCol="bigrams")
    hashingTF = HashingTF(inputCol="bigrams", 
                        outputCol="bigrams_tf", 
                        numFeatures=FEATURE_PARAMS["bigram_features"])
    idf = IDF(inputCol="bigrams_tf", outputCol="bigrams_features")
    lr = LogisticRegression(featuresCol="bigrams_features", 
                          labelCol="label",
                          maxIter=FEATURE_PARAMS["max_iter"],
                          regParam=FEATURE_PARAMS["reg_param"])
    
    return Pipeline(stages=[tokenizer, remover, ngram, hashingTF, idf, lr])
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StandardScaler, StringIndexer
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import os
import numpy as np
import cv2
import pandas as pd
import time

# 1. Start Spark session
spark = SparkSession.builder \
    .appName("EuroSAT Land Use Classification") \
    .getOrCreate()

# 2. Set EuroSAT image path
image_dir = os.path.join("Bigdataassessment2", "data", "train")

if not os.path.exists(image_dir):
    raise FileNotFoundError(f"Image directory not found: {image_dir}")

if not os.listdir(image_dir):
    raise FileNotFoundError(f"No subdirectories found in the image directory: {image_dir}")

# 3. Flatten images and extract labels
images, labels = [], []
for label in os.listdir(image_dir):
    label_dir = os.path.join(image_dir, label)
    if not os.listdir(label_dir):
        raise FileNotFoundError(f"No images found in the subdirectory: {label_dir}")
    for file in os.listdir(label_dir):
        path = os.path.join(label_dir, file)
        img = cv2.imread(path)
        if img is not None:
            flat = cv2.resize(img, (64, 64)).flatten()
            images.append(flat)
            labels.append(label)

# 4. Convert to DataFrame and save to CSV
df = pd.DataFrame(images)
df["label"] = labels
df.to_csv("eurosat_flat.csv", index=False)

# 5. Load CSV into Spark
data = spark.read.csv("eurosat_flat.csv", header=True, inferSchema=True)

# 6. Assemble features
assembler = VectorAssembler(inputCols=[col for col in data.columns if col != "label"], outputCol="features")
indexer = StringIndexer(inputCol="label", outputCol="labelIndex")
scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures")

# 7. Random Forest Classifier
rf = RandomForestClassifier(labelCol="labelIndex", featuresCol="scaledFeatures", numTrees=100, maxDepth=15, maxBins=128, featureSubsetStrategy="sqrt")

# 8. Build pipeline
pipeline = Pipeline(stages=[assembler, indexer, scaler, rf])

# 9. Train/test split
train, test = data.randomSplit([0.8, 0.2], seed=42)

# 10. Train model
start_time = time.time()
model = pipeline.fit(train)
end_time = time.time()
training_duration = end_time - start_time
print(f"Training Duration: {training_duration:.2f} seconds")

# Save runtime to a file
with open("spark_training_runtime.txt", "w") as f:
    f.write(f"Training Duration: {training_duration:.2f} seconds\n")

# 11. Evaluate
predictions = model.transform(test)
evaluator = MulticlassClassificationEvaluator(
    labelCol="labelIndex", predictionCol="prediction", metricName="accuracy")
acc = evaluator.evaluate(predictions)

print(f"Spark MLlib Accuracy: {acc:.4f}")

# Save predictions to a CSV file
predictions.select("labelIndex", "prediction").write.csv("spark_predictions.csv", header=True)

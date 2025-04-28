import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.decomposition import PCA
from preprocess import create_generators
from utils import plot_confusion_matrix, print_classification_report, plot_evaluation_metrics
import joblib

def flatten_generator(generator):
    data, labels = [], []
    for batch_data, batch_labels in generator:
        data.append(batch_data.reshape(batch_data.shape[0], -1))
        labels.append(batch_labels)
        if len(data) * generator.batch_size >= generator.samples:
            break
    data = np.vstack(data)[:generator.samples]
    labels = np.argmax(np.vstack(labels), axis=1)[:generator.samples]
    return data, labels

data_dir = os.path.join("data", "train")
train_gen, val_gen = create_generators(data_dir, img_size=(64, 64), batch_size=32)  # Ensure consistent image size

X_train, y_train = flatten_generator(train_gen)
X_val, y_val = flatten_generator(val_gen)

if len(X_train) == 0 or len(y_train) == 0:
    raise ValueError("Training data is empty. Ensure the dataset is correctly set up.")

# Optional: Apply PCA
pca = PCA(n_components=100)  # Reduce dimensionality to 100 components
X_train = pca.fit_transform(X_train)
X_val = pca.transform(X_val)

clf = RandomForestClassifier(n_estimators=200, max_depth=20, min_samples_leaf=2)  # Adjusted hyperparameters
clf.fit(X_train, y_train)

y_pred = clf.predict(X_val)
accuracy = accuracy_score(y_val, y_pred)
print(f"Validation Accuracy: {accuracy}")

# Compute evaluation metrics
precision = precision_score(y_val, y_pred, average='weighted', zero_division=0)
recall = recall_score(y_val, y_pred, average='weighted', zero_division=0)
f1 = f1_score(y_val, y_pred, average='weighted', zero_division=0)

# Plot evaluation metrics
metrics = {"Accuracy": accuracy, "Precision": precision, "Recall": recall, "F1 Score": f1}
plot_evaluation_metrics(metrics, model_name="Random Forest")

# Save evaluation metrics plot
plot_evaluation_metrics(metrics, model_name="Random Forest", save_path="Bigdataassessment2/random_forest_evaluation_metrics.png")
print("Random Forest Evaluation Metrics saved as random_forest_evaluation_metrics.png")

labels = sorted(set(y_val))  # Ensure only labels present in y_val are used
plot_confusion_matrix(y_val, y_pred, labels)

# Save confusion matrix
plot_confusion_matrix(y_val, y_pred, labels, title="Random Forest Confusion Matrix", save_path="Bigdataassessment2/random_forest_confusion_matrix.png")
print("Random Forest Confusion Matrix saved as random_forest_confusion_matrix.png")

print_classification_report(y_val, y_pred, labels)

# Save the trained Random Forest model
joblib.dump(clf, "random_forest_model.pkl")

# Save predictions and true labels
np.savez("random_forest_predictions.npz", y_true=y_val, y_pred=y_pred)

# Log feature importance
feature_importances = clf.feature_importances_
np.save("random_forest_feature_importances.npy", feature_importances)

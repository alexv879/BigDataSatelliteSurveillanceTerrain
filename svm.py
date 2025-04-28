import numpy as np
import os
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.decomposition import PCA
from preprocess import create_generators
from utils import plot_confusion_matrix, print_classification_report, plot_evaluation_metrics
from sklearn.utils.class_weight import compute_class_weight
import json

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

# Optional: Apply PCA
pca = PCA(n_components=100)
X_train = pca.fit_transform(X_train)
X_val = pca.transform(X_val)

# Save PCA components
np.save("pca_components.npy", pca.components_)
# Save PCA components and explained variance ratio
np.save("pca_explained_variance_ratio.npy", pca.explained_variance_ratio_)

# Compute class weights
if len(y_train) == 0:
    raise ValueError("Training labels are empty. Ensure the dataset is correctly set up.")
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights_dict = {i: weight for i, weight in enumerate(class_weights)}

# Check if training data is empty
if len(X_train) == 0 or len(y_train) == 0:
    raise ValueError("Training data is empty. Ensure the dataset is correctly set up.")

# Train SVM with class weights and different C values
clf = SVC(kernel='rbf', C=20, gamma='scale', class_weight=class_weights_dict)  # Adjusted C value
clf.fit(X_train, y_train)

y_pred = clf.predict(X_val)
accuracy = accuracy_score(y_val, y_pred)
print(f"Validation Accuracy: {accuracy}")

# Compute evaluation metrics
precision = precision_score(y_val, y_pred, average='weighted', zero_division=0)
recall = recall_score(y_val, y_pred, average='weighted', zero_division=0)
f1 = f1_score(y_val, y_pred, average='weighted', zero_division=0)

# Save predictions and true labels
np.savez("svm_predictions.npz", y_true=y_val, y_pred=y_pred)

# Plot evaluation metrics
metrics = {"Accuracy": accuracy, "Precision": precision, "Recall": recall, "F1 Score": f1}
plot_evaluation_metrics(metrics, model_name="SVM")

# Save evaluation metrics plot
plot_evaluation_metrics(metrics, model_name="SVM", save_path="Bigdataassessment2/svm_evaluation_metrics.png")
print("SVM Evaluation Metrics saved as svm_evaluation_metrics.png")

# Log evaluation metrics
with open("svm_metrics.json", "w") as f:
    json.dump(metrics, f)

# Fixed label mapping
unique_labels = sorted(set(y_val))  # Ensure only labels present in y_val are used
plot_confusion_matrix(y_val, y_pred, unique_labels)

# Save confusion matrix
plot_confusion_matrix(y_val, y_pred, unique_labels, title="SVM Confusion Matrix", save_path="Bigdataassessment2/svm_confusion_matrix.png")
print("SVM Confusion Matrix saved as svm_confusion_matrix.png")

print_classification_report(y_val, y_pred, unique_labels)

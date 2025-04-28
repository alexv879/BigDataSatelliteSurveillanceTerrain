import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score
import os

def plot_confusion_matrix(y_true, y_pred, labels, title="Confusion Matrix", normalize=False, save_path=None):
    """
    Plot a confusion matrix using seaborn heatmap and save it as an image file.
    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        labels: List of label names.
        title: Title for the plot.
        normalize: Whether to normalize the confusion matrix.
        save_path: Path to save the plot (optional).
    """
    # Map numeric labels to class names
    unique_labels = sorted(set(y_true))  # Use only labels present in y_true
    cm = confusion_matrix(y_true, y_pred, labels=unique_labels, normalize='true' if normalize else None)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='.2f' if normalize else 'd', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    if save_path:
        save_path = os.path.join("Bigdataassessment2", os.path.basename(save_path))  # Ensure plots are saved in the folder
        os.makedirs(os.path.dirname(save_path), exist_ok=True)  # Ensure the directory exists
        plt.savefig(save_path)
    plt.close()  # Close the plot to avoid displaying it

def print_classification_report(y_true, y_pred, labels):
    """
    Print the classification report.
    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        labels: List of label names.
    """
    # Map numeric labels to class names
    unique_labels = sorted(set(y_true))
    target_names = [labels[label] for label in unique_labels]
    report = classification_report(y_true, y_pred, target_names=target_names)
    print("Classification Report:\n", report)

def evaluate_model(model, generator, class_indices):
    """
    Evaluate the model on a validation generator and print metrics.
    Args:
        model: Trained model to evaluate.
        generator: Data generator for validation data.
        class_indices: Dictionary mapping class names to numeric labels.
    """
    y_true = []
    y_pred = []

    for batch_data, batch_labels in generator:
        predictions = model.predict(batch_data)
        y_true.extend(batch_labels.argmax(axis=1))
        y_pred.extend(predictions.argmax(axis=1))
        # Ensure we only process the number of samples in the generator
        if len(y_true) >= generator.samples:
            y_true = y_true[:generator.samples]
            y_pred = y_pred[:generator.samples]
            break

    # Map numeric labels back to class names using class_indices
    label_names = {v: k for k, v in class_indices.items()}  # Reverse the class_indices mapping
    unique_labels = sorted(set(y_true))  # Ensure only labels present in y_true are used
    target_names = [label_names[label] for label in unique_labels]  # Map numeric labels to class names

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")

    # Plot confusion matrix with proper labels
    plot_confusion_matrix(y_true, y_pred, unique_labels)
    print_classification_report(y_true, y_pred, target_names)

def plot_training_curves(history, title="Training Curves", save_path=None):
    """
    Plot training and validation loss/accuracy curves and save them as an image file.
    Args:
        history: Keras History object from model.fit().
        title: Title for the plot.
        save_path: Path to save the plot (optional).
    """
    plt.figure(figsize=(12, 5))
    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f"{title} - Loss")
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title(f"{title} - Accuracy")
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    if save_path:
        save_path = os.path.join("Bigdataassessment2", os.path.basename(save_path))  # Ensure plots are saved in the folder
        os.makedirs(os.path.dirname(save_path), exist_ok=True)  # Ensure the directory exists
        plt.savefig(save_path)
    plt.close()  # Close the plot to avoid displaying it

def plot_evaluation_metrics(metrics, model_name="Model", save_path=None):
    """
    Plot a bar chart comparing evaluation metrics and save it as an image file.
    Args:
        metrics: Dictionary with metric names as keys and their values.
        model_name: Name of the model being evaluated.
        save_path: Path to save the plot (optional).
    """
    plt.figure(figsize=(8, 5))
    plt.bar(metrics.keys(), metrics.values(), color='skyblue')
    plt.title(f"{model_name} - Evaluation Metrics")
    plt.ylabel('Score')
    plt.ylim(0, 1)  # Metrics are between 0 and 1
    if save_path:
        save_path = os.path.join("Bigdataassessment2", os.path.basename(save_path))  # Ensure plots are saved in the folder
        os.makedirs(os.path.dirname(save_path), exist_ok=True)  # Ensure the directory exists
        plt.savefig(save_path)
    plt.close()  # Close the plot to avoid displaying it

def compare_model_metrics(metrics_dict, save_path=None):
    """
    Compare evaluation metrics across multiple models and save the plots as image files.
    Args:
        metrics_dict: Dictionary where keys are model names and values are dictionaries of metrics.
        save_path: Path to save the plots (optional).
    """
    metrics = list(next(iter(metrics_dict.values())).keys())  # Get metric names
    for model, model_metrics in metrics_dict.items():
        if set(metrics) != set(model_metrics.keys()):
            raise ValueError(f"Metrics for model '{model}' do not match the expected metrics: {metrics}")

    model_names = list(metrics_dict.keys())
    for metric in metrics:
        plt.figure(figsize=(8, 5))
        values = [metrics_dict[model][metric] for model in model_names]
        plt.bar(model_names, values, color='skyblue')
        plt.title(f"Comparison of {metric.capitalize()} Across Models")
        plt.ylabel(metric.capitalize())
        plt.ylim(0, 1)
        if save_path:
            metric_save_path = os.path.join("Bigdataassessment2", f"{os.path.basename(save_path)}_{metric.lower()}.png")
            os.makedirs(os.path.dirname(metric_save_path), exist_ok=True)
            plt.savefig(metric_save_path)
        plt.close()

import os
import torch
from torch.utils.data import DataLoader
from dataset import MyCOCODataset
from data_split import split_annotation_ids, class_names as coco_class_names
from settings import DATA_ROOT, ANNOTATIONS, OUTPUT_DIR, BATCH_SIZE
from pycocotools.coco import COCO
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns

def build_confusion_matrix(predicted, ground_truth, num_classes):
    """
    Build a confusion matrix where:
      Rows = actual class
      Columns = predicted class
    """
    matrix = torch.zeros(num_classes, num_classes, dtype=torch.int64)
    for p, g in zip(predicted, ground_truth):
        matrix[g, p] += 1
    return matrix

def plot_confusion_matrix_sklearn(cm, class_names, title='Confusion Matrix'):
    """
    Plots a confusion matrix using seaborn heatmap.
    """
    cm_normalized = cm.astype('float') / np.maximum(cm.sum(axis=1, keepdims=True), 1)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Greens',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title(title)
    plt.tight_layout()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_DIR / 'decision_tree_confusion_matrix.png')
    plt.show()

def main():
    data_root = str(DATA_ROOT)
    annofile = str(ANNOTATIONS)
    coco = COCO(annofile)
    partitions = split_annotation_ids(coco)
    train_ids, val_ids, test_ids = (partitions[name] for name in ('train', 'val', 'test'))
    class_names = coco_class_names(coco)
    print(f"Training instances: {len(train_ids)}; validation: {len(val_ids)}; test: {len(test_ids)}")

    # Create datasets
    train_set = MyCOCODataset(
        data_root=data_root,
        annofile=annofile,
        ids=train_ids
    )
    val_set = MyCOCODataset(
        data_root=data_root,
        annofile=annofile,
        ids=val_ids
    )
    test_set = MyCOCODataset(
        data_root=data_root,
        annofile=annofile,
        ids=test_ids
    )

    # Create DataLoaders
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=False)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

    # Prepare training data for Decision Tree
    X_train = []
    y_train = []
    for images, labels in train_loader:
        images = images.float() / 255.0
        images = images.permute(0, 3, 1, 2)
        X_train.append(flatten_images(images))
        y_train.append(labels.numpy())
    X_train = np.concatenate(X_train, axis=0)
    y_train = np.concatenate(y_train, axis=0)

    # Prepare validation data for Decision Tree
    X_val = []
    y_val = []
    for images, labels in val_loader:
        images = images.float() / 255.0
        images = images.permute(0, 3, 1, 2)
        X_val.append(flatten_images(images))
        y_val.append(labels.numpy())
    X_val = np.concatenate(X_val, axis=0)
    y_val = np.concatenate(y_val, axis=0)

    # Initialize and train Decision Tree
    clf = DecisionTreeClassifier(random_state=42)
    clf.fit(X_train, y_train)

    # Prepare test data for Decision Tree
    X_test = []
    y_test = []
    for images, labels in test_loader:
        images = images.float() / 255.0
        images = images.permute(0, 3, 1, 2)
        X_test.append(flatten_images(images))
        y_test.append(labels.numpy())
    X_test = np.concatenate(X_test, axis=0)
    y_test = np.concatenate(y_test, axis=0)

    # Predict on test set
    y_pred = clf.predict(X_test)

    # Calculate accuracy
    accuracy = np.mean(y_pred == y_test)
    print(f"Decision Tree Test Accuracy: {accuracy:.4f}")

    # Confusion Matrix using custom function
    cm = build_confusion_matrix(torch.tensor(y_pred), torch.tensor(y_test), num_classes=len(class_names))
    print("Decision Tree Confusion Matrix:")
    print(cm)

    # Plot confusion matrix
    plot_confusion_matrix_sklearn(cm.numpy(), class_names, title='Decision Tree Normalized Confusion Matrix')

    # Classification Report using sklearn
    print("Decision Tree Classification Report:")
    print(classification_report(y_test, y_pred, labels=list(range(len(class_names))), target_names=class_names, zero_division=0))

def flatten_images(images):
    """
    Flatten images for Decision Tree input.
    Args:
        images (torch.Tensor): Batch of images with shape [N, C, H, W]
    Returns:
        np.ndarray: Flattened images with shape [N, C*H*W]
    """
    # Use reshape instead of view to avoid RuntimeError for non-contiguous tensors
    return images.reshape(images.size(0), -1).numpy()

if __name__ == "__main__":
    main()
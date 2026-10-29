import os
import torch
from torch.utils.data import DataLoader
from model import AlexNet
from dataset import MyCOCODataset
from data_split import split_annotation_ids, class_names as coco_class_names
from settings import DATA_ROOT, ANNOTATIONS, OUTPUT_DIR, DEVICE, BATCH_SIZE
from pycocotools.coco import COCO
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report
import numpy as np

checkpoint_path = OUTPUT_DIR / "best_model.pth"  # Path to the best model weights
device = DEVICE  

def confusion_matrix(predicted, ground_truth, num_classes):
    """
    Build a confusion matrix where:
      Rows = actual class
      Columns = predicted class
    """
    matrix = torch.zeros(num_classes, num_classes, dtype=torch.int64)
    for p, g in zip(predicted, ground_truth):
        matrix[g, p] += 1
    return matrix

def plot_confusion_matrix(cm, class_names, title='Confusion Matrix'):
    """
    Plots a confusion matrix using seaborn heatmap.
    """
    cm = cm.numpy()
    cm_normalized = cm.astype('float') / np.maximum(cm.sum(axis=1, keepdims=True), 1)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title(title)
    plt.tight_layout()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_DIR / 'confusion_matrix.png')
    plt.show()

def main():
    coco = COCO(str(ANNOTATIONS))
    num_classes = len(coco.cats)
    class_names = coco_class_names(coco)
    # Reconstruct the model
    model = AlexNet(num_classes=num_classes).to(device)

    # Load checkpoint
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"No model file found at {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt)
    model.eval()

    data_root = str(DATA_ROOT)
    annofile = str(ANNOTATIONS)
    partitions = split_annotation_ids(coco)
    test_ids = partitions['test']
    print(f"Test instances: {len(test_ids)}")

    # Create a test dataset
    test_set = MyCOCODataset(
        data_root=data_root,
        annofile=annofile,
        ids=test_ids
    )
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.float() / 255.0
            images = images.permute(0, 3, 1, 2)
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

    # Concatenate all predictions
    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    # Calculate accuracy
    correct = (all_preds == all_labels).sum().item()
    accuracy = correct / len(all_labels)
    print(f"Test Accuracy: {accuracy:.4f}")

    # Confusion matrix
    cm = confusion_matrix(all_preds, all_labels, num_classes)
    print("Confusion Matrix:")
    print(cm)

    # Plot confusion matrix
    plot_confusion_matrix(cm, class_names, title='Normalized Confusion Matrix')

    # Classification Report
    print("Classification Report:")
    print(classification_report(all_labels.numpy(), all_preds.numpy(), labels=list(range(num_classes)), target_names=class_names, zero_division=0))

if __name__ == "__main__":
    main()
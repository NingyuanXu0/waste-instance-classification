import os
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader

from model import AlexNet
from dataset import MyCOCODataset
from data_split import split_annotation_ids
from settings import DATA_ROOT, ANNOTATIONS, OUTPUT_DIR, DEVICE, BATCH_SIZE, EPOCHS

# Hyperparameters
lr = 1e-3
device = DEVICE  
batch_size = BATCH_SIZE
num_epochs = EPOCHS  
output_dir = str(OUTPUT_DIR)

def main():
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    from pycocotools.coco import COCO
    data_root = str(DATA_ROOT)
    annofile = str(ANNOTATIONS)
    coco = COCO(annofile)
    partitions = split_annotation_ids(coco)
    train_ids = partitions['train']
    val_ids = partitions['val']
    num_classes = len(coco.cats)
    model = AlexNet(num_classes=num_classes).to(device)
    print(f"Training instances: {len(train_ids)}; validation instances: {len(val_ids)}")

    # 3. Create dataset instances
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

    # 4. Create DataLoaders
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    # 5. Setup optimizer and loss
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_accuracy = -1.0  # Track best accuracy across epochs

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for i, (images, labels) in enumerate(train_loader):
            # Normalize images to [0,1]
            images = images.float() / 255.0
            # [N, H, W, C] -> [N, C, H, W]
            images = images.permute(0, 3, 1, 2)

            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

            if (i + 1) % 10 == 0:
                avg_loss = running_loss / ((i+1) * images.size(0))
                print(f"Epoch [{epoch+1}/{num_epochs}], "
                      f"Step [{i+1}/{len(train_loader)}], "
                      f"Loss: {loss.item():.4f}")

        # Validation
        model.eval()
        val_preds = []
        val_labels = []
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.float() / 255.0
                images = images.permute(0, 3, 1, 2)
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                val_preds.append(preds.cpu())
                val_labels.append(labels.cpu())

        # Calculate the accuracy on the validation set
        val_preds = torch.cat(val_preds, dim=0)
        val_labels = torch.cat(val_labels, dim=0)
        correct = (val_preds == val_labels).sum().item()
        accuracy = correct / len(val_labels)

        print(f"Epoch {epoch+1} Validation Accuracy: {accuracy:.4f}")

        # Save best model if accuracy improves
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            ckpt_path = os.path.join(output_dir, "best_model.pth")
            torch.save(model.state_dict(), ckpt_path)
            print(f"Best model saved at epoch {epoch+1} with accuracy {accuracy:.4f}.")

    print("Training complete.")


if __name__ == "__main__":
    main()

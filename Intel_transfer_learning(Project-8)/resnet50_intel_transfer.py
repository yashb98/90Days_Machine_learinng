"""
Day 20 — Fine-Tuning ResNet50 on Intel Image Classification Dataset
Author: Yash Bishnoi
Goal: Fine-tune deeper layers of a pre-trained ResNet50 to achieve >94% accuracy
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import numpy as np
import time
import copy
from tqdm import tqdm

# ---------------------------
# Device setup
# ---------------------------
device = torch.device("mps" if torch.backends.mps.is_available(
) else "cuda" if torch.cuda.is_available() else "cpu")
print(f"💻 Using device: {device}")

# ---------------------------
# Data Preparation
# ---------------------------
data_dir = "Intel_transfer_learning"
train_dir = f"{data_dir}/seg_train"
val_dir = f"{data_dir}/seg_test"

# Data augmentation and normalization
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
val_dataset = datasets.ImageFolder(val_dir, transform=val_transforms)

train_loader = DataLoader(train_dataset, batch_size=32,
                          shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=32,
                        shuffle=False, num_workers=0)

class_names = train_dataset.classes
print(f" Classes: {class_names}")
print(
    f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")

# ---------------------------
#  Model Setup: Fine-Tuning ResNet50
# ---------------------------
model = models.resnet50(pretrained=True)

# Freeze all layers first
for param in model.parameters():
    param.requires_grad = False

# Unfreeze deeper layers (layer3, layer4)
for name, param in model.named_parameters():
    if "layer3" in name or "layer4" in name or "fc" in name:
        param.requires_grad = True

# Replace classifier
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, len(class_names))

model = model.to(device)

# ---------------------------
#  Loss, Optimizer, Scheduler
# ---------------------------
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

# ---------------------------
#  Training Function with Early Stopping
# ---------------------------


def train_model(model, criterion, optimizer, scheduler, num_epochs=10, patience=3):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    patience_counter = 0

    train_loss_history, val_loss_history = [], []
    train_acc_history, val_acc_history = [], []

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 40)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
                dataloader = train_loader
            else:
                model.eval()
                dataloader = val_loader

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data
            for inputs, labels in tqdm(dataloader, desc=f"{phase.capitalize()} Epoch {epoch+1}"):
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_acc = running_corrects.float() / len(dataloader.dataset)

            if phase == 'train':
                scheduler.step()
                train_loss_history.append(epoch_loss)
                train_acc_history.append(epoch_acc.item())
            else:
                val_loss_history.append(epoch_loss)
                val_acc_history.append(epoch_acc.item())

            print(
                f"{phase.capitalize()} | Loss: {epoch_loss:.4f} | Accuracy: {epoch_acc:.4f}")

            # deep copy the model if it improves
            if phase == 'val':
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    patience_counter = 0
                else:
                    patience_counter += 1

        # Early stopping
        if patience_counter >= patience:
            print(" Early stopping triggered.")
            break

    time_elapsed = time.time() - since
    print(
        f"\n Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
    print(f" Best Validation Accuracy: {best_acc:.4f}")

    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), "resnet50_intel_finetuned.pth")
    print(" Model saved as resnet50_intel_finetuned.pth")

    return model, (train_loss_history, val_loss_history, train_acc_history, val_acc_history)


# ---------------------------
# Train the Model
# ---------------------------
model, history = train_model(
    model, criterion, optimizer, scheduler, num_epochs=10)

# ---------------------------
# Visualization
# ---------------------------
train_loss, val_loss, train_acc, val_acc = history

epochs = range(1, len(train_loss) + 1)

plt.figure(figsize=(12, 5))

# Loss
plt.subplot(1, 2, 1)
plt.plot(epochs, train_loss, 'o-', label='Train Loss')
plt.plot(epochs, val_loss, 'o-', label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training vs Validation Loss')
plt.legend()

# Accuracy
plt.subplot(1, 2, 2)
plt.plot(epochs, train_acc, 'o-', label='Train Accuracy')
plt.plot(epochs, val_acc, 'o-', label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training vs Validation Accuracy')
plt.legend()

plt.tight_layout()
plt.show()

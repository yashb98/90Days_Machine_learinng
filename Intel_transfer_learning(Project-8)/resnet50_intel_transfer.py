# ============================================================
# Day 19 - Transfer Learning with ResNet50 on Intel Dataset
# Author: Yash Bishnoi
# macOS Compatible (Fix for multiprocessing issue)
# ============================================================

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import copy
import time
from PIL import Image

# ------------------------------------------------------------
# 1  Dataset Path Setup
# ------------------------------------------------------------
data_dir = "Intel_transfer_learning"
train_dir = os.path.join(data_dir, "seg_train")
val_dir = os.path.join(data_dir, "seg_test")

# ------------------------------------------------------------
# 2  Data Transformations
# ------------------------------------------------------------
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
}

# ------------------------------------------------------------
# 3 Training Function
# ------------------------------------------------------------


def train_model(model, dataloaders, dataset_sizes, criterion, optimizer, device, num_epochs=10):
    start_time = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 40)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            loop = tqdm(dataloaders[phase], desc=f"{phase} phase", leave=False)

            for inputs, labels in loop:
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

                loop.set_postfix(loss=loss.item())

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.float() / dataset_sizes[phase]

            print(
                f"{phase.capitalize()} | Loss: {epoch_loss:.4f} | Accuracy: {epoch_acc:.4f}")

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

    total_time = time.time() - start_time
    print(
        f"\n Training complete in {total_time//60:.0f}m {total_time%60:.0f}s")
    print(f" Best Validation Accuracy: {best_acc:.4f}")

    model.load_state_dict(best_model_wts)
    return model


# ------------------------------------------------------------
# 4 Main Entry Point (Fix for macOS multiprocessing)
# ------------------------------------------------------------
if __name__ == "__main__":
    print(" Starting Transfer Learning on Intel Dataset...")

    # Load dataset
    image_datasets = {
        'train': datasets.ImageFolder(train_dir, transform=data_transforms['train']),
        'val': datasets.ImageFolder(val_dir, transform=data_transforms['val'])
    }

    dataloaders = {
        # 👈 num_workers=0 fixes macOS issue
        x: DataLoader(image_datasets[x], batch_size=32,
                      shuffle=True, num_workers=0)
        for x in ['train', 'val']
    }

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes

    # Device configuration
    device = torch.device("cuda:0" if torch.cuda.is_available(
    ) else "mps" if torch.backends.mps.is_available() else "cpu")

    print(f" Classes: {class_names}")
    print(
        f" Training samples: {dataset_sizes['train']}, Validation samples: {dataset_sizes['val']}")
    print(f" Using device: {device}")

    # Load pretrained model
    from torchvision.models import ResNet50_Weights
    model = models.resnet50(weights=ResNet50_Weights.DEFAULT)

    # Freeze layers
    for param in model.parameters():
        param.requires_grad = False

    # Replace final layer
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_features, 256),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(256, len(class_names))
    )

    model = model.to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

    # Train model
    trained_model = train_model(
        model, dataloaders, dataset_sizes, criterion, optimizer, device, num_epochs=10)

    # Save model
    torch.save(trained_model.state_dict(), "resnet50_intel_best.pth")
    print("\n Model saved as resnet50_intel_best.pth")

    # Optional test
    def predict_image(model, image_path):
        model.eval()
        img = Image.open(image_path).convert("RGB")
        transform = data_transforms['val']
        img_t = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(img_t)
            _, preds = torch.max(outputs, 1)
            pred_class = class_names[preds[0]]

        print(
            f" Image: {os.path.basename(image_path)} → 🔍 Predicted class: {pred_class}")

    # Example:
    # predict_image(trained_model, "Intel_transfer_learning/seg_test/mountain/12345.jpg")

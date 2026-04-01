import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
from collections import Counter

# ---------------- CONFIG ----------------

DATASET_PATH = "/colored_images" # Change this to your dataset path, e.g., "data/colored_images"
BATCH_SIZE = 32
EPOCHS = 30
LR = 3e-4
MODEL_SAVE_PATH = ".pth" # Change this to your desired path, e.g., "best_model.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ---------------- TRANSFORMS ----------------

train_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])

val_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])

# ---------------- DATASET ----------------

dataset = datasets.ImageFolder(DATASET_PATH, transform=train_transform)

print("Dataset size:", len(dataset))
print("Class mapping:", dataset.class_to_idx)

# ---------------- TRAIN / VAL SPLIT ----------------

dataset_size = len(dataset)
train_size = int(0.8 * dataset_size)
val_size = dataset_size - train_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

val_dataset.dataset.transform = val_transform

# ---------------- HANDLE CLASS IMBALANCE ----------------

targets = [dataset.samples[i][1] for i in train_dataset.indices]

class_counts = Counter(targets)
print("Class counts:", class_counts)

class_weights = {cls: 1.0/count for cls, count in class_counts.items()}
sample_weights = [class_weights[label] for label in targets]

sampler = WeightedRandomSampler(
    sample_weights,
    num_samples=len(sample_weights),
    replacement=True
)

# ---------------- DATALOADERS ----------------

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    sampler=sampler,
    num_workers=0,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=True
)

# ---------------- MODEL ----------------

model = models.efficientnet_b2(weights=models.EfficientNet_B2_Weights.DEFAULT)

num_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(num_features, 5)

model = model.to(device)

# ---------------- LOSS FUNCTION ----------------

weights_tensor = torch.tensor(
    [1/class_counts[i] for i in range(len(class_counts))],
    dtype=torch.float
).to(device)

criterion = nn.CrossEntropyLoss(weight=weights_tensor)

# ---------------- OPTIMIZER ----------------

optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

# Learning rate scheduler
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='max',
    patience=3,
    factor=0.3
)

# ---------------- TRAINING ----------------

best_acc = 0

for epoch in range(EPOCHS):

    model.train()
    running_loss = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    epoch_loss = running_loss / len(train_loader)

    # ---------------- VALIDATION ----------------

    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total

    scheduler.step(accuracy)

    print(f"Epoch [{epoch+1}/{EPOCHS}]  Loss: {epoch_loss:.4f}  Val Accuracy: {accuracy:.2f}%")

    # ---------------- SAVE BEST MODEL ----------------

    if accuracy > best_acc:
        best_acc = accuracy
        torch.save(model.state_dict(), MODEL_SAVE_PATH)

print("\nTraining complete")
print("Best accuracy:", best_acc)
print("Model saved as:", MODEL_SAVE_PATH)
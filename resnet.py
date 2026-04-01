import os
import copy
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from torchvision import models

import albumentations as A
from albumentations.pytorch import ToTensorV2

# ---------------- CONFIG ----------------

DATA_DIR = "final_dataset" # Change this to your dataset path
MODEL_PATH = "resnet.pth" # Path to save the best model 

BATCH_SIZE = 16
EPOCHS = 40
LR = 5e-5
PATIENCE = 5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------- CLASSES ----------------

classes = [
    "No_DR",
    "Mild",
    "Moderate",
    "Severe",
    "Proliferative_DR"
]

# ---------------- PREPROCESSING ----------------

def circular_crop(img):
    h, w, _ = img.shape
    center = (int(w/2), int(h/2))
    radius = min(center[0], center[1])
    mask = np.zeros((h, w), np.uint8)
    cv2.circle(mask, center, radius, 255, -1)
    return cv2.bitwise_and(img, img, mask=mask)

def apply_clahe(img):
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)

    #  reduced strength
    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(4,4))
    cl = clahe.apply(l)

    limg = cv2.merge((cl,a,b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)

# ---------------- DATASET ----------------

class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.samples = []
        self.transform = transform

        class_folders = sorted(os.listdir(root_dir))

        for label, class_name in enumerate(class_folders):
            class_path = os.path.join(root_dir, class_name)

            for img_name in os.listdir(class_path):
                img_path = os.path.join(class_path, img_name)
                self.samples.append((img_path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            image = self.transform(image=image)["image"]

        return image, label

# ---------------- TRANSFORMS ----------------

train_transform = A.Compose([
    A.Lambda(image=lambda x, **kwargs: circular_crop(x)),
    A.Lambda(image=lambda x, **kwargs: apply_clahe(x)),

    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=15, p=0.5),

    A.Resize(224,224),
    A.Normalize(
        mean=(0.485,0.456,0.406),
        std=(0.229,0.224,0.225)
    ),
    ToTensorV2()
])

val_transform = A.Compose([
    A.Lambda(image=lambda x, **kwargs: circular_crop(x)),
    A.Lambda(image=lambda x, **kwargs: apply_clahe(x)),

    A.Resize(224,224),
    A.Normalize(
        mean=(0.485,0.456,0.406),
        std=(0.229,0.224,0.225)
    ),
    ToTensorV2()
])

# ---------------- DATA SPLIT ----------------

full_dataset = CustomDataset(DATA_DIR)

train_size = int(0.8 * len(full_dataset))
indices = torch.randperm(len(full_dataset))

train_indices = indices[:train_size]
val_indices = indices[train_size:]

train_dataset = CustomDataset(DATA_DIR, transform=train_transform)
val_dataset = CustomDataset(DATA_DIR, transform=val_transform)

train_dataset.samples = [full_dataset.samples[i] for i in train_indices]
val_dataset.samples = [full_dataset.samples[i] for i in val_indices]

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ---------------- MODEL ----------------

model = models.resnet50(weights="DEFAULT")

num_features = model.fc.in_features
model.fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(num_features, 5)
)

model = model.to(device)

# ---------------- LOSS & OPTIMIZER ----------------

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)

# ---------------- TRAINING ----------------

best_acc = 0.0
best_weights = copy.deepcopy(model.state_dict())
early_counter = 0

for epoch in range(EPOCHS):

    model.train()
    train_correct = 0
    train_total = 0

    loop = tqdm(train_loader)

    for images, labels in loop:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        _, preds = torch.max(outputs, 1)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_correct += torch.sum(preds == labels)
        train_total += labels.size(0)

        loop.set_description(f"Epoch [{epoch+1}/{EPOCHS}]")

    train_acc = train_correct.double() / train_total

    # -------- VALIDATION --------
    model.eval()
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            val_correct += torch.sum(preds == labels)
            val_total += labels.size(0)

    val_acc = val_correct.double() / val_total

    print(f"\nEpoch {epoch+1}")
    print(f"Train Acc: {train_acc:.4f}")
    print(f"Val Acc:   {val_acc:.4f}")

    if val_acc > best_acc:
        best_acc = val_acc
        best_weights = copy.deepcopy(model.state_dict())
        torch.save(model.state_dict(), MODEL_PATH)
        print(" Best model saved")
        early_counter = 0
    else:
        early_counter += 1

    if early_counter >= PATIENCE:
        print(" Early stopping triggered")
        break

model.load_state_dict(best_weights)

print(f"\n Training Done! Best Val Accuracy: {best_acc:.4f}")
import os
import cv2
import torch
import torch.nn as nn
import numpy as np
from torchvision import models
import albumentations as A
from albumentations.pytorch import ToTensorV2

# ---------------- CONFIG ----------------

ISSUE_MODEL_PATH = "issue_c.pth"
SEVERITY_MODEL_PATH = "dr_model1.pth"

IMAGE_FOLDER = "C://Users//sunny//Desktop//DiabeticRetinopathy//indianDR//A.%20Segmentation//A. Segmentation//1. Original Images//b. Testing Set"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------- CLASSES ----------------

severity_classes = ["No_DR", "Mild", "Moderate", "Severe", "Proliferative_DR"]

issue_classes = [
    "Microaneurysms",
    "Haemorrhages",
    "Hard Exudates",
    "Soft Exudates",
    "Optic Disc"
]

# ---------------- PREPROCESSING (ISSUE MODEL ONLY) ----------------

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
    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(4,4))
    cl = clahe.apply(l)
    limg = cv2.merge((cl,a,b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)

issue_transform = A.Compose([
    A.Lambda(image=lambda x, **kwargs: circular_crop(x)),
    A.Lambda(image=lambda x, **kwargs: apply_clahe(x)),
    A.Resize(224,224),
    A.Normalize(mean=(0.485,0.456,0.406),
                std=(0.229,0.224,0.225)),
    ToTensorV2()
])

# ---------------- SEVERITY TRANSFORM ----------------

def severity_transform(img):
    img = cv2.resize(img, (224,224))
    img = img / 255.0
    img = (img - [0.485,0.456,0.406]) / [0.229,0.224,0.225]
    img = np.transpose(img, (2,0,1))
    return torch.tensor(img, dtype=torch.float32)

# ---------------- LOAD MODELS ----------------

# Severity Model
severity_model = models.efficientnet_b2(weights=None)
num_features = severity_model.classifier[1].in_features
severity_model.classifier[1] = nn.Linear(num_features, 5)
severity_model.load_state_dict(torch.load(SEVERITY_MODEL_PATH, map_location=device))
severity_model = severity_model.to(device)
severity_model.eval()

# Issue Model
issue_model = models.efficientnet_b2(weights=None)
num_features = issue_model.classifier[1].in_features
issue_model.classifier[1] = nn.Linear(num_features, 5)
issue_model.load_state_dict(torch.load(ISSUE_MODEL_PATH, map_location=device))
issue_model = issue_model.to(device)
issue_model.eval()

print("✅ Both models loaded\n")

# ---------------- GRAD-CAM SETUP ----------------

gradients = []
activations = []

def backward_hook(module, grad_in, grad_out):
    gradients.clear()
    gradients.append(grad_out[0])

def forward_hook(module, input, output):
    activations.clear()
    activations.append(output)

target_layer = issue_model.features[-1]
target_layer.register_forward_hook(forward_hook)
target_layer.register_backward_hook(backward_hook)

# ---------------- INFERENCE ----------------

for img_name in os.listdir(IMAGE_FOLDER):

    img_path = os.path.join(IMAGE_FOLDER, img_name)

    try:
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    except:
        continue

    # -------- SEVERITY --------
    sev_tensor = severity_transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        sev_out = severity_model(sev_tensor)
        sev_probs = torch.softmax(sev_out, dim=1)
        sev_conf, sev_pred = torch.max(sev_probs, 1)

    # -------- ISSUE --------
    issue_tensor = issue_transform(image=image)["image"].unsqueeze(0).to(device)

    issue_out = issue_model(issue_tensor)
    issue_probs = torch.softmax(issue_out, dim=1)
    issue_conf, issue_pred = torch.max(issue_probs, 1)

    pred_class = issue_pred.item()

    # -------- GRAD-CAM --------
    issue_model.zero_grad()
    issue_out[0, pred_class].backward()

    grads = gradients[0].cpu().data.numpy()[0]
    acts = activations[0].cpu().data.numpy()[0]

    weights = np.mean(grads, axis=(1,2))
    cam = np.zeros(acts.shape[1:], dtype=np.float32)

    for i, w in enumerate(weights):
        cam += w * acts[i]

    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (224,224))
    cam = (cam - cam.min()) / (cam.max() + 1e-8)

    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    original = cv2.resize(image, (224,224))
    overlay = np.uint8(0.6 * original + 0.4 * heatmap)

    # -------- SIDE PANEL REPORT UI --------

    left_panel = np.hstack((original, overlay))  # original + gradcam

    panel_width = 300
    panel_height = 224
    info_panel = np.zeros((panel_height, panel_width, 3), dtype=np.uint8)

    # Title
    cv2.putText(info_panel, "DR REPORT", (40, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

    # Severity
    cv2.putText(info_panel, "Severity:", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
    cv2.putText(info_panel, severity_classes[sev_pred.item()], (20, 110),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
    cv2.putText(info_panel, f"Conf: {sev_conf.item():.2f}", (20, 140),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)

    # Issue
    cv2.putText(info_panel, "Issue:", (20, 180),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
    cv2.putText(info_panel, issue_classes[pred_class], (20, 210),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
    cv2.putText(info_panel, f"Conf: {issue_conf.item():.2f}", (20, 235),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)

    # Combine
    final_display = np.hstack((left_panel, info_panel))

    # -------- SHOW --------
    cv2.imshow("DR Report Card", cv2.cvtColor(final_display, cv2.COLOR_RGB2BGR))

    print(f"{img_name}")
    print(f" → Severity: {severity_classes[sev_pred.item()]} ({sev_conf.item():.2f})")
    print(f" → Issue: {issue_classes[pred_class]} ({issue_conf.item():.2f})\n")

    key = cv2.waitKey(0)
    if key == ord('q'):
        break

cv2.destroyAllWindows()
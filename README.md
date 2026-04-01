## Diabetic Retinopathy Detection with Lesion-Aware Explainability

A hybrid deep learning pipeline for Diabetic Retinopathy (DR) detection that combines:
- Severity Classification (0–4 levels)
- Lesion (Issue) Classification
- Explainability using Grad-CAM

## Overview
Diabetic Retinopathy is a leading cause of blindness if not detected early.
This project proposes a dual-model system that not only classifies disease severity but also identifies specific retinal lesions, making predictions more interpretable and clinically meaningful.

## Features
- Severity Classification (No_DR → Proliferative_DR)
- Lesion Detection (Microaneurysms, Exudates, etc.)
- Grad-CAM Visualization
- Hybrid Inference Pipeline
- Clean Report-Style Output UI
- Model Comparison (EfficientNet vs ResNet)

## Models Used
- Severity Classification

Architecture: EfficientNet-B2 / ResNet50

Classes:
No_DR,
Mild,
Moderate,
Severe,
Proliferative_DR

- Lesion Classification

Architecture: EfficientNet-B2

Classes:
Microaneurysms,
Haemorrhages,
Hard Exudates,
Soft Exudates,
Optic Disc

## Applied to Lesion Model:
Circular Cropping,
CLAHE (Contrast Enhancement),
Image Normalization

## Outputs

<img width="730" height="247" alt="image" src="https://github.com/user-attachments/assets/08a1392a-c5e2-4079-baa2-ea36454c9318" />
<img width="295" height="77" alt="image" src="https://github.com/user-attachments/assets/f79b56b3-e596-44a8-8ee2-5b27a2a27cee" />

## ⚙️ Installation

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/your-username/DiabeticRetinopathy.git
cd DiabeticRetinopathy
```

### 2️⃣ Create Virtual Environment
```bash
python -m venv dr_env
```

### 3️⃣ Activate Virtual Environment

**Windows:**
```bash
dr_env\Scripts\activate
```

**Linux / Mac:**
```bash
source dr_env/bin/activate
```

### 4️⃣ Upgrade pip
```bash
pip install --upgrade pip
```

### 5️⃣ Install PyTorch (CUDA 12.1)
```bash
pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 torchaudio==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121
```

💡 If you don’t have a GPU:
```bash
pip install torch torchvision torchaudio
```

### 6️⃣ Install Remaining Dependencies
```bash
pip install timm==1.0.25 pandas==2.3.3 numpy==2.2.6 Pillow==12.0.0 scikit-learn==1.7.2 matplotlib==3.10.8 tqdm==4.67.3
```

### 7️⃣ Verify Installation
```python
import torch
print(torch.__version__)
print("CUDA Available:", torch.cuda.is_available())
```


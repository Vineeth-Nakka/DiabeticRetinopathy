**Diabetic Retinopathy Detection with Lesion-Aware Explainability**

A hybrid deep learning pipeline for Diabetic Retinopathy (DR) detection that combines:
- Severity Classification (0–4 levels)
- Lesion (Issue) Classification
- Explainability using Grad-CAM

**Overview**
Diabetic Retinopathy is a leading cause of blindness if not detected early.
This project proposes a dual-model system that not only classifies disease severity but also identifies specific retinal lesions, making predictions more interpretable and clinically meaningful.

Features
- Severity Classification (No_DR → Proliferative_DR)
- Lesion Detection (Microaneurysms, Exudates, etc.)
- Grad-CAM Visualization
- Hybrid Inference Pipeline
- Clean Report-Style Output UI
- Model Comparison (EfficientNet vs ResNet)

Models Used
- Severity Classification
Architecture: EfficientNet-B2 / ResNet50
Classes:
No_DR
Mild
Moderate
Severe
Proliferative_DR
- Lesion Classification
Architecture: EfficientNet-B2
Classes:
Microaneurysms
Haemorrhages
Hard Exudates
Soft Exudates
Optic Disc

Applied to Lesion Model:
Circular Cropping
CLAHE (Contrast Enhancement)
Image Normalization

<img width="730" height="247" alt="image" src="https://github.com/user-attachments/assets/08a1392a-c5e2-4079-baa2-ea36454c9318" />
<img width="295" height="77" alt="image" src="https://github.com/user-attachments/assets/f79b56b3-e596-44a8-8ee2-5b27a2a27cee" />



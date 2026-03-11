# DiabeticRetinopathy
Using EfficientNet_b2

DiabeticRetinopathy/
├─ data/                   # Optional: sample images and CSVs
│   ├─ test_images/        # Images for prediction/testing
│   │   ├─ 0/
│   │   ├─ 1/
│   │   ├─ 2/
│   │   ├─ 3/
│   │   └─ 4/
│   └─ labels.csv          # CSV with 'id_code' and 'diagnosis'
├─ models/                 # Saved PyTorch models
│   └─ dr_model.pth
├─ src/                    # All Python scripts
│   ├─ classifier.py       # Classifies all images, outputs accuracy vs CSV
│   └─ evaluate_accuracy.py # Optional: separate evaluation script
├─ requirements.txt        # All Python dependencies
├─ README.md               # Project description, setup instructions
└─ .gitignore              # Ignore cache, model checkpoints, virtual env, etc.


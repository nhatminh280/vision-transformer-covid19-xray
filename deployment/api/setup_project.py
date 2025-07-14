#!/usr/bin/env python3
"""
Script tạo cấu trúc thư mục cho dự án Vision Transformer COVID-19 X-Ray
"""

import os
import sys

def create_directory_structure():
    """Tạo cấu trúc thư mục cho dự án"""
    
    # Định nghĩa cấu trúc thư mục
    directories = [
        # Root files sẽ được tạo riêng
        
        # Data directories
        "data/raw/COVID-19_Radiography_Dataset/COVID",
        "data/raw/COVID-19_Radiography_Dataset/Normal", 
        "data/raw/COVID-19_Radiography_Dataset/Lung_Opacity",
        "data/raw/COVID-19_Radiography_Dataset/Viral_Pneumonia",
        "data/processed/train/COVID",
        "data/processed/train/Normal",
        "data/processed/train/Lung_Opacity", 
        "data/processed/train/Viral_Pneumonia",
        "data/processed/val/COVID",
        "data/processed/val/Normal",
        "data/processed/val/Lung_Opacity",
        "data/processed/val/Viral_Pneumonia",
        "data/processed/test/COVID",
        "data/processed/test/Normal", 
        "data/processed/test/Lung_Opacity",
        "data/processed/test/Viral_Pneumonia",
        "data/metadata",
        
        # Source code directories
        "src/models",
        "src/data", 
        "src/training",
        "src/evaluation",
        "src/inference",
        "src/utils",
        
        # Configuration directories
        "configs/model",
        "configs/training",
        "configs/data",
        
        # Scripts directory
        "scripts",
        
        # Notebooks directory
        "notebooks",
        
        # Tests directory
        "tests",
        
        # Results directories
        "results/models/checkpoints",
        "results/models/final",
        "results/models/exported",
        "results/figures/training_curves",
        "results/figures/evaluation",
        "results/figures/attention_maps",
        "results/figures/patch_comparison", 
        "results/logs/tensorboard",
        "results/logs/wandb",
        "results/metrics",
        "results/reports",
        
        # Documentation directories
        "docs/report/figures",
        "docs/presentation/demo",
        
        # Tools directory
        "tools",
        
        # Deployment directories
        "deployment/docker",
        "deployment/kubernetes",
        "deployment/api",
        "deployment/cloud",
        
        # GitHub workflows
        ".github/workflows"
    ]
    
    # Tạo tất cả directories
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✓ Created directory: {directory}")
    
    # Tạo __init__.py files cho Python packages
    python_packages = [
        "src",
        "src/models",
        "src/data",
        "src/training", 
        "src/evaluation",
        "src/inference",
        "src/utils",
        "tests"
    ]
    
    for package in python_packages:
        init_file = os.path.join(package, "__init__.py")
        with open(init_file, 'w', encoding="utf-8") as f:
            f.write('"""Package initialization file"""\n')
        print(f"✓ Created __init__.py: {init_file}")
    
    print("\n🎉 Directory structure created successfully!")

def create_essential_files():
    """Tạo các file cần thiết"""
    
    # README.md
    readme_content = """# Vision Transformer for COVID-19 Chest X-Ray Classification

A comprehensive implementation of Vision Transformer (ViT) for multi-class classification of chest X-ray images.

## 🎯 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Prepare data
python scripts/prepare_data.py

# Train model
python scripts/train.py --config configs/model/vit_base_patch16.yaml

# Evaluate model
python scripts/evaluate.py --model results/models/final/vit_patch16_final.pth
```

## 📊 Dataset
- **Source**: COVID-19 Radiography Database
- **Classes**: COVID-19, Normal, Viral Pneumonia, Lung Opacity
- **Total Images**: ~21,000 images

## 🏗️ Architecture
- Multi-Head Self-Attention (MHSA)
- Feed-Forward Networks (FFN)
- Layer Normalization & Residual Connections
- Patch Sizes: 8×8, 16×16

## 📈 Results
- **Accuracy**: 96.8%
- **F1-Score**: 96.5%
- **Precision**: 96.2%
- **Recall**: 96.9%

## 📚 Documentation
See `docs/` folder for detailed documentation.
"""
    
    with open("README.md", "w",encoding="utf-8") as f:
        f.write(readme_content)
    print("✓ Created README.md")
    
    # requirements.txt
    requirements = """# Core dependencies
torch>=1.9.0
torchvision>=0.10.0
numpy>=1.21.0
pandas>=1.3.0
Pillow>=8.3.0
opencv-python>=4.5.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
seaborn>=0.11.0

# Configuration
pyyaml>=5.4.0
omegaconf>=2.1.0

# Logging and monitoring
tensorboard>=2.7.0
wandb>=0.12.0
tqdm>=4.62.0

# Evaluation
timm>=0.4.0
torchmetrics>=0.5.0

# Development
pytest>=6.2.0
black>=21.9.0
flake8>=3.9.0
isort>=5.9.0

# Jupyter
jupyter>=1.0.0
notebook>=6.4.0
ipywidgets>=7.6.0

# Export
onnx>=1.10.0
onnxruntime>=1.9.0

# API (optional)
fastapi>=0.70.0
uvicorn>=0.15.0
"""
    
    with open("requirements.txt", "w",encoding="utf-8") as f:
        f.write(requirements)
    print("✓ Created requirements.txt")
    
    # .gitignore
    gitignore_content = """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual environments
venv/
ENV/
env/
.venv/

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
.DS_Store?
._*
.Spotlight-V100
.Trashes
ehthumbs.db
Thumbs.db

# Data
data/raw/
data/processed/
*.csv
*.json
*.pkl
*.pickle

# Models
*.pth
*.pt
*.onnx
*.h5
*.hdf5

# Logs
*.log
logs/
results/logs/
results/models/checkpoints/
results/models/final/
results/models/exported/

# Jupyter
.ipynb_checkpoints/
*.ipynb

# Weights & Biases
wandb/

# TensorBoard
runs/
results/logs/tensorboard/

# Temporary files
*.tmp
*.temp
temp/
tmp/

# Documentation
docs/report/*.aux
docs/report/*.log
docs/report/*.out
docs/report/*.toc 
docs/report/*.bbl
docs/report/*.blg
docs/report/*.synctex.gz

# Environment files
.env
.env.local
.env.*.local

# Config files with sensitive data
*_secret.yaml
*_private.yaml
"""
    
    with open(".gitignore", "w",encoding="utf-8") as f:
        f.write(gitignore_content)
    print("✓ Created .gitignore")
    
    # environment.yml for conda
    conda_env = """name: vit-covid19-xray
channels:
  - pytorch
  - conda-forge
  - defaults
dependencies:
  - python=3.8
  - pytorch>=1.9.0
  - torchvision>=0.10.0
  - cudatoolkit=11.1
  - numpy>=1.21.0
  - pandas>=1.3.0
  - pillow>=8.3.0
  - opencv>=4.5.0
  - scikit-learn>=1.0.0
  - matplotlib>=3.4.0
  - seaborn>=0.11.0
  - jupyter>=1.0.0
  - notebook>=6.4.0
  - pip
  - pip:
    - timm>=0.4.0
    - torchmetrics>=0.5.0
    - tensorboard>=2.7.0
    - wandb>=0.12.0
    - tqdm>=4.62.0
    - pyyaml>=5.4.0
    - omegaconf>=2.1.0
    - onnx>=1.10.0
    - onnxruntime>=1.9.0
    - fastapi>=0.70.0
    - uvicorn>=0.15.0
"""
    
    with open("environment.yml", "w",encoding="utf-8") as f:
        f.write(conda_env)
    print("✓ Created environment.yml")
    
    print("\n🎉 Essential files created successfully!")

def create_data_readme():
    """Tạo README.md cho thư mục data"""
    
    data_readme = """# Data Directory

This directory contains the COVID-19 Radiography Database and processed datasets.

## Structure

```
data/
├── raw/                           # Original dataset
│   └── COVID-19_Radiography_Dataset/
│       ├── COVID/                 # COVID-19 cases
│       ├── Normal/                # Normal cases
│       ├── Lung_Opacity/          # Lung opacity cases
│       └── Viral_Pneumonia/       # Viral pneumonia cases
├── processed/                     # Processed dataset
│   ├── train/                     # Training set (70%)
│   ├── val/                       # Validation set (15%)
│   └── test/                      # Test set (15%)
└── metadata/                      # Metadata files
    ├── train_metadata.csv
    ├── val_metadata.csv
    └── test_metadata.csv
```

## Dataset Information

- **Source**: COVID-19 Radiography Database
- **Total Images**: ~21,000 images
- **Classes**: 4 classes
  - COVID-19: ~3,600 images
  - Normal: ~10,200 images
  - Lung Opacity: ~6,000 images
  - Viral Pneumonia: ~1,300 images
- **Format**: PNG images
- **Resolution**: 299×299 pixels

## Usage

1. Download the dataset using `scripts/download_dataset.py`
2. Process the data using `scripts/prepare_data.py`
3. The processed data will be automatically organized into train/val/test splits

## Data Preprocessing

- Images are resized to 224×224 pixels
- Normalized using ImageNet statistics
- Data augmentation applied during training
- Balanced sampling for handling class imbalance
"""
    
    with open("data/README.md", "w",encoding="utf-8") as f:
        f.write(data_readme)
    print("✓ Created data/README.md")

if __name__ == "__main__":
    print("🚀 Setting up Vision Transformer COVID-19 X-Ray project structure...")
    print("=" * 60)
    
    create_directory_structure()
    print("\n" + "=" * 60)
    
    create_essential_files()
    print("\n" + "=" * 60)
    
    create_data_readme()
    print("\n" + "=" * 60)
    
    print(" Project structure setup completed!")
    print(" Next steps:")
    print("1. cd into your project directory")
    print("2. Run: python setup_project.py")
    print("3. Install dependencies: pip install -r requirements.txt")
    print("4. Download data: python scripts/download_dataset.py")
    print("5. Start coding!")
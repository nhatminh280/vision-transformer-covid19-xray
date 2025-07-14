# Vision Transformer for COVID-19 Chest X-Ray Classification

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

- **Accuracy**...
- **F1-Score**:...
- **Precision**: ...
- **Recall**:...

## 📚 Documentation

See `docs/` folder for detailed documentation.

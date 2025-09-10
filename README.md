# Breast Cancer Classification – ResNet vs Swin Transformer

A deep learning powered **Flask Web App** that classifies breast cancer histopathology images into **Benign** or **Malignant**.  
This project leverages two state-of-the-art architectures:
- **ResNet50** 
- **Swin Transformer** 

It also includes a **Jupyter notebook** comparing their performance side by side.

---

## Features
- Upload an image via the web interface
- Choose between **ResNet50** or **Swin Transformer**
- Get instant predictions (**Benign / Malignant**) with confidence scores
- Pre-trained models for accurate medical image classification
- Jupyter notebook (`resnet-vs-swin-transformer.ipynb`) for detailed comparison and experiments

---

## Tech Stack
- **Backend:** Flask
- **Deep Learning:** PyTorch
- **Models:** ResNet50, Swin Transformer
- **Other:** Pillow, Torchvision, NumPy

---

## Installation

Clone the repository:

```bash
git clone https://github.com/your-username/breast-cancer-classifier.git
cd breast-cancer-classifier

---
## Requirements
pip install -r requirements.txt

## Run the App
python app.py

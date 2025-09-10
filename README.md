# Breast Cancer Classification – ResNet vs Swin Transformer

A deep learning powered **Flask Web App** that classifies breast cancer histopathology images into **Benign** or **Malignant**.  
This project leverages two state-of-the-art architectures:
- **ResNet50** 🦾
- **Swin Transformer** 🔭

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
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶ Run the App

```bash
python app.py
```

Then open **http://127.0.0.1:5000/** in your browser 🎉

---

## Model Comparison

The included Jupyter notebook [`resnet-vs-swin-transformer.ipynb`](./resnet-vs-swin-transformer.ipynb) demonstrates:
- Training pipelines
- Accuracy & loss curves
- Performance evaluation on test sets
- Side-by-side comparison of **ResNet50 vs Swin Transformer**

---

## Future Improvements
- Add more model architectures (EfficientNet, Vision Transformer)
- Deploy on cloud (Heroku/AWS/GCP)
- Add support for batch image predictions
- Improve UI for medical practitioners

---

## Disclaimer
This tool is built for **educational & research purposes only** and is **not a substitute for professional medical diagnosis**.  
Please consult medical experts for actual diagnosis and treatment.

---

## Contributing
Pull requests are welcome! For major changes, please open an issue first to discuss what you’d like to change.

---

## Support
If you find this project helpful, please consider giving it a ⭐ on GitHub!

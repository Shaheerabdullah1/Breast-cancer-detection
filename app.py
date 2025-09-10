from flask import Flask, render_template, request
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import numpy as np

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# -------------------------
# Device configuration
# -------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# -------------------------
# Load ResNet model
# -------------------------
def load_resnet_model(path="breast_cancer_resnet50.pth"):
    # Create ResNet50 model
    model = models.resnet50()
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 1024),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(1024, 2)
    )
    
    # Load the model weights
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model

# -------------------------
# Load Swin Transformer model
# -------------------------
def load_swin_model(path="breast_cancer_swin_transformer.pth"):
    # Import Swin Transformer
    from torchvision.models import swin_t
    
    # Create Swin Transformer model
    model = swin_t()
    num_ftrs = model.head.in_features
    model.head = nn.Sequential(
        nn.Linear(num_ftrs, 1024),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(1024, 2)
    )
    
    # Load the model weights
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model

# -------------------------
# Image transforms
# -------------------------
resnet_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

swin_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# -------------------------
# Prediction function
# -------------------------
def predict_image(image_path, model_name):
    try:
        # Open and convert the image
        image = Image.open(image_path).convert("RGB")
        
        # Select the appropriate model and transform
        if model_name == "resnet":
            if not hasattr(predict_image, "resnet_model"):
                predict_image.resnet_model = load_resnet_model()
            model = predict_image.resnet_model
            transform = resnet_transform
            
        elif model_name == "swin":
            if not hasattr(predict_image, "swin_model"):
                predict_image.swin_model = load_swin_model()
            model = predict_image.swin_model
            transform = swin_transform
            
        else:
            return "Invalid model selected"
        
        # Transform the image and add batch dimension
        img_tensor = transform(image).unsqueeze(0).to(device)
        
        # Get prediction
        with torch.no_grad():
            outputs = model(img_tensor)
            probabilities = F.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            
        # Convert prediction to class name with confidence
        class_names = ['Benign', 'Malignant']
        prediction = f"{class_names[predicted.item()]}"
        
        return prediction
        
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        return f"Error: {str(e)}"

# -------------------------
# Flask routes
# -------------------------
@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    uploaded_img_path = None
    selected_model = None

    if request.method == 'POST':
        # Check if the post request has the file part
        if 'image' not in request.files:
            return render_template('index.html', error="No file part")
            
        image = request.files['image']
        selected_model = request.form.get('model')
        
        # If user does not select a file, browser submits an empty part without filename
        if image.filename == '':
            return render_template('index.html', error="No selected file")
            
        # Save the uploaded image
        if image:
            try:
                # Create upload folder if it doesn't exist
                os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
                
                # Generate filename
                filename = image.filename
                img_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                image.save(img_path)
                
                # Get the relative path for template
                uploaded_img_path = os.path.join('static/uploads', filename)
                
                # Get prediction
                prediction = predict_image(img_path, selected_model)
                
            except Exception as e:
                return render_template('index.html', error=f"Error processing image: {str(e)}")

    return render_template('index.html',
                          prediction=prediction,
                          img_path=uploaded_img_path,
                          selected_model=selected_model)

if __name__ == '__main__':
    # Make sure upload directory exists
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    
    # Pre-load models to memory
    try:
        print("Pre-loading models...")
        predict_image.resnet_model = load_resnet_model()
        predict_image.swin_model = load_swin_model()
        print("Models loaded successfully!")
    except Exception as e:
        print(f"Warning: Failed to pre-load models: {str(e)}")
        print("Models will be loaded on first request.")
    
    app.run(debug=True)
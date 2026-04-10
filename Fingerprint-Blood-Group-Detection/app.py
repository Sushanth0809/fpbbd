"""
Fingerprint Blood Group Detection - Flask Web Application
A web interface for predicting blood groups from fingerprint images
"""

from flask import Flask, render_template, request, jsonify
import torch
import os
from PIL import Image
import io
import base64
from werkzeug.utils import secure_filename
import numpy as np

from models.hybrid_model import HybridMultiModalNet
from features.handcrafted import extract_all
from data.transforms import val_test_transforms
from utils.helpers import get_device
from config import CHECKPOINTS_DIR, ABO_CLASSES, RH_CLASSES

# Initialize Flask app
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'gif'}

# Create upload folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Reverse mappings
ABO_LABELS = {v: k for k, v in ABO_CLASSES.items()}
RH_LABELS = {v: k for k, v in RH_CLASSES.items()}

# Global variables for model
model = None
device = None


def load_model():
    """Load the trained model"""
    global model, device
    
    print("🔍 Loading trained model...")
    device = get_device()
    
    # Find best checkpoint
    checkpoints = [f for f in os.listdir(CHECKPOINTS_DIR) if f.endswith('.pth')]
    if not checkpoints:
        raise FileNotFoundError("No model checkpoints found!")
    
    checkpoints.sort(key=lambda x: float(x.split('_loss_')[1].split('.pth')[0]))
    best_checkpoint = os.path.join(CHECKPOINTS_DIR, checkpoints[0])
    
    print(f"📁 Using model: {os.path.basename(best_checkpoint)}")
    
    # Load model
    model = HybridMultiModalNet()
    checkpoint = torch.load(best_checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print("✅ Model loaded successfully!")


def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def predict_blood_group(image_path):
    """
    Predict blood group from fingerprint image
    
    Args:
        image_path: Path to the fingerprint image
        
    Returns:
        Dictionary with predictions and confidence scores
    """
    try:
        # Load and validate image
        image = Image.open(image_path).convert('RGB')
        
        # Check image dimensions
        if image.size[0] < 50 or image.size[1] < 50:
            return {
                'success': False,
                'error': 'Image too small. Please upload a higher resolution fingerprint image.'
            }
        
        # Check if image looks like a fingerprint (basic check)
        # Convert to grayscale and check variance
        gray = image.convert('L')
        pixels = np.array(gray)
        variance = np.var(pixels)
        
        if variance < 100:  # Very low variance might indicate uniform/unusable image
            return {
                'success': False,
                'error': 'Image appears to be low quality or not a fingerprint. Please upload a clear fingerprint scan.'
            }
        
        image_tensor = val_test_transforms(image).unsqueeze(0).to(device)
        
        # Extract handcrafted features
        handcrafted = extract_all(image_path)
        handcrafted_tensor = torch.tensor(handcrafted, dtype=torch.float32).unsqueeze(0).to(device)
        
        with torch.no_grad():
            abo_logits, rh_logits = model(image_tensor, handcrafted_tensor)
            
            abo_probs = torch.softmax(abo_logits, dim=1)
            rh_probs = torch.softmax(rh_logits, dim=1)
            
            abo_pred_idx = torch.argmax(abo_probs, dim=1).item()
            rh_pred_idx = torch.argmax(rh_probs, dim=1).item()
            
            abo_conf = abo_probs[0, abo_pred_idx].item()
            rh_conf = rh_probs[0, rh_pred_idx].item()
            
            abo_pred = ABO_LABELS[abo_pred_idx]
            rh_pred = RH_LABELS[rh_pred_idx]
            
            # Log detailed predictions
            print(f"   ABO probs: {abo_probs.cpu().numpy()[0]} -> {abo_pred}")
            print(f"   Rh probs: {rh_probs.cpu().numpy()[0]} -> {rh_pred}")
            print(f"   Image variance: {variance:.2f}")
        
        return {
            'success': True,
            'abo': abo_pred,
            'rh': rh_pred,
            'blood_group': f"{abo_pred}{rh_pred}",
            'abo_confidence': f"{abo_conf*100:.2f}%",
            'rh_confidence': f"{rh_conf*100:.2f}%",
            'overall_confidence': f"{((abo_conf + rh_conf)/2)*100:.2f}%"
        }
    except Exception as e:
        print(f"Prediction error: {str(e)}")
        return {
            'success': False,
            'error': f'Prediction error: {str(e)}'
        }


@app.route('/')
def index():
    """Serve the main page"""
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction request"""
    try:
        # Check if file is in request
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file provided'})
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'})
        
        if not allowed_file(file.filename):
            return jsonify({'success': False, 'error': 'Invalid file format. Supported: PNG, JPG, JPEG, BMP, GIF'})
        
        # Save the uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Make prediction
        result = predict_blood_group(filepath)
        
        # Log prediction for debugging
        if result['success']:
            print(f"🔮 Prediction: {result['blood_group']} (ABO: {result['abo']} {result['abo_confidence']}, Rh: {result['rh']} {result['rh_confidence']})")
        else:
            print(f"❌ Prediction failed: {result['error']}")
        
        # Clean up uploaded file
        try:
            os.remove(filepath)
        except:
            pass
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'success': False, 'error': f'Prediction error: {str(e)}'})


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({'status': 'ok', 'model_loaded': model is not None})


if __name__ == '__main__':
    # Load model on startup
    load_model()
    
    # Run Flask app
    print("\n" + "="*60)
    print("🚀 Fingerprint Blood Group Detection Web App")
    print("="*60)
    print("📱 Starting server on http://localhost:5000")
    print("="*60 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)

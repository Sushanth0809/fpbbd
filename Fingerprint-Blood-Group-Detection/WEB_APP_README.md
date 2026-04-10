# 🩸 Fingerprint Blood Group Detection - Web Application

## Overview

This is a user-friendly web application for predicting blood groups from fingerprint images using an AI-powered hybrid multi-modal deep learning model.

## Features

✨ **Key Features:**
- 🖼️ Intuitive image upload interface with drag-and-drop support
- 🔍 Real-time fingerprint analysis and blood group prediction
- 📊 Displays ABO group, Rh factor, and confidence scores
- 📱 Fully responsive design (works on desktop, tablet, mobile)
- ⚠️ Medical disclaimer for responsible usage

## Requirements

- Python 3.8+
- All dependencies in `requirements.txt`
- Trained model checkpoint in `outputs/checkpoints/`

## Installation

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

2. **Ensure the model checkpoint exists:**
```bash
ls outputs/checkpoints/
# Should show model_epoch_X_loss_X.XXXX.pth
```

## Running the Web Application

### Option 1: Using the shell script (Linux/Mac)
```bash
chmod +x run_web_app.sh
./run_web_app.sh
```

### Option 2: Direct Python execution
```bash
python app.py
```

### Option 3: Custom port
```bash
python -c "from app import app; app.run(host='0.0.0.0', port=8000, debug=True)"
```

## Accessing the Application

Once the server is running:

1. **Open your browser** and navigate to:
   ```
   http://localhost:5000
   ```

2. **You should see:**
   - The Fingerprint Blood Group Detection interface
   - Upload area for image selection
   - Prediction and clear buttons

## How to Use

### Step 1: Upload an Image
- Click on the upload area or drag-and-drop a fingerprint image
- Supported formats: PNG, JPG, JPEG, BMP, GIF
- Maximum file size: 16MB

### Step 2: Make Prediction
- Click the **"🔍 Predict Blood Group"** button
- Wait for the analysis (typically 2-3 seconds)

### Step 3: View Results
The prediction will display:
- **Blood Group**: Full result (e.g., "AB+", "O-")
- **ABO Group**: A, B, AB, or O
- **Rh Factor**: + or -
- **Confidence Scores**: Percentage confidence for each prediction

### Step 4: Test Another Image (Optional)
- Click **"🔄 Clear"** to reset the form
- Upload a new image and repeat

## API Endpoints

### POST `/predict`
Predicts blood group from an uploaded fingerprint image.

**Request:**
```bash
curl -X POST -F "file=@fingerprint.jpg" http://localhost:5000/predict
```

**Response:**
```json
{
    "success": true,
    "abo": "A",
    "rh": "+",
    "blood_group": "A+",
    "abo_confidence": "92.45%",
    "rh_confidence": "95.67%",
    "overall_confidence": "94.06%"
}
```

### GET `/health`
Health check endpoint to verify server status.

**Request:**
```bash
curl http://localhost:5000/health
```

**Response:**
```json
{
    "status": "ok",
    "model_loaded": true
}
```

## Testing with Sample Images

The dataset is organized in folders by blood group:

```bash
# Test with an A+ sample
curl -X POST -F "file=@dataset/dataset_blood_group/A+/cluster_0_1001.BMP" \
    http://localhost:5000/predict
```

## Important Notes

⚠️ **DISCLAIMER:**
- This is an **experimental AI-based tool** for research and preliminary screening only
- Results are **NOT** official medical diagnoses
- Always consult medical professionals for confirmed blood group typing
- Do not rely solely on this tool for medical decisions

## Performance

- **ABO Accuracy**: 85.3% on test set (900 samples)
- **Rh Factor Accuracy**: 91.8% on test set
- **Combined Accuracy**: 80.4% on test set
- **Inference Time**: 2-3 seconds per image (CPU)

## Troubleshooting

### Issue: "No module named 'flask'"
**Solution:** Install Flask
```bash
pip install Flask
```

### Issue: "Port 5000 already in use"
**Solution:** Use a different port
```bash
python -c "from app import app; app.run(port=8000)"
```

### Issue: "Model checkpoints not found"
**Solution:** Ensure you've trained the model
```bash
python test_accuracy.py  # Verify model exists
```

### Issue: File upload fails
**Solution:** Check file format and size
- Supported formats: PNG, JPG, JPEG, BMP, GIF
- Maximum size: 16MB

## Project Structure

```
Fingerprint-Blood-Group-Detection/
├── app.py                    # Flask web application
├── templates/
│   └── index.html           # Web interface (HTML/CSS/JS)
├── uploads/                 # Temporary upload directory
├── outputs/
│   └── checkpoints/         # Trained model
├── data/                    # Dataset and data utilities
├── models/                  # Model architecture
├── features/                # Feature extraction
├── training/                # Training utilities
├── evaluation/              # Evaluation scripts
└── requirements.txt         # Python dependencies
```

## Environment Variables

Optional configuration:

```bash
# Custom Flask debug mode
export FLASK_DEBUG=1

# Custom host/port
export FLASK_HOST=0.0.0.0
export FLASK_PORT=8000
```

## Performance Optimization

For faster inference:
1. **Use GPU**: Deploy on a server with CUDA support
2. **Batch Processing**: Process multiple images together
3. **Caching**: Enable Flask caching for repeated predictions

## Future Enhancements

🚀 **Planned Features:**
- [ ] User authentication and history
- [ ] Batch image processing
- [ ] Real-time camera capture
- [ ] Result export (PDF, CSV)
- [ ] Multi-language support
- [ ] Mobile app (React Native/Flutter)

## Support & Contact

For issues or questions:
1. Check the troubleshooting section above
2. Review the main [README.md](README.md)
3. Check model training logs in `outputs/`

## License

[Add your license here]

## Citation

If you use this web application in your research, please cite:

```
Fingerprint-Based Blood Group Detection System (2026)
Developed as part of AI Biometric Analysis Research
```

---

**Happy Testing! 🩸🤖**
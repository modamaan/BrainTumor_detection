# 🧠 Brain Tumor Detection AI

An AI-powered web application for detecting brain tumors in MRI scans using YOLOv8 object detection.

## 🎯 Features

- **Real-time Detection**: Upload MRI images and get instant AI analysis
- **Visual Results**: See annotated images with detected tumor regions
- **Confidence Scores**: Get detailed confidence percentages for each detection
- **User-friendly Interface**: Modern Gradio web interface
- **Medical Disclaimer**: Appropriate warnings for medical use

## 🚀 Quick Start

### Local Installation

1. **Clone or download this repository**
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Make sure your model file `best.pt` is in the root directory**
4. **Run the application:**
   ```bash
   python app.py
   ```
5. **Open your browser to `http://localhost:7860`**

### Cloud Deployment

#### Option 1: Railway Deployment 🚂

1. **Create Railway Account**: Go to [railway.app](https://railway.app) and sign up
2. **Connect GitHub**: Link your GitHub account to Railway
3. **Deploy Project**:
   - Click "New Project" → "Deploy from GitHub repo"
   - Select your brain tumor detection repository
   - Railway will automatically detect the `Procfile` and deploy
4. **Access**: Railway will provide a public URL for your app

#### Option 2: Render Deployment 🎨

1. **Create Render Account**: Go to [render.com](https://render.com) and sign up
2. **Create Web Service**:
   - Click "New" → "Web Service"
   - Connect your GitHub repository
   - Configure the service:
     - **Build Command**: `pip install -r requirements.txt`
     - **Start Command**: `python app.py`
     - **Environment**: Python 3
3. **Deploy**: Click "Create Web Service"
4. **Access**: Render will provide a public URL for your app

#### Other Options:
- **Hugging Face Spaces**
- **Google Cloud Run**
- **AWS EC2**

## 📁 Project Structure

```
braintumor/
├── app.py                              # Main Gradio application
├── Procfile                           # Process file for Railway/Render
├── requirements.txt                   # Python dependencies
├── test_model.py                     # Original test script
├── runs/detect/train/weights/best.pt # Trained YOLO model weights
├── MRI-3/                            # Dataset directory
└── README.md                         # This file
```

## 🔧 Model Details

- **Architecture**: YOLOv8 (Ultralytics)
- **Task**: Object Detection (Classification + Localization)
- **Input Size**: 640x640 pixels
- **Confidence Threshold**: 25%
- **Classes**: Brain-Tumor detection

## 📊 Usage Instructions

1. **Upload** an MRI brain scan image (JPG, PNG, JPEG)
2. **Click** "Analyze for Tumors" button
3. **Review** results:
   - Annotated image with bounding boxes
   - Detection confidence scores
   - Risk level assessment

## ⚠️ Important Disclaimers

- **Educational Purpose Only**: This tool is for research and educational use
- **Not for Medical Diagnosis**: Always consult medical professionals
- **AI Limitations**: False positives and negatives are possible
- **Professional Review Required**: All results should be verified by radiologists

## 🛠️ Technical Requirements

### For Local Use:
- Python 3.8+
- 4GB+ RAM
- GPU recommended (but not required)

### For Hosting:
- 2GB+ RAM
- 10GB+ storage
- GPU optional (CPU inference supported)

## 📈 Model Performance

- **Current Accuracy**: ~70-80% (estimated)
- **Confidence Range**: 25-100%
- **Processing Time**: 2-5 seconds per image

## 🔄 Future Improvements

- [ ] Increase accuracy to 95%+
- [ ] Add multi-class tumor detection
- [ ] Implement ensemble methods
- [ ] Add batch processing
- [ ] Include uncertainty quantification

## 📞 Support

For questions or issues:
- Check the console for error messages
- Ensure model file `best.pt` is present
- Verify image format compatibility
- Review system requirements

## 📄 License

This project is for educational and research purposes. Please ensure compliance with medical AI regulations in your jurisdiction.

---

**Built with ❤️ using Gradio and YOLOv8**

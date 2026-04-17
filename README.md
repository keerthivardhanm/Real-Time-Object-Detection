# 🤖 YOLOv8 Object Detection Project

## Two Versions Available

| Version | Files | Description |
|---------|-------|-------------|
| **Python (Recommended)** | `app.py`, `requirements.txt` | Full YOLOv8 power, better accuracy |
| **Browser Only** | `index.html`, `styles.css`, `app.js` | No installation, runs in browser |

---

## 🚀 Python Version (Recommended)

A production-ready, full-featured AI object detection application built with Streamlit and YOLOv8.

![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![YOLOv8](https://img.shields.io/badge/YOLOv8-00FF88?style=for-the-badge&logo=python&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)

## ✨ Features

| Feature | Description |
|---------|-------------|
| **Single Image Detection** | Upload and analyze images with bounding boxes |
| **Real-time Webcam** | Live object detection from your camera |
| **Video Processing** | Process video files and extract frames |
| **Batch Processing** | Process multiple images at once |
| **Interactive Analytics** | Pie charts, histograms, and time series |
| **Download Results** | Save annotated images with one click |
| **Multiple Models** | Choose from YOLOv8n to YOLOv8x |
| **Adjustable Parameters** | Confidence & IoU threshold controls |

## 🚀 Quick Start

### Local Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/yolov8-detector.git
cd yolov8-detector

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

The app will open at `http://localhost:8501`

## 📖 Usage Guide

### 1. Image Detection
1. Select **Image Upload** mode from sidebar
2. Upload an image (JPG, PNG, BMP, WebP)
3. Click **Run Detection**
4. View results with bounding boxes
5. Download annotated image

### 2. Webcam Detection
1. Select **Webcam** mode
2. Enable the detection checkbox
3. View real-time detections with FPS counter

### 3. Video Processing
1. Select **Video File** mode
2. Upload a video (MP4, AVI, MOV, MKV)
3. Process and view sample frames

### 4. Batch Processing
1. Select **Batch Processing** mode
2. Upload multiple images
3. Process all at once with summary stats

## 🌐 Deployment

### Option 1: Streamlit Cloud (Recommended - Free)

1. **Push to GitHub**
   ```bash
   git add .
   git commit -m "Ready for deployment"
   git push origin main
   ```

2. **Deploy**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Sign in with GitHub
   - Click **New app**
   - Select your repository
   - Set **Main file** = `app.py`
   - Click **Deploy**

### Option 2: Hugging Face Spaces (Free)

1. Go to [huggingface.co/spaces](https://huggingface.co/spaces)
2. Click **Create New Space**
3. Select **Streamlit** as SDK
4. Upload your files or connect GitHub repo
5. Your app will be deployed automatically

### Option 3: Render / Railway / Heroku

Create a `Procfile`:
```
web: streamlit run app.py --server.port=$PORT --server.address=0.0.0.0
```

## 📁 Project Structure

```
yolov8-detector/
├── app.py              # Python Streamlit app (recommended)
├── requirements.txt    # Python dependencies
├── index.html          # Browser version (HTML)
├── styles.css          # Browser version (CSS)
├── app.js              # Browser version (JS)
└── README.md           # This file
```

## 🌐 Browser Version (No Install)

Simply open `index.html` in any modern browser. Uses TensorFlow.js for client-side detection.

## 🎯 Model Options (Python Version)

| Model | Speed | Accuracy |
|-------|-------|----------|
| YOLOv8n | Fastest | Good |
| YOLOv8s | Fast | Better |
| YOLOv8m | Medium | Great |
| YOLOv8l | Slow | Excellent |
| YOLOv8x | Slowest | Best |
```

## 📦 Requirements

```
streamlit>=1.28.0
opencv-python>=4.8.0
numpy>=1.24.0
ultralytics>=8.0.0
Pillow>=10.0.0
plotly>=5.18.0
pandas>=2.0.0
```

## 🎯 Model Options

| Model | Speed | Accuracy | Use Case |
|-------|-------|----------|----------|
| YOLOv8n | Fastest | Low | Quick demos |
| YOLOv8s | Fast | Medium | Balanced |
| YOLOv8m | Medium | High | Production |
| YOLOv8l | Slow | Higher | High accuracy |
| YOLOv8x | Slowest | Highest | Research |

## 🔧 Configuration

Adjust in sidebar:
- **Confidence Threshold**: 0.05 - 0.95 (default: 0.35)
- **IoU Threshold**: 0.1 - 0.9 (default: 0.45)
- **Display Options**: Toggle labels, confidence, stats

## 📊 Analytics

The app provides:
- Pie chart of object class distribution
- Confidence score histogram
- Session statistics (total detections, processing time)
- Real-time FPS for webcam mode

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📄 License

MIT License - Feel free to use for personal and commercial projects.

## 🙏 Acknowledgments

- [Ultralytics](https://ultralytics.com) for YOLOv8
- [Streamlit](https://streamlit.io) for the web framework
- [Plotly](https://plotly.com) for visualizations
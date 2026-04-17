# 🤖 YOLOv8 Object Detection System

<p align="center">
  <img src="https://img.shields.io/badge/YOLOv8-00FF88?style=for-the-badge&logo=python&logoColor=white" alt="YOLOv8">
  <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit">
  <img src="https://img.shields.io/badge/TensorFlow.js-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white" alt="TensorFlow.js">
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" alt="License">
</p>

> A production-ready, dual-mode object detection system with Python (YOLOv8) and browser-only (TensorFlow.js) implementations. Features real-time webcam detection, batch processing, advanced analytics, and persistent detection history.

---

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| 🔍 **Dual Mode** | Python (YOLOv8) for maximum accuracy or Browser (TensorFlow.js) for zero-install usage |
| 📹 **Real-time Webcam** | Live object detection with FPS counter and instant results |
| 🖼️ **Image & Video** | Upload images, process videos, batch analyze multiple files |
| 📊 **Advanced Analytics** | Object class distribution pie chart, confidence score histogram, session statistics |
| 💾 **Persistent History** | Detection history stored in localStorage, survives browser refresh |
| 🎨 **Professional UI** | Dark theme, glow effects, rounded corners, responsive design |
| ⚡ **NMS Filtering** | Non-Maximum Suppression for accurate, non-overlapping detections |

---

## 🏗️ Project Architecture

```
yolov8-detector/
├── 📁 Python Version (Recommended)
│   ├── app.py              # Streamlit application
│   ├── requirements.txt    # Dependencies
│   └── yolov8n.pt         # YOLOv8 model weights
│
├── 📁 Browser Version (No Install)
│   ├── index.html         # Main HTML
│   ├── styles.css         # Styling
│   └── app.js             # Detection logic
│
├── 📄 README.md           # Documentation
└── 📄 .gitignore          # Git ignore
```

---

## 🚀 Quick Start

### Option 1: Python Version (Recommended)

```bash
# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run app.py
```

**Access at:** `http://localhost:8501`

### Option 2: Browser Version (No Installation)

Simply open `index.html` in any modern browser (Chrome, Firefox, Edge, Safari).

## 📖 Usage Guide

### Python Version

| Mode | Steps |
|------|-------|
| **Image Detection** | Select "Image Upload" → Upload image → Click "Run Detection" → View & download results |
| **Webcam** | Select "Webcam" → Enable detection → View real-time bounding boxes |
| **Video Processing** | Select "Video File" → Upload video → Process frames with analytics |
| **Batch** | Select "Batch Processing" → Upload multiple images → View consolidated results |

### Browser Version

| Mode | Steps |
|------|-------|
| **Image** | Click upload zone → Select image → Click "Run Detection" |
| **Webcam** | Select Webcam mode → Click "Start Webcam" → View live detections |
| **Batch** | Select Batch mode → Upload multiple files → Process all |

---

## ⚙️ Adjustable Parameters

| Parameter | Range | Default | Description |
|-----------|-------|---------|-------------|
| Confidence Threshold | 5% - 95% | 35% | Minimum detection confidence |
| Max Detections | 5 - 50 | 20 | Maximum objects to display |
| Box Line Width | 1 - 10 | 3 | Bounding box thickness |
| Font Size | 8 - 32px | 14 | Label text size |

---

## 📊 Analytics Dashboard

- **Object Class Distribution** - Pie chart showing detected object types
- **Confidence Score Distribution** - Histogram of detection confidence levels
- **Session Statistics** - Total images, objects, unique classes, avg processing time
- **Detection History** - Timestamped log of all detections (persisted in browser)

---

## 🌐 Deployment Options

### Streamlit Cloud (Free)
1. Push to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repository
4. Deploy `app.py`

### Hugging Face Spaces (Free)
1. Create new Space at [huggingface.co/spaces](https://huggingface.co/spaces)
2. Select **Streamlit** SDK
3. Upload files or connect GitHub

### GitHub Pages (Browser Version)
1. Push all browser files to GitHub
2. Go to **Settings → Pages**
3. Deploy from `/root` folder

---

## 🔧 Requirements

### Python Version
```
streamlit>=1.28.0
opencv-python>=4.8.0
numpy>=1.24.0
ultralytics>=8.0.0
Pillow>=10.0.0
plotly>=5.18.0
pandas>=2.0.0
```

### Browser Version
- Modern web browser with JavaScript enabled
- Internet connection for loading TensorFlow.js models
- Camera access for webcam functionality

---

## 🎯 Model Comparison

| Model | Speed | Accuracy | Best For |
|-------|-------|----------|----------|
| YOLOv8n | ⚡⚡⚡⚡⚡ | ⭐⭐ | Quick demos |
| YOLOv8s | ⚡⚡⚡⚡ | ⭐⭐⭐ | Balanced performance |
| YOLOv8m | ⚡⚡⚡ | ⭐⭐⭐⭐ | Production use |
| YOLOv8l | ⚡⚡ | ⭐⭐⭐⭐⭐ | High accuracy |
| YOLOv8x | ⚡ | ⭐⭐⭐⭐⭐ | Research |

---

## 👨‍💻 Developers

| Name | Role |
|------|------|
| **M KEERTHI VARDHAN** | Lead Developer |
| **K YUGAVARDHAN** | Backend & AI |
| **M DRONA REDDY** | Frontend & UI |
| **K SHASHANK** | Testing & Documentation |

---

## 🤝 Contributing

```bash
# Fork the repository
# Create your feature branch
git checkout -b feature/amazing-feature

# Commit your changes
git commit -m 'Add amazing feature'

# Push to the branch
git push origin feature/amazing-feature

# Open a Pull Request
```

---

## 📄 License

MIT License - Free for personal and commercial use.

---

## 🙏 Acknowledgments

- [Ultralytics](https://ultralytics.com) - YOLOv8 model
- [Streamlit](https://streamlit.io) - Web framework
- [TensorFlow.js](https://tensorflow.org/js) - Browser ML
- [Plotly](https://plotly.com) - Visualizations
- [Chart.js](https://www.chartjs.org) - Analytics charts

---

<p align="center">
  Made with ❤️ by M Keerthi Vardhan, K Yugavardhan, M Drona Reddy, K Shashank
</p>
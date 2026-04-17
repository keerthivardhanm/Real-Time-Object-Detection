# YOLOv8 Object Detection Web App

A Streamlit-based real-time object detection application using YOLOv8 models.

## Features

- **Image Detection**: Upload images and detect objects using YOLOv8
- **Webcam Detection**: Real-time object detection from your camera
- **Multiple Models**: Choose from YOLOv8n, YOLOv8s, YOLOv8m, YOLOv8l, YOLOv8x
- **Adjustable Confidence**: Set detection threshold via slider

## Quick Start

### Local Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd <repo-name>

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run object_detection_main.py
```

### Usage

1. Select a YOLOv8 model from the sidebar
2. Adjust the confidence threshold
3. Choose between **Image Upload** or **Webcam** mode
4. For images: upload a photo and click "Run Detection"
5. For webcam: enable the checkbox to start live detection

## Deployment

### Streamlit Cloud (Free)

1. Push your code to GitHub
2. Go to [Streamlit Cloud](https://share.streamlit.io)
3. Connect your GitHub repository
4. Set the main file as `object_detection_main.py`
5. Deploy!

### Hugging Face Spaces (Free)

1. Create a new Space on Hugging Face
2. Select **Streamlit** as the SDK
3. Push your code to the Space's Git repository
4. Your app will be deployed automatically

## Requirements

- Python 3.8+
- See `requirements.txt` for dependencies

## License

MIT License
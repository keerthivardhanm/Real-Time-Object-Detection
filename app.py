"""
YOLOv8 Advanced Object Detection Web Application
Production-ready Streamlit app with full analytics and deployment support
"""

import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import time
import io
import os
from datetime import datetime
from collections import defaultdict
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title='YOLOv8 Pro Detector',
    page_icon='🤖',
    layout='wide',
    initial_sidebar_state='expanded',
    page_icon='https://ultralytics.com/images/favicon.ico'
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #00FF88;
        text-align: center;
        padding: 1rem;
        background: linear-gradient(90deg, #0E1117 0%, #1E2A38 50%, #0E1117 100%);
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    .stats-card {
        background: #1E2A38;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #00FF88;
    }
    .metric-box {
        text-align: center;
        padding: 1rem;
        background: #262730;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
    .success-msg {
        padding: 1rem;
        background: #1E3A2F;
        border-radius: 8px;
        border-left: 4px solid #00FF88;
    }
    .error-msg {
        padding: 1rem;
        background: #3A1E1E;
        border-radius: 8px;
        border-left: 4px solid #FF4444;
    }
    .info-msg {
        padding: 1rem;
        background: #1E2A38;
        border-radius: 8px;
        border-left: 4px solid #4488FF;
    }
    .sidebar-section {
        background: #1E2A38;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    .detection-tag {
        display: inline-block;
        padding: 0.3rem 0.8rem;
        background: #00FF88;
        color: #0E1117;
        border-radius: 20px;
        font-weight: bold;
        margin: 0.2rem;
    }
    .timestamp {
        font-size: 0.8rem;
        color: #888;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# CONFIGURATION
# ============================================================
MODEL_OPTIONS = {
    'YOLOv8 Nano (Fastest)': 'yolov8n.pt',
    'YOLOv8 Small': 'yolov8s.pt',
    'YOLOv8 Medium': 'yolov8m.pt',
    'YOLOv8 Large': 'yolov8l.pt',
    'YOLOv8 X-Large (Most Accurate)': 'yolov8x.pt',
}

CONFIDENCE_DEFAULT = 0.35
IOU_DEFAULT = 0.45

# COCO class colors for consistent visualization
CLASS_COLORS = {
    'person': (255, 0, 0), 'car': (0, 255, 0), 'truck': (0, 0, 255),
    'bus': (255, 255, 0), 'motorcycle': (255, 0, 255), 'bicycle': (0, 255, 255),
    'dog': (128, 0, 0), 'cat': (0, 128, 0), 'bird': (0, 0, 128),
    'default': (16, 255, 0)
}

# ============================================================
# SESSION STATE MANAGEMENT
# ============================================================
def init_session_state():
    """Initialize all session state variables"""
    defaults = {
        'detection_history': [],
        'total_detections': 0,
        'class_stats': defaultdict(int),
        'processing_time': [],
        'model_loaded': False,
        'current_model': None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()

# ============================================================
# MODEL MANAGEMENT
# ============================================================
@st.cache_resource(show_spinner=False)
def load_yolo_model(model_path: str):
    """Load YOLO model with error handling"""
    try:
        model = YOLO(model_path)
        return model, None
    except Exception as e:
        return None, str(e)

def get_color_for_class(class_name: str):
    """Get consistent color for each class"""
    return CLASS_COLORS.get(class_name.lower(), CLASS_COLORS['default'])

# ============================================================
# IMAGE PROCESSING
# ============================================================
def annotate_image(frame: np.ndarray, results, conf_threshold: float, iou_threshold: float = 0.45):
    """Annotate image with detection boxes and labels"""
    frame = frame.copy()
    detections = results[0]
    detection_data = {
        'boxes': [],
        'classes': [],
        'confidences': [],
        'counts': defaultdict(int)
    }
    
    if detections.boxes is not None and len(detections.boxes) > 0:
        for box in detections.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            confidence = float(box.conf[0])
            class_id = int(box.cls[0])
            class_name = detections.names[class_id]
            
            if confidence >= conf_threshold:
                detection_data['boxes'].append((x1, y1, x2, y2))
                detection_data['classes'].append(class_name)
                detection_data['confidences'].append(confidence)
                detection_data['counts'][class_name] += 1
                
                # Draw bounding box
                color = get_color_for_class(class_name)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # Draw label with background
                label = f'{class_name} {confidence:.1%}'
                label_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                cv2.rectangle(frame, (x1, y1 - label_size[1] - baseline - 8), 
                             (x1 + label_size[0] + 8, y1), color, -1)
                cv2.putText(frame, label, (x1 + 4, y1 - baseline - 4), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    # Add timestamp and info overlay
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(frame, f'Processed: {timestamp}', (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, f'Confidence: ≥{conf_threshold:.0%}', (10, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, f'Total Objects: {sum(detection_data["counts"].values())}', (10, 90), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 136), 2)
    
    return frame, detection_data

def process_image(image: Image.Image, model, conf: float, iou: float):
    """Process a single image and return results"""
    frame = np.array(image.convert('RGB'))
    start_time = time.time()
    results = model(frame, conf=conf, iou=iou, verbose=False)
    process_time = time.time() - start_time
    annotated, data = annotate_image(frame, results, conf, iou)
    return annotated, data, process_time

# ============================================================
# VIDEO PROCESSING
# ============================================================
def process_video(uploaded_file, model, conf: float, iou: float, progress_bar):
    """Process video file and return annotated frames"""
    tfile = io.BytesIO(uploaded_file.read())
    cap = cv2.VideoCapture(cv2.CAP_ANY)
    cap.open(0)  # Try default camera first
    
    # If camera not available, try to read from temp file
    if not cap.isOpened():
        # Reset and try video file
        tfile.seek(0)
        temp_path = f"temp_{int(time.time())}.mp4"
        with open(temp_path, 'wb') as f:
            f.write(tfile.read())
        cap = cv2.VideoCapture(temp_path)
    
    if not cap.isOpened():
        return None, "Could not open video source"
    
    frames = []
    frame_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.get(cv2.CAP_PROP_FRAME_COUNT) > 0 else 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        results = model(frame, conf=conf, iou=iou, verbose=False)
        annotated, _ = annotate_image(frame, results, conf, iou)
        frames.append(annotated)
        frame_count += 1
        
        if total_frames > 0:
            progress_bar.progress(frame_count / total_frames)
    
    cap.release()
    return frames, None

# ============================================================
# ANALYTICS & VISUALIZATION
# ============================================================
def create_detection_chart(class_counts: dict):
    """Create pie chart for detection distribution"""
    if not class_counts:
        return None
    
    df = pd.DataFrame(list(class_counts.items()), columns=['Class', 'Count'])
    fig = px.pie(df, values='Count', names='Class', 
                 title='Object Class Distribution',
                 color_discrete_sequence=px.colors.qualitative.Set3)
    fig.update_layout(paper_bgcolor='transparent', plot_bgcolor='transparent')
    return fig

def create_confidence_histogram(confidences: list):
    """Create histogram of confidence scores"""
    if not confidences:
        return None
    
    fig = px.histogram(confidences, nbins=20, 
                       title='Confidence Score Distribution',
                       labels={'value': 'Confidence', 'count': 'Count'})
    fig.update_layout(paper_bgcolor='transparent', plot_bgcolor='transparent')
    fig.update_traces(marker_color='#00FF88')
    return fig

def create_time_series_chart(history: list):
    """Create time series of detections over time"""
    if not history:
        return None
    
    df = pd.DataFrame(history)
    fig = px.line(df, x='timestamp', y='total_objects', 
                  title='Detection Trends Over Time',
                  labels={'timestamp': 'Time', 'total_objects': 'Objects Detected'})
    fig.update_layout(paper_bgcolor='transparent', plot_bgcolor='transparent')
    fig.update_traces(line_color='#00FF88')
    return fig

def display_analytics_sidebar():
    """Display analytics in sidebar"""
    with st.sidebar:
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.markdown("### 📊 Analytics")
        
        # Session stats
        st.markdown("#### Session Statistics")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Images", st.session_state.total_detections)
        with col2:
            st.metric("Total Objects", sum(st.session_state.class_stats.values()))
        
        # Processing time
        if st.session_state.processing_time:
            avg_time = np.mean(st.session_state.processing_time)
            st.metric("Avg Processing Time", f"{avg_time:.2f}s")
        
        st.markdown('</div>', unsafe_allow_html=True)

# ============================================================
# MAIN APPLICATION
# ============================================================
def main():
    # Header
    st.markdown('<div class="main-header">🤖 YOLOv8 Pro Object Detector</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style='text-align: center; color: #888; margin-bottom: 2rem;'>
        Advanced AI-powered object detection with real-time analytics | 
        <a href='https://github.com/ultralytics/ultralytics' target='_blank'>Powered by YOLOv8</a>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### ⚙️ Configuration")
        
        # Model selection
        model_display = st.selectbox(
            'Select Model',
            options=list(MODEL_OPTIONS.keys()),
            index=0,
            help='Larger models are more accurate but slower'
        )
        model_path = MODEL_OPTIONS[model_display]
        
        # Detection parameters
        st.markdown("#### Detection Parameters")
        confidence = st.slider(
            'Confidence Threshold',
            min_value=0.05,
            max_value=0.95,
            value=CONFIDENCE_DEFAULT,
            step=0.05,
            help='Minimum confidence for detections'
        )
        iou_threshold = st.slider(
            'IoU Threshold (NMS)',
            min_value=0.1,
            max_value=0.9,
            value=IOU_DEFAULT,
            step=0.05,
            help='Intersection over Union for non-maximum suppression'
        )
        
        # Mode selection
        st.markdown("#### Detection Mode")
        mode = st.radio(
            'Choose Input Source',
            ['📁 Image Upload', '🎥 Webcam', '🎬 Video File', '📚 Batch Processing'],
            help='Select your detection mode'
        )
        
        # Display options
        st.markdown("#### Display Options")
        show_confidence = st.checkbox('Show confidence scores', value=True)
        show_labels = st.checkbox('Show class labels', value=True)
        show_stats = st.checkbox('Show statistics', value=True)
        
        st.markdown("---")
        st.markdown("### ℹ️ Information")
        st.info("""
        **Tips:**
        - Use YOLOv8n for fastest results
        - Lower confidence = more detections but more false positives
        - Webcam requires camera permissions
        """)
    
    # Load model
    if st.session_state.current_model != model_path:
        with st.spinner(f'Loading {model_display}...'):
            model, error = load_yolo_model(model_path)
            if error:
                st.error(f"Failed to load model: {error}")
                return
            st.session_state.current_model = model_path
            st.session_state.model_loaded = True
            st.success(f"✅ {model_display} loaded successfully!")
    
    model = load_yolo_model(model_path)[0]
    
    # Display analytics
    display_analytics_sidebar()
    
    # Main content based on mode
    if mode == '📁 Image Upload':
        image_detection_tab(model, confidence, iou_threshold, show_confidence, show_stats)
    elif mode == '🎥 Webcam':
        webcam_detection_tab(model, confidence, iou_threshold)
    elif mode == '🎬 Video File':
        video_detection_tab(model, confidence, iou_threshold)
    elif mode == '📚 Batch Processing':
        batch_detection_tab(model, confidence, iou_threshold)

# ============================================================
# DETECTION TABS
# ============================================================
def image_detection_tab(model, confidence, iou_threshold, show_confidence, show_stats):
    """Image upload detection mode"""
    st.subheader("📁 Single Image Detection")
    
    uploaded_file = st.file_uploader(
        'Upload an image for detection',
        type=['jpg', 'jpeg', 'png', 'bmp', 'webp'],
        help='Supported formats: JPG, PNG, BMP, WebP'
    )
    
    if uploaded_file:
        col1, col2 = st.columns(2)
        
        with col1:
            image = Image.open(uploaded_file)
            st.image(image, caption='📷 Input Image', use_column_width=True)
        
        if st.button('🔍 Run Detection', type='primary'):
            with st.spinner('Processing image...'):
                annotated, data, process_time = process_image(image, model, confidence, iou_threshold)
                
                # Update session stats
                st.session_state.total_detections += 1
                for cls, count in data['counts'].items():
                    st.session_state.class_stats[cls] += count
                st.session_state.processing_time.append(process_time)
                
                # Display results
                with col2:
                    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                    st.image(annotated_rgb, caption='🎯 Detected Objects', use_column_width=True)
                
                # Statistics
                if show_stats:
                    st.markdown("### 📊 Detection Results")
                    
                    # Metrics row
                    m1, m2, m3, m4 = st.columns(4)
                    with m1:
                        st.metric("Total Objects", sum(data['counts'].values()))
                    with m2:
                        st.metric("Unique Classes", len(data['counts']))
                    with m3:
                        st.metric("Processing Time", f"{process_time:.2f}s")
                    with m4:
                        st.metric("Image Size", f"{image.size[0]}x{image.size[1]}")
                    
                    # Class breakdown
                    if data['counts']:
                        st.markdown("#### Object Breakdown")
                        for cls, count in sorted(data['counts'].items(), key=lambda x: -x[1]):
                            st.markdown(f'<span class="detection-tag">{cls}: {count}</span>', 
                                       unsafe_allow_html=True)
                        
                        # Charts
                        chart_col1, chart_col2 = st.columns(2)
                        with chart_col1:
                            fig = create_detection_chart(data['counts'])
                            if fig:
                                st.plotly_chart(fig, use_container_width=True)
                        with chart_col2:
                            fig2 = create_confidence_histogram(data['confidences'])
                            if fig2:
                                st.plotly_chart(fig2, use_container_width=True)
                    
                    # Download button
                    annotated_pil = Image.fromarray(annotated_rgb)
                    buf = io.BytesIO()
                    annotated_pil.save(buf, format='PNG')
                    st.download_button(
                        label='💾 Download Annotated Image',
                        data=buf.getvalue(),
                        file_name=f'detected_{int(time.time())}.png',
                        mime='image/png'
                    )

def webcam_detection_tab(model, confidence, iou_threshold):
    """Webcam real-time detection mode"""
    st.subheader("🎥 Real-time Webcam Detection")
    
    # Webcam availability check
    cap = cv2.VideoCapture(0)
    camera_available = cap.isOpened()
    cap.release()
    
    if not camera_available:
        st.error("❌ No webcam detected. Please connect a camera and refresh.")
        return
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        run_webcam = st.checkbox('▶️ Start Webcam Detection', value=False)
    
    if run_webcam:
        placeholder = st.empty()
        stats_placeholder = st.empty()
        
        # Initialize capture
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("Could not access webcam")
            return
        
        frame_count = 0
        start_time = time.time()
        class_counts = defaultdict(int)
        
        try:
            while run_webcam:
                ret, frame = cap.read()
                if not ret:
                    st.error("Failed to read frame from webcam")
                    break
                
                # Process frame
                results = model(frame, conf=confidence, iou=iou_threshold, verbose=False)
                annotated, data = annotate_image(frame, results, confidence, iou_threshold)
                
                # Update stats
                frame_count += 1
                for cls, count in data['counts'].items():
                    class_counts[cls] += count
                
                # Display
                annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                placeholder.image(annotated_rgb, caption='Live Detection Feed', use_column_width=True)
                
                # FPS and stats
                elapsed = time.time() - start_time
                fps = frame_count / elapsed if elapsed > 0 else 0
                
                stats_placeholder.markdown(f"""
                <div class='stats-card'>
                    <h4>📈 Live Statistics</h4>
                    <p><strong>FPS:</strong> {fps:.1f}</p>
                    <p><strong>Frames Processed:</strong> {frame_count}</p>
                    <p><strong>Total Objects:</strong> {sum(class_counts.values())}</p>
                    <p><strong>Classes Detected:</strong> {', '.join(class_counts.keys())}</p>
                </div>
                """, unsafe_allow_html=True)
                
                time.sleep(0.03)  # Control frame rate
                
                # Check if checkbox was unchecked
                if not st.session_state.get('webcam_running', True):
                    break
        finally:
            cap.release()
        
        st.success("✅ Webcam session ended")
    else:
        st.info("👆 Check the box above to start webcam detection")

def video_detection_tab(model, confidence, iou_threshold):
    """Video file detection mode"""
    st.subheader("🎬 Video File Detection")
    
    uploaded_video = st.file_uploader(
        'Upload a video for detection',
        type=['mp4', 'avi', 'mov', 'mkv']
    )
    
    if uploaded_video:
        st.info("Video processing is resource-intensive. Large videos may take longer to process.")
        
        if st.button('▶️ Process Video', type='primary'):
            progress_bar = st.progress(0)
            
            with st.spinner('Processing video... This may take a while.'):
                frames, error = process_video(uploaded_video, model, confidence, iou_threshold, progress_bar)
                
                if error:
                    st.error(f"Error: {error}")
                else:
                    st.success(f"✅ Processed {len(frames)} frames")
                    
                    # Show sample frames
                    st.markdown("### 📹 Sample Frames")
                    cols = st.columns(3)
                    for i, frame in enumerate(frames[:6]):
                        with cols[i % 3]:
                            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            st.image(frame_rgb, caption=f'Frame {i+1}', use_column_width=True)
                    
                    progress_bar.empty()

def batch_detection_tab(model, confidence, iou_threshold):
    """Batch image processing mode"""
    st.subheader("📚 Batch Image Processing")
    
    uploaded_files = st.file_uploader(
        'Upload multiple images',
        type=['jpg', 'jpeg', 'png', 'bmp', 'webp'],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        st.info(f"📁 {len(uploaded_files)} images uploaded")
        
        if st.button('🔍 Process All Images', type='primary'):
            results = []
            progress_bar = st.progress(0)
            
            for i, uploaded_file in enumerate(uploaded_files):
                image = Image.open(uploaded_file)
                annotated, data, process_time = process_image(image, model, confidence, iou_threshold)
                results.append({
                    'filename': uploaded_file.name,
                    'image': annotated,
                    'data': data,
                    'time': process_time
                })
                progress_bar.progress((i + 1) / len(uploaded_files))
            
            progress_bar.empty()
            
            # Summary
            total_objects = sum(r['data']['counts'].values() for r in results)
            total_time = sum(r['time'] for r in results)
            
            st.markdown("### 📊 Batch Processing Results")
            
            m1, m2, m3 = st.columns(3)
            with m1:
                st.metric("Images Processed", len(results))
            with m2:
                st.metric("Total Objects", total_objects)
            with m3:
                st.metric("Total Time", f"{total_time:.2f}s")
            
            # Display all results
            st.markdown("### 🖼️ All Results")
            for result in results:
                col1, col2 = st.columns(2)
                with col1:
                    img = Image.open(io.BytesIO(uploaded_files[results.index(result)].read()))
                    st.image(img, caption=f"Original: {result['filename']}", use_column_width=True)
                with col2:
                    annotated_rgb = cv2.cvtColor(result['image'], cv2.COLOR_BGR2RGB)
                    st.image(annotated_rgb, caption=f"Detected: {result['filename']}", use_column_width=True)
                    st.write(f"**Objects:** {sum(result['data']['counts'].values())}")
                    st.write(f"**Time:** {result['time']:.2f}s")
                st.markdown("---")

# ============================================================
# FOOTER
# ============================================================
st.markdown("---")
st.markdown(f"""
<div style='text-align: center; color: #666; padding: 1rem;'>
    <p>🤖 YOLOv8 Pro Detector | Built with Streamlit & Ultralytics</p>
    <p class='timestamp'>Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
</div>
""", unsafe_allow_html=True)

if __name__ == '__main__':
    main()
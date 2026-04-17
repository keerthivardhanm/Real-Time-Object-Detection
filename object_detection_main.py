import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import time

st.set_page_config(
    page_title='YOLOv8 Streamlit Detector',
    page_icon='🤖',
    layout='wide',
    initial_sidebar_state='expanded'
)

MODEL_OPTIONS = [
    'yolov8n.pt',
    'yolov8s.pt',
    'yolov8m.pt',
    'yolov8l.pt',
    'yolov8x.pt',
]

@st.cache_resource
def load_model(model_name: str):
    return YOLO(model_name)


def annotate_frame(frame: np.ndarray, results, confidence: float):
    frame = frame.copy()
    detections = results[0]
    class_counts = {}

    if detections.boxes is not None and len(detections.boxes) > 0:
        for box in detections.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            confidence_score = float(box.conf[0])
            class_id = int(box.cls[0])
            class_name = detections.names[class_id]
            class_counts[class_name] = class_counts.get(class_name, 0) + 1

            color = (16, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f'{class_name} {confidence_score:.2f}'
            label_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x1, y1 - label_size[1] - baseline - 4), (x1 + label_size[0] + 4, y1), color, -1)
            cv2.putText(frame, label, (x1 + 2, y1 - baseline - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    cv2.putText(frame, f'Confidence >= {confidence:.2f}', (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return frame, class_counts


def format_class_counts(counts: dict) -> str:
    if not counts:
        return 'No detections yet.'
    return '\n'.join([f'{name}: {count}' for name, count in sorted(counts.items(), key=lambda item: -item[1])])


st.title('YOLOv8 Streamlit Object Detection')
st.markdown(
    'Use this simple Streamlit app for object detection with a single YOLOv8 script. ' 
    'Choose an image or run the local webcam to see detections live.'
)

with st.sidebar:
    st.header('Settings')
    model_name = st.selectbox('YOLOv8 Model', MODEL_OPTIONS, index=0)
    confidence = st.slider('Confidence Threshold', 0.05, 0.90, 0.35, step=0.05)
    mode = st.radio('Detection Mode', ['Image Upload', 'Webcam'])
    st.markdown('---')
    st.markdown('### Notes')
    st.markdown(
        '- Models download automatically on first run.\n'
        '- Webcam mode uses the local camera connected to your machine.\n'
        '- If the webcam is busy, close other camera applications first.'
    )

model = load_model(model_name)

if mode == 'Image Upload':
    st.subheader('Upload an Image')
    uploaded_file = st.file_uploader('Choose a photo to detect', type=['jpg', 'jpeg', 'png', 'bmp'])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        frame = np.array(image)
        st.image(frame, caption='Input image', use_column_width=True)

        if st.button('Run Detection'):
            with st.spinner('Running YOLOv8...'):
                results = model(frame, conf=confidence, verbose=False)
                annotated, counts = annotate_frame(frame, results, confidence)
                annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                st.image(annotated_rgb, caption='Detected image', use_column_width=True)
                st.markdown('### Detection Summary')
                st.text(format_class_counts(counts))
                st.write(f'**Total objects:** {sum(counts.values())}')

else:
    st.subheader('Local Webcam Detection')
    run_camera = st.checkbox('Start webcam detection')

    if 'camera_active' not in st.session_state:
        st.session_state.camera_active = False
    if 'frame_count' not in st.session_state:
        st.session_state.frame_count = 0
    if 'start_time' not in st.session_state:
        st.session_state.start_time = time.time()

    if run_camera:
        st.session_state.camera_active = True
    else:
        st.session_state.camera_active = False
        st.session_state.frame_count = 0
        st.session_state.start_time = time.time()

    webcam_placeholder = st.empty()
    metrics_placeholder = st.empty()

    if st.session_state.camera_active:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error('Unable to access the webcam. Ensure it is connected and not used by another application.')
        else:
            ret, frame = cap.read()
            if not ret:
                st.error('Could not read a frame from the webcam. Try restarting the app or closing other camera apps.')
            else:
                st.session_state.frame_count += 1
                results = model(frame, conf=confidence, verbose=False)
                annotated, counts = annotate_frame(frame, results, confidence)
                annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                webcam_placeholder.image(annotated_rgb, caption='Webcam feed', use_column_width=True)
                elapsed = max(time.time() - st.session_state.start_time, 1e-3)
                fps = st.session_state.frame_count / elapsed
                metrics_placeholder.markdown(
                    f'**FPS:** {fps:.1f}  \n'
                    f'**Detected objects:** {sum(counts.values())}  \n'
                    f'**Classes:**\n{format_class_counts(counts)}'
                )
            cap.release()
            time.sleep(0.05)
            st.experimental_rerun()
    else:
        st.info('Enable webcam detection to start the live feed.')

st.markdown('---')
st.caption('Streamlit app powered by YOLOv8 and OpenCV.')

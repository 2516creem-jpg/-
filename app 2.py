import cv2
import numpy as np
import streamlit as st
import tempfile
import time
import math

import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python import BaseOptions

# ===============================
# Streamlit UI
# ===============================
st.set_page_config(layout="wide")
st.title("🚶‍♂️ Gait Analysis (MediaPipe + Streamlit)")

uploaded_file = st.file_uploader(
    "📂 เลือกไฟล์วิดีโอ (.mp4)",
    type=["mp4"]
)

if uploaded_file is None:
    st.stop()

# ===============================
# Save uploaded video temporarily
# ===============================
tfile = tempfile.NamedTemporaryFile(delete=False)
tfile.write(uploaded_file.read())
video_path = tfile.name

# ===============================
# Load MediaPipe PoseLandmarker
# ===============================
@st.cache_resource
def load_pose_model():
    base_options = BaseOptions(
        model_asset_path="pose_landmarker_heavy.task"
    )

    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    return vision.PoseLandmarker.create_from_options(options)

pose_landmarker = load_pose_model()

# ===============================
# OpenCV Video
# ===============================
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)

FRAME_WINDOW = st.image([], channels="BGR")

# ===============================
# Drawing utilities (เหมือนเดิม)
# ===============================
mp_drawing = mp.solutions.drawing_utils
mp_pose_style = mp.solutions.drawing_styles
mp_pose_connections = mp.solutions.pose.POSE_CONNECTIONS

timestamp_ms = 0

# ===============================
# Video loop
# ===============================
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # MediaPipe ต้องการ timestamp (ms)
    timestamp_ms += int(1000 / fps)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=frame_rgb
    )

    result = pose_landmarker.detect_for_video(
        mp_image,
        timestamp_ms
    )

    # ===============================
    # Draw landmarks (เหมือน mediapipe เดิม)
    # ===============================
    if result.pose_landmarks:
        for pose_landmarks in result.pose_landmarks:
            landmark_list = mp.framework.formats.landmark_pb2.NormalizedLandmarkList(
                landmark=[
                    mp.framework.formats.landmark_pb2.NormalizedLandmark(
                        x=l.x,
                        y=l.y,
                        z=l.z,
                        visibility=l.visibility
                    )
                    for l in pose_landmarks
                ]
            )

            mp_drawing.draw_landmarks(
                frame,
                landmark_list,
                mp_pose_connections,
                landmark_drawing_spec=mp_pose_style.get_default_pose_landmarks_style()
            )

    FRAME_WINDOW.image(frame, channels="BGR")
    time.sleep(1 / fps)

cap.release()
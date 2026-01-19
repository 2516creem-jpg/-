import cv2
import mediapipe as mp
import math
import os
import numpy as np
import streamlit as st
import tempfile

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
# MediaPipe setup
# ===============================
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ===============================
# Helper functions
# ===============================
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - \
              np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180:
        angle = 360 - angle
    return angle

# ===============================
# Calibration
# ===============================
pixels_per_cm = 5.0

peaks_shoulder, peaks_foot, peaks_knee, peaks_fh = [], [], [], []
temp_max_shoulder = temp_max_foot = temp_max_knee = 0
temp_max_fh_L = temp_max_fh_R = -100
avg_max_shoulder = avg_max_foot = avg_max_knee = avg_max_fh = 0
prev_shoulder = prev_foot = prev_knee = prev_fh_L = prev_fh_R = 0

# ===============================
# Video
# ===============================
cap = cv2.VideoCapture(video_path)
frame_area = st.empty()
info_area = st.empty()

while cap.isOpened():
    success, image = cap.read()
    if not success:
        break

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    h, w, _ = image.shape

    if results.pose_landmarks:
        lm = results.pose_landmarks.landmark

        # Shoulder
        x1, y1 = int(lm[11].x*w), int(lm[11].y*h)
        x2, y2 = int(lm[23].x*w), int(lm[23].y*h)
        dist_sh = abs(x2-x1)

        if dist_sh < prev_shoulder-2 and temp_max_shoulder > 5:
            peaks_shoulder.append(temp_max_shoulder/pixels_per_cm)
            peaks_shoulder = sorted(peaks_shoulder, reverse=True)[:5]
            avg_max_shoulder = sum(peaks_shoulder)/len(peaks_shoulder)
            temp_max_shoulder = 0

        temp_max_shoulder = max(temp_max_shoulder, dist_sh)
        prev_shoulder = dist_sh

        # Knee
        k1 = [lm[23].x*w, lm[23].y*h]
        k2 = [lm[25].x*w, lm[25].y*h]
        k3 = [lm[27].x*w, lm[27].y*h]
        angle = calculate_angle(k1, k2, k3)

        if angle < prev_knee-2 and temp_max_knee > 50:
            peaks_knee.append(temp_max_knee)
            peaks_knee = sorted(peaks_knee, reverse=True)[:5]
            avg_max_knee = sum(peaks_knee)/len(peaks_knee)
            temp_max_knee = 0

        temp_max_knee = max(temp_max_knee, angle)
        prev_knee = angle

        cv2.putText(image, f"Knee: {int(angle)}",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 0), 2)

    # ===============================
    # Show on Streamlit
    # ===============================
    frame_area.image(image, channels="BGR")

    info_area.markdown(f"""
    **AVG Shoulder (cm):** {avg_max_shoulder:.2f}  
    **AVG Knee (deg):** {avg_max_knee:.1f}  
    """)

cap.release()
st.success("✅ ประมวลผลเสร็จแล้ว")
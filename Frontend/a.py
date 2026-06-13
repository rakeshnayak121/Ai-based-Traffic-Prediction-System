import streamlit as st
import cv2
import tempfile
import numpy as np
from ultralytics import YOLO
from norfair import Detection, Tracker

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Traffic Flow Optimization",
    layout="wide"
)

st.title("🚦 AI Traffic Flow Optimization")
st.subheader("Emergency Vehicle Detection System")

# -----------------------------
# LOAD MODELS
# -----------------------------
@st.cache_resource
def load_models():
    emergency_model = YOLO("best_emergency_vehicle_model.pt")
    vehicle_model = YOLO("yolov8n.pt")
    return emergency_model, vehicle_model

try:
    emergency_model, vehicle_model = load_models()
except Exception as e:
    st.error(f"Model Loading Failed: {e}")
    st.stop()

# -----------------------------
# LABELS
# -----------------------------
EMERGENCY_LABELS = [
    "Police Car",
    "Police Van",
    "Fire Truck",
    "Ambulance"
]

# -----------------------------
# FILE UPLOAD
# -----------------------------
uploaded_files = st.file_uploader(
    "Upload up to 4 traffic videos",
    type=["mp4", "avi", "mov", "mkv"],
    accept_multiple_files=True
)

# -----------------------------
# CREATE DETECTIONS
# -----------------------------
def create_detections(results):

    detections = []

    for box in results.boxes:

        x1, y1, x2, y2 = map(int, box.xyxy[0])

        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2

        label = results.names[int(box.cls)]

        detections.append(
            Detection(
                points=np.array([[center_x, center_y]]),
                data={
                    "label": label,
                    "box": (x1, y1, x2, y2)
                }
            )
        )

    return detections

# -----------------------------
# PROCESS VIDEOS
# -----------------------------
if uploaded_files:

    total_clearance_time = 0

    for video_no, uploaded_file in enumerate(uploaded_files[:4], start=1):

        st.markdown(f"## Video {video_no}")

        temp_video = tempfile.NamedTemporaryFile(delete=False)

        temp_video.write(uploaded_file.read())

        cap = cv2.VideoCapture(temp_video.name)

        frame_placeholder = st.empty()

        tracker = Tracker(
            distance_function="euclidean",
            distance_threshold=30
        )

        emergency_ids = set()
        normal_ids = set()

        emergency_detected = False

        while cap.isOpened():

            success, frame = cap.read()

            if not success:
                break

            emergency_results = emergency_model(frame)[0]

            normal_results = vehicle_model(frame)[0]

            detections = (
                create_detections(emergency_results)
                + create_detections(normal_results)
            )

            tracked_objects = tracker.update(detections)

            for obj in tracked_objects:

                data = obj.last_detection.data

                label = data["label"]

                x1, y1, x2, y2 = data["box"]

                if label in EMERGENCY_LABELS:

                    emergency_detected = True

                    emergency_ids.add(obj.id)

                    color = (0, 0, 255)

                else:

                    normal_ids.add(obj.id)

                    color = (0, 255, 0)

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    color,
                    2
                )

                cv2.putText(
                    frame,
                    f"{label} ID:{obj.id}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2
                )

            frame_placeholder.image(
                frame,
                channels="BGR",
                use_container_width=True
            )

        cap.release()

        emergency_count = len(emergency_ids)
        vehicle_count = len(normal_ids)

        clearance_time = max(
            0,
            (vehicle_count - emergency_count) * 3
        )

        total_clearance_time += clearance_time

        if emergency_detected:
            st.warning(
                f"🚨 Emergency vehicle detected in Video {video_no}"
            )

        st.success(
            f"""
            Emergency Vehicles: {emergency_count}

            Non-Emergency Vehicles: {vehicle_count}

            Clearance Time: {clearance_time} sec
            """
        )

    st.markdown(
        f"# Total Clearance Time: {total_clearance_time} sec"
    )
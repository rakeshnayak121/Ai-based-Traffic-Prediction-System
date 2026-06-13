
import cv2
import tempfile
import numpy as np
import pandas as pd
import streamlit as st

from ultralytics import YOLO
from norfair import Detection, Tracker

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------

st.set_page_config(
    page_title="Advanced Traffic Flow Optimization",
    layout="wide"
)

st.title("🚦 Advanced Traffic Flow Optimization")
st.subheader("Intelligent Traffic Monitoring System")

# ---------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------

@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

try:
    model = load_model()
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

# ---------------------------------------------------
# VEHICLE CLASSES
# ---------------------------------------------------

VEHICLE_CLASSES = [
    "car",
    "bus",
    "truck",
    "motorcycle"
]

# ---------------------------------------------------
# FILE UPLOADER
# ---------------------------------------------------

uploaded_files = st.file_uploader(
    "Upload up to 4 Videos",
    type=["mp4", "avi", "mov", "mkv"],
    accept_multiple_files=True
)

# ---------------------------------------------------
# DETECTION CONVERTER
# ---------------------------------------------------

def create_detections(results):

    detections = []

    for box in results.boxes:

        cls_id = int(box.cls[0])
        label = results.names[cls_id]

        if label not in VEHICLE_CLASSES:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])

        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2

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

# ---------------------------------------------------
# PROCESS VIDEOS
# ---------------------------------------------------

if uploaded_files:

    total_clearance_time = 0
    video_results = []

    for index, uploaded_file in enumerate(uploaded_files[:4], start=1):

        st.markdown(f"## 🎥 Video {index}: {uploaded_file.name}")

        temp_video = tempfile.NamedTemporaryFile(delete=False)
        temp_video.write(uploaded_file.read())

        cap = cv2.VideoCapture(temp_video.name)

        frame_placeholder = st.empty()

        tracker = Tracker(
    distance_function="mean_euclidean",
    distance_threshold=30
)

        unique_vehicle_ids = set()

        while cap.isOpened():

            success, frame = cap.read()

            if not success:
                break

            results = model(frame)[0]

            detections = create_detections(results)

            tracked_objects = tracker.update(detections)

            for obj in tracked_objects:

                label = obj.last_detection.data["label"]

                x1, y1, x2, y2 = obj.last_detection.data["box"]

                unique_vehicle_ids.add(obj.id)

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    frame,
                    f"{label} ID:{obj.id}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

            frame_placeholder.image(
                frame,
                channels="BGR",
                use_container_width=True
            )

        cap.release()

        vehicle_count = len(unique_vehicle_ids)

        clearance_time = vehicle_count * 3

        total_clearance_time += clearance_time

        video_results.append(
            {
                "Video Name": uploaded_file.name,
                "Vehicle Count": vehicle_count,
                "Estimated Clearance Time (Seconds)": clearance_time
            }
        )

    # ---------------------------------------------------
    # SUMMARY TABLE
    # ---------------------------------------------------

    st.markdown("## 📊 Detection Summary")

    df = pd.DataFrame(video_results)

    st.dataframe(
        df,
        use_container_width=True
    )

    # ---------------------------------------------------
    # TOTAL CLEARANCE TIME
    # ---------------------------------------------------

    st.markdown(
        f"### ⏱️ Total Road Clearance Time: {total_clearance_time} Seconds"
    )

    # ---------------------------------------------------
    # MOST CONGESTED ROUTE
    # ---------------------------------------------------

    busiest_route = df.loc[df["Vehicle Count"].idxmax()]

    st.warning(
        f"🚨 Route Priority: {busiest_route['Video Name']} "
        f"(Vehicle Count: {busiest_route['Vehicle Count']})"
    )


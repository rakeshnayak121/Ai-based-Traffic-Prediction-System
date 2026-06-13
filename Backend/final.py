
import os
import cv2
import tempfile
import numpy as np
import streamlit as st

from ultralytics import YOLO
from norfair import Detection, Tracker

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------

st.set_page_config(
    page_title="Emergency Vehicle Detection",
    layout="wide"
)

st.title("🚨 Emergency Vehicle Detection System")
st.subheader("AI-Based Traffic Flow Optimization")

# ---------------------------------------------------
# LOAD MODELS
# ---------------------------------------------------

@st.cache_resource
def load_models():

    emergency_model = YOLO(
        "Frontend/best_emergency_vehicle_model.pt"
    )

    vehicle_model = YOLO(
        "Backend/yolov8n.pt"
    )

    return emergency_model, vehicle_model


try:
    emergency_model, vehicle_model = load_models()

except Exception as e:
    st.error(f"Model loading failed: {e}")
    st.stop()

# ---------------------------------------------------
# LABELS
# ---------------------------------------------------

EMERGENCY_LABELS = [
    "Police Car",
    "Police Van",
    "Fire Truck",
    "Ambulance"
]

NON_EMERGENCY_LABELS = [
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
# DETECTION CREATOR
# ---------------------------------------------------

def create_detections(results):

    detections = []

    for box in results.boxes:

        x1, y1, x2, y2 = map(
            int,
            box.xyxy[0]
        )

        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2

        label = results.names[
            int(box.cls[0])
        ]

        conf = float(box.conf[0])

        detections.append(
            Detection(
                points=np.array(
                    [[center_x, center_y]]
                ),
                data={
                    "label": label,
                    "conf": conf,
                    "box": (
                        x1,
                        y1,
                        x2,
                        y2
                    )
                }
            )
        )

    return detections

# ---------------------------------------------------
# MAIN PROCESSING
# ---------------------------------------------------

if uploaded_files:

    total_clearance_time = 0

    for idx, uploaded_file in enumerate(
        uploaded_files[:4],
        start=1
    ):

        st.markdown(
            f"## 🎥 Processing Video {idx}"
        )

        temp_file = tempfile.NamedTemporaryFile(
            delete=False
        )

        temp_file.write(
            uploaded_file.read()
        )

        cap = cv2.VideoCapture(
            temp_file.name
        )

        frame_placeholder = st.empty()

        tracker = Tracker(
            distance_function="mean_euclidean",
            distance_threshold=30
        )

        emergency_detected = False

        emergency_ids = set()
        non_emergency_ids = set()

        while cap.isOpened():

            success, frame = cap.read()

            if not success:
                break

            emergency_results = emergency_model(
                frame
            )[0]

            vehicle_results = vehicle_model(
                frame
            )[0]

            detections = (
                create_detections(
                    emergency_results
                )
                +
                create_detections(
                    vehicle_results
                )
            )

            tracked_objects = tracker.update(
                detections
            )

            for obj in tracked_objects:

                data = obj.last_detection.data

                label = data["label"]

                x1, y1, x2, y2 = data["box"]

                if label in EMERGENCY_LABELS:

                    emergency_ids.add(
                        obj.id
                    )

                    emergency_detected = True

                    color = (0, 0, 255)

                elif label in NON_EMERGENCY_LABELS:

                    non_emergency_ids.add(
                        obj.id
                    )

                    color = (0, 255, 0)

                else:
                    continue

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    color,
                    2
                )

                cv2.putText(
                    frame,
                    f"{label} {obj.id}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    color,
                    2
                )

            frame_placeholder.image(
                frame,
                channels="BGR",
                use_container_width=True
            )

        cap.release()

        emergency_count = len(
            emergency_ids
        )

        non_emergency_count = len(
            non_emergency_ids
        )

        clearance_time = max(
            0,
            (
                non_emergency_count
                -
                emergency_count
            ) * 3
        )

        total_clearance_time += (
            clearance_time
        )

        if emergency_detected:

            st.warning(
                f"🚨 Emergency vehicle detected in Video {idx}"
            )

        st.success(
            f"""
            Emergency Vehicles: {emergency_count}

            Non-Emergency Vehicles: {non_emergency_count}

            Estimated Clearance Time:
            {clearance_time} seconds
            """
        )

    st.markdown(
        f"# ⏱️ Total Clearance Time: {total_clearance_time} seconds"
    )


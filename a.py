import cv2
import torch
import tempfile
from ultralytics import YOLO
import streamlit as st
from norfair import Detection, Tracker
import numpy as np

# ----------------------------- #
# ✅ Streamlit Page Configuration
# ----------------------------- #
st.set_page_config(page_title="AI Traffic Optimization", layout="wide")
st.title("🚦 Advanced Traffic Flow Optimization with Emergency Vehicle Detection")

st.info(
    "Upload up to **4 video files** (MP4, MOV, AVI, MKV).\n\n"
    "The system will detect **emergency** and **non-emergency** vehicles "
    "and estimate road clearance time."
)

# ----------------------------- #
# ✅ Cache and Load Models Once
# ----------------------------- #
@st.cache_resource
def load_models():
    try:
        yolo_v8 = YOLO('best_emergency_vehicle_model.pt')
    except Exception as e:
        st.error(f"Error loading YOLOv8 model: {e}")
        st.stop()

    try:
        yolo_v5 = torch.hub.load('ultralytics/yolov5', 'yolov5s', trust_repo=True)
    except Exception as e:
        st.error(f"Error loading YOLOv5 model: {e}")
        st.stop()

    return yolo_v8, yolo_v5


yolo_v8_emergency, yolo_v5_non_emergency = load_models()

# ----------------------------- #
# ✅ Define Labels
# ----------------------------- #
emergency_labels = ['Police Car', 'Police Van', 'Fire Truck', 'Ambulance']
non_emergency_labels = ['car', 'bus', 'truck', 'motorcycle']

# ----------------------------- #
# ✅ Helper: Convert YOLO Results → Norfair Detections
# ----------------------------- #
def create_detections(results, labels, model_type="yolov8"):
    detections = []
    if model_type == "yolov8":
        if isinstance(results, list):
            results = results[0]

        if hasattr(results, 'boxes'):
            for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                centroid = np.array([[(x1 + x2) / 2, (y1 + y2) / 2]])
                label = labels[int(box.cls)]
                conf = float(box.conf[0].item())

                if label in emergency_labels + non_emergency_labels:
                    detections.append(
                        Detection(
                            centroid,
                            data={"label": label, "conf": conf, "box": (x1, y1, x2, y2)},
                        )
                    )

    elif model_type == "yolov5":
        if hasattr(results, 'xyxy'):
            for result in results.xyxy[0]:
                if len(result) >= 6:
                    x1, y1, x2, y2, conf, cls = result[:6]
                    label = labels[int(cls)]
                    centroid = np.array([[(x1 + x2) / 2, (y1 + y2) / 2]])
                    if label in emergency_labels + non_emergency_labels:
                        detections.append(
                            Detection(
                                centroid,
                                data={"label": label, "conf": float(conf),
                                      "box": (int(x1), int(y1), int(x2), int(y2))},
                            )
                        )
    return detections


# ----------------------------- #
# ✅ File Uploader
# ----------------------------- #
uploaded_files = st.file_uploader(
    "Upload Video Files",
    type=["mp4", "mov", "avi", "mkv"],
    accept_multiple_files=True,
)

# ----------------------------- #
# ✅ Process Each Video
# ----------------------------- #
if uploaded_files:
    total_clearance_time = 0

    for idx, uploaded_file in enumerate(uploaded_files[:4]):
        st.write(f"### 🎥 Processing Video {idx + 1}: `{uploaded_file.name}`")

        # Save to temporary file
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())

        # OpenCV VideoCapture
        cap = cv2.VideoCapture(tfile.name)
        if not cap.isOpened():
            st.error(f"Failed to open video file: {uploaded_file.name}")
            continue

        tracker = Tracker(distance_function="euclidean", distance_threshold=30)
        stframe = st.empty()

        unique_emergency_ids = set()
        unique_non_emergency_ids = set()
        emergency_detected = False

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            # YOLO Inference
            try:
                emergency_results = yolo_v8_emergency(frame)
                non_emergency_results = yolo_v5_non_emergency(frame)
            except Exception as e:
                st.error(f"Model inference failed: {e}")
                break

            # Detections + Tracking
            detections = (
                create_detections(emergency_results, yolo_v8_emergency.names, "yolov8")
                + create_detections(non_emergency_results, yolo_v5_non_emergency.names, "yolov5")
            )
            tracked_objects = tracker.update(detections)

            # Draw detections
            for obj in tracked_objects:
                data = obj.last_detection.data
                label = data["label"]
                x1, y1, x2, y2 = data["box"]

                if label in emergency_labels:
                    unique_emergency_ids.add(obj.id)
                    emergency_detected = True
                    color = (0, 0, 255)
                else:
                    unique_non_emergency_ids.add(obj.id)
                    color = (0, 255, 0)

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    frame,
                    f"{label} ID:{obj.id}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    color,
                    2,
                )

            # Show video frame safely
            if isinstance(frame, np.ndarray) and frame.ndim == 3:
                stframe.image(frame, channels="BGR", use_container_width=True)

        cap.release()

        # ----------------------------- #
        # ✅ Summarize Results Per Video
        # ----------------------------- #
        non_emergency_count = len(unique_non_emergency_ids)
        emergency_count = len(unique_emergency_ids)
        clearance_time = max(0, (non_emergency_count - emergency_count) * 3)
        total_clearance_time += clearance_time

        if emergency_detected:
            st.warning(f"🚨 Emergency vehicles detected in {uploaded_file.name}! Clear the route!")

        st.success(f"**Non-Emergency Vehicles:** {non_emergency_count}")
        st.success(f"**Estimated Road Clearance Time:** {clearance_time} seconds")

    st.write("---")
    st.write(f"## ⏱️ Total Estimated Road Clearance Time: {total_clearance_time} seconds")

else:
    st.info("Please upload at least one video to begin processing.")

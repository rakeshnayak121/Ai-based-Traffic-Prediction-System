import cv2
import torch
import tempfile
from ultralytics import YOLO
import streamlit as st
from norfair import Detection, Tracker
import numpy as np

# --------------------------------------------------
# ✅ Streamlit Page Setup
# --------------------------------------------------
st.set_page_config(page_title="AI Traffic Flow Optimization", layout="wide")
st.title("🚦 Advanced Traffic Flow Optimization with Emergency Vehicle Detection")

st.info(
    "Upload up to **4 videos** (MP4, MOV, AVI, MKV).\n\n"
    "The system detects **emergency** and **non-emergency** vehicles "
    "and estimates total road clearance time."
)

# --------------------------------------------------
# ✅ Cached Model Loading
# --------------------------------------------------
@st.cache_resource
def load_models():
    try:
        yolo_v8 = YOLO("best_emergency_vehicle_model.pt")
    except Exception as e:
        st.error(f"❌ Failed to load YOLOv8 model: {e}")
        st.stop()

    try:
        yolo_v5 = torch.hub.load("ultralytics/yolov5", "yolov5s", trust_repo=True)
    except Exception as e:
        st.error(f"❌ Failed to load YOLOv5 model: {e}")
        st.stop()

    return yolo_v8, yolo_v5


yolo_v8_emergency, yolo_v5_non_emergency = load_models()

# --------------------------------------------------
# ✅ Labels
# --------------------------------------------------
emergency_labels = ["Police Car", "Police Van", "Fire Truck", "Ambulance"]
non_emergency_labels = ["car", "bus", "truck", "motorcycle"]

# --------------------------------------------------
# ✅ Helper: YOLO → Norfair Detections
# --------------------------------------------------
def create_detections(results, labels, model_type="yolov8"):
    detections = []

    if model_type == "yolov8":
        if isinstance(results, list):
            results = results[0]

        if hasattr(results, "boxes"):
            for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                label = labels[int(box.cls)]
                conf = float(box.conf[0])
                centroid = np.array([[(x1 + x2) / 2, (y1 + y2) / 2]])

                if label in emergency_labels + non_emergency_labels:
                    detections.append(
                        Detection(
                            centroid,
                            data={"label": label, "conf": conf, "box": (x1, y1, x2, y2)},
                        )
                    )

    elif model_type == "yolov5":
        if hasattr(results, "xyxy"):
            for r in results.xyxy[0]:
                if len(r) >= 6:
                    x1, y1, x2, y2, conf, cls = r[:6]
                    label = labels[int(cls)]
                    centroid = np.array([[(x1 + x2) / 2, (y1 + y2) / 2]])

                    if label in emergency_labels + non_emergency_labels:
                        detections.append(
                            Detection(
                                centroid,
                                data={
                                    "label": label,
                                    "conf": float(conf),
                                    "box": (int(x1), int(y1), int(x2), int(y2)),
                                },
                            )
                        )

    return detections


# --------------------------------------------------
# ✅ File Upload
# --------------------------------------------------
uploaded_files = st.file_uploader(
    "Upload Video Files",
    type=["mp4", "mov", "avi", "mkv"],
    accept_multiple_files=True,
)

# --------------------------------------------------
# ✅ Main Video Processing
# --------------------------------------------------
if uploaded_files:
    total_clearance_time = 0

    for idx, uploaded_file in enumerate(uploaded_files[:4]):
        st.write(f"### 🎥 Processing Video {idx + 1}: `{uploaded_file.name}`")

        # Write to a temporary file
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())

        cap = cv2.VideoCapture(tfile.name)
        if not cap.isOpened():
            st.error(f"❌ Could not open video file: {uploaded_file.name}")
            continue

        tracker = Tracker(distance_function="euclidean", distance_threshold=30)
        stframe = st.empty()

        unique_emergency_ids = set()
        unique_non_emergency_ids = set()
        emergency_detected = False

        while cap.isOpened():
            ret, frame = cap.read()

            # Validate frame before use
            if not ret or frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
                break

            # Optional: resize to keep performance stable
            frame = cv2.resize(frame, (640, 360))

            try:
                emergency_results = yolo_v8_emergency(frame)
                non_emergency_results = yolo_v5_non_emergency(frame)
            except Exception as e:
                st.error(f"Model inference failed: {e}")
                break

            detections = (
                create_detections(emergency_results, yolo_v8_emergency.names, "yolov8")
                + create_detections(non_emergency_results, yolo_v5_non_emergency.names, "yolov5")
            )
            tracked_objects = tracker.update(detections)

            for obj in tracked_objects:
                data = obj.last_detection.data
                label = data["label"]
                x1, y1, x2, y2 = data["box"]

                if label in emergency_labels:
                    color = (0, 0, 255)
                    unique_emergency_ids.add(obj.id)
                    emergency_detected = True
                else:
                    color = (0, 255, 0)
                    unique_non_emergency_ids.add(obj.id)

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    frame,
                    f"{label} ID:{obj.id}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    color,
                    2,
                )

            # Safe rendering guard
            if isinstance(frame, np.ndarray) and frame.ndim == 3 and frame.size > 0:
                stframe.image(frame, channels="BGR", use_container_width=True)
            else:
                continue

        cap.release()

        # --------------------------------------------------
        # ✅ Summary for Each Video
        # --------------------------------------------------
        non_emergency_count = len(unique_non_emergency_ids)
        emergency_count = len(unique_emergency_ids)
        clearance_time = max(0, (non_emergency_count - emergency_count) * 3)
        total_clearance_time += clearance_time

        if emergency_detected:
            st.warning(f"🚨 Emergency vehicles detected in {uploaded_file.name}! Clear the route!")

        st.success(f"**Non-Emergency Vehicles:** {non_emergency_count}")
        st.success(f"**Estimated Clearance Time:** {clearance_time} seconds")

    # --------------------------------------------------
    # ✅ Final Total
    # --------------------------------------------------
    st.write("---")
    st.subheader(f"⏱️ Total Estimated Road Clearance Time: {total_clearance_time} seconds")

else:
    st.info("Please upload one or more videos to start processing.")


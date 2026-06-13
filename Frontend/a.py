import cv2
import torch
import tempfile
import numpy as np
import streamlit as st
from ultralytics import YOLO
from norfair import Detection, Tracker

# Load models
yolo_v8_emergency = YOLO('best_emergency_vehicle_model.pt')  # Emergency model
yolo_v5_non_emergency = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # Non-emergency model

# Label definitions
emergency_labels = ['Police Car', 'Police Van', 'Fire Truck', 'Ambulance']
non_emergency_labels = ['car', 'bus', 'truck', 'motorcycle']

# Streamlit UI
st.title("🚦 Advanced Traffic Flow Optimization")
st.subheader("Emergency Vehicle Detection System")

uploaded_files = st.file_uploader(
    "Upload up to 4 Videos", type=["mp4", "mov", "avi", "mkv"], accept_multiple_files=True
)

# Detection conversion for Norfair tracking
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
                conf = box.conf[0].item()
                if label in emergency_labels + non_emergency_labels:
                    detections.append(
                        Detection(
                            centroid,
                            data={"label": label, "conf": conf, "box": (x1, y1, x2, y2)}
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
                                data={"label": label, "conf": conf, "box": (int(x1), int(y1), int(x2), int(y2))}
                            )
                        )
    return detections

# Main logic
if uploaded_files:
    total_clearance_time = 0

    for idx, uploaded_file in enumerate(uploaded_files[:4]):
        st.write(f"### 🎥 Processing Video {idx + 1}: `{uploaded_file.name}`")

        # Temp file
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        cap = cv2.VideoCapture(tfile.name)
        stframe = st.empty()

        emergency_detected = False
        unique_emergency_ids = set()
        unique_non_emergency_ids = set()

        tracker = Tracker(distance_function="euclidean", distance_threshold=30)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Model inference
            emergency_results = yolo_v8_emergency(frame)
            non_emergency_results = yolo_v5_non_emergency(frame)

            # Detection processing
            detections = create_detections(emergency_results, yolo_v8_emergency.names, "yolov8") + \
                         create_detections(non_emergency_results, yolo_v5_non_emergency.names, "yolov5")

            tracked_objects = tracker.update(detections)

            # Draw bounding boxes
            for obj in tracked_objects:
                label = obj.last_detection.data["label"]
                x1, y1, x2, y2 = obj.last_detection.data["box"]

                if label in emergency_labels:
                    if obj.id not in unique_emergency_ids:
                        unique_emergency_ids.add(obj.id)
                        emergency_detected = True

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(frame, f'{label} {obj.id}', (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

                elif label in non_emergency_labels:
                    if obj.id not in unique_non_emergency_ids:
                        unique_non_emergency_ids.add(obj.id)

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f'{label} {obj.id}', (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            stframe.image(frame, channels="BGR", use_container_width=True)

        cap.release()

        # Video summary
        if emergency_detected:
            st.warning(f"🚨 Emergency vehicle detected in Video {idx + 1}! Take action to clear the path.")

        non_emergency_count = len(unique_non_emergency_ids)
        emergency_count = len(unique_emergency_ids)
        clearance_time = max(0, (non_emergency_count - emergency_count) * 3)
        total_clearance_time += clearance_time

        st.markdown(f"### ✅ Results for Video {idx + 1}")
        st.write(f"• Final Non-Emergency Vehicles: {non_emergency_count}")
        st.write(f"• Estimated Clearance Time: **{clearance_time} seconds**")

    st.markdown(f"## 🧮 Total Clearance Time Across All Videos: **{total_clearance_time} seconds**")

import streamlit as st
import cv2
import torch
import tempfile
from ultralytics import YOLO
from norfair import Detection, Tracker
import numpy as np
import pandas as pd

# Load YOLOv5 model for non-emergency vehicles
yolo_v5_non_emergency = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # Pre-trained YOLOv5 model

# Define labels for non-emergency vehicles
non_emergency_labels = ['car', 'bus', 'truck', 'motorcycle']

# Streamlit App
st.set_page_config(page_title="AI-Based Traffic Flow Optimization", layout="wide")
st.title("🚦 Advanced Traffic Flow Optimization for Intelligent Traffic System")

st.markdown("""
### Instructions:
1. Upload up to **4 videos** showing traffic or emergency vehicles.  
2. The system will analyze each video using YOLOv5 and display detection results.  
3. The app calculates and displays **road clearance time** for better traffic management.
""")

# File uploader for multiple videos
uploaded_files = st.file_uploader(
    "Upload up to 4 Videos", type=["mp4", "mov", "avi", "mkv"], accept_multiple_files=True
)

def create_detections(results, labels, model_type="yolov5"):
    """Convert YOLO detection results to Norfair detections for tracking."""
    detections = []
    if model_type == "yolov5":
        if hasattr(results, 'xyxy'):
            for result in results.xyxy[0]:
                if len(result) >= 6:
                    x1, y1, x2, y2, conf, cls = result[:6]
                    label = labels[int(cls)]
                    centroid = np.array([[(x1 + x2) / 2, (y1 + y2) / 2]])
                    if label in non_emergency_labels:
                        detections.append(
                            Detection(
                                centroid,
                                data={"label": label, "conf": conf, "box": (int(x1), int(y1), int(x2), int(y2))}
                            )
                        )
    return detections

if uploaded_files:
    total_clearance_time = 0
    video_results = []

    for idx, uploaded_file in enumerate(uploaded_files[:4]):  # Process up to 4 videos
        st.write(f"### Processing Video {idx + 1}: {uploaded_file.name}")

        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())

        cap = cv2.VideoCapture(tfile.name)
        stframe = st.empty()

        unique_non_emergency_ids = set()
        tracker = Tracker(distance_function="euclidean", distance_threshold=30)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            non_emergency_results = yolo_v5_non_emergency(frame)
            detections = create_detections(non_emergency_results, yolo_v5_non_emergency.names, model_type="yolov5")
            tracked_objects = tracker.update(detections)

            for obj in tracked_objects:
                label = obj.last_detection.data["label"]
                x1, y1, x2, y2 = obj.last_detection.data["box"]

                if label in non_emergency_labels:
                    if obj.id not in unique_non_emergency_ids:
                        unique_non_emergency_ids.add(obj.id)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f'{label} {obj.id}', (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            stframe.image(frame, channels="BGR", use_container_width=True)

        cap.release()

        non_emergency_count = len(unique_non_emergency_ids)
        clearance_time = non_emergency_count * 3  # 3 sec per vehicle
        total_clearance_time += clearance_time

        video_results.append({
            "Video Name": uploaded_file.name,
            "Vehicle Count": non_emergency_count,
            "Estimated Road Clearance Time (seconds)": clearance_time
        })

    # Create summary table
    video_df = pd.DataFrame(video_results)
    st.write("### 📊 Video Detection Summary")
    st.table(video_df)

    st.write(f"### 🕒 Total Road Clearance Time for All Videos: **{total_clearance_time} seconds**")

    # Highlight the most congested route
    max_vehicle_video = video_df.loc[video_df["Vehicle Count"].idxmax()]
    st.success(f"🚧 Route to Clear First: **{max_vehicle_video['Video Name']}** — {max_vehicle_video['Vehicle Count']} vehicles")

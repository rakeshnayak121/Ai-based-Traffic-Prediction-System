
import cv2
import torch
import tempfile
from ultralytics import YOLO
import streamlit as st
from norfair import Detection, Tracker
import numpy as np
import torch.amp

# Load the YOLOv8 model for emergency vehicles
yolo_v8_emergency = YOLO('best_emergency_vehicle_model.pt')  # Adjust path if needed
# Load the YOLOv5 model for non-emergency vehicles
yolo_v5_non_emergency = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # Pre-trained YOLOv5 model

# Define labels for emergency and non-emergency vehicles
emergency_labels = ['Police Car', 'Police Van', 'Fire Truck', 'Ambulance']
non_emergency_labels = ['car', 'bus', 'truck', 'motorcycle']

st.title("ADVANCED TRAFFIC FLOW OPTIMIZATION FOR INTELLIGENT TRAFFIC SYSTEM - Emergency Vehicle Detection")
st.info(
    "👉 **Upload an image or video of any emergency vehicle to get the detection results.**\n\n"
    "Supported formats: MP4, MOV, AVI, MKV, JPEG, PNG | Max 200 MB per file"
)


# File uploader for multiple videos
uploaded_files = st.file_uploader(
    "Upload up to 4 Videos", type=["mp4", "mov", "avi", "mkv"], accept_multiple_files=True
)

def create_detections(results, labels, model_type="yolov8"):
    """Convert YOLO detection results to Norfair detections for tracking."""
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

if uploaded_files:
    for idx, uploaded_file in enumerate(uploaded_files[:4]):
        st.write(f"### Processing Video {idx + 1}: {uploaded_file.name}")
        total_clearance_time = 0
        
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

            emergency_results = yolo_v8_emergency(frame)
            non_emergency_results = yolo_v5_non_emergency(frame)
            
            detections = create_detections(emergency_results, yolo_v8_emergency.names, model_type="yolov8") + \
                         create_detections(non_emergency_results, yolo_v5_non_emergency.names, model_type="yolov5")
            
            tracked_objects = tracker.update(detections)

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

        if emergency_detected:
            st.warning(f"🚨 Emergency vehicle detected in Video {idx + 1}. Please clear the road!")

        non_emergency_count = len(unique_non_emergency_ids)
        emergency_count = len(unique_emergency_ids)
        clearance_time = max(0, (non_emergency_count - emergency_count) * 3)
        total_clearance_time += clearance_time 

        st.write(f"### Results for Video {idx + 1}")
        st.write(f"Final Non-Emergency Vehicles: {non_emergency_count}")
        st.write(f"Estimated Road Clearance Time: {clearance_time} seconds")

    st.write(f"### Total Road Clearance Time for All Videos: {total_clearance_time} seconds")




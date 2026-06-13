import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
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

st.title("Vehicle Detection with ByteTrack - Multiple Videos")

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
                
                # Check if the label matches any emergency or non-emergency vehicle
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
    for idx, uploaded_file in enumerate(uploaded_files[:4]):  # Process up to 4 videos
        st.write(f"### Processing Video {idx + 1}: {uploaded_file.name}")
        
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())

        cap = cv2.VideoCapture(tfile.name)
        stframe = st.empty()
        emergency_detected = False  # Reset for each video

        # Unique IDs for each video
        unique_emergency_ids = set()
        unique_non_emergency_ids = set()

        # Initialize ByteTrack tracker
        tracker = Tracker(distance_function="euclidean", distance_threshold=30)

        # Process each frame of the video
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Run YOLOv8 model for emergency vehicle detection
            emergency_results = yolo_v8_emergency(frame)
            # Run YOLOv5 model for non-emergency vehicle detection
            non_emergency_results = yolo_v5_non_emergency(frame)
            
            # Generate detections from both models
            detections = create_detections(emergency_results, yolo_v8_emergency.names, model_type="yolov8") + \
                         create_detections(non_emergency_results, yolo_v5_non_emergency.names, model_type="yolov5")
            
            # Update tracked objects
            tracked_objects = tracker.update(detections)

            # Draw bounding boxes on the frame
            for obj in tracked_objects:
                label = obj.last_detection.data["label"]
                x1, y1, x2, y2 = obj.last_detection.data["box"]

                # If it’s an emergency vehicle
                if label in emergency_labels:
                    if obj.id not in unique_emergency_ids:
                        unique_emergency_ids.add(obj.id)
                        emergency_detected = True  # Trigger notification if emergency detected
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(frame, f'{label} {obj.id}', (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

                # If it’s a non-emergency vehicle
                elif label in non_emergency_labels:
                    if obj.id not in unique_non_emergency_ids:
                        unique_non_emergency_ids.add(obj.id)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f'{label} {obj.id}', (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # Update the frame in Streamlit
            stframe.image(frame, channels="BGR", use_container_width=True)

        cap.release()

        # After the video ends, display the results
        if emergency_detected:
            st.warning(f"🚨 Emergency vehicle detected in Video {idx + 1}. Please clear the road!")

        # Calculate and display road clearance time for this video
        non_emergency_count = len(unique_non_emergency_ids)
        emergency_count = len(unique_emergency_ids)
        clearance_time = max(0, (non_emergency_count - emergency_count) * 3)

        st.write(f"### Results for Video {idx + 1}")
        st.write(f"Final Non-Emergency Vehicles: {non_emergency_count}")
        st.write(f"Estimated Road Clearance Time: {clearance_time} seconds")

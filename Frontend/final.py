import cv2
import torch
import tempfile
from ultralytics import YOLO
import streamlit as st
from norfair import Detection, Tracker
import numpy as np
import pandas as pd

# Load the YOLOv5 model for non-emergency vehicles
yolo_v5_non_emergency = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # Pre-trained YOLOv5 model

# Define labels for non-emergency vehicles
non_emergency_labels = ['car', 'bus', 'truck', 'motorcycle']

st.title("Advanced Traffic Flow Optimization for Intelligent Traffic System")

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
    total_clearance_time = 0  # Initialize total clearance time
    video_results = []  # Store the results for the table

    for idx, uploaded_file in enumerate(uploaded_files[:4]):  # Process up to 4 videos
        st.write(f"### Processing Video {idx + 1}: {uploaded_file.name}")
        
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())

        cap = cv2.VideoCapture(tfile.name)
        stframe = st.empty()

        # Initialize the vehicle ID tracking set
        unique_non_emergency_ids = set()

        # Initialize ByteTrack tracker
        tracker = Tracker(distance_function="euclidean", distance_threshold=30)

        # Process each frame of the video
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Run YOLOv5 model for non-emergency vehicle detection
            non_emergency_results = yolo_v5_non_emergency(frame)
            
            # Generate detections from YOLOv5 model
            detections = create_detections(non_emergency_results, yolo_v5_non_emergency.names, model_type="yolov5")
            
            # Update tracked objects
            tracked_objects = tracker.update(detections)

            # Draw bounding boxes on the frame
            for obj in tracked_objects:
                label = obj.last_detection.data["label"]
                x1, y1, x2, y2 = obj.last_detection.data["box"]

                # If it’s a non-emergency vehicle
                if label in non_emergency_labels:
                    if obj.id not in unique_non_emergency_ids:
                        unique_non_emergency_ids.add(obj.id)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f'{label} {obj.id}', (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # Update the frame in Streamlit
            stframe.image(frame, channels="BGR", use_container_width=True)

        cap.release()

        # Calculate and display road clearance time for this video
        non_emergency_count = len(unique_non_emergency_ids)
        # Use the max vehicle count to determine the clearance time
        clearance_time = non_emergency_count * 3  # Each vehicle requires 3 seconds for clearance
        total_clearance_time += clearance_time 

        video_results.append({
            "Video Name": uploaded_file.name,
            "Vehicle Count": non_emergency_count,
            "Estimated Road Clearance Time (seconds)": clearance_time
        })

    # Create a DataFrame for displaying the table
    video_df = pd.DataFrame(video_results)
    st.write("### Video Detection Summary")
    st.table(video_df)

    # Display the total road clearance time for all videos
    st.write(f"### Total Road Clearance Time for All Videos: {total_clearance_time} seconds")

    # Find the video with the maximum vehicle count
    max_vehicle_video = video_df.loc[video_df["Vehicle Count"].idxmax()]
    st.write(f"### Route to Clear: {max_vehicle_video['Video Name']} (Max Vehicle Count: {max_vehicle_video['Vehicle Count']})")

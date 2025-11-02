import os
import tempfile
import cv2
import torch
import streamlit as st
import numpy as np
from ultralytics import YOLO
from norfair import Detection, Tracker

st.set_page_config(page_title="AI Traffic Flow Optimization", layout="wide")
st.title("🚦 Advanced Traffic Flow Optimization - Emergency Vehicle Detection")

@st.cache_resource
def load_models(yolov8_path="best_emergency_vehicle_model.pt"):
    try:
        yv8 = YOLO(yolov8_path)
        yv5 = torch.hub.load("ultralytics/yolov5", "yolov5s", trust_repo=True)
        return yv8, yv5
    except Exception as e:
        st.error(f"Model load error: {e}")
        st.stop()

yolo_v8_emergency, yolo_v5_non_emergency = load_models()

emergency_labels = ["Police Car", "Police Van", "Fire Truck", "Ambulance"]
non_emergency_labels = ["car", "bus", "truck", "motorcycle"]
ALL_TARGETS = emergency_labels + non_emergency_labels

def create_detections(results, labels, model_type="yolov8"):
    detections = []
    if results is None:
        return detections
    if model_type == "yolov8":
        if isinstance(results, list):
            results = results[0]
        if hasattr(results, "boxes"):
            for box in results.boxes:
                try:
                    xy = box.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = [int(v) for v in xy.tolist()]
                    cls_idx = int(box.cls[0].item()) if hasattr(box, "cls") else int(box.cls)
                    conf = float(box.conf[0].item()) if hasattr(box, "conf") else float(box.conf)
                    label = labels[cls_idx] if labels and cls_idx < len(labels) else str(cls_idx)
                    if label in ALL_TARGETS:
                        centroid = np.array([[(x1 + x2) / 2.0, (y1 + y2) / 2.0]])
                        detections.append(Detection(centroid, data={"label": label, "conf": conf, "box": (x1, y1, x2, y2)}))
                except Exception:
                    continue
    elif model_type == "yolov5":
        if hasattr(results, "xyxy"):
            for r in results.xyxy[0]:
                if len(r) >= 6:
                    x1, y1, x2, y2, conf, cls = r[:6]
                    label = labels[int(cls.item())] if labels and int(cls.item()) < len(labels) else str(int(cls.item()))
                    if label in ALL_TARGETS:
                        centroid = np.array([[(x1 + x2) / 2.0, (y1 + y2) / 2.0]])
                        detections.append(Detection(centroid, data={"label": label, "conf": float(conf.item()), "box": (int(x1.item()), int(y1.item()), int(x2.item()), int(y2.item()))}))
    return detections

uploaded_files = st.file_uploader("Upload up to 4 videos", type=["mp4", "mov", "avi", "mkv"], accept_multiple_files=True)

if uploaded_files:
    total_clearance_time = 0
    for idx, uploaded in enumerate(uploaded_files[:4]):
        st.markdown(f"### Video {idx+1}: `{uploaded.name}`")
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded.name)[1]) as tf:
            tf.write(uploaded.read())
            temp_path = tf.name

        cap = cv2.VideoCapture(temp_path)
        if not cap.isOpened():
            st.error(f"Failed to open {uploaded.name}")
            continue

        stframe = st.empty()
        tracker = Tracker(distance_function="euclidean", distance_threshold=30)
        unique_emergency_ids, unique_non_emergency_ids = set(), set()
        emergency_detected = False

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            if not isinstance(frame, np.ndarray) or frame.size == 0:
                continue

            if frame.ndim == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            frame = cv2.resize(frame, (640, 360))

            try:
                emergency_results = yolo_v8_emergency(frame)
                non_emergency_results = yolo_v5_non_emergency(frame)
            except Exception as e:
                st.error(f"Inference failed: {e}")
                break

            detections = create_detections(emergency_results, yolo_v8_emergency.names, "yolov8") + \
                         create_detections(non_emergency_results, yolo_v5_non_emergency.names, "yolov5")

            tracked_objects = tracker.update(detections)
            for obj in tracked_objects:
                data = obj.last_detection.data
                label, (x1, y1, x2, y2) = data["label"], data["box"]
                color = (0, 0, 255) if label in emergency_labels else (0, 255, 0)
                if label in emergency_labels:
                    unique_emergency_ids.add(obj.id)
                    emergency_detected = True
                else:
                    unique_non_emergency_ids.add(obj.id)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{label} ID:{obj.id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            try:
                stframe.image(frame_rgb, width=640)
            except TypeError:
                stframe.image(frame_rgb)

        cap.release()
        os.remove(temp_path)

        non_emergency_count = len(unique_non_emergency_ids)
        emergency_count = len(unique_emergency_ids)
        clearance_time = max(0, (non_emergency_count - emergency_count) * 3)
        total_clearance_time += clearance_time

        if emergency_detected:
            st.warning(f"🚨 Emergency vehicle(s) detected in `{uploaded.name}`")

        st.write(f"**Non-Emergency Vehicles:** {non_emergency_count}")
        st.write(f"**Clearance Time:** {clearance_time} seconds")

    st.markdown("---")
    st.subheader(f"Total Estimated Road Clearance Time: {total_clearance_time} seconds")
else:
    st.info("Please upload at least one video file.")

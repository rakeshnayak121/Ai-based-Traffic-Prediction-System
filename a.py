import cv2
import torch
import tempfile
from ultralytics import YOLO
import streamlit as st
from norfair import Detection, Tracker
import numpy as np
import os

# -------------------------
# Streamlit page setup
# -------------------------
st.set_page_config(page_title="AI Traffic Flow Optimization", layout="wide")
st.title("🚦 Advanced Traffic Flow Optimization - Emergency Vehicle Detection")
st.caption("Upload up to 4 videos. The app will detect emergency and non-emergency vehicles and estimate road clearance time.")

# -------------------------
# Models (cached)
# -------------------------
@st.cache_resource
def load_models(yolov8_path="best_emergency_vehicle_model.pt"):
    # Load YOLOv8 (custom) and YOLOv5 (pretrained) once and cache the models
    try:
        yv8 = YOLO(yolov8_path)
    except Exception as e:
        st.error(f"Failed to load YOLOv8 model from '{yolov8_path}': {e}")
        st.stop()

    try:
        # trust_repo=True avoids some hub warnings; adjust if your environment blocks hub
        yv5 = torch.hub.load("ultralytics/yolov5", "yolov5s", trust_repo=True)
    except Exception as e:
        st.error(f"Failed to load YOLOv5 model (torch.hub): {e}")
        st.stop()

    return yv8, yv5

yolo_v8_emergency, yolo_v5_non_emergency = load_models()

# -------------------------
# Labels
# -------------------------
emergency_labels = ['Police Car', 'Police Van', 'Fire Truck', 'Ambulance']
non_emergency_labels = ['car', 'bus', 'truck', 'motorcycle']

# -------------------------
# Helper: convert detections
# -------------------------
def create_detections(results, labels, model_type="yolov8"):
    detections = []

    if model_type == "yolov8":
        if isinstance(results, list):
            results = results[0]

        if hasattr(results, "boxes"):
            for box in results.boxes:
                try:
                    # box.xyxy may be a tensor, convert safely
                    xy = box.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = map(int, xy.tolist())
                    cls_idx = int(box.cls[0].item()) if hasattr(box, "cls") else int(box.cls)
                    conf = float(box.conf[0].item()) if hasattr(box, "conf") else float(box.conf)
                    label = labels[cls_idx] if labels is not None and cls_idx < len(labels) else str(cls_idx)
                except Exception:
                    continue

                centroid = np.array([[(x1 + x2) / 2.0, (y1 + y2) / 2.0]])
                if label in emergency_labels + non_emergency_labels:
                    detections.append(Detection(centroid, data={"label": label, "conf": conf, "box": (x1, y1, x2, y2)}))

    elif model_type == "yolov5":
        if hasattr(results, "xyxy"):
            try:
                xyxy_tensor = results.xyxy[0]  # tensor of boxes for the frame
                for r in xyxy_tensor:
                    if len(r) >= 6:
                        x1, y1, x2, y2, conf, cls = r[:6]
                        x1, y1, x2, y2 = int(x1.item()), int(y1.item()), int(x2.item()), int(y2.item())
                        conf = float(conf.item())
                        cls_idx = int(cls.item())
                        label = labels[cls_idx] if labels is not None and cls_idx < len(labels) else str(cls_idx)
                        centroid = np.array([[(x1 + x2) / 2.0, (y1 + y2) / 2.0]])
                        if label in emergency_labels + non_emergency_labels:
                            detections.append(Detection(centroid, data={"label": label, "conf": conf, "box": (x1, y1, x2, y2)}))
            except Exception:
                # if results.xyxy not structured as expected, skip gracefully
                pass

    return detections

# -------------------------
# File uploader
# -------------------------
uploaded_files = st.file_uploader("Upload up to 4 videos (mp4, mov, avi, mkv)", type=["mp4", "mov", "avi", "mkv"], accept_multiple_files=True)

# -------------------------
# Main processing
# -------------------------
if uploaded_files:
    total_clearance_time = 0

    for idx, uploaded_file in enumerate(uploaded_files[:4]):
        st.markdown(f"### Processing Video {idx+1}: `{uploaded_file.name}`")
        # write uploaded file to temp file for OpenCV
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as t:
            t.write(uploaded_file.read())
            temp_path = t.name

        cap = cv2.VideoCapture(temp_path)
        if not cap.isOpened():
            st.error(f"Could not open video file: {uploaded_file.name}")
            os.remove(temp_path)
            continue

        stframe = st.empty()
        tracker = Tracker(distance_function="euclidean", distance_threshold=30)
        unique_emergency_ids = set()
        unique_non_emergency_ids = set()
        emergency_detected = False

        frame_index = 0
        while cap.isOpened():
            ret, frame = cap.read()
            frame_index += 1

            # Strict validation of frame
            if not ret or frame is None:
                # End of file or read error
                break
            if not isinstance(frame, np.ndarray):
                st.warning(f"Skipping non-array frame at index {frame_index}. Type: {type(frame)}")
                continue
            if frame.size == 0:
                st.warning(f"Skipping empty frame at index {frame_index}.")
                continue
            if frame.ndim not in (2, 3):
                st.warning(f"Skipping frame with invalid ndim {frame.ndim} at index {frame_index}.")
                continue

            # If grayscale, convert to BGR for model/display
            if frame.ndim == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

            # Optional: resize to stabilize memory and processing speed
            try:
                frame = cv2.resize(frame, (640, 360))
            except Exception:
                # if resize fails, continue with original (but still display guard below)
                pass

            # Run detection safely
            try:
                emergency_results = yolo_v8_emergency(frame)    # Ultralytics YOLOv8
            except Exception as e:
                st.error(f"YOLOv8 inference error on frame {frame_index}: {e}")
                # stop processing this video (model inference failing means nothing useful will come next)
                break

            try:
                non_emergency_results = yolo_v5_non_emergency(frame)  # YOLOv5 via torch.hub
            except Exception as e:
                st.error(f"YOLOv5 inference error on frame {frame_index}: {e}")
                break

            # Convert to Norfair detections and update tracker
            detections = create_detections(emergency_results, yolo_v8_emergency.names, model_type="yolov8") + \
                         create_detections(non_emergency_results, yolo_v5_non_emergency.names, model_type="yolov5")

            try:
                tracked_objects = tracker.update(detections)
            except Exception:
                # if tracker fails for some reason, continue after warning
                st.warning("Tracker update failed for a frame; continuing.")
                tracked_objects = []

            # Draw boxes & labels
            for obj in tracked_objects:
                try:
                    data = obj.last_detection.data
                    label = data["label"]
                    x1, y1, x2, y2 = data["box"]
                except Exception:
                    continue

                if label in emergency_labels:
                    color = (0, 0, 255)
                    if obj.id not in unique_emergency_ids:
                        unique_emergency_ids.add(obj.id)
                        emergency_detected = True
                else:
                    color = (0, 255, 0)
                    if obj.id not in unique_non_emergency_ids:
                        unique_non_emergency_ids.add(obj.id)

                # Make sure box coords are ints and within frame boundaries
                try:
                    h, w = frame.shape[:2]
                    x1c, y1c = max(0, int(x1)), max(0, int(y1))
                    x2c, y2c = min(w - 1, int(x2)), min(h - 1, int(y2))
                    cv2.rectangle(frame, (x1c, y1c), (x2c, y2c), color, 2)
                    cv2.putText(frame, f"{label} ID:{obj.id}", (x1c, max(0, y1c - 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                except Exception:
                    continue

            # Final safety guard before display
            if isinstance(frame, np.ndarray) and frame.ndim == 3 and frame.shape[2] == 3 and frame.size > 0:
                stframe.image(frame, channels="BGR", use_container_width=True)
            else:
                st.warning(f"Skipped displaying invalid frame at index {frame_index}: type={type(frame)}, shape={getattr(frame, 'shape', None)}")

        cap.release()
        # remove temporary file
        try:
            os.remove(temp_path)
        except Exception:
            pass

        # Summarize results for this video
        non_emergency_count = len(unique_non_emergency_ids)
        emergency_count = len(unique_emergency_ids)
        clearance_time = max(0, (non_emergency_count - emergency_count) * 3)
        total_clearance_time += clearance_time

        if emergency_detected:
            st.warning(f"🚨 Emergency vehicle(s) detected in `{uploaded_file.name}` — please clear the route!")

        st.write(f"**Non-Emergency Vehicles (unique):** {non_emergency_count}")
        st.write(f"**Estimated Road Clearance Time (this video):** {clearance_time} seconds")

    st.markdown("---")
    st.subheader(f"Total Estimated Road Clearance Time (all videos): {total_clearance_time} seconds")

else:
    st.info("Please upload at least one video file to begin processing.")


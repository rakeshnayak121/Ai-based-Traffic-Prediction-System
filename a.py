import os
import tempfile
import cv2
import torch
import streamlit as st
import numpy as np
from ultralytics import YOLO
from norfair import Detection, Tracker

# -------------------------
# Page setup
# -------------------------
st.set_page_config(page_title="AI Traffic Flow Optimization", layout="wide")
st.title("🚦 Advanced Traffic Flow Optimization - Emergency Vehicle Detection")
st.caption("Upload up to 4 videos (mp4, mov, avi, mkv).")

# -------------------------
# Model loading (cached)
# -------------------------
@st.cache_resource
def load_models(yolov8_path: str = "best_emergency_vehicle_model.pt"):
    try:
        yv8 = YOLO(yolov8_path)
    except Exception as e:
        st.error(f"Failed to load YOLOv8 model from '{yolov8_path}': {e}")
        st.stop()

    try:
        yv5 = torch.hub.load("ultralytics/yolov5", "yolov5s", trust_repo=True)
    except Exception as e:
        st.error(f"Failed to load YOLOv5 model (torch.hub): {e}")
        st.stop()

    return yv8, yv5

yolo_v8_emergency, yolo_v5_non_emergency = load_models()

# -------------------------
# Labels
# -------------------------
emergency_labels = ["Police Car", "Police Van", "Fire Truck", "Ambulance"]
non_emergency_labels = ["car", "bus", "truck", "motorcycle"]
ALL_TARGETS = emergency_labels + non_emergency_labels

# -------------------------
# Convert model results -> Norfair detections
# -------------------------
def create_detections(results, labels, model_type="yolov8"):
    detections = []
    if results is None:
        return detections

    if model_type == "yolov8":
        # ultralytics YOLO returns a Results object or list of Results
        if isinstance(results, list):
            results = results[0]

        if hasattr(results, "boxes"):
            for box in results.boxes:
                try:
                    # box.xyxy is tensor-like
                    xy = box.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = [int(v) for v in xy.tolist()]
                    # class index might be tensor-like
                    cls_idx = int(box.cls[0].item()) if hasattr(box, "cls") else int(box.cls)
                    conf = float(box.conf[0].item()) if hasattr(box, "conf") else float(box.conf)
                    label = labels[cls_idx] if labels and cls_idx < len(labels) else str(cls_idx)
                except Exception:
                    continue

                if label in ALL_TARGETS:
                    centroid = np.array([[(x1 + x2) / 2.0, (y1 + y2) / 2.0]])
                    detections.append(Detection(centroid, data={"label": label, "conf": conf, "box": (x1, y1, x2, y2)}))

    elif model_type == "yolov5":
        # torch.hub yolov5 returns a Results object with .xyxy (tensor)
        if hasattr(results, "xyxy"):
            try:
                rows = results.xyxy[0]
                for r in rows:
                    if len(r) >= 6:
                        x1, y1, x2, y2, conf, cls = r[:6]
                        x1, y1, x2, y2 = int(x1.item()), int(y1.item()), int(x2.item()), int(y2.item())
                        conf = float(conf.item())
                        cls_idx = int(cls.item())
                        label = labels[cls_idx] if labels and cls_idx < len(labels) else str(cls_idx)
                        if label in ALL_TARGETS:
                            centroid = np.array([[(x1 + x2) / 2.0, (y1 + y2) / 2.0]])
                            detections.append(Detection(centroid, data={"label": label, "conf": conf, "box": (x1, y1, x2, y2)}))
            except Exception:
                pass

    return detections

# -------------------------
# File uploader
# -------------------------
uploaded_files = st.file_uploader("Upload up to 4 videos", type=["mp4", "mov", "avi", "mkv"], accept_multiple_files=True)

# -------------------------
# Processing
# -------------------------
if uploaded_files:
    total_clearance_time = 0

    for idx, uploaded in enumerate(uploaded_files[:4]):
        st.markdown(f"### Video {idx+1}: `{uploaded.name}`")
        # Save to temp file (preserve extension)
        suffix = os.path.splitext(uploaded.name)[1] or ".mp4"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tf:
            tf.write(uploaded.read())
            temp_path = tf.name

        cap = cv2.VideoCapture(temp_path)
        if not cap.isOpened():
            st.error(f"Could not open file `{uploaded.name}` with OpenCV.")
            try:
                os.remove(temp_path)
            except Exception:
                pass
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

            # Strict validation
            if not ret or frame is None:
                break
            if not isinstance(frame, np.ndarray):
                # Skip non-array frames (this prevents TypeError)
                st.warning(f"Skipping invalid frame type at index {frame_index}: {type(frame)}")
                continue
            if frame.size == 0:
                st.warning(f"Skipping empty frame at index {frame_index}")
                continue
            if frame.ndim not in (2, 3):
                st.warning(f"Skipping unexpected frame ndim={frame.ndim} at index {frame_index}")
                continue

            # Convert grayscale to BGR
            if frame.ndim == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

            # Optional resize for stability/perf
            try:
                frame = cv2.resize(frame, (640, 360))
            except Exception:
                pass

            # Run inference safely
            try:
                emergency_results = yolo_v8_emergency(frame)
            except Exception as e:
                st.error(f"YOLOv8 inference failed on frame {frame_index}: {e}")
                break

            try:
                non_emergency_results = yolo_v5_non_emergency(frame)
            except Exception as e:
                st.error(f"YOLOv5 inference failed on frame {frame_index}: {e}")
                break

            # Build detections & update tracker
            detections = create_detections(emergency_results, yolo_v8_emergency.names, model_type="yolov8") + \
                         create_detections(non_emergency_results, yolo_v5_non_emergency.names, model_type="yolov5")

            try:
                tracked_objects = tracker.update(detections)
            except Exception:
                st.warning("Tracker update failed for a frame; continuing.")
                tracked_objects = []

            # Draw boxes
            for obj in tracked_objects:
                try:
                    data = obj.last_detection.data
                    label = data["label"]
                    x1, y1, x2, y2 = data["box"]
                except Exception:
                    continue

                if label in emergency_labels:
                    color = (0, 0, 255)
                    unique_emergency_ids.add(obj.id)
                    emergency_detected = True
                else:
                    color = (0, 255, 0)
                    unique_non_emergency_ids.add(obj.id)

                # Clamp coordinates and draw
                h, w = frame.shape[:2]
                x1c, y1c = max(0, int(x1)), max(0, int(y1))
                x2c, y2c = min(w - 1, int(x2)), min(h - 1, int(y2))
                cv2.rectangle(frame, (x1c, y1c), (x2c, y2c), color, 2)
                cv2.putText(frame, f"{label} ID:{obj.id}", (x1c, max(0, y1c - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            # Convert BGR -> RGB for Streamlit and display (safe guard)
            if isinstance(frame, np.ndarray) and frame.ndim == 3 and frame.shape[2] == 3 and frame.size > 0:
                try:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    stframe.image(frame_rgb, use_container_width=True)
                except Exception as e:
                    st.warning(f"Skipping display at frame {frame_index} due to conversion/display error: {e}")
            else:
                st.warning(f"Skipped displaying invalid frame at index {frame_index}: shape={getattr(frame,'shape',None)}")

        cap.release()
        try:
            os.remove(temp_path)
        except Exception:
            pass

        # Summary per video
        non_emergency_count = len(unique_non_emergency_ids)
        emergency_count = len(unique_emergency_ids)
        clearance_time = max(0, (non_emergency_count - emergency_count) * 3)
        total_clearance_time += clearance_time

        if emergency_detected:
            st.warning(f"🚨 Emergency vehicle(s) detected in `{uploaded.name}` — clear the route!")

        st.write(f"**Unique Non-Emergency Vehicles:** {non_emergency_count}")
        st.write(f"**Estimated Clearance Time (this video):** {clearance_time} seconds")

    st.markdown("---")
    st.subheader(f"Total Estimated Road Clearance Time (all videos): {total_clearance_time} seconds")
else:
    st.info("Please upload at least one video.")

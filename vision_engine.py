"""FocusGuard V3 Vision Engine.

Embedded real-time facial analysis and attention state classification using
OpenCV, MediaPipe FaceMesh, and streamlit-webrtc.
"""

import json
import math
import os
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import av
import cv2
import mediapipe as mp
import numpy as np
from streamlit_webrtc import VideoProcessorBase


# Landmark indices for eyes and face orientation
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# Outer eye corners for distance estimation
LEFT_OUTER_CANTHUS = 33
RIGHT_OUTER_CANTHUS = 263

# Key landmarks for head pose
NOSE_TIP = 1
CHIN = 152
NOSE_BRIDGE = 168

# Default paths
V3_DIR = Path(__file__).resolve().parent
PROJECT_DIR = V3_DIR.parent
DATA_JSON_PATH = PROJECT_DIR / "data.json"


def atomic_write_json(file_path: Path, data: dict) -> None:
    """Write data to a JSON file atomically using a temporary file."""
    try:
        tmp_path = file_path.with_name(file_path.stem + ".tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, file_path)
    except Exception:
        pass


def dist_2d(p1: np.ndarray, p2: np.ndarray) -> float:
    """Euclidean distance between two 2D points."""
    return float(np.linalg.norm(p1 - p2))


def compute_ear(landmarks: Any, w: int, h: int) -> float:
    """Calculate Eye Aspect Ratio (EAR) averaged over left and right eyes."""
    def pt(idx: int) -> np.ndarray:
        lm = landmarks[idx]
        return np.array([lm.x * w, lm.y * h], dtype=np.float32)

    # Left eye
    l_corner1 = pt(LEFT_EYE[0])
    l_corner2 = pt(LEFT_EYE[3])
    l_top = pt(LEFT_EYE[1])
    l_bottom = pt(LEFT_EYE[5])

    # Right eye
    r_corner1 = pt(RIGHT_EYE[0])
    r_corner2 = pt(RIGHT_EYE[3])
    r_top = pt(RIGHT_EYE[1])
    r_bottom = pt(RIGHT_EYE[5])

    left_ear = dist_2d(l_top, l_bottom) / (2.0 * dist_2d(l_corner1, l_corner2) + 1e-6)
    right_ear = dist_2d(r_top, r_bottom) / (2.0 * dist_2d(r_corner1, r_corner2) + 1e-6)

    return float((left_ear + right_ear) / 2.0)


def estimate_distance_cm(landmarks: Any, w: int, h: int) -> float:
    """Estimate viewing distance in cm based on outer eye corner distance."""
    p1 = np.array([landmarks[LEFT_OUTER_CANTHUS].x * w, landmarks[LEFT_OUTER_CANTHUS].y * h])
    p2 = np.array([landmarks[RIGHT_OUTER_CANTHUS].x * w, landmarks[RIGHT_OUTER_CANTHUS].y * h])
    pixel_dist = dist_2d(p1, p2)
    # Focal constant calibrated for standard 640x480 webcam ~ 5500
    if pixel_dist < 5.0:
        return 60.0
    dist_cm = 5500.0 / pixel_dist
    return float(round(max(20.0, min(120.0, dist_cm)), 1))


def estimate_head_pose(landmarks: Any, w: int, h: int) -> Tuple[float, float]:
    """Estimate head yaw and pitch angles in degrees."""
    left_eye = np.array([landmarks[LEFT_OUTER_CANTHUS].x * w, landmarks[LEFT_OUTER_CANTHUS].y * h])
    right_eye = np.array([landmarks[RIGHT_OUTER_CANTHUS].x * w, landmarks[RIGHT_OUTER_CANTHUS].y * h])
    eye_mid = (left_eye + right_eye) / 2.0
    eye_span = dist_2d(left_eye, right_eye) + 1e-6

    nose = np.array([landmarks[NOSE_TIP].x * w, landmarks[NOSE_TIP].y * h])
    chin = np.array([landmarks[CHIN].x * w, landmarks[CHIN].y * h])

    # Yaw: nose horizontal offset relative to eye center normalized by eye span
    # Note: right eye in mirror view has higher x or lower x depending on orientation
    dx_nose = nose[0] - eye_mid[0]
    yaw = float(np.clip((dx_nose / eye_span) * 75.0, -60.0, 60.0))

    # Pitch: vertical relation between nose and eye-chin axis
    face_height = dist_2d(eye_mid, chin) + 1e-6
    dy_nose = (nose[1] - eye_mid[1]) / face_height
    # In neutral head position dy_nose is typically ~0.35 to 0.45
    pitch = float(np.clip((dy_nose - 0.40) * 110.0, -60.0, 60.0))

    return round(yaw, 1), round(pitch, 1)


# Global singleton holding the latest measurement for immediate Streamlit access
_global_lock = threading.Lock()
_global_latest_measurement: Dict[str, Any] = {
    "timestamp_ms": int(time.time() * 1000),
    "face_detected": False,
    "focus_score": 0.0,
    "attention": "No Face / Calibrating",
    "ear": None,
    "blink_rate": 0,
    "distance_cm": None,
    "yaw": 0.0,
    "pitch": 0.0,
}


def get_latest_vision_measurement() -> Dict[str, Any]:
    """Retrieve thread-safe snapshot of the latest measurement."""
    with _global_lock:
        return dict(_global_latest_measurement)


def set_latest_vision_measurement(data: Dict[str, Any]) -> None:
    """Update global thread-safe measurement and atomically persist to data.json."""
    with _global_lock:
        _global_latest_measurement.update(data)
    # Also write data.json atomically as compatibility output
    atomic_write_json(DATA_JSON_PATH, data)


class FocusVisionProcessor(VideoProcessorBase):
    """Real-time video processor integrating MediaPipe FaceMesh with FocusGuard attention metrics."""

    def __init__(self) -> None:
        super().__init__()
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_draw = mp.solutions.drawing_utils

        # Initialize FaceMesh once per processor
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        # Drawing specs
        self.contour_spec = self.mp_draw.DrawingSpec(
            color=(0, 230, 140), thickness=1, circle_radius=1
        )
        self.mesh_spec = self.mp_draw.DrawingSpec(
            color=(0, 180, 100), thickness=1
        )

        # Blink state machine
        # OPEN -> CLOSED -> OPEN
        # Closed when EAR < 0.20, Open when EAR > 0.23
        self.eye_state = "OPEN"
        self.closed_start_time = 0.0
        self.blink_count = 0
        self.blink_timestamps: List[float] = []

        # Session tracking
        self.start_time = time.time()
        self.last_json_write_time = 0.0

        # Rolling 1.5 - 2.0s smoothing window
        # Contains tuples: (timestamp, face_detected, raw_score, candidate_state, ear, distance_cm, yaw, pitch)
        self.smoothing_window: deque = deque(maxlen=60)

        # Configurable thresholds
        self.settings = {
            "blink_threshold": 25,
            "yaw_threshold": 25,
            "pitch_threshold": 25,
            "minimum_distance_cm": 45,
        }

    def update_settings(self, settings: Dict[str, Any]) -> None:
        """Update configurable thresholds from app settings."""
        self.settings.update(settings)

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        """Process incoming WebRTC video frame."""
        img = frame.to_ndarray(format="bgr24")
        h, w, _ = img.shape
        now = time.time()

        # Convert to RGB for MediaPipe
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)

        face_detected = False
        ear: Optional[float] = None
        distance_cm: Optional[float] = None
        yaw = 0.0
        pitch = 0.0

        if results.multi_face_landmarks:
            face_detected = True
            face_landmarks = results.multi_face_landmarks[0]
            lm = face_landmarks.landmark

            # Draw visible face contours
            self.mp_draw.draw_landmarks(
                image=img,
                landmark_list=face_landmarks,
                connections=self.mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=self.contour_spec,
                connection_drawing_spec=self.mesh_spec,
            )

            # Compute biometric signals
            ear = round(compute_ear(lm, w, h), 3)
            distance_cm = estimate_distance_cm(lm, w, h)
            yaw, pitch = estimate_head_pose(lm, w, h)

            # --- Accurate Blink Detection State Machine ---
            # OPEN -> CLOSED -> OPEN
            # Do NOT increment repeatedly while eyes stay closed!
            if self.eye_state == "OPEN":
                if ear < 0.20:
                    self.eye_state = "CLOSED"
                    self.closed_start_time = now
            elif self.eye_state == "CLOSED":
                if ear > 0.23:
                    # Valid blink duration between 0.05s and 0.8s
                    closure_duration = now - self.closed_start_time
                    if 0.04 <= closure_duration <= 0.85:
                        self.blink_count += 1
                        self.blink_timestamps.append(now)
                    self.eye_state = "OPEN"

        # Clean blink timestamps older than 60 seconds
        cutoff = now - 60.0
        self.blink_timestamps = [t for t in self.blink_timestamps if t >= cutoff]

        # Calculate stable blink rate (blinks per minute)
        elapsed = now - self.start_time
        if elapsed >= 45.0:
            blink_rate = len(self.blink_timestamps)
        elif elapsed >= 12.0:
            # Conservative extrapolation during first 45 seconds
            blink_rate = int(round(len(self.blink_timestamps) * (60.0 / elapsed)))
        else:
            # Warm-up period baseline
            blink_rate = min(20, len(self.blink_timestamps) * 4)

        # Guard against unrealistic blink spikes
        blink_rate = max(0, min(50, blink_rate))

        # --- Instantaneous Focus Score Calculation ---
        raw_score = 0.0
        candidate_state = "No Face / Calibrating"

        if face_detected and ear is not None and distance_cm is not None:
            # 1. Eye openness component: normal EAR (0.24 - 0.35) -> 1.0
            eye_comp = float(np.clip((ear - 0.19) / 0.09, 0.0, 1.0))

            # 2. Blink rate component: normal (<= 22) -> 1.0, drops if abnormally high
            blink_threshold = max(10, self.settings["blink_threshold"])
            if blink_rate <= 22:
                blink_comp = 1.0
            else:
                blink_comp = float(np.clip(1.0 - (blink_rate - 22) / float(blink_threshold), 0.0, 1.0))

            # 3. Head pose component
            yaw_thresh = max(10, self.settings["yaw_threshold"])
            pitch_thresh = max(10, self.settings["pitch_threshold"])
            pose_penalty = min(
                1.0,
                (abs(yaw) / yaw_thresh + abs(pitch) / pitch_thresh) / 2.0,
            )
            pose_comp = float(np.clip(1.0 - pose_penalty, 0.0, 1.0))

            # 4. Viewing distance component
            min_dist = max(25, self.settings["minimum_distance_cm"])
            if distance_cm >= min_dist:
                dist_comp = 1.0
            else:
                dist_ratio = distance_cm / float(min_dist)
                dist_comp = float(np.clip(dist_ratio ** 1.6, 0.0, 1.0))

            # Weighted sum for instantaneous score
            raw_score = (
                0.35 * eye_comp
                + 0.30 * pose_comp
                + 0.20 * dist_comp
                + 0.15 * blink_comp
            )
            raw_score = round(float(np.clip(raw_score, 0.0, 1.0)), 3)

            # Instantaneous candidate state
            # Distracted if head turned or looking away
            if abs(yaw) > yaw_thresh or abs(pitch) > pitch_thresh:
                candidate_state = "Distracted"
            # Over-engaged if too close or excessive blink/fatigue
            elif distance_cm < min_dist or blink_rate > blink_threshold or raw_score < 0.35:
                candidate_state = "Over-engaged"
            else:
                candidate_state = "Focused"

        # Record in smoothing window
        self.smoothing_window.append({
            "time": now,
            "face_detected": face_detected,
            "raw_score": raw_score,
            "candidate_state": candidate_state,
            "ear": ear,
            "distance_cm": distance_cm,
            "yaw": yaw,
            "pitch": pitch,
        })

        # --- Rolling 1-2 Second Smoothing Window ---
        # Window of recent samples within last 1.8 seconds
        recent_window = [s for s in self.smoothing_window if now - s["time"] <= 1.8]
        if not recent_window:
            recent_window = list(self.smoothing_window)

        # Determine smoothed attention state and score
        detected_count = sum(1 for s in recent_window if s["face_detected"])
        ratio_detected = detected_count / max(1, len(recent_window))

        if ratio_detected < 0.35:
            # Face absent for majority of smoothing window
            attention = "No Face / Calibrating"
            focus_score = 0.0
        else:
            # Average focus score over window
            valid_scores = [s["raw_score"] for s in recent_window if s["face_detected"]]
            avg_score = float(np.mean(valid_scores)) if valid_scores else 0.0
            focus_score = round(float(np.clip(avg_score, 0.0, 1.0)), 2)

            # Check sustained conditions over the 1-2 second window
            distracted_count = sum(1 for s in recent_window if s["candidate_state"] == "Distracted")
            over_engaged_count = sum(1 for s in recent_window if s["candidate_state"] == "Over-engaged")
            total_samples = len(recent_window)

            # Sustained head-turn or look-away -> Distracted
            if distracted_count / total_samples >= 0.50:
                attention = "Distracted"
            # Sustained too-close or abnormal fatigue or low score -> Over-engaged
            elif over_engaged_count / total_samples >= 0.50 or focus_score < 0.35:
                attention = "Over-engaged"
            elif focus_score >= 0.65:
                attention = "Focused"
            else:
                # In between 0.35 and 0.64
                attention = "Distracted"

        # Build clean latest measurement schema
        measurement = {
            "timestamp_ms": int(now * 1000),
            "face_detected": face_detected,
            "focus_score": focus_score,
            "attention": attention,
            "ear": ear,
            "blink_rate": blink_rate,
            "distance_cm": distance_cm,
            "yaw": yaw,
            "pitch": pitch,
        }

        # Update global state and persist data.json at ~5-10Hz throttle
        if now - self.last_json_write_time >= 0.15:
            set_latest_vision_measurement(measurement)
            self.last_json_write_time = now

        return av.VideoFrame.from_ndarray(img, format="bgr24")

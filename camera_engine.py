import cv2
import mediapipe as mp
import time
import json
import numpy as np
import math

# Mediapipe setup
mp_face_mesh = mp.solutions.face_mesh
mp_draw = mp.solutions.drawing_utils

face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

# Eye landmark indices
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

def dist(a, b):
    return math.dist(a, b)

def EAR(landmarks, w, h):
    def p(i):
        lm = landmarks[i]
        return np.array([lm.x * w, lm.y * h])

    L = p(LEFT_EYE[0]); R = p(LEFT_EYE[3])
    LT = p(LEFT_EYE[1]); LB = p(LEFT_EYE[5])

    RE = p(RIGHT_EYE[0]); RR = p(RIGHT_EYE[3])
    RT = p(RIGHT_EYE[1]); RB = p(RIGHT_EYE[5])

    left_ear = dist(LT, LB) / (2 * dist(L, R) + 1e-6)
    right_ear = dist(RT, RB) / (2 * dist(RE, RR) + 1e-6)
    return (left_ear + right_ear) / 2

def estimate_distance(landmarks, w, h):
    p1 = np.array([landmarks[33].x * w, landmarks[33].y * h])
    p2 = np.array([landmarks[263].x * w, landmarks[263].y * h])
    return 1200 / (dist(p1, p2) + 1e-6)

def head_pose(landmarks, w, h):
    nose = landmarks[1]
    dx = (nose.x * w) - (w / 2)
    dy = (nose.y * h) - (h / 2)
    yaw = -dx / w * 60
    pitch = dy / h * 60
    return yaw, pitch

def classify(ear, blink_rate, yaw, pitch):
    if blink_rate > 25:
        return "Overengaged"
    if abs(yaw) > 25 or abs(pitch) > 25:
        return "Distracted"
    if ear < 0.18:
        return "Blinking"
    return "Focused"

cap = cv2.VideoCapture(0)

blink_count = 0
last_blink = time.time()
blink_window = []

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w, _ = rgb.shape
    results = face_mesh.process(rgb)

    ear = None
    distance = None
    yaw = 0
    pitch = 0

    if results.multi_face_landmarks:
        face_landmarks = results.multi_face_landmarks[0]
        lm = face_landmarks.landmark

        # ⭐ SAFE FACE MESH DRAWING (NO TESSELLATION) ⭐
        mp_draw.draw_landmarks(
            frame,
            face_landmarks,
            mp_face_mesh.FACEMESH_CONTOURS,
            mp_draw.DrawingSpec(color=(0,255,0), thickness=1, circle_radius=1),
            mp_draw.DrawingSpec(color=(0,255,0), thickness=1)
        )

        # Compute metrics
        ear = EAR(lm, w, h)
        distance = estimate_distance(lm, w, h)
        yaw, pitch = head_pose(lm, w, h)

        # Blink detection
        if ear < 0.18 and time.time() - last_blink > 0.25:
            blink_count += 1
            last_blink = time.time()

    # Blink rate per minute
    blink_window.append((time.time(), blink_count))
    blink_window = [(t, c) for (t, c) in blink_window if time.time() - t <= 60]

    if len(blink_window) > 1:
        blink_rate = blink_window[-1][1] - blink_window[0][1]
    else:
        blink_rate = 0

    attention = classify(ear or 0, blink_rate, yaw, pitch)

    # Write data.json
    data = {
        "ear": ear,
        "distance_cm": distance,
        "blink_rate": blink_rate,
        "yaw": yaw,
        "pitch": pitch,
        "attention": attention,
        "timestamp_ms": int(time.time() * 1000)
    }

    with open("data.json", "w") as f:
        json.dump(data, f)

    # Show camera window
    cv2.imshow("Camera + Face Mesh", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

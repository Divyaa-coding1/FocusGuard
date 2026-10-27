from camera.webcam import Webcam
from vision.face_mesh import FaceMeshDetector
from vision.eye_metrics import EyeTracker, LEFT_EYE, RIGHT_EYE
from vision.head_pose import HeadPose
from ai.ai_engine import AIEngine
from utils.logger import DataLogger, log_attention_state
from logic.focus_state import track_distraction
from utils.notifier import send_notification

import subprocess
import sys
import time

# --- Initialize modules ---
cam = Webcam()
mesh = FaceMeshDetector()
eyes = EyeTracker()
pose = HeadPose()
ai = AIEngine()
logger = DataLogger()

print("FocusGuard AI running...")
print("Press Ctrl + C to stop and generate graphs")

try:
    while True:
        # 1. Read camera frame
        frame = cam.read()
        faces = mesh.process(frame)
        fps = cam.fps()

        # 2. Only run AI if a face is detected
        if faces:
            landmarks = faces[0]

            left_ear = eyes.ear(landmarks, LEFT_EYE)
            right_ear = eyes.ear(landmarks, RIGHT_EYE)
            blink_rate = eyes.blink_rate(left_ear, right_ear) or 0

            yaw, pitch = pose.estimate(landmarks)

            features = [
                left_ear,
                right_ear,
                blink_rate,
                yaw,
                pitch,
                fps
            ]

            # 3. AI inference
            state, score = ai.update(features)
            logger.log(score, state)

            # 4. Distraction tracking
            attention_event = track_distraction(state)

            # 5. Timeline logging
            log_attention_state(time.time(), state)

            # 6. Notifications
            if attention_event == "REMINDER":
                send_notification(
                    "FocusGuard Reminder",
                    "You appear distracted. Try refocusing for better productivity."
                )

            elif attention_event == "ALERT":
                send_notification(
                    "FocusGuard Alert",
                    "Prolonged distraction detected. Consider taking a short break."
                )
                # trigger glitch / LED / screen effect here

            print(f"STATE: {state} | SCORE: {round(score, 2)}")

except KeyboardInterrupt:
    print("\nStopping FocusGuard...")
    print("Generating graphs...")

    subprocess.run(["/opt/homebrew/bin/python3.11", "plot_result.py"])

    print("Graphs generated. Program ended.")
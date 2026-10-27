# ai/ai_engine.py

import numpy as np

class AIEngine:
    def __init__(self):
        self.prev_score = 0.5

    def update(self, features):
        """
        features:
        [left_ear, right_ear, blink_rate, yaw, pitch, fps]
        """

        left_ear, right_ear, blink_rate, yaw, pitch, fps = features

        # -------------------------------
        # HARD BIOLOGICAL RULES (INSTANT)
        # -------------------------------

        # Look-away detection (instant distraction)
        # Look-away detection (ONLY left/right, more forgiving)
        if abs(yaw) > 25:
            return "Distracted", 0.3

        # Excessive blinking = fatigue
        if blink_rate > 25:
            return "Distracted", 0.3

        # -------------------------------
        # SOFT FOCUS SCORING (AI-LIKE)
        # -------------------------------

        ear_avg = (left_ear + right_ear) / 2

        score = (
            0.6 * np.clip((ear_avg - 0.18) * 5, 0, 1) +
            0.2 * np.clip(1 - blink_rate / 30, 0, 1) +
            0.2 * np.clip(1 - (abs(yaw) + abs(pitch)) / 30, 0, 1)
        )

        # Smooth score slightly (but NOT for look-away)
        score = 0.7 * self.prev_score + 0.3 * score
        self.prev_score = score

        # -------------------------------
        # STATE DECISION
        # -------------------------------

        if score > 0.65:
            state = "Focused"
        elif score > 0.4:
            state = "Distracted"
        else:
            state = "Over-engaged"

        return state, round(score, 2)

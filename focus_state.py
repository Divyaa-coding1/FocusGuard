import time

DISTRACTION_THRESHOLD = 0.4
REMINDER_1_TIME = 1
REMINDER_2_TIME = 2

distracted_start_time = None
current_state = "Focused"

class FocusInterpreter:
    STATES = {0: "Focused", 1: "Distracted", 2: "Over-engaged"}

    def interpret(self, ml_class, yaw, pitch):
        # HARD RULE: Looking away = distracted instantly
        if abs(yaw) > 15 or abs(pitch) > 15:
            return "Distracted"

        # Otherwise trust ML
        return self.STATES[ml_class]


def track_distraction(focus_state):
    global distracted_start_time, current_state

    current_time = time.time()

    if focus_state == "Distracted":
        if distracted_start_time is None:
            distracted_start_time = current_time

        distracted_duration = current_time - distracted_start_time
        current_state = "Distracted"

        if REMINDER_1_TIME <= distracted_duration < REMINDER_2_TIME:
            return "REMINDER"

        elif distracted_duration >= REMINDER_2_TIME:
            return "ALERT"

        else:
            return "DISTRACTED"

    else:
        distracted_start_time = None
        current_state = "Focused"
        return "FOCUSED"

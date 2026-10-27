import csv
import time
import os

timeline_times = []
timeline_states = []  # 0 = Focused, 1 = Distracted

def log_attention_state(timestamp, state):
    timeline_times.append(timestamp)
    timeline_states.append(1 if state == "Distracted" else 0)

class DataLogger:
    def __init__(self, filename="focus_data.csv"):
        self.filename = filename

        # Create file + header if it doesn't exist
        if not os.path.exists(self.filename):
            with open(self.filename, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["timestamp", "focus_score", "state"])

    def log(self, score, state):
        with open(self.filename, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([time.time(), score, state])

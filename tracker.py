import cv2
import numpy as np
import torch
from ultralytics import YOLO
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
import os
import csv
from utils import detect_players, extract_features

# Configuration
VIDEO_PATH = "15sec_input_720p.mp4"
OUTPUT_DIR = "tracking_output"
LOG_DIR = "tracking_logs"
MODEL_PATH = "best.pt"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# Load YOLO Model
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 Using device: {device.upper()}")
model = YOLO(MODEL_PATH).to(device)

# CSV Setup 
csv_file = open(os.path.join(LOG_DIR, "player_tracking_log.csv"), mode="w", newline="")
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["Frame", "PlayerID", "X1", "Y1", "X2", "Y2", "Time"])

# Tracking Logic 
video = cv2.VideoCapture(VIDEO_PATH)
prev_features, prev_ids = np.array([]), []
player_id_counter = 0
id_history = {}
frame_index = 0

while True:
    ret, frame = video.read()
    if not ret:
        break

    detections = detect_players(frame, model)
    features = extract_features(frame, detections)
    current_ids = {}

    if prev_features.size == 0:
        for i in range(len(detections)):
            current_ids[i] = player_id_counter
            id_history[player_id_counter] = [frame_index]
            player_id_counter += 1
    else:
        if features.shape[0] > 0 and prev_features.shape[0] > 0:
            dist = cdist(features, prev_features, metric="cosine")
            row_ind, col_ind = linear_sum_assignment(dist)

            used = set()
            for r, c in zip(row_ind, col_ind):
                if dist[r][c] < 0.5:
                    current_ids[r] = prev_ids[c]
                    id_history[prev_ids[c]].append(frame_index)
                    used.add(c)
                else:
                    current_ids[r] = player_id_counter
                    id_history[player_id_counter] = [frame_index]
                    player_id_counter += 1

            for i in range(len(detections)):
                if i not in current_ids:
                    current_ids[i] = player_id_counter
                    id_history[player_id_counter] = [frame_index]
                    player_id_counter += 1

    prev_features = features
    prev_ids = [current_ids[i] for i in range(len(detections))]

    timestamp = f"{frame_index / 30:.2f}s"
    for i, d in enumerate(detections):
        x1, y1, x2, y2 = d['bbox']
        pid = current_ids.get(i, -1)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"Player {pid}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, f"Time: {timestamp}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        csv_writer.writerow([frame_index, pid, x1, y1, x2, y2, timestamp])

    cv2.imshow("Real-Time Player Tracker", frame)  # Show frame
    cv2.imwrite(f"{OUTPUT_DIR}/frame_{frame_index:03}.png", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    frame_index += 1

csv_file.close()
video.release()
cv2.destroyAllWindows()

print(f"\n🎯 Total players tracked: {player_id_counter}")
print(f"📊 ID history lengths: {dict((k, len(v)) for k,v in id_history.items())}")


import cv2
import numpy as np
from ultralytics import YOLO

# Player Detection
def detect_players(frame, model):
    """Detect players using YOLOv11 model. Filters detections to class 0 (players only)."""
    results = model(frame)
    detections = []
    for r in results[0].boxes:
        if int(r.cls) == 0:  # Class 0 = player
            x1, y1, x2, y2 = map(int, r.xyxy[0])
            detections.append({'bbox': (x1, y1, x2, y2)})
    return detections

# Feature Extraction
def get_color_histogram(crop):
    """Compute normalized HSV histogram for the cropped player region."""
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [30, 32], [0, 180, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()

def extract_features(frame, detections):
    """Extract visual features (HSV histograms) for each detected player.Returns a NumPy array of feature vectors."""
    features = []
    for d in detections:
        x1, y1, x2, y2 = d['bbox']
        crop = frame[y1:y2, x1:x2]
        if crop.size > 0:
            features.append(get_color_histogram(crop))
    return np.array(features)

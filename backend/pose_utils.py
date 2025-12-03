# backend/pose_utils.py
from typing import List, Optional, Tuple

from .config import SMOOTHING_ALPHA, MIN_POSE_SCORE

def extract_landmarks(results, frame_width: int, frame_height: int):
    if not results.pose_landmarks:
        return None

    landmarks = []
    for lm in results.pose_landmarks.landmark:
        x = lm.x * frame_width
        y = lm.y * frame_height
        vis = lm.visibility
        ok = vis is not None and vis >= MIN_POSE_SCORE
        landmarks.append((x, y, vis, ok))
    return landmarks


def smooth_landmarks(raw_landmarks, prev_landmarks):
    if prev_landmarks is None:
        return list(raw_landmarks)

    smoothed = []
    for (x, y, vis, ok), (px, py, pvis, pok) in zip(raw_landmarks, prev_landmarks):
        if not ok and pok:
            smoothed.append((px, py, pvis, pok))
            continue
        if ok and not pok:
            smoothed.append((x, y, vis, ok))
            continue
        if not ok and not pok:
            smoothed.append((x, y, vis, ok))
            continue

        sx = px + SMOOTHING_ALPHA * (x - px)
        sy = py + SMOOTHING_ALPHA * (y - py)
        smoothed.append((sx, sy, vis, ok))

    return smoothed

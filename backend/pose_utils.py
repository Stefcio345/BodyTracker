# backend/pose_utils.py
from typing import List, Optional, Tuple

from .config import SMOOTHING_ALPHA, MIN_POSE_SCORE


def _convert_single_landmarks(pose_landmarks, frame_width: int, frame_height: int):
    """Convert a Mediapipe landmark list into pixel coordinates."""
    if pose_landmarks is None:
        return None

    landmarks = []
    for lm in pose_landmarks.landmark:
        x = lm.x * frame_width
        y = lm.y * frame_height
        vis = lm.visibility
        ok = vis is not None and vis >= MIN_POSE_SCORE
        landmarks.append((x, y, vis, ok))
    return landmarks


def extract_landmarks(results, frame_width: int, frame_height: int):
    """Backward compatible single-pose extraction."""
    if not results.pose_landmarks:
        return None
    return _convert_single_landmarks(results.pose_landmarks, frame_width, frame_height)


def extract_multiple_landmarks(results, frame_width: int, frame_height: int) -> Optional[List[List[Tuple[float, float, float, bool]]]]:
    """
    Extract a list of landmark sets from a multi-pose result.

    Returns a list of landmark lists, or None when nothing is found.
    """
    if not getattr(results, "pose_landmarks", None):
        return None

    all_landmarks: List[List[Tuple[float, float, float, bool]]] = []
    for lm_list in results.pose_landmarks:
        converted = _convert_single_landmarks(lm_list, frame_width, frame_height)
        if converted:
            all_landmarks.append(converted)

    return all_landmarks or None


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


def smooth_landmarks_batch(raw_batch, prev_batch):
    """Apply smoothing across a list of poses, index-aligned."""
    if raw_batch is None:
        return None

    if prev_batch is None:
        return [list(lm_set) for lm_set in raw_batch]

    smoothed_sets: List[List[Tuple[float, float, float, bool]]] = []
    for idx, lm_set in enumerate(raw_batch):
        prev = prev_batch[idx] if idx < len(prev_batch) else None
        smoothed_sets.append(smooth_landmarks(lm_set, prev))

    return smoothed_sets

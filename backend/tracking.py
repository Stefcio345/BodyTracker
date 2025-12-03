# backend/tracking.py
import time
import math
from typing import Dict, List, Optional, Tuple
import os

import cv2
import numpy as np
from deepface import DeepFace

from .config import (
    FACE_MATCH_MAX_DIST,
    DESCRIPTOR_SIM_THRESHOLD,
    EMOTION_UPDATE_INTERVAL,
    FACE_SNAPSHOT_DIR,
)

# --- Utility helpers --------------------------------------------------------

def clamp(v: float, vmin: float, vmax: float) -> float:
    return max(vmin, min(v, vmax))

def warmup_deepface():
    dummy = np.zeros((160, 160, 3), dtype=np.uint8)
    try:
        DeepFace.analyze(
            dummy,
            actions=["age", "emotion"],
            enforce_detection=False
        )
    except Exception:
        pass


def is_real_face(face_crop_bgr):
    # 1) Check if crop is valid
    if face_crop_bgr is None:
        return False
    if face_crop_bgr.size == 0:
        return False

    h, w = face_crop_bgr.shape[:2]
    if h < 60 or w < 60:
        # too small, DeepFace really struggles on tiny crops
        return False

    # 2) Convert BGR -> RGB
    face_rgb = cv2.cvtColor(face_crop_bgr, cv2.COLOR_BGR2RGB)

    # 3) Make sure we have uint8 0–255
    if face_rgb.dtype != np.uint8:
        face_rgb = np.clip(face_rgb, 0, 255).astype(np.uint8)

    # 4) Call DeepFace – use at least one action
    try:
        _ = DeepFace.analyze(
            face_rgb,
            actions=["age"],          # anything non-empty is fine
            enforce_detection=True,   # if no face detected -> exception
        )
        return True
    except Exception as e:
        # Uncomment to see what’s going on:
        # print("DeepFace error:", e)
        return False

class FaceTracker:
    def __init__(
        self,
        match_max_dist: float = FACE_MATCH_MAX_DIST,
        descriptor_sim_threshold: float = DESCRIPTOR_SIM_THRESHOLD,
        track_ttl: float = 2.0,              # was 1.0
        emotion_update_interval: float = EMOTION_UPDATE_INTERVAL,
        smooth_alpha: float = 0.6,          # NEW: smoothing for position / bbox
    ):
        self.match_max_dist = match_max_dist
        self.descriptor_sim_threshold = descriptor_sim_threshold
        self.track_ttl = track_ttl
        self.emotion_update_interval = emotion_update_interval
        self.smooth_alpha = smooth_alpha
        self.age_list = []

        # [{ "id": int, "hist": np.ndarray }, ...]
        self.stored_face_descriptors: List[Dict] = []

        # id -> {"age": int/None, "emotion": str/None, "last_ts": float}
        self.face_attributes: Dict[int, Dict] = {}

        # Active tracks:
        # [{ "id": int, "cx": float, "cy": float, "bbox": (x1,y1,x2,y2), "last_seen": float }, ...]
        self.tracked_faces: List[Dict] = []

        self.next_face_id: int = 1

    # --- descriptor / identity helpers -------------------------------------

    @staticmethod
    def compute_face_descriptor(bgr_crop):
        """
        Very simple image descriptor (HSV histogram).
        Used only for dedup of saved faces.
        """
        if bgr_crop is None or bgr_crop.size == 0:
            return None

        crop_resized = cv2.resize(bgr_crop, (64, 64), interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(crop_resized, cv2.COLOR_BGR2HSV)

        hist = cv2.calcHist([hsv], [0, 1], None, [16, 16], [0, 180, 0, 256])
        cv2.normalize(hist, hist)
        return hist

    def find_similar_stored_face(self, descriptor) -> Optional[int]:
        """
        Compare descriptor to all stored descriptors.
        Returns an existing face_id if similarity >= threshold, else None.
        """
        if descriptor is None:
            return None

        best_id = None
        best_score = -1.0

        for entry in self.stored_face_descriptors:
            stored_hist = entry["hist"]
            score = cv2.compareHist(stored_hist, descriptor, cv2.HISTCMP_CORREL)
            if score > best_score:
                best_score = score
                best_id = entry["id"]

        if best_score >= self.descriptor_sim_threshold:
            return best_id
        return None

    def analyze_face_attributes(self, bgr_crop) -> Tuple[Optional[int], Optional[str]]:
        """
        Use DeepFace to estimate age + emotion for a cropped face.
        This is approximate and not identity recognition.
        """
        try:
            if bgr_crop is None or bgr_crop.size == 0:
                return None, None

            rgb = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2RGB)

            result = DeepFace.analyze(
                rgb,
                actions=["age", "emotion"],
                enforce_detection=False
            )

            if isinstance(result, list):
                result = result[0]

            if len(self.age_list) < 10:
                self.age_list.append(result.get("age", None))

            emotion_dict = result.get("emotion", {})
            if isinstance(emotion_dict, dict) and emotion_dict:
                emotion = max(emotion_dict, key=emotion_dict.get)
            else:
                emotion = result.get("dominant_emotion", None)

            age = sum(self.age_list) / len(self.age_list)


            return age, emotion
        except Exception as e:
            print(f"[WARN] DeepFace error: {e}")
            return None, None

    # --- tracking helpers ---------------------------------------------------

    @staticmethod
    def detection_to_bbox_px(detection, frame_w: int, frame_h: int) -> Tuple[int, int, int, int]:
        relative_box = detection.location_data.relative_bounding_box
        x = relative_box.xmin
        y = relative_box.ymin
        w = relative_box.width
        h = relative_box.height

        x1 = int(clamp(x, 0.0, 1.0) * frame_w)
        y1 = int(clamp(y, 0.0, 1.0) * frame_h)
        x2 = int(clamp(x + w, 0.0, 1.0) * frame_w)
        y2 = int(clamp(y + h, 0.0, 1.0) * frame_h)

        x1, x2 = sorted((clamp(x1, 0, frame_w - 1), clamp(x2, 0, frame_w - 1)))
        y1, y2 = sorted((clamp(y1, 0, frame_h - 1), clamp(y2, 0, frame_h - 1)))

        return int(x1), int(y1), int(x2), int(y2)

    def _find_best_tracked_face(self, cx: float, cy: float) -> Optional[Dict]:
        """Return the best matching tracked face by center distance."""
        best_face = None
        best_dist = self.match_max_dist

        for face in self.tracked_faces:
            dist = math.hypot(cx - face["cx"], cy - face["cy"])
            if dist < best_dist:
                best_dist = dist
                best_face = face

        return best_face

    def _create_new_face(self, face_crop, cx: float, cy: float, now: float, descriptor, bbox):
        """Create a truly new face identity."""
        face_id = self.next_face_id
        self.next_face_id += 1

        self.tracked_faces.append({
            "id": face_id,
            "cx": cx,
            "cy": cy,
            "bbox": bbox,       # store bbox here
            "last_seen": now,
        })

        age, emotion = self.analyze_face_attributes(face_crop)
        self.face_attributes[face_id] = {
            "age": age,
            "emotion": emotion,
            "last_ts": now,
        }

        if descriptor is not None:
            self.stored_face_descriptors.append({
                "id": face_id,
                "hist": descriptor,
            })

        if face_crop is not None and face_crop.size > 0:
            filename = os.path.join(FACE_SNAPSHOT_DIR, f"face_{face_id}.jpg")
            cv2.imwrite(filename, face_crop)
            print(f"[INFO] Saved new face #{face_id} -> {filename}")
            print(f"       Age≈{age}, Emotion={emotion}")

        return face_id

    def _update_attributes_if_needed(self, face_id: int, face_crop, now: float):
        attrs = self.face_attributes.get(face_id)
        if attrs is None:
            return

        last_ts = attrs.get("last_ts", 0.0)
        if now - last_ts >= self.emotion_update_interval:
            age, new_emotion = self.analyze_face_attributes(face_crop)
            if new_emotion is not None:
                attrs["emotion"] = new_emotion
            attrs["last_ts"] = now

            attrs["age"] = int(age)

    def _update_track_smoothly(self, track: Dict, cx: float, cy: float, bbox, now: float):
        """
        Smoothly update center and bbox using EMA to avoid jitter.
        """
        ax = self.smooth_alpha

        # Smooth center
        track["cx"] = track["cx"] + ax * (cx - track["cx"])
        track["cy"] = track["cy"] + ax * (cy - track["cy"])

        # Smooth bbox
        old_x1, old_y1, old_x2, old_y2 = track["bbox"]
        x1, y1, x2, y2 = bbox

        new_x1 = old_x1 + ax * (x1 - old_x1)
        new_y1 = old_y1 + ax * (y1 - old_y1)
        new_x2 = old_x2 + ax * (x2 - old_x2)
        new_y2 = old_y2 + ax * (y2 - old_y2)

        track["bbox"] = (new_x1, new_y1, new_x2, new_y2)
        track["last_seen"] = now

    def cleanup_tracks(self, now: float):
        """Remove tracks that haven't been seen for a while (track_ttl)."""
        self.tracked_faces = [
            f for f in self.tracked_faces
            if now - f["last_seen"] <= self.track_ttl
        ]

    # --- public API ---------------------------------------------------------

    def track(self, frame, detections) -> List[Dict]:
        """
        Main tracking method.

        Face flow for each detection:
          1) bbox + Haar check (only for new tracks)
          2) track by position
          3) if new track → dedup by descriptor
          4) if truly new → DeepFace(age+emotion) once
          5) if known → update emotion at most once per interval

        Returns:
            current_faces: list of {"id": int, "bbox": (x1, y1, x2, y2)}
            (one per active track, smoothed over time)
        """
        frame_h, frame_w = frame.shape[:2]
        now = time.time()

        for det in detections:
            x1, y1, x2, y2 = self.detection_to_bbox_px(det, frame_w, frame_h)
            w_box = x2 - x1
            h_box = y2 - y1

            # Optional simple filter: ignore tiny or very weird aspect boxes
            if w_box < 60 or h_box < 60:
                continue
            ratio = w_box / float(h_box)
            if ratio < 0.4 or ratio > 2.5:
                continue

            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            face_crop = frame[y1:y2, x1:x2]
            bbox = (x1, y1, x2, y2)

            # 2) Try to match existing tracked face by position
            best_face = self._find_best_tracked_face(cx, cy)

            if best_face is not None:
                # Known track — smooth update
                self._update_track_smoothly(best_face, cx, cy, bbox, now)
                face_id = best_face["id"]
            else:
                # 1) Second AI gate - cheap filter before any descriptor / DeepFace
                if not is_real_face(face_crop):
                    continue

                print("face is real")
                # 3) Descriptor for dedup (only if it's not matching any active track)
                descriptor = self.compute_face_descriptor(face_crop)
                existing_id = self.find_similar_stored_face(descriptor)

                if existing_id is not None:
                    # We've seen this face before in history
                    face_id = existing_id
                    self.tracked_faces.append({
                        "id": face_id,
                        "cx": cx,
                        "cy": cy,
                        "bbox": bbox,
                        "last_seen": now,
                    })

                    # If somehow we don't have attributes yet (edge case), compute them now
                    if face_id not in self.face_attributes:
                        age, emotion = self.analyze_face_attributes(face_crop)
                        self.face_attributes[face_id] = {
                            "age": age,
                            "emotion": emotion,
                            "last_ts": now,
                        }
                else:
                    # Truly new-looking face
                    face_id = self._create_new_face(face_crop, cx, cy, now, descriptor, bbox)

            # 4) For already-known faces: update emotion once per interval
            self._update_attributes_if_needed(face_id, face_crop, now)

        # Prune stale tracks so we don't carry ghosts forever
        self.cleanup_tracks(now)

        # Build current_faces from smoothed tracks, not raw detections
        current_faces: List[Dict] = []
        for f in self.tracked_faces:
            x1, y1, x2, y2 = f["bbox"]
            current_faces.append({
                "id": f["id"],
                "bbox": (int(x1), int(y1), int(x2), int(y2)),
            })

        return current_faces

    # --- read-only helpers for drawing -------------------------------------

    @property
    def unique_faces_count(self) -> int:
        # IDs are assigned sequentially from 1
        return self.next_face_id - 1

    def get_attributes(self, face_id: int) -> Optional[Dict]:
        return self.face_attributes.get(face_id)

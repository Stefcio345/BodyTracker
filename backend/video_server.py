# backend/video_server.py
import base64
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import mediapipe as mp
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from .config import (
    VIDEO_WIDTH,
    VIDEO_HEIGHT,
    mp_face,
    MAX_POSES,
    POSE_LANDMARKER_MODEL,
    POSE_LANDMARKER_MODEL_URL,
)
from .tracking import FaceTracker, warmup_deepface
from .pose_utils import (
    extract_multiple_landmarks,
    smooth_landmarks_batch,
)
from .draw_utils import draw_stickman, draw_faces_and_counts

app = FastAPI()

# ---------- Static frontend setup ----------

BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"
INDEX_HTML = FRONTEND_DIR / "index.html"

# Serve static assets (CSS, JS) under /static
app.mount(
    "/static",
    StaticFiles(directory=str(FRONTEND_DIR)),
    name="static",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or your domain
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def index():
    # Serve the main HTML
    return FileResponse(str(INDEX_HTML))


# ---------- Video stream ----------

tracker = FaceTracker()
prev_landmarks_stream = None  # smoothing for /video_feed
pose_landmarker = None
pose_timestamp_http = 0


def _ensure_pose_model():
    if POSE_LANDMARKER_MODEL.exists():
        return

    try:
        import urllib.request

        print("[INFO] Downloading pose landmarker model...")
        urllib.request.urlretrieve(str(POSE_LANDMARKER_MODEL_URL), str(POSE_LANDMARKER_MODEL))
        print(f"[INFO] Model saved to {POSE_LANDMARKER_MODEL}")
    except Exception as exc:
        raise RuntimeError(f"Failed to download pose landmarker model: {exc}")


def _get_pose_landmarker():
    global pose_landmarker
    if pose_landmarker is not None:
        return pose_landmarker

    _ensure_pose_model()

    base_options = mp_python.BaseOptions(model_asset_path=str(POSE_LANDMARKER_MODEL))
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_poses=MAX_POSES,
        min_pose_detection_confidence=0.6,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.6,
    )
    pose_landmarker = vision.PoseLandmarker.create_from_options(options)
    return pose_landmarker


def _match_faces_to_poses(poses, faces):
    """
    Naive association between pose landmarks and face boxes using the nose point.
    Returns a dict[idx -> bbox or None].
    """
    assignments = {}
    if not poses:
        return assignments

    for idx, lm_set in enumerate(poses):
        if len(lm_set) == 0:
            continue
        nose_x, nose_y, _, nose_ok = lm_set[mp.solutions.pose.PoseLandmark.NOSE.value]
        if not nose_ok:
            continue

        best_bbox = None
        best_dist = float("inf")
        for face in faces or []:
            x1, y1, x2, y2 = face.get("bbox", (0, 0, 0, 0))
            fx = (x1 + x2) / 2
            fy = (y1 + y2) / 2
            dist = (nose_x - fx) ** 2 + (nose_y - fy) ** 2
            if dist < best_dist:
                best_dist = dist
                best_bbox = (x1, y1, x2, y2)

        if best_bbox is not None:
            assignments[idx] = best_bbox

    return assignments


def gen_frames():
    global prev_landmarks_stream, tracker

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, VIDEO_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, VIDEO_HEIGHT)

    warmup_deepface()

    if not cap.isOpened():
        print("Could not open webcam")
        return

    pose_detector = _get_pose_landmarker()
    timestamp_ms = 0

    with mp_face.FaceDetection(
        model_selection=0,
        min_detection_confidence=0.5,
    ) as face_detector:

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_h, frame_w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Pose (multi-person)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms += 33  # ~30 FPS pacing for the landmarker
            pose_results = pose_detector.detect_for_video(mp_image, timestamp_ms)

            raw_landmarks = extract_multiple_landmarks(pose_results, frame_w, frame_h)
            if raw_landmarks is not None:
                smoothed = smooth_landmarks_batch(raw_landmarks, prev_landmarks_stream)
                prev_landmarks_stream = smoothed
            else:
                smoothed = None
                prev_landmarks_stream = None

            # Faces + tracking
            face_results = face_detector.process(rgb)
            detections = face_results.detections if face_results.detections else []
            current_faces = tracker.track(frame, detections)

            # Stickmen for each pose
            if smoothed is not None:
                face_by_pose = _match_faces_to_poses(smoothed, current_faces)
                for idx, lm_set in enumerate(smoothed):
                    draw_stickman(frame, lm_set, face_bbox=face_by_pose.get(idx))

            # Mirror + draw faces/counts
            frame_flipped = cv2.flip(frame, 1)
            draw_faces_and_counts(frame_flipped, current_faces, tracker, mirrored=True)

            # Encode as JPEG for MJPEG stream
            ok, buffer = cv2.imencode(".jpg", frame_flipped)
            if not ok:
                continue

            frame_bytes = buffer.tobytes()
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
            )

    cap.release()


@app.get("/video_feed")
def video_feed():
    return StreamingResponse(
        gen_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


# ---------- /analyze endpoint (single frame API) ----------

class AnalyzeRequest(BaseModel):
    # base64-encoded image, with or without data URL prefix
    image: str


def _decode_base64_image(data: str) -> np.ndarray:
    """
    Decode a base64-encoded image (optionally with a data URL prefix)
    into a BGR OpenCV frame.
    """
    if "," in data and data.strip().startswith("data:"):
        # data:image/png;base64,XXXX
        _, b64data = data.split(",", 1)
    else:
        b64data = data

    try:
        img_bytes = base64.b64decode(b64data)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid base64 image payload")

    np_arr = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=400, detail="Could not decode image")
    return frame


def _serialize_landmarks(landmarks: Any) -> Any:
    """
    Make whatever extract_landmarks returns JSON-safe.
    Supports dict[name] = (x, y, ...) or list/tuple of points.
    """
    if landmarks is None:
        return None

    if isinstance(landmarks, dict):
        out = {}
        for k, v in landmarks.items():
            if isinstance(v, (list, tuple)) and len(v) >= 2:
                out[k] = [float(v[0]), float(v[1])] + [
                    float(x) for x in v[2:]
                ]
            else:
                out[k] = v
        return out

    if isinstance(landmarks, (list, tuple)):
        out_list = []
        for p in landmarks:
            if isinstance(p, (list, tuple)) and len(p) >= 2:
                out_list.append([float(p[0]), float(p[1])] + [
                    float(x) for x in p[2:]
                ])
            else:
                out_list.append(p)
        return out_list

    return landmarks


def _serialize_faces(faces: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Take FaceTracker output and make it JSON-safe.
    Assumes each face is a dict with at least a 'bbox' key.
    """
    serialized = []
    for f in faces:
        bbox = f.get("bbox", None)
        if bbox is not None:
            bbox = [float(x) for x in bbox]

        serialized.append(
            {
                "id": f.get("id"),
                "bbox": bbox,
                "confidence": float(f.get("confidence", 0.0))
                if f.get("confidence") is not None
                else None,
                "emotion": f.get("emotion"),
                "raw": {
                    k: v
                    for k, v in f.items()
                    if k not in {"bbox", "confidence", "emotion", "id"}
                },
            }
        )
    return serialized


# Separate smoothing state for HTTP-based analyze
prev_landmarks_http: Optional[Any] = None


@app.post("/analyze")
def analyze(req: AnalyzeRequest):
    """
    Analyze a single frame sent as base64, return face + posture info.

    Request body:
    {
        "image": "<base64 or data:image/...;base64,...>"
    }
    """
    global prev_landmarks_http

    # Decode frame
    frame = _decode_base64_image(req.image)
    frame_h, frame_w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Pose
    pose_detector = _get_pose_landmarker()
    global pose_timestamp_http
    pose_timestamp_http += 1
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    pose_results = pose_detector.detect_for_video(mp_image, pose_timestamp_http)

    raw_landmarks = extract_multiple_landmarks(pose_results, frame_w, frame_h)

    if raw_landmarks is not None:
        smoothed = smooth_landmarks_batch(raw_landmarks, prev_landmarks_http)
        prev_landmarks_http = smoothed
    else:
        smoothed = None
        prev_landmarks_http = None

    # Faces + tracking
    with mp_face.FaceDetection(
        model_selection=0,
        min_detection_confidence=0.5,
    ) as face_detector:
        face_results = face_detector.process(rgb)
        detections = face_results.detections if face_results.detections else []
        # We don't need the drawn frame here, just tracker info
        faces = tracker.track(frame, detections)

    return {
        "faces": _serialize_faces(faces),
        "posture": {
            # For now we expose only landmarks; you can plug in derived metrics later.
            "landmarks": _serialize_landmarks(smoothed),
        },
    }

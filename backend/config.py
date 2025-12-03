# backend/config.py
import os
import cv2
import mediapipe as mp
from pathlib import Path

VIDEO_WIDTH = 640
VIDEO_HEIGHT = 480

SMOOTHING_ALPHA = 0.8
MIN_POSE_SCORE = 0.6
MAX_POSES = 4

FACE_MATCH_MAX_DIST = 80
DESCRIPTOR_SIM_THRESHOLD = 0.8
TRACK_TTL = 0.3
EMOTION_UPDATE_INTERVAL = 1.0

USE_SECOND_AI = True

FACE_SNAPSHOT_DIR = "faces"
os.makedirs(FACE_SNAPSHOT_DIR, exist_ok=True)

# Model assets
MODELS_DIR = Path(__file__).resolve().parent / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
POSE_LANDMARKER_MODEL = MODELS_DIR / "pose_landmarker_full.task"
POSE_LANDMARKER_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_full/float16/1/pose_landmarker_full.task"
)

mp_pose = mp.solutions.pose
mp_face = mp.solutions.face_detection
LM = mp_pose.PoseLandmark

FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
EYE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_eye_tree_eyeglasses.xml"
)

# backend/config.py
import os
import cv2
import mediapipe as mp

VIDEO_WIDTH = 640
VIDEO_HEIGHT = 480

SMOOTHING_ALPHA = 0.8
MIN_POSE_SCORE = 0.6

FACE_MATCH_MAX_DIST = 80
DESCRIPTOR_SIM_THRESHOLD = 0.8
TRACK_TTL = 0.3
EMOTION_UPDATE_INTERVAL = 1.0

USE_SECOND_AI = True

FACE_SNAPSHOT_DIR = "faces"
os.makedirs(FACE_SNAPSHOT_DIR, exist_ok=True)

mp_pose = mp.solutions.pose
mp_face = mp.solutions.face_detection
LM = mp_pose.PoseLandmark

FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
EYE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_eye_tree_eyeglasses.xml"
)

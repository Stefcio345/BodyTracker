# backend/draw_utils.py
from typing import Dict, List, Optional

import cv2

from .config import LM
from .tracking import FaceTracker

def draw_stickman(frame, landmarks, face_bbox=None):
    """
    Cartoon stickman with elbows and knees.

    If face_bbox is provided, we use it to position the TOP of the head:
      - top of circle ≈ top of face box
      - radius is still based on shoulder width (so proportions are sane)
    """
    if landmarks is None:
        return

    def pt(lm_enum):
        idx = lm_enum.value
        x, y, _, ok = landmarks[idx]
        return x, y, ok

    # Grab relevant joints
    nose_x, nose_y, nose_ok = pt(LM.NOSE)

    l_sh_x, l_sh_y, l_sh_ok = pt(LM.LEFT_SHOULDER)
    r_sh_x, r_sh_y, r_sh_ok = pt(LM.RIGHT_SHOULDER)

    l_el_x, l_el_y, l_el_ok = pt(LM.LEFT_ELBOW)
    r_el_x, r_el_y, r_el_ok = pt(LM.RIGHT_ELBOW)

    l_wr_x, l_wr_y, l_wr_ok = pt(LM.LEFT_WRIST)
    r_wr_x, r_wr_y, r_wr_ok = pt(LM.RIGHT_WRIST)

    l_hip_x, l_hip_y, l_hip_ok = pt(LM.LEFT_HIP)
    r_hip_x, r_hip_y, r_hip_ok = pt(LM.RIGHT_HIP)

    l_kn_x, l_kn_y, l_kn_ok = pt(LM.LEFT_KNEE)
    r_kn_x, r_kn_y, r_kn_ok = pt(LM.RIGHT_KNEE)

    l_an_x, l_an_y, l_an_ok = pt(LM.LEFT_ANKLE)
    r_an_x, r_an_y, r_an_ok = pt(LM.RIGHT_ANKLE)

    # Require shoulders & hips for torso
    if not (l_sh_ok and r_sh_ok and l_hip_ok and r_hip_ok):
        return

    # Midpoints for cartoon style
    neck_x = (l_sh_x + r_sh_x) / 2
    neck_y = (l_sh_y + r_sh_y) / 2

    hips_x = (l_hip_x + r_hip_x) / 2
    hips_y = (l_hip_y + r_hip_y) / 2

    def line(ax, ay, bx, by, ok_a, ok_b):
        if ok_a and ok_b:
            cv2.line(
                frame,
                (int(ax), int(ay)),
                (int(bx), int(by)),
                (255, 255, 255),
                4,
                lineType=cv2.LINE_AA,
            )

    # Torso
    line(neck_x, neck_y, hips_x, hips_y, True, True)

    # Arms (shoulder midpoint → elbow → wrist)
    # Left arm
    line(neck_x, neck_y, l_el_x, l_el_y, True, l_el_ok)
    line(l_el_x, l_el_y, l_wr_x, l_wr_y, l_el_ok, l_wr_ok)
    # Right arm
    line(neck_x, neck_y, r_el_x, r_el_y, True, r_el_ok)
    line(r_el_x, r_el_y, r_wr_x, r_wr_y, r_el_ok, r_wr_ok)

    # Legs (hips midpoint → knee → ankle)
    # Left leg
    line(hips_x, hips_y, l_kn_x, l_kn_y, True, l_kn_ok)
    line(l_kn_x, l_kn_y, l_an_x, l_an_y, l_kn_ok, l_an_ok)
    # Right leg
    line(hips_x, hips_y, r_kn_x, r_kn_y, True, r_kn_ok)
    line(r_kn_x, r_kn_y, r_an_x, r_an_y, r_kn_ok, r_an_ok)

    # --- Head: circle between face top and neck ---
    if face_bbox is not None:
        fx1, fy1, fx2, fy2 = face_bbox

        # top of head = top of bbox
        top_y = fy1 - 20
        # bottom of head = neck
        bottom_y = neck_y

        # radius and center
        head_r = abs(bottom_y - top_y) / 2
        head_cy = (bottom_y + top_y) / 2
        head_cx = neck_x

        # draw it
        cv2.circle(
            frame,
            (int(head_cx), int(head_cy)),
            int(head_r),
            (255, 255, 255),
            3,
            cv2.LINE_AA
        )

def draw_faces_and_counts(frame, current_faces: List[Dict], tracker: FaceTracker, mirrored: bool = False):
    """
    Draw boxes, IDs, age & emotion, and person counts.
    If mirrored=True, adjust x coordinates for flipped frame.
    """
    h, w = frame.shape[:2]
    current_count = len(current_faces)
    unique_count = tracker.unique_faces_count

    for face in current_faces:
        face_id = face["id"]
        x1, y1, x2, y2 = face["bbox"]

        if mirrored:
            nx1 = w - 1 - x2
            nx2 = w - 1 - x1
            x1, x2 = nx1, nx2

        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

        # First line: ID
        label_y = max(int(y1) - 25, 15)
        cv2.putText(
            frame,
            f"ID {face_id}",
            (int(x1), label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        # Second line: age + emotion (if we have them)
        attrs = tracker.get_attributes(face_id)
        if attrs:
            age = attrs.get("age")
            emotion = attrs.get("emotion") or "?"
            age_str = f"{age}y" if isinstance(age, (int, float)) else "age?"
            text2 = f"{age_str}, {emotion}"
            cv2.putText(
                frame,
                text2,
                (int(x1), label_y + 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )

    # Bottom counts
    text_current = f"Current people: {current_count}"
    text_unique = f"Unique people: {unique_count}"

    cv2.putText(
        frame,
        text_current,
        (10, h - 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        text_unique,
        (10, h - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

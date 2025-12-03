# BodyTracker

BodyTracker is a webcam-based posture and face analysis demo built with FastAPI, OpenCV, and MediaPipe. It serves a lightweight frontend and exposes APIs for live MJPEG streaming or single-frame analysis that return pose landmarks, detected faces, and basic attributes.

## Features
- Live MJPEG webcam stream with overlaid stick-figure pose drawing and face tracking
- Face ID assignment with simple deduplication, optional DeepFace-based age/emotion estimation, and snapshot saving
- Single-frame `/analyze` API that accepts a base64 image payload and returns pose landmarks plus face metadata
- Smooth landmark output for less jitter in both the stream and HTTP analyze endpoint
- Static frontend (HTML/JS/CSS) served directly by FastAPI for quick local experimentation

## Requirements
- Python 3.10+
- A webcam accessible to OpenCV
- System packages needed by OpenCV/MediaPipe (e.g., `libglib2.0-0`, `libsm6`, `libxrender1`, `libxext6` on Debian/Ubuntu)

## Installation
1. Create and activate a virtual environment (recommended):
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```
2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Running the app
Start the FastAPI server (serves both the API and frontend):
```bash
uvicorn backend.video_server:app --reload
```
The server defaults to port 8000. Open http://localhost:8000 to load the frontend, which pulls the live MJPEG stream from `/video_feed`.

### API quick test
Send a base64-encoded image to the `/analyze` endpoint:
```bash
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"image": "<base64-data>"}'
```
The response includes `faces` (with bounding boxes and attributes) and `posture.landmarks` (smoothed pose keypoints).

## Project structure
- `backend/` – FastAPI app, pose/face utilities, tracking logic, model configuration
- `frontend/` – Static assets (HTML, JS, CSS) served by the backend
- `requirements.txt` – Python dependencies

## Notes
- The DeepFace warm-up call runs on startup to reduce first-inference latency.
- Face snapshots are stored in the `backend/faces/` directory for deduplication and inspection.

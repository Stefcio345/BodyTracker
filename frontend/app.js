// frontend/app.js

const statusEl = document.getElementById("status");
const img = document.getElementById("video-stream");

// Optional: elements for client-side analysis
const webcamVideo = document.getElementById("webcam");
const analysisStatusEl = document.getElementById("analysis-status");
const analysisOutputEl = document.getElementById("analysis-output");
const startAnalyzeBtn = document.getElementById("start-analyze");

function setStatus(text, ok = true) {
  statusEl.textContent = text;
  statusEl.style.borderColor = ok
    ? "rgba(45, 212, 191, 0.4)"
    : "rgba(248, 113, 113, 0.4)";
  statusEl.style.backgroundColor = ok
    ? "rgba(15, 118, 110, 0.15)"
    : "rgba(127, 29, 29, 0.3)";
  statusEl.style.color = ok ? "#a5f3fc" : "#fecaca";
}

function setAnalysisStatus(text, ok = true) {
  if (!analysisStatusEl) {
    // Fallback if element not present
    console.log("[analysis-status]", text);
    return;
  }
  analysisStatusEl.textContent = text;
  analysisStatusEl.style.borderColor = ok
    ? "rgba(56, 189, 248, 0.4)"
    : "rgba(248, 113, 113, 0.4)";
  analysisStatusEl.style.backgroundColor = ok
    ? "rgba(8, 47, 73, 0.6)"
    : "rgba(127, 29, 29, 0.4)";
  analysisStatusEl.style.color = ok ? "#bae6fd" : "#fecaca";
}

function setAnalysisOutput(data) {
  if (!analysisOutputEl) {
    console.log("[analysis-output]", data);
    return;
  }
  analysisOutputEl.textContent = JSON.stringify(data, null, 2);
}

// ---------- SERVER → FRONTEND: MJPEG stream ----------

setStatus("Connecting to /video_feed...");

// Kick off the MJPEG stream
img.src = "/video_feed";

img.addEventListener("load", () => {
  setStatus("Stream running");
});

img.addEventListener("error", () => {
  setStatus("Stream error – is backend running?", false);
});

// ---------- FRONTEND → SERVER: webcam → /analyze ----------

// Offscreen canvas for grabbing frames
const canvas = document.createElement("canvas");
const ctx = canvas.getContext("2d");

// Adjust these to match backend expectations
const CAM_WIDTH = 640;
const CAM_HEIGHT = 480;
canvas.width = CAM_WIDTH;
canvas.height = CAM_HEIGHT;

async function startWebcam() {
  if (!webcamVideo) {
    console.warn("No #webcam element found, cannot start client-side analysis webcam.");
    return;
  }

  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { width: CAM_WIDTH, height: CAM_HEIGHT }
    });
    webcamVideo.srcObject = stream;
    setAnalysisStatus("Webcam started, analyzing…");
  } catch (err) {
    console.error("Could not start webcam:", err);
    setAnalysisStatus("Cannot access webcam (permission denied?)", false);
    throw err;
  }
}

function frameToBase64() {
  if (!webcamVideo || webcamVideo.readyState < webcamVideo.HAVE_ENOUGH_DATA) {
    return null;
  }
  ctx.drawImage(webcamVideo, 0, 0, CAM_WIDTH, CAM_HEIGHT);
  // JPEG = smaller payload, faster
  return canvas.toDataURL("image/jpeg", 0.7);
}

let analyzeLoopRunning = false;

async function analyzeLoop() {
  if (analyzeLoopRunning) return;
  analyzeLoopRunning = true;

  while (true) {
    const base64 = frameToBase64();
    if (base64) {
      try {
        const res = await fetch("/analyze", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ image: base64 })
        });

        if (!res.ok) {
          const text = await res.text();
          console.error("Analyze error:", res.status, text);
          setAnalysisStatus(`Analyze error: ${res.status}`, false);
        } else {
          const data = await res.json();
          setAnalysisOutput(data);
          setAnalysisStatus("Analyzing…");
        }
      } catch (err) {
        console.error("Analyze request failed:", err);
        setAnalysisStatus("Analyze request failed", false);
      }
    }

    // Throttle – 5 fps (200 ms). Tune as needed.
    await new Promise((r) => setTimeout(r, 200));
  }
}

// Hook up button (user gesture is needed for getUserMedia on some browsers)
if (startAnalyzeBtn) {
  startAnalyzeBtn.addEventListener("click", async () => {
    try {
      await startWebcam();
      analyzeLoop();
    } catch {
      // Error already handled in startWebcam
    }
  });
} else {
  console.warn("No #start-analyze button – client-side /analyze won't start automatically.");
}

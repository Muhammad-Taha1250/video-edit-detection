"""
app.py – Flask HTTP Service for Video Edit Detection (Port 5001).

Endpoints:
  GET  /health                      Service health and status
  POST /analyze                     Start async video analysis: {video_path, job_id}
  GET  /jobs/<job_id>               Poll analysis status, progress, and results
  GET  /frames/<job_id>/<filename>  Serve extracted event junction frames
"""

from __future__ import annotations

import os
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any

import cv2
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

_HERE = Path(__file__).resolve().parent
AI_SERVICE = _HERE.parent
if str(AI_SERVICE) not in sys.path:
    sys.path.insert(0, str(AI_SERVICE))

from service.inference import (
    EXTRACTED_FRAMES_DIR,
    analyze_video,
    get_active_model_info,
    get_models,
)

app = Flask(__name__)
CORS(app)

# In-memory job registry
jobs: dict[str, dict[str, Any]] = {}
jobs_lock = threading.Lock()


def _process_job(job_id: str, video_path: str):
    """Worker task executed in background thread."""
    with jobs_lock:
        if job_id in jobs:
            jobs[job_id]["status"] = "processing"
            jobs[job_id]["progress"] = 5

    def progress_callback(pct: int, msg: str):
        with jobs_lock:
            if job_id in jobs:
                jobs[job_id]["progress"] = pct

    try:
        result = analyze_video(
            video_path=video_path,
            progress_cb=progress_callback,
            job_id=job_id,
        )
        with jobs_lock:
            if job_id in jobs:
                jobs[job_id]["status"] = "completed"
                jobs[job_id]["progress"] = 100
                jobs[job_id]["result"] = result
                jobs[job_id]["error"] = None
    except Exception as exc:
        with jobs_lock:
            if job_id in jobs:
                jobs[job_id]["status"] = "failed"
                jobs[job_id]["progress"] = 0
                jobs[job_id]["error"] = str(exc)


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint confirming service status and model readiness."""
    models_ready = False
    info = get_active_model_info()
    try:
        get_models()
        models_ready = True
    except Exception:
        models_ready = False

    return jsonify({
        "status": "ok",
        "service": "video-edit-detection-ai",
        "version": info.get("model_version", "3.0.0"),
        "models_loaded": models_ready,
        "model_set": info["model_set"],
        "training_data_description": info["training_data_description"],
    }), 200


@app.route("/analyze", methods=["POST"])
def analyze():
    """
    Starts an asynchronous video analysis job.
    Expects JSON: { "video_path": "...", "job_id": "..." (optional) }
    """
    data = request.get_json(silent=True)
    if not data or "video_path" not in data:
        return jsonify({
            "error": "Invalid request. Must provide JSON body with 'video_path'."
        }), 400

    raw_path = data.get("video_path")
    if not raw_path or not isinstance(raw_path, str) or not raw_path.strip():
        return jsonify({"error": "Field 'video_path' must be a non-empty string."}), 400

    v_path = Path(raw_path).expanduser().resolve()
    if not v_path.exists() or not v_path.is_file():
        return jsonify({
            "error": f"Video file not found at path: {raw_path}"
        }), 400

    # Probe readability with OpenCV
    cap = cv2.VideoCapture(str(v_path))
    if not cap.isOpened():
        cap.release()
        return jsonify({
            "error": f"Cannot open video file (unsupported format or corrupted): {raw_path}"
        }), 400
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()

    if frame_count < 2:
        return jsonify({
            "error": f"Video has {frame_count} frames; minimum 2 frames required for edit detection."
        }), 400

    job_id = data.get("job_id")
    if not job_id or not isinstance(job_id, str) or not job_id.strip():
        job_id = uuid.uuid4().hex[:12]
    else:
        job_id = job_id.strip()

    with jobs_lock:
        jobs[job_id] = {
            "job_id": job_id,
            "status": "pending",
            "progress": 0,
            "result": None,
            "error": None,
            "created_at": time.time(),
            "video_path": str(v_path),
        }

    worker = threading.Thread(
        target=_process_job,
        args=(job_id, str(v_path)),
        daemon=True,
    )
    worker.start()

    info = get_active_model_info()

    return jsonify({
        "job_id": job_id,
        "status": "pending",
        "message": "Analysis started in background.",
        "model_set": info["model_set"],
        "training_data_description": info["training_data_description"],
    }), 202


@app.route("/jobs/<job_id>", methods=["GET"])
def get_job(job_id: str):
    """Returns the current status, progress, and results (if ready) for a job."""
    with jobs_lock:
        job = jobs.get(job_id)

    if job is None:
        return jsonify({
            "error": f"Job not found: '{job_id}'"
        }), 404

    return jsonify({
        "job_id": job_id,
        "status": job["status"],
        "progress": job["progress"],
        "result": job["result"],
        "error": job["error"],
    }), 200


@app.route("/frames/<job_id>/<filename>", methods=["GET"])
def get_frame(job_id: str, filename: str):
    """Serves an extracted junction JPG frame."""
    target_dir = EXTRACTED_FRAMES_DIR / job_id
    target_file = target_dir / filename

    if not target_file.exists() or not target_file.is_file():
        return jsonify({
            "error": f"Frame '{filename}' not found for job '{job_id}'"
        }), 404

    return send_from_directory(str(target_dir), filename, mimetype="image/jpeg")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    print(f"Starting Video Edit Detection AI Service on port {port}...")
    app.run(host="127.0.0.1", port=port, debug=False, threaded=True)


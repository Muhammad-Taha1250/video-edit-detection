"""
test_service.py – Verification script for the Video Edit Detection AI Service.

Tests:
  1. Health check (GET /health)
  2. Error handling (POST /analyze with non-existent / invalid video path)
  3. Analysis of 3 videos:
     - 1 Clean video (unedited_clip_01_landscape_slow_r1.mp4)
     - 1 Edited video: Deletion (deletion_clip_01_landscape_slow_r1.mp4)
     - 1 Edited video: Insertion (insertion_clip_01_landscape_slow_r1.mp4)
  4. Polling /jobs/<job_id> until completed
  5. Frame retrieval check (GET /frames/<job_id>/<filename>)
  6. Prints JSON results
"""

from __future__ import annotations

import json
import time
from pathlib import Path
import urllib.request
import urllib.error

BASE_URL = "http://127.0.0.1:5001"
AI_SERVICE = Path(__file__).resolve().parent.parent
GENERATED_DIR = AI_SERVICE / "dataset" / "generated"


def http_get(endpoint: str) -> tuple[int, dict | bytes]:
    url = f"{BASE_URL}{endpoint}"
    req = urllib.request.Request(url)
    try:
        with urllib.request.urlopen(req) as resp:
            content_type = resp.headers.get("Content-Type", "")
            data = resp.read()
            if "application/json" in content_type:
                return resp.status, json.loads(data.decode("utf-8"))
            return resp.status, data
    except urllib.error.HTTPError as e:
        data = e.read()
        try:
            return e.code, json.loads(data.decode("utf-8"))
        except Exception:
            return e.code, {"raw_error": data.decode("utf-8", errors="ignore")}


def http_post(endpoint: str, payload: dict) -> tuple[int, dict]:
    url = f"{BASE_URL}{endpoint}"
    data_bytes = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data_bytes,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        data = e.read()
        try:
            return e.code, json.loads(data.decode("utf-8"))
        except Exception:
            return e.code, {"raw_error": data.decode("utf-8", errors="ignore")}


def main():
    print("=" * 70)
    print("TESTING AI SERVICE (PORT 5001)")
    print("=" * 70)

    # 1. Health check
    print("\n[1] Checking GET /health ...")
    status, health_resp = http_get("/health")
    print(f"    HTTP {status}: {json.dumps(health_resp)}")
    assert status == 200, f"Health check failed with {status}"
    assert health_resp.get("models_loaded") is True, "Models not loaded!"

    # 2. Error handling checks
    print("\n[2] Checking Error Handling ...")
    status_bad, bad_resp = http_post("/analyze", {"video_path": "/path/to/nonexistent/file.mp4"})
    print(f"    POST bad path -> HTTP {status_bad}: {bad_resp}")
    assert status_bad == 400, "Expected HTTP 400 for bad path"

    status_404, bad_job = http_get("/jobs/non_existent_job_12345")
    print(f"    GET bad job id -> HTTP {status_404}: {bad_job}")
    assert status_404 == 404, "Expected HTTP 404 for nonexistent job"

    # 3. Test 3 Videos
    test_videos = [
        {
            "category": "Clean (Unedited)",
            "file": "unedited_clip_01_landscape_slow_r1.mp4",
            "job_id": "job_clean_01",
        },
        {
            "category": "Edited (Deletion)",
            "file": "deletion_clip_01_landscape_slow_r1.mp4",
            "job_id": "job_deletion_01",
        },
        {
            "category": "Edited (Insertion)",
            "file": "insertion_clip_01_landscape_slow_r1.mp4",
            "job_id": "job_insertion_01",
        },
    ]

    print("\n[3] Submitting 3 Analysis Jobs to POST /analyze ...")
    for tv in test_videos:
        vpath = (GENERATED_DIR / tv["file"]).resolve()
        assert vpath.exists(), f"Video file not found: {vpath}"
        status_sub, sub_resp = http_post("/analyze", {
            "video_path": str(vpath),
            "job_id": tv["job_id"],
        })
        print(f"    Submitted [{tv['category']}] {tv['file']} -> HTTP {status_sub}: {sub_resp}")
        assert status_sub == 202, f"Expected HTTP 202, got {status_sub}"

    print("\n[4] Polling GET /jobs/<job_id> until completion ...")
    completed_results = {}
    for tv in test_videos:
        jid = tv["job_id"]
        print(f"\n  Polling job: {jid} ({tv['category']})")
        while True:
            st, job_info = http_get(f"/jobs/{jid}")
            status_val = job_info.get("status")
            progress_val = job_info.get("progress")
            print(f"    status: {status_val:<10} progress: {progress_val}%")
            if status_val == "completed":
                completed_results[jid] = job_info
                break
            elif status_val == "failed":
                raise RuntimeError(f"Job {jid} failed: {job_info.get('error')}")
            time.sleep(0.5)

    print("\n[5] Checking Frame Retrieval (GET /frames/<job_id>/<filename>) ...")
    for jid, job_info in completed_results.items():
        res = job_info.get("result", {})
        events = res.get("events", [])
        print(f"    Job {jid}: {len(events)} events detected.")
        for ev in events:
            frames_dict = ev.get("extracted_frames", {})
            for role, finfo in frames_dict.items():
                url = finfo["url"]
                fst, fdata = http_get(url)
                is_valid_jpeg = isinstance(fdata, bytes) and fdata.startswith(b"\xff\xd8")
                print(f"      Frame [{role}] GET {url} -> HTTP {fst} (JPEG valid: {is_valid_jpeg}, bytes: {len(fdata):,})")
                assert fst == 200, f"Failed to retrieve frame from {url}"
                assert is_valid_jpeg, "Frame is not a valid JPEG"

    print("\n" + "=" * 70)
    print("FINAL JSON RESULTS FOR THE THREE TEST VIDEOS")
    print("=" * 70)
    for tv in test_videos:
        jid = tv["job_id"]
        job_info = completed_results[jid]
        res = job_info.get("result", {})
        # Create a display copy with cut_probabilities summarized to avoid flooding stdout
        display_res = dict(res)
        probs = display_res.get("cut_probabilities", [])
        display_res["cut_probabilities_summary"] = {
            "total_points": len(probs),
            "first_3_points": probs[:3],
            "last_3_points": probs[-3:],
            "max_probability": max([p["prob"] for p in probs]) if probs else 0.0,
        }
        del display_res["cut_probabilities"]

        print(f"\n--- {tv['category']}: {tv['file']} (Job ID: {jid}) ---")
        print(json.dumps({
            "job_id": jid,
            "status": job_info["status"],
            "progress": job_info["progress"],
            "result": display_res,
        }, indent=2))

    print("\n" + "=" * 70)
    print("ALL TESTS PASSED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    main()


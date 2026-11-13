// ============================================
// Video Edit Detection - Express API Server
// ============================================
import express from "express";
import cors from "cors";
import mysql from "mysql2/promise";
import multer from "multer";
import axios from "axios";
import path from "path";
import fs from "fs";
import dotenv from "dotenv";

dotenv.config();

const app = express();
app.use(cors());
app.use(express.json());

const PORT = process.env.PORT || 4000;
const FLASK_URL = process.env.AI_SERVICE_URL || "http://127.0.0.1:5001";

// Ensure uploads directory exists
const UPLOADS_DIR = path.resolve("uploads");
if (!fs.existsSync(UPLOADS_DIR)) {
  fs.mkdirSync(UPLOADS_DIR, { recursive: true });
}

// MySQL connection pool
const pool = mysql.createPool({
  host: process.env.DB_HOST || "localhost",
  user: process.env.DB_USER || "root",
  password: process.env.DB_PASSWORD || "",
  database: process.env.DB_NAME || "video_edit_detection",
  waitForConnections: true,
  connectionLimit: 10,
});

// Auto-run Migration 003: ensure nullable evidence_z column exists in edit_segments
(async function ensureEvidenceZColumn() {
  try {
    const [cols] = await pool.query(`
      SELECT COLUMN_NAME 
      FROM INFORMATION_SCHEMA.COLUMNS 
      WHERE TABLE_SCHEMA = DATABASE() 
        AND TABLE_NAME = 'edit_segments' 
        AND COLUMN_NAME = 'evidence_z'
    `);
    if (cols.length === 0) {
      console.log("[Migration 003] Adding nullable evidence_z column to edit_segments...");
      await pool.query("ALTER TABLE edit_segments ADD COLUMN evidence_z FLOAT NULL DEFAULT NULL");
      console.log("[Migration 003] Column evidence_z added successfully.");
    }
  } catch (err) {
    console.warn("[Migration 003] Notice checking/adding evidence_z column:", err.message);
  }
})();

// Configure Multer storage and file filter
const storage = multer.diskStorage({
  destination: (req, file, cb) => {
    cb(null, UPLOADS_DIR);
  },
  filename: (req, file, cb) => {
    const ext = path.extname(file.originalname).toLowerCase();
    const unique = `${Date.now()}-${Math.random().toString(36).substring(2, 9)}${ext}`;
    cb(null, unique);
  },
});

const ALLOWED_EXTS = new Set([".mp4", ".mov", ".avi", ".mkv", ".webm"]);

const fileFilter = (req, file, cb) => {
  const ext = path.extname(file.originalname).toLowerCase();
  if (ALLOWED_EXTS.has(ext)) {
    cb(null, true);
  } else {
    cb(new Error(`Invalid file type (${ext}). Only mp4, mov, avi, mkv, webm are supported.`));
  }
};

const upload = multer({
  storage,
  limits: { fileSize: 500 * 1024 * 1024 }, // 500MB
  fileFilter,
});

// ==========================================================
// BACKGROUND ANALYSIS WORKER & POLLER
// ==========================================================

async function startAnalysisJob(videoId, filePath) {
  const absolutePath = path.resolve(filePath);
  console.log(`[Job ${videoId}] Starting AI analysis for: ${absolutePath}`);

  try {
    await pool.query(
      "UPDATE videos SET status = 'processing', progress = 5, error_message = NULL WHERE video_id = ?",
      [videoId]
    );

    // Call Flask POST /analyze
    const analyzeResp = await axios.post(
      `${FLASK_URL}/analyze`,
      {
        video_path: absolutePath,
        job_id: String(videoId),
      },
      { timeout: 15000 }
    );

    console.log(`[Job ${videoId}] Flask analysis initiated:`, analyzeResp.data);

    // Start background poller
    pollJobStatus(videoId);
  } catch (err) {
    const errorMsg =
      err.code === "ECONNREFUSED"
        ? `AI Service (Flask) is offline at ${FLASK_URL}. Please start the Flask service on port 5001.`
        : err.response?.data?.error || err.message || "Failed to start AI analysis";

    console.error(`[Job ${videoId}] Error initiating analysis:`, errorMsg);

    try {
      await pool.query(
        "UPDATE videos SET status = 'failed', progress = 0, error_message = ? WHERE video_id = ?",
        [errorMsg, videoId]
      );
    } catch (dbErr) {
      console.error(`[Job ${videoId}] Database update failed:`, dbErr);
    }
  }
}

function pollJobStatus(videoId) {
  const maxAttempts = 600; // 10 minutes max
  let attempts = 0;

  const intervalId = setInterval(async () => {
    attempts++;
    if (attempts > maxAttempts) {
      clearInterval(intervalId);
      console.error(`[Job ${videoId}] Polling timed out`);
      await pool
        .query(
          "UPDATE videos SET status = 'failed', error_message = 'Analysis timed out after 10 minutes' WHERE video_id = ?",
          [videoId]
        )
        .catch(() => {});
      return;
    }

    try {
      const resp = await axios.get(`${FLASK_URL}/jobs/${videoId}`, { timeout: 10000 });
      const jobData = resp.data;

      if (!jobData) return;

      if (jobData.status === "processing") {
        const progress = typeof jobData.progress === "number" ? jobData.progress : 15;
        await pool.query(
          "UPDATE videos SET status = 'processing', progress = ? WHERE video_id = ?",
          [progress, videoId]
        );
      } else if (jobData.status === "completed") {
        clearInterval(intervalId);
        console.log(`[Job ${videoId}] Analysis completed in AI Service. Writing results to DB...`);
        await saveCompletedJobResults(videoId, jobData.result);
      } else if (jobData.status === "failed") {
        clearInterval(intervalId);
        const errMsg = jobData.error || "Analysis failed in AI Service";
        console.error(`[Job ${videoId}] Analysis reported failure:`, errMsg);
        await pool.query(
          "UPDATE videos SET status = 'failed', progress = 0, error_message = ? WHERE video_id = ?",
          [errMsg, videoId]
        );
      }
    } catch (pollErr) {
      const errMsg =
        pollErr.code === "ECONNREFUSED"
          ? "AI Service connection lost during polling"
          : pollErr.message || "Polling error";
      console.error(`[Job ${videoId}] Polling error:`, errMsg);

      if (pollErr.code === "ECONNREFUSED" || (attempts >= 5 && pollErr.response?.status === 404)) {
        clearInterval(intervalId);
        await pool
          .query(
            "UPDATE videos SET status = 'failed', error_message = ? WHERE video_id = ?",
            [errMsg, videoId]
          )
          .catch(() => {});
      }
    }
  }, 500);
}

// Cache for active model metadata passed from Flask analysis results
const jobModelMeta = new Map();

async function saveCompletedJobResults(videoId, result) {
  if (!result) return;
  if (result.model_set || result.training_data_description) {
    jobModelMeta.set(String(videoId), {
      model_set: result.model_set,
      training_data_description: result.training_data_description,
    });
  }
  const conn = await pool.getConnection();
  try {
    await conn.beginTransaction();

    const fps = result.fps || 30.0;
    const totalFrames = result.total_frames || 0;
    const duration = result.duration || 0;
    const events = Array.isArray(result.events) ? result.events : [];
    const cutProbs = Array.isArray(result.cut_probabilities) ? result.cut_probabilities : [];

    // 1. Update videos table
    await conn.query(
      `UPDATE videos 
       SET status = 'completed', progress = 100, fps = ?, total_frames = ?, duration = ?, cut_probabilities = ?, error_message = NULL
       WHERE video_id = ?`,
      [fps, totalFrames, duration, JSON.stringify(cutProbs), videoId]
    );

    // 2. Average confidence score (0-100)
    let avgConfidence = 95.0;
    if (events.length > 0) {
      const sum = events.reduce((acc, ev) => acc + (ev.confidence || 0), 0);
      avgConfidence = Math.round((sum / events.length) * 1000) / 10;
    }

    // 3. Upsert analysis_results
    const [existingRes] = await conn.query(
      "SELECT result_id FROM analysis_results WHERE video_id = ?",
      [videoId]
    );
    let resultId;
    if (existingRes.length > 0) {
      resultId = existingRes[0].result_id;
      await conn.query(
        `UPDATE analysis_results 
         SET total_frames = ?, edits_detected = ?, confidence_score = ?, model_name = 'VideoEditDetect v3', model_version = '3.0.0', cut_probabilities = ?, analysis_date = NOW()
         WHERE result_id = ?`,
        [totalFrames, events.length, avgConfidence, JSON.stringify(cutProbs), resultId]
      );
    } else {
      const [insRes] = await conn.query(
        `INSERT INTO analysis_results (video_id, total_frames, edits_detected, confidence_score, model_name, model_version, cut_probabilities)
         VALUES (?, ?, ?, ?, 'VideoEditDetect v3', '3.0.0', ?)`,
        [videoId, totalFrames, events.length, avgConfidence, JSON.stringify(cutProbs)]
      );
      resultId = insRes.insertId;
    }

    // 4. Clear previous segments and detected_frames
    await conn.query("DELETE FROM edit_segments WHERE video_id = ?", [videoId]);
    await conn.query("DELETE FROM detected_frames WHERE result_id = ?", [resultId]);

    // 5. Insert edit_segments & detected_frames
    for (const ev of events) {
      const eType = ev.type || ev.edit_type || "cut";
      const sFrame = ev.start_frame || 0;
      const eFrame = ev.end_frame || sFrame;
      const sTime = ev.start_time !== undefined ? ev.start_time : sFrame / fps;
      const eTime = ev.end_time !== undefined ? ev.end_time : eFrame / fps;
      const conf = ev.confidence || 0.0;
      const evidenceZ = ev.evidence_z !== undefined && ev.evidence_z !== null ? Number(ev.evidence_z) : null;
      const framesObj = ev.extracted_frames || {};

      await conn.query(
        `INSERT INTO edit_segments (video_id, result_id, edit_type, start_frame, end_frame, start_time, end_time, confidence, evidence_z, extracted_frames)
         VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
        [videoId, resultId, eType, sFrame, eFrame, sTime, eTime, conf, evidenceZ, JSON.stringify(framesObj)]
      );

      const mm = Math.floor(sTime / 60)
        .toString()
        .padStart(2, "0");
      const ss = Math.floor(sTime % 60)
        .toString()
        .padStart(2, "0");
      const timeStr = `00:${mm}:${ss}`;

      await conn.query(
        `INSERT INTO detected_frames (result_id, frame_number, timestamp, difference_score, is_suspicious)
         VALUES (?, ?, ?, ?, TRUE)`,
        [resultId, sFrame, timeStr, conf]
      );
    }

    await conn.commit();
    console.log(`[Job ${videoId}] Stored ${events.length} edit_segments successfully.`);
  } catch (err) {
    await conn.rollback();
    console.error(`[Job ${videoId}] Failed saving completed analysis results:`, err);
  } finally {
    conn.release();
  }
}

// ==========================================================
// ROUTES
// ==========================================================

// ---------- Health check ----------
app.get("/api/health", (req, res) => {
  res.json({ status: "ok", message: "API is running", service: "express-gateway" });
});

// ---------- POST: upload video file & start analysis ----------
app.post("/api/videos/upload", (req, res) => {
  const uploadSingle = upload.fields([
    { name: "video", maxCount: 1 },
    { name: "file", maxCount: 1 },
  ]);

  uploadSingle(req, res, async (err) => {
    if (err instanceof multer.MulterError) {
      if (err.code === "LIMIT_FILE_SIZE") {
        return res.status(400).json({ error: "File exceeds maximum size limit of 500MB" });
      }
      return res.status(400).json({ error: `Upload error: ${err.message}` });
    } else if (err) {
      return res.status(400).json({ error: err.message });
    }

    const file =
      (req.files?.video && req.files.video[0]) || (req.files?.file && req.files.file[0]);

    if (!file) {
      return res.status(400).json({
        error: "No video file provided. Please upload a file under field 'video'.",
      });
    }

    try {
      const ext = path.extname(file.originalname).replace(".", "").toUpperCase();
      const [insertResult] = await pool.query(
        `INSERT INTO videos (user_id, filename, file_size, format, file_path, status, progress)
         VALUES (?, ?, ?, ?, ?, 'pending', 0)`,
        [1, file.originalname, file.size, ext, file.path]
      );
      const videoId = insertResult.insertId;

      // Kick off background analysis job
      startAnalysisJob(videoId, file.path);

      res.status(201).json({
        video_id: videoId,
        id: videoId,
        filename: file.originalname,
        status: "pending",
        message: "Video uploaded successfully. Analysis queued.",
      });
    } catch (dbErr) {
      console.error("DB error during upload:", dbErr);
      res.status(500).json({ error: "Failed to record uploaded video in database" });
    }
  });
});

// ---------- GET: poll video analysis status ----------
app.get("/api/videos/:id/status", async (req, res) => {
  try {
    const { id } = req.params;
    const [[video]] = await pool.query(
      "SELECT video_id, status, progress, error_message FROM videos WHERE video_id = ?",
      [id]
    );
    if (!video) {
      return res.status(404).json({ error: `Video with ID ${id} not found` });
    }
    res.json({
      video_id: video.video_id,
      status: video.status,
      progress: video.progress,
      error: video.error_message,
    });
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: "Failed to fetch video status" });
  }
});

// ---------- GET: fetch full analysis results (events, frames, series, summary) ----------
app.get("/api/videos/:id/results", async (req, res) => {
  try {
    const { id } = req.params;
    const [[video]] = await pool.query("SELECT * FROM videos WHERE video_id = ?", [id]);
    if (!video) {
      return res.status(404).json({ error: `Video with ID ${id} not found` });
    }

    const [[result]] = await pool.query(
      "SELECT * FROM analysis_results WHERE video_id = ?",
      [id]
    );

    const [segments] = await pool.query(
      "SELECT * FROM edit_segments WHERE video_id = ? ORDER BY start_frame ASC",
      [id]
    );

    const [detectedFrames] = result
      ? await pool.query(
          "SELECT * FROM detected_frames WHERE result_id = ? ORDER BY frame_number ASC",
          [result.result_id]
        )
      : [[]];

    const events = segments.map((s) => ({
      ...s,
      evidence_z: s.evidence_z !== null && s.evidence_z !== undefined ? Number(s.evidence_z) : null,
      extracted_frames:
        typeof s.extracted_frames === "string" ? JSON.parse(s.extracted_frames) : s.extracted_frames,
    }));

    let series = [];
    if (video.cut_probabilities) {
      series =
        typeof video.cut_probabilities === "string"
          ? JSON.parse(video.cut_probabilities)
          : video.cut_probabilities;
    } else if (result?.cut_probabilities) {
      series =
        typeof result.cut_probabilities === "string"
          ? JSON.parse(result.cut_probabilities)
          : result.cut_probabilities;
    }

    const frames = [];
    events.forEach((ev) => {
      if (ev.extracted_frames) {
        Object.entries(ev.extracted_frames).forEach(([role, fInfo]) => {
          frames.push({
            event_id: ev.segment_id,
            edit_type: ev.edit_type,
            role,
            frame: fInfo.frame,
            url: `/api/videos/${id}/frames/${id}/${fInfo.filename}`,
            flask_url: fInfo.url,
            relative_path: fInfo.relative_path,
          });
        });
      }
    });

    // Retrieve model metadata from job cache, Flask /health, or active model card
    let meta = jobModelMeta.get(String(id));
    if (!meta || !meta.training_data_description) {
      try {
        const healthResp = await axios.get(`${FLASK_URL}/health`, { timeout: 1000 });
        if (healthResp.data) {
          meta = {
            model_set: healthResp.data.model_set,
            training_data_description: healthResp.data.training_data_description,
          };
          jobModelMeta.set(String(id), meta);
        }
      } catch (e) {}
    }

    if (!meta || !meta.training_data_description) {
      try {
        const candidates = [
          path.resolve("ai-service/models/final_real/model_card.json"),
          path.resolve("../ai-service/models/final_real/model_card.json"),
          path.resolve("ai-service/models/final_v3/model_card.json"),
          path.resolve("../ai-service/models/final_v3/model_card.json"),
        ];
        for (const p of candidates) {
          if (fs.existsSync(p)) {
            const card = JSON.parse(fs.readFileSync(p, "utf8"));
            meta = {
              model_set: p.includes("final_real") ? "final_real" : "final_v3",
              training_data_description: card.training_data_description || card.training_data,
            };
            break;
          }
        }
      } catch (e) {}
    }

    const modelSet = meta?.model_set || "final_v3";
    const trainingDataDesc = meta?.training_data_description || "synthetic smoke-test clips";

    const summary = {
      video_id: video.video_id,
      filename: video.filename,
      status: video.status,
      duration: video.duration,
      fps: video.fps,
      total_frames: video.total_frames,
      edits_detected: result?.edits_detected ?? events.length,
      confidence_score: result?.confidence_score ?? 0,
      model_name: result?.model_name || "VideoEditDetect v3",
      model_version: result?.model_version || "3.0.0",
      model_set: modelSet,
      training_data_description: trainingDataDesc,
      analysis_date: result?.analysis_date || video.upload_date,
      error: video.error_message,
    };

    res.json({
      video_id: Number(id),
      status: video.status,
      summary,
      events,
      frames,
      series,
      detected_frames: detectedFrames,
      model_set: modelSet,
      training_data_description: trainingDataDesc,
    });
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: "Failed to fetch video results" });
  }
});

// ---------- GET: proxy extracted frame JPG from Flask ----------
app.get("/api/videos/:id/frames/:job/:file", async (req, res) => {
  const { job, file } = req.params;
  try {
    const flaskUrl = `${FLASK_URL}/frames/${encodeURIComponent(job)}/${encodeURIComponent(file)}`;
    const streamResp = await axios.get(flaskUrl, { responseType: "stream" });
    res.setHeader("Content-Type", streamResp.headers["content-type"] || "image/jpeg");
    streamResp.data.pipe(res);
  } catch (err) {
    if (err.response?.status === 404) {
      return res.status(404).json({ error: `Frame '${file}' for job '${job}' not found` });
    }
    console.error("Frame proxy error:", err.message);
    res.status(502).json({ error: "Failed to proxy frame from AI Service" });
  }
});

// ---------- GET: fetch all videos with their analysis results (history list) ----------
app.get("/api/videos", async (req, res) => {
  try {
    const [rows] = await pool.query(`
      SELECT 
        v.video_id, v.filename, v.file_size, v.duration, v.format,
        v.upload_date, v.status, v.progress, v.fps, v.total_frames, v.error_message,
        r.result_id, r.edits_detected, r.confidence_score, r.analysis_date
      FROM videos v
      LEFT JOIN analysis_results r ON v.video_id = r.video_id
      ORDER BY v.upload_date DESC
    `);
    res.json(rows);
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: "Failed to fetch videos" });
  }
});

// ---------- GET: fetch single video (existing shape for analysis.tsx / audit) ----------
app.get("/api/videos/:id", async (req, res) => {
  try {
    const { id } = req.params;

    const [[video]] = await pool.query("SELECT * FROM videos WHERE video_id = ?", [id]);
    if (!video) return res.status(404).json({ error: "Video not found" });

    const [[result]] = await pool.query(
      "SELECT * FROM analysis_results WHERE video_id = ?",
      [id]
    );

    const [segments] = await pool.query(
      "SELECT * FROM edit_segments WHERE video_id = ? ORDER BY start_frame ASC",
      [id]
    );

    let frames = [];
    if (result) {
      const [frameRows] = await pool.query(
        "SELECT * FROM detected_frames WHERE result_id = ? ORDER BY frame_number ASC",
        [result.result_id]
      );
      frames = frameRows;
    }

    res.json({ video, result, segments, frames });
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: "Failed to fetch video details" });
  }
});

// ---------- DELETE: remove a video (also removes file and edit_segments) ----------
app.delete("/api/videos/:id", async (req, res) => {
  try {
    const { id } = req.params;
    const [[video]] = await pool.query("SELECT file_path FROM videos WHERE video_id = ?", [id]);
    if (!video) {
      return res.status(404).json({ error: "Video not found" });
    }

    if (video.file_path && fs.existsSync(video.file_path)) {
      try {
        fs.unlinkSync(video.file_path);
        console.log(`Deleted file from disk: ${video.file_path}`);
      } catch (fErr) {
        console.warn(`Could not delete file ${video.file_path}:`, fErr.message);
      }
    }

    await pool.query("DELETE FROM edit_segments WHERE video_id = ?", [id]);
    await pool.query("DELETE FROM videos WHERE video_id = ?", [id]);
    res.json({ message: "Video and associated results deleted successfully" });
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: "Failed to delete video" });
  }
});

// ---------- GET: stats summary (for history page cards) ----------
app.get("/api/stats", async (req, res) => {
  try {
    const [[stats]] = await pool.query(`
      SELECT 
        COUNT(*) AS total,
        SUM(CASE WHEN r.edits_detected > 5 THEN 1 ELSE 0 END) AS suspicious,
        SUM(CASE WHEN r.edits_detected <= 5 OR r.edits_detected IS NULL THEN 1 ELSE 0 END) AS clean
      FROM videos v
      LEFT JOIN analysis_results r ON v.video_id = r.video_id
    `);
    res.json(stats);
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: "Failed to fetch stats" });
  }
});

// ---------- POST: legacy mock endpoint (kept for backward compatibility) ----------
app.post("/api/videos", async (req, res) => {
  const conn = await pool.getConnection();
  try {
    const {
      filename,
      file_size,
      duration,
      format,
      total_frames,
      edits_detected,
      confidence_score,
      frames,
      user_id = 1,
    } = req.body;

    if (!filename) {
      return res.status(400).json({ error: "filename is required" });
    }

    await conn.beginTransaction();

    const [videoResult] = await conn.query(
      `INSERT INTO videos (user_id, filename, file_size, duration, format, status)
       VALUES (?, ?, ?, ?, ?, 'completed')`,
      [user_id, filename, file_size || 0, duration || 0, format || "MP4"]
    );
    const video_id = videoResult.insertId;

    const [analysisResult] = await conn.query(
      `INSERT INTO analysis_results (video_id, total_frames, edits_detected, confidence_score)
       VALUES (?, ?, ?, ?)`,
      [video_id, total_frames || 0, edits_detected || 0, confidence_score || 0]
    );
    const result_id = analysisResult.insertId;

    if (Array.isArray(frames) && frames.length > 0) {
      const values = frames.map((f) => [
        result_id,
        f.frame_number,
        f.timestamp,
        f.difference_score,
        f.is_suspicious || false,
      ]);
      await conn.query(
        `INSERT INTO detected_frames (result_id, frame_number, timestamp, difference_score, is_suspicious)
         VALUES ?`,
        [values]
      );
    }

    await conn.commit();
    res.status(201).json({ video_id, result_id, message: "Video saved successfully" });
  } catch (err) {
    await conn.rollback();
    console.error(err);
    res.status(500).json({ error: "Failed to save video" });
  } finally {
    conn.release();
  }
});

app.listen(PORT, () => {
  console.log(`✅ API server running at http://localhost:${PORT}`);
});

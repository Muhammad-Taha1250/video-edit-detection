// ============================================
// Video Edit Detection - Express API Server
// ============================================
import express from "express";
import cors from "cors";
import mysql from "mysql2/promise";

const app = express();
app.use(cors());
app.use(express.json());

const PORT = process.env.PORT || 4000;

// MySQL connection pool
const pool = mysql.createPool({
  host: process.env.DB_HOST || "localhost",
  user: process.env.DB_USER || "root",
  password: process.env.DB_PASSWORD || "",
  database: process.env.DB_NAME || "video_edit_detection",
  waitForConnections: true,
  connectionLimit: 10,
});

// ---------- Health check ----------
app.get("/api/health", (req, res) => {
  res.json({ status: "ok", message: "API is running" });
});

// ---------- GET: fetch all videos with their analysis results ----------
app.get("/api/videos", async (req, res) => {
  try {
    const [rows] = await pool.query(`
      SELECT 
        v.video_id, v.filename, v.file_size, v.duration, v.format,
        v.upload_date, v.status,
        r.result_id, r.total_frames, r.edits_detected, r.confidence_score, r.analysis_date
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

// ---------- GET: fetch single video with full analysis + frames ----------
app.get("/api/videos/:id", async (req, res) => {
  try {
    const { id } = req.params;

    const [[video]] = await pool.query(
      "SELECT * FROM videos WHERE video_id = ?",
      [id]
    );
    if (!video) return res.status(404).json({ error: "Video not found" });

    const [[result]] = await pool.query(
      "SELECT * FROM analysis_results WHERE video_id = ?",
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

    res.json({ video, result, frames });
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: "Failed to fetch video details" });
  }
});

// ---------- POST: upload a video record + analysis result ----------
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
      frames, // optional array of { frame_number, timestamp, difference_score, is_suspicious }
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

// ---------- DELETE: remove a video (cascades to results + frames) ----------
app.delete("/api/videos/:id", async (req, res) => {
  try {
    const { id } = req.params;
    await pool.query("DELETE FROM videos WHERE video_id = ?", [id]);
    res.json({ message: "Video deleted" });
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

app.listen(PORT, () => {
  console.log(`✅ API server running at http://localhost:${PORT}`);
});

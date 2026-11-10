-- ============================================
-- Video Edit Detection & Frame Extraction System
-- Database Schema (MySQL)
-- ============================================

CREATE DATABASE IF NOT EXISTS video_edit_detection;
USE video_edit_detection;

DROP TABLE IF EXISTS detected_frames;
DROP TABLE IF EXISTS analysis_results;
DROP TABLE IF EXISTS videos;
DROP TABLE IF EXISTS users;

CREATE TABLE users (
  user_id INT AUTO_INCREMENT PRIMARY KEY,
  username VARCHAR(100) UNIQUE NOT NULL,
  email VARCHAR(150) UNIQUE NOT NULL,
  password_hash VARCHAR(255) NOT NULL,
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE videos (
  video_id INT AUTO_INCREMENT PRIMARY KEY,
  user_id INT,
  filename VARCHAR(255) NOT NULL,
  file_size BIGINT,
  duration FLOAT,
  format VARCHAR(10),
  upload_date DATETIME DEFAULT CURRENT_TIMESTAMP,
  status ENUM('pending', 'processing', 'completed') DEFAULT 'pending',
  FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
);

CREATE TABLE analysis_results (
  result_id INT AUTO_INCREMENT PRIMARY KEY,
  video_id INT NOT NULL,
  total_frames INT DEFAULT 0,
  edits_detected INT DEFAULT 0,
  confidence_score FLOAT DEFAULT 0,
  analysis_date DATETIME DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY (video_id) REFERENCES videos(video_id) ON DELETE CASCADE
);

CREATE TABLE detected_frames (
  frame_id INT AUTO_INCREMENT PRIMARY KEY,
  result_id INT NOT NULL,
  frame_number INT,
  timestamp VARCHAR(20),
  difference_score FLOAT,
  is_suspicious BOOLEAN DEFAULT FALSE,
  FOREIGN KEY (result_id) REFERENCES analysis_results(result_id) ON DELETE CASCADE
);

-- ============================================
-- Seed Data (Demo Records)
-- ============================================

INSERT INTO users (username, email, password_hash) VALUES
('demo_user', 'demo@videoguard.com', 'hashed_demo_password');

INSERT INTO videos (user_id, filename, file_size, duration, format, upload_date, status) VALUES
(1, 'interview_clip_01.mp4', 45200000, 124.5, 'MP4', '2026-06-20 10:15:00', 'completed'),
(1, 'news_segment_evening.mp4', 78500000, 210.0, 'MP4', '2026-06-22 14:30:00', 'completed'),
(1, 'product_review_unboxing.mov', 62300000, 180.2, 'MOV', '2026-06-24 09:05:00', 'completed'),
(1, 'security_footage_lobby.avi', 102400000, 300.0, 'AVI', '2026-06-26 18:45:00', 'completed'),
(1, 'social_media_clip.mp4', 18900000, 45.8, 'MP4', '2026-06-28 11:20:00', 'completed'),
(1, 'court_evidence_video.mp4', 95600000, 264.3, 'MP4', '2026-06-29 16:00:00', 'completed');

INSERT INTO analysis_results (video_id, total_frames, edits_detected, confidence_score, analysis_date) VALUES
(1, 2988, 3, 96.5, '2026-06-20 10:18:00'),
(2, 5040, 12, 89.2, '2026-06-22 14:35:00'),
(3, 4325, 1, 98.7, '2026-06-24 09:09:00'),
(4, 7200, 0, 99.1, '2026-06-26 18:50:00'),
(5, 1100, 8, 84.4, '2026-06-28 11:23:00'),
(6, 6343, 15, 91.8, '2026-06-29 16:06:00');

INSERT INTO detected_frames (result_id, frame_number, timestamp, difference_score, is_suspicious) VALUES
(1, 412, '00:00:13', 0.81, TRUE),
(1, 890, '00:00:29', 0.76, TRUE),
(1, 2104, '00:01:10', 0.69, TRUE),
(2, 120, '00:00:04', 0.72, TRUE),
(2, 980, '00:00:32', 0.88, TRUE),
(2, 3300, '00:01:50', 0.91, TRUE),
(3, 2200, '00:01:13', 0.65, TRUE),
(5, 230, '00:00:09', 0.79, TRUE),
(5, 540, '00:00:21', 0.83, TRUE),
(6, 1500, '00:01:02', 0.94, TRUE),
(6, 4800, '00:03:18', 0.87, TRUE);

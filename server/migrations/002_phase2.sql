-- ============================================
-- Migration 002: Phase 2 Real ML Video Analysis Schema
-- Safe to run multiple times without data loss
-- ============================================

USE video_edit_detection;

-- 1. Create edit_segments table if it doesn't already exist
CREATE TABLE IF NOT EXISTS edit_segments (
  segment_id INT AUTO_INCREMENT PRIMARY KEY,
  video_id INT NOT NULL,
  result_id INT NULL,
  edit_type VARCHAR(50) NOT NULL,
  start_frame INT NOT NULL,
  end_frame INT NOT NULL,
  start_time FLOAT NOT NULL,
  end_time FLOAT NOT NULL,
  confidence FLOAT DEFAULT 0,
  extracted_frames JSON NULL,
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY (video_id) REFERENCES videos(video_id) ON DELETE CASCADE
);

-- 2. Helper procedure to safely add columns if they don't already exist
DROP PROCEDURE IF EXISTS AddColumnIfNotExists;
DELIMITER $$
CREATE PROCEDURE AddColumnIfNotExists(
    IN dbName VARCHAR(64),
    IN tableName VARCHAR(64),
    IN colName VARCHAR(64),
    IN colDef TEXT
)
BEGIN
    IF NOT EXISTS (
        SELECT * FROM INFORMATION_SCHEMA.COLUMNS
        WHERE TABLE_SCHEMA = dbName
          AND TABLE_NAME = tableName
          AND COLUMN_NAME = colName
    ) THEN
        SET @sql = CONCAT('ALTER TABLE `', dbName, '`.`', tableName, '` ADD COLUMN `', colName, '` ', colDef);
        PREPARE stmt FROM @sql;
        EXECUTE stmt;
        DEALLOCATE PREPARE stmt;
    END IF;
END$$
DELIMITER ;

-- 3. Safely add Phase 2 columns to videos table
CALL AddColumnIfNotExists('video_edit_detection', 'videos', 'file_path', 'VARCHAR(500) NULL');
CALL AddColumnIfNotExists('video_edit_detection', 'videos', 'progress', 'INT DEFAULT 0');
CALL AddColumnIfNotExists('video_edit_detection', 'videos', 'error_message', 'TEXT NULL');
CALL AddColumnIfNotExists('video_edit_detection', 'videos', 'fps', 'FLOAT DEFAULT 30.0');
CALL AddColumnIfNotExists('video_edit_detection', 'videos', 'total_frames', 'INT DEFAULT 0');
CALL AddColumnIfNotExists('video_edit_detection', 'videos', 'cut_probabilities', 'JSON NULL');

-- 4. Update status ENUM to include 'failed'
ALTER TABLE videos MODIFY COLUMN status ENUM('pending', 'processing', 'completed', 'failed') DEFAULT 'pending';

-- 5. Safely add Phase 2 columns to analysis_results
CALL AddColumnIfNotExists('video_edit_detection', 'analysis_results', 'model_name', "VARCHAR(50) DEFAULT 'VideoEditDetect v3'");
CALL AddColumnIfNotExists('video_edit_detection', 'analysis_results', 'model_version', "VARCHAR(20) DEFAULT '3.0.0'");
CALL AddColumnIfNotExists('video_edit_detection', 'analysis_results', 'cut_probabilities', 'JSON NULL');

-- 6. Clean up procedure
DROP PROCEDURE IF EXISTS AddColumnIfNotExists;


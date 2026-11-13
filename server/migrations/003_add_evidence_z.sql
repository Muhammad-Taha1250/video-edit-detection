-- ============================================
-- Migration 003: Add nullable evidence_z to edit_segments
-- Safe to run multiple times without data loss
-- ============================================

USE video_edit_detection;

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

CALL AddColumnIfNotExists('video_edit_detection', 'edit_segments', 'evidence_z', 'FLOAT NULL DEFAULT NULL');

DROP PROCEDURE IF EXISTS AddColumnIfNotExists;


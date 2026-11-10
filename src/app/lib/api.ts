// ============================================
// API client for Video Edit Detection backend
// ============================================

const API_BASE = "http://localhost:4000/api";

export interface VideoRecord {
  video_id: number;
  filename: string;
  file_size: number;
  duration: number;
  format: string;
  upload_date: string;
  status: string;
  result_id?: number;
  total_frames?: number;
  edits_detected?: number;
  confidence_score?: number;
  analysis_date?: string;
}

export interface FrameRecord {
  frame_id: number;
  result_id: number;
  frame_number: number;
  timestamp: string;
  difference_score: number;
  is_suspicious: boolean;
}

export interface VideoDetail {
  video: VideoRecord;
  result: {
    result_id: number;
    video_id: number;
    total_frames: number;
    edits_detected: number;
    confidence_score: number;
    analysis_date: string;
  } | null;
  frames: FrameRecord[];
}

export interface Stats {
  total: number;
  suspicious: number;
  clean: number;
}

export async function fetchVideos(): Promise<VideoRecord[]> {
  const res = await fetch(`${API_BASE}/videos`);
  if (!res.ok) throw new Error("Failed to fetch videos");
  return res.json();
}

export async function fetchVideoDetail(id: string | number): Promise<VideoDetail> {
  const res = await fetch(`${API_BASE}/videos/${id}`);
  if (!res.ok) throw new Error("Failed to fetch video detail");
  return res.json();
}

export async function fetchStats(): Promise<Stats> {
  const res = await fetch(`${API_BASE}/stats`);
  if (!res.ok) throw new Error("Failed to fetch stats");
  return res.json();
}

export interface UploadPayload {
  filename: string;
  file_size: number;
  duration: number;
  format: string;
  total_frames: number;
  edits_detected: number;
  confidence_score: number;
  frames?: Array<{
    frame_number: number;
    timestamp: string;
    difference_score: number;
    is_suspicious: boolean;
  }>;
}

export async function uploadVideo(payload: UploadPayload) {
  const res = await fetch(`${API_BASE}/videos`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) throw new Error("Failed to save video");
  return res.json();
}

export async function deleteVideo(id: string | number) {
  const res = await fetch(`${API_BASE}/videos/${id}`, { method: "DELETE" });
  if (!res.ok) throw new Error("Failed to delete video");
  return res.json();
}

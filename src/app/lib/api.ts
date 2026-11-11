// ============================================
// API client for Video Edit Detection backend
// ============================================

export const SERVER_URL = (
  (typeof import.meta !== "undefined" && import.meta.env && import.meta.env.VITE_API_URL) ||
  "http://localhost:4000"
).replace(/\/$/, "");

export const API_BASE = `${SERVER_URL}/api`;

export function getFrameUrl(urlOrPath: string): string {
  if (!urlOrPath) return "";
  if (urlOrPath.startsWith("http://") || urlOrPath.startsWith("https://")) {
    return urlOrPath;
  }
  const clean = urlOrPath.startsWith("/") ? urlOrPath : `/${urlOrPath}`;
  return `${SERVER_URL}${clean}`;
}

export interface VideoRecord {
  video_id: number;
  filename: string;
  file_size: number;
  duration: number;
  format: string;
  upload_date: string;
  status: "pending" | "processing" | "completed" | "failed" | string;
  progress?: number;
  fps?: number;
  total_frames?: number;
  error_message?: string | null;
  result_id?: number;
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

export interface ExtractedFrameInfo {
  frame: number;
  role: string;
  filename: string;
  relative_path: string;
  url: string;
}

export interface EditEvent {
  segment_id: number;
  video_id: number;
  result_id?: number;
  edit_type: string;
  type?: string;
  start_frame: number;
  end_frame: number;
  start_time: number;
  end_time: number;
  confidence: number;
  extracted_frames?: Record<string, ExtractedFrameInfo>;
  created_at?: string;
}

export interface FlattenedFrame {
  event_id: number;
  edit_type: string;
  role: string;
  frame: number;
  url: string;
  flask_url?: string;
  relative_path?: string;
}

export interface ProbabilityPoint {
  frame: number;
  prob: number;
  time: number;
}

export interface AnalysisSummary {
  video_id: number;
  filename: string;
  status: string;
  duration: number;
  fps: number;
  total_frames: number;
  edits_detected: number;
  confidence_score: number;
  model_name: string;
  model_version: string;
  analysis_date: string;
  error?: string | null;
}

export interface AnalysisResultsResponse {
  video_id: number;
  status: string;
  summary: AnalysisSummary;
  events: EditEvent[];
  frames: FlattenedFrame[];
  series: ProbabilityPoint[];
  detected_frames: FrameRecord[];
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
  segments?: EditEvent[];
  frames: FrameRecord[];
}

export interface Stats {
  total: number;
  suspicious: number;
  clean: number;
}

export interface VideoStatusResponse {
  video_id: number;
  status: "pending" | "processing" | "completed" | "failed";
  progress: number;
  error?: string | null;
}

export interface UploadResponse {
  video_id: number;
  id: number;
  filename: string;
  status: string;
  message: string;
}

// --------------------------------------------
// API Functions
// --------------------------------------------

export function uploadVideo(
  file: File,
  onProgress?: (percent: number) => void
): Promise<UploadResponse> {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("POST", `${API_BASE}/videos/upload`);

    if (xhr.upload && onProgress) {
      xhr.upload.onprogress = (e) => {
        if (e.lengthComputable) {
          const pct = Math.round((e.loaded / e.total) * 100);
          onProgress(pct);
        }
      };
    }

    xhr.onload = () => {
      if (xhr.status >= 200 && xhr.status < 300) {
        try {
          const data = JSON.parse(xhr.responseText);
          resolve(data);
        } catch {
          reject(new Error("Invalid response from server"));
        }
      } else {
        try {
          const errData = JSON.parse(xhr.responseText);
          reject(new Error(errData.error || `Upload failed (HTTP ${xhr.status})`));
        } catch {
          reject(new Error(`Upload failed (HTTP ${xhr.status})`));
        }
      }
    };

    xhr.onerror = () => {
      reject(new Error("Network error during file upload. Is the API server running?"));
    };

    const formData = new FormData();
    formData.append("video", file);
    xhr.send(formData);
  });
}

export async function getStatus(id: string | number): Promise<VideoStatusResponse> {
  const res = await fetch(`${API_BASE}/videos/${id}/status`);
  if (!res.ok) {
    const errData = await res.json().catch(() => ({}));
    throw new Error(errData.error || `Failed to fetch status (HTTP ${res.status})`);
  }
  return res.json();
}

export async function getResults(id: string | number): Promise<AnalysisResultsResponse> {
  const res = await fetch(`${API_BASE}/videos/${id}/results`);
  if (!res.ok) {
    const errData = await res.json().catch(() => ({}));
    throw new Error(errData.error || `Failed to fetch analysis results (HTTP ${res.status})`);
  }
  return res.json();
}

export async function deleteVideo(id: string | number): Promise<{ message: string }> {
  const res = await fetch(`${API_BASE}/videos/${id}`, { method: "DELETE" });
  if (!res.ok) {
    const errData = await res.json().catch(() => ({}));
    throw new Error(errData.error || `Failed to delete video (HTTP ${res.status})`);
  }
  return res.json();
}

export async function listVideos(): Promise<VideoRecord[]> {
  const res = await fetch(`${API_BASE}/videos`);
  if (!res.ok) throw new Error("Failed to fetch videos");
  return res.json();
}

export async function fetchVideos(): Promise<VideoRecord[]> {
  return listVideos();
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

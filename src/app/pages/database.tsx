import { useState, useEffect } from "react";
import { Database, Link2, CheckCircle2, Table as TableIcon, Layers } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "../components/ui/card";
import { Badge } from "../components/ui/badge";
import { fetchVideos, fetchStats, VideoRecord, Stats } from "../lib/api";

interface Column {
  name: string;
  type: string;
  note: string;
  isPK?: boolean;
  isFK?: boolean;
}

interface TableDefinition {
  name: string;
  icon: string;
  desc: string;
  accent: string;
  columns: Column[];
}

const tables: TableDefinition[] = [
  {
    name: "users",
    icon: "👤",
    desc: "Stores registered user accounts and authentication credentials",
    accent: "from-blue-500 to-blue-600",
    columns: [
      { name: "user_id", type: "INT", note: "Primary Key, Auto Increment", isPK: true },
      { name: "username", type: "VARCHAR(100)", note: "Unique user handle" },
      { name: "email", type: "VARCHAR(150)", note: "Unique email address" },
      { name: "password_hash", type: "VARCHAR(255)", note: "Password hash" },
      { name: "created_at", type: "DATETIME", note: "Account creation timestamp" },
    ],
  },
  {
    name: "videos",
    icon: "🎬",
    desc: "Stores uploaded video files, metadata, and background analysis status",
    accent: "from-purple-500 to-purple-600",
    columns: [
      { name: "video_id", type: "INT", note: "Primary Key, Auto Increment", isPK: true },
      { name: "user_id", type: "INT", note: "References users.user_id", isFK: true },
      { name: "filename", type: "VARCHAR(255)", note: "Original uploaded file name" },
      { name: "file_size", type: "BIGINT", note: "File size in bytes" },
      { name: "duration", type: "FLOAT", note: "Video duration in seconds" },
      { name: "format", type: "VARCHAR(10)", note: "MP4, MOV, AVI, MKV, WEBM" },
      { name: "file_path", type: "VARCHAR(500)", note: "Server-side disk storage path" },
      { name: "fps", type: "FLOAT", note: "Decoded frame rate (OpenCV)" },
      { name: "total_frames", type: "INT", note: "Total frames in video" },
      { name: "status", type: "ENUM", note: "pending | processing | completed | failed" },
      { name: "progress", type: "INT", note: "Processing progress percentage (0-100)" },
      { name: "cut_probabilities", type: "JSON", note: "Downsampled probability series for charting" },
      { name: "error_message", type: "TEXT", note: "Error explanation if analysis failed" },
      { name: "upload_date", type: "DATETIME", note: "Upload timestamp" },
    ],
  },
  {
    name: "analysis_results",
    icon: "📊",
    desc: "Stores aggregate model results and detection metrics per video",
    accent: "from-cyan-500 to-cyan-600",
    columns: [
      { name: "result_id", type: "INT", note: "Primary Key, Auto Increment", isPK: true },
      { name: "video_id", type: "INT", note: "References videos.video_id", isFK: true },
      { name: "total_frames", type: "INT", note: "Frames analyzed by AI pipeline" },
      { name: "edits_detected", type: "INT", note: "Total edit segments detected" },
      { name: "confidence_score", type: "FLOAT", note: "Aggregate model detection score percentage" },
      { name: "model_name", type: "VARCHAR(50)", note: "e.g. VideoEditDetect v3" },
      { name: "model_version", type: "VARCHAR(20)", note: "e.g. 3.0.0" },
      { name: "cut_probabilities", type: "JSON", note: "Cached cut probability series" },
      { name: "analysis_date", type: "DATETIME", note: "When analysis was completed" },
    ],
  },
  {
    name: "edit_segments",
    icon: "✂️",
    desc: "Stores discrete edit intervals, types, and extracted junction frames (Phase 2)",
    accent: "from-emerald-500 to-emerald-600",
    columns: [
      { name: "segment_id", type: "INT", note: "Primary Key, Auto Increment", isPK: true },
      { name: "video_id", type: "INT", note: "References videos.video_id", isFK: true },
      { name: "result_id", type: "INT", note: "References analysis_results.result_id", isFK: true },
      { name: "edit_type", type: "VARCHAR(50)", note: "scene_cut | deletion | insertion | duplication | frame_drop" },
      { name: "start_frame", type: "INT", note: "Boundary start frame index" },
      { name: "end_frame", type: "INT", note: "Boundary end frame index" },
      { name: "start_time", type: "FLOAT", note: "Start timestamp in seconds" },
      { name: "end_time", type: "FLOAT", note: "End timestamp in seconds" },
      { name: "confidence", type: "FLOAT", note: "Detection score (0.0 to 1.0)" },
      { name: "evidence_z", type: "FLOAT", note: "Robust z-score pixel MAD evidence (Migration 003)" },
      { name: "extracted_frames", type: "JSON", note: "URLs & paths of before/junction/after JPGs" },
      { name: "created_at", type: "DATETIME", note: "Creation timestamp" },
    ],
  },
  {
    name: "detected_frames",
    icon: "🖼️",
    desc: "Stores per-frame anomaly flags for backward compatibility",
    accent: "from-rose-500 to-rose-600",
    columns: [
      { name: "frame_id", type: "INT", note: "Primary Key, Auto Increment", isPK: true },
      { name: "result_id", type: "INT", note: "References analysis_results.result_id", isFK: true },
      { name: "frame_number", type: "INT", note: "Frame index in video" },
      { name: "timestamp", type: "VARCHAR(20)", note: "e.g. 00:01:07" },
      { name: "difference_score", type: "FLOAT", note: "Detection score (0.0 to 1.0)" },
      { name: "is_suspicious", type: "BOOLEAN", note: "True if flagged as suspicious" },
    ],
  },
];

const relationships = [
  { from: "users", to: "videos", type: "One-to-Many" },
  { from: "videos", to: "analysis_results", type: "One-to-One" },
  { from: "videos", to: "edit_segments", type: "One-to-Many" },
  { from: "analysis_results", to: "detected_frames", type: "One-to-Many" },
];

export function DatabasePage() {
  const [videoCount, setVideoCount] = useState<number | null>(null);
  const [stats, setStats] = useState<Stats | null>(null);
  const [isConnected, setIsConnected] = useState<boolean>(true);

  useEffect(() => {
    Promise.all([fetchVideos(), fetchStats()])
      .then(([vList, st]) => {
        setVideoCount(vList.length);
        setStats(st);
        setIsConnected(true);
      })
      .catch((err) => {
        console.warn("Could not query DB stats:", err);
        setIsConnected(false);
      });
  }, []);

  return (
    <div className="space-y-8">
      {/* Header */}
      <div className="space-y-2">
        <div className="flex items-center gap-2">
          <div className="inline-flex items-center gap-2 bg-blue-50 border border-blue-200 rounded-full px-3 py-1">
            <Database className="size-3.5 text-blue-600" />
            <span className="text-xs font-medium text-blue-600">
              MySQL Relational Schema (video_edit_detection)
            </span>
          </div>
          <Badge variant="outline" className={isConnected ? "text-green-700 border-green-300 bg-green-50" : "text-amber-700"}>
            <CheckCircle2 className="size-3 mr-1" />
            {isConnected ? "Database Connected" : "Connection Check Failed"}
          </Badge>
        </div>
        <h2 className="text-3xl font-bold">Database Schema & Architecture</h2>
        <p className="text-muted-foreground text-sm">
          Relational schema and tables powering video ingestion, model inference results, and junction frame storage.
        </p>
      </div>

      {/* Live Record Counts */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <Card className="p-4 flex items-center justify-between">
          <div>
            <p className="text-xs text-muted-foreground">Registered Videos</p>
            <p className="text-2xl font-bold">{videoCount !== null ? videoCount : "—"}</p>
          </div>
          <TableIcon className="size-6 text-purple-600" />
        </Card>
        <Card className="p-4 flex items-center justify-between">
          <div>
            <p className="text-xs text-muted-foreground">Edited Clips Flagged</p>
            <p className="text-2xl font-bold text-red-600">{stats ? stats.suspicious : "—"}</p>
          </div>
          <Layers className="size-6 text-red-600" />
        </Card>
        <Card className="p-4 flex items-center justify-between">
          <div>
            <p className="text-xs text-muted-foreground">No Edits Detected</p>
            <p className="text-2xl font-bold text-green-600">{stats ? stats.clean : "—"}</p>
          </div>
          <CheckCircle2 className="size-6 text-green-600" />
        </Card>
        <Card className="p-4 flex items-center justify-between">
          <div>
            <p className="text-xs text-muted-foreground">Schema Version</p>
            <p className="text-2xl font-bold font-mono">002_phase2</p>
          </div>
          <Database className="size-6 text-blue-600" />
        </Card>
      </div>

      {/* Relationships */}
      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="flex items-center gap-2 text-base">
            <Link2 className="size-4 text-blue-600" />
            Foreign Key Table Relationships
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex flex-wrap gap-3">
            {relationships.map((r, i) => (
              <div
                key={i}
                className="flex items-center gap-2 bg-slate-50 border rounded-lg px-3 py-2 text-sm"
              >
                <span className="font-mono font-semibold text-blue-600">{r.from}</span>
                <span className="text-muted-foreground">→</span>
                <span className="font-mono font-semibold text-blue-600">{r.to}</span>
                <Badge variant="secondary" className="text-xs">{r.type}</Badge>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* Tables Grid */}
      <div className="grid md:grid-cols-2 gap-6">
        {tables.map((table) => (
          <Card key={table.name} className="overflow-hidden border border-slate-200 shadow-sm flex flex-col">
            <CardHeader
              className={`bg-gradient-to-br ${table.accent} text-white rounded-t-xl pb-4`}
            >
              <div className="flex items-center gap-3">
                <span className="text-2xl">{table.icon}</span>
                <div>
                  <CardTitle className="font-mono text-white text-base">
                    {table.name}
                  </CardTitle>
                  <CardDescription className="text-white/80 text-xs">
                    {table.desc}
                  </CardDescription>
                </div>
              </div>
            </CardHeader>
            <CardContent className="p-0 flex-1">
              <div className="overflow-x-auto">
                <table className="w-full text-xs">
                  <thead className="bg-slate-50 border-b text-slate-500">
                    <tr>
                      <th className="py-2.5 px-3 text-left font-semibold">Column</th>
                      <th className="py-2.5 px-3 text-left font-semibold">Type</th>
                      <th className="py-2.5 px-3 text-left font-semibold">Description</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-100">
                    {table.columns.map((col) => (
                      <tr key={col.name} className="hover:bg-slate-50/50">
                        <td className="py-2 px-3 font-mono font-semibold text-slate-800">
                          <div className="flex items-center gap-1.5">
                            {col.name}
                            {col.isPK && (
                              <Badge variant="outline" className="text-[10px] py-0 px-1 text-blue-600 border-blue-200">
                                PK
                              </Badge>
                            )}
                            {col.isFK && (
                              <Badge variant="outline" className="text-[10px] py-0 px-1 text-purple-600 border-purple-200">
                                FK
                              </Badge>
                            )}
                          </div>
                        </td>
                        <td className="py-2 px-3 font-mono text-slate-500">{col.type}</td>
                        <td className="py-2 px-3 text-slate-600">{col.note}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </CardContent>
          </Card>
        ))}
      </div>
    </div>
  );
}

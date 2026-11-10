import { Database, Key, Link2 } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "../components/ui/card";
import { Badge } from "../components/ui/badge";

interface Column {
  name: string;
  type: string;
  note: string;
  isPK?: boolean;
  isFK?: boolean;
}

interface Table {
  name: string;
  icon: string;
  desc: string;
  accent: string;
  columns: Column[];
}

const tables: Table[] = [
  {
    name: "users",
    icon: "👤",
    desc: "Stores registered user accounts",
    accent: "from-blue-500 to-blue-600",
    columns: [
      { name: "user_id", type: "INT", note: "Primary Key, Auto Increment", isPK: true },
      { name: "username", type: "VARCHAR(100)", note: "Unique username" },
      { name: "email", type: "VARCHAR(150)", note: "Unique email address" },
      { name: "password_hash", type: "VARCHAR(255)", note: "Hashed password" },
      { name: "created_at", type: "DATETIME", note: "Account creation time" },
    ],
  },
  {
    name: "videos",
    icon: "🎬",
    desc: "Stores uploaded video file information",
    accent: "from-purple-500 to-purple-600",
    columns: [
      { name: "video_id", type: "INT", note: "Primary Key, Auto Increment", isPK: true },
      { name: "user_id", type: "INT", note: "References users.user_id", isFK: true },
      { name: "filename", type: "VARCHAR(255)", note: "Original file name" },
      { name: "file_size", type: "BIGINT", note: "Size in bytes" },
      { name: "duration", type: "FLOAT", note: "Video duration in seconds" },
      { name: "format", type: "VARCHAR(10)", note: "MP4, AVI, MOV" },
      { name: "upload_date", type: "DATETIME", note: "When video was uploaded" },
      { name: "status", type: "ENUM", note: "pending / processing / completed" },
    ],
  },
  {
    name: "analysis_results",
    icon: "📊",
    desc: "Stores overall analysis output for each video",
    accent: "from-cyan-500 to-cyan-600",
    columns: [
      { name: "result_id", type: "INT", note: "Primary Key, Auto Increment", isPK: true },
      { name: "video_id", type: "INT", note: "References videos.video_id", isFK: true },
      { name: "total_frames", type: "INT", note: "Total frames extracted" },
      { name: "edits_detected", type: "INT", note: "Number of suspicious frames" },
      { name: "confidence_score", type: "FLOAT", note: "Model confidence (0-100%)" },
      { name: "analysis_date", type: "DATETIME", note: "When analysis was run" },
    ],
  },
  {
    name: "detected_frames",
    icon: "🖼️",
    desc: "Stores per-frame detection details",
    accent: "from-rose-500 to-rose-600",
    columns: [
      { name: "frame_id", type: "INT", note: "Primary Key, Auto Increment", isPK: true },
      { name: "result_id", type: "INT", note: "References analysis_results.result_id", isFK: true },
      { name: "frame_number", type: "INT", note: "Frame index in video" },
      { name: "timestamp", type: "VARCHAR(20)", note: "e.g. 00:01:07" },
      { name: "difference_score", type: "FLOAT", note: "Edit score (0.0 to 1.0)" },
      { name: "is_suspicious", type: "BOOLEAN", note: "True if edit detected" },
    ],
  },
];

const relationships = [
  { from: "users", to: "videos", type: "One-to-Many" },
  { from: "videos", to: "analysis_results", type: "One-to-One" },
  { from: "analysis_results", to: "detected_frames", type: "One-to-Many" },
];

export function DatabasePage() {
  return (
    <div className="space-y-8">
      {/* Header */}
      <div className="space-y-2">
        <div className="inline-flex items-center gap-2 bg-blue-50 border border-blue-200 rounded-full px-3 py-1">
          <Database className="size-3.5 text-blue-600" />
          <span className="text-xs font-medium text-blue-600">
            Phase-I · Core Feature — Database Schema
          </span>
        </div>
        <h2 className="text-2xl font-bold">Database Schema</h2>
        <p className="text-muted-foreground text-sm">
          Relational schema for the Video Edit Detection & Frame Extraction System
        </p>
      </div>

      {/* Relationships */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-base">
            <Link2 className="size-4 text-blue-600" />
            Table Relationships
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex flex-wrap gap-3">
            {relationships.map((r, i) => (
              <div
                key={i}
                className="flex items-center gap-2 bg-slate-50 border rounded-lg px-3 py-2"
              >
                <span className="font-mono text-sm font-semibold text-blue-600">
                  {r.from}
                </span>
                <span className="text-muted-foreground">→</span>
                <span className="font-mono text-sm font-semibold text-blue-600">
                  {r.to}
                </span>
                <Badge variant="secondary">{r.type}</Badge>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* Tables Grid */}
      <div className="grid md:grid-cols-2 gap-6">
        {tables.map((table) => (
          <Card key={table.name} className="overflow-hidden">
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
            <CardContent className="pt-4">
              <div className="grid grid-cols-[1.1fr_1fr_1.3fr] gap-2 pb-2 border-b text-xs font-semibold uppercase text-muted-foreground tracking-wide">
                <span>Column</span>
                <span>Type</span>
                <span>Description</span>
              </div>
              {table.columns.map((col, i) => (
                <div
                  key={col.name}
                  className={`grid grid-cols-[1.1fr_1fr_1.3fr] gap-2 py-2 text-sm ${
                    i % 2 === 1 ? "bg-slate-50" : ""
                  }`}
                >
                  <span
                    className={`font-mono flex items-center gap-1 ${
                      col.isPK
                        ? "text-amber-600 font-semibold"
                        : col.isFK
                        ? "text-cyan-600 font-semibold"
                        : "text-foreground"
                    }`}
                  >
                    {col.isPK && <Key className="size-3" />}
                    {col.isFK && <Link2 className="size-3" />}
                    {col.name}
                  </span>
                  <span className="font-mono text-muted-foreground text-xs flex items-center">
                    {col.type}
                  </span>
                  <span className="text-muted-foreground text-xs flex items-center">
                    {col.note}
                  </span>
                </div>
              ))}
            </CardContent>
          </Card>
        ))}
      </div>

      {/* Legend */}
      <Card>
        <CardContent className="pt-6 flex flex-wrap items-center gap-6 text-sm">
          <span className="font-semibold text-muted-foreground">Legend:</span>
          <span className="flex items-center gap-1.5 text-amber-600">
            <Key className="size-3.5" /> Primary Key (PK)
          </span>
          <span className="flex items-center gap-1.5 text-cyan-600">
            <Link2 className="size-3.5" /> Foreign Key (FK) — links tables together
          </span>
          <span className="text-muted-foreground">
            ENUM — fixed set of allowed values
          </span>
        </CardContent>
      </Card>
    </div>
  );
}

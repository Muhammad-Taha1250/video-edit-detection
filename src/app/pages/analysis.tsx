import { useState, useEffect } from "react";
import { useParams, Link } from "react-router";
import { Button } from "../components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "../components/ui/card";
import { Badge } from "../components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "../components/ui/tabs";
import { ScrollArea } from "../components/ui/scroll-area";
import { Separator } from "../components/ui/separator";
import {
  Download,
  AlertTriangle,
  CheckCircle,
  FileVideo,
  Film,
  Activity,
  ArrowLeft,
  Loader2,
  Clock,
  Layers,
  Sparkles,
  Info,
} from "lucide-react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  ReferenceLine,
} from "recharts";
import { getResults, getFrameUrl, AnalysisResultsResponse, EditEvent } from "../lib/api";

function FrameImage({
  src,
  alt,
  className = "w-full h-full object-contain",
}: {
  src: string;
  alt: string;
  className?: string;
}) {
  const [hasError, setHasError] = useState(false);

  useEffect(() => {
    setHasError(false);
  }, [src]);

  if (hasError || !src) {
    return (
      <div className="w-full h-full min-h-[140px] flex flex-col items-center justify-center p-3 text-center bg-slate-900 border border-slate-800 text-slate-300">
        <AlertTriangle className="size-5 text-amber-500 mb-1.5 shrink-0" />
        <span className="text-xs font-semibold text-slate-200">Frame unavailable</span>
        <span
          className="text-[10px] text-slate-400 font-mono mt-1 break-all px-2 select-all line-clamp-3"
          title={src}
        >
          {src || "No URL provided"}
        </span>
      </div>
    );
  }

  return (
    <img
      src={src}
      alt={alt}
      className={className}
      loading="lazy"
      onError={() => setHasError(true)}
    />
  );
}

export function AnalysisPage() {
  const { id } = useParams();
  const [data, setData] = useState<AnalysisResultsResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!id) return;
    setLoading(true);
    setError(null);
    getResults(id)
      .then(setData)
      .catch((err) => {
        console.error(err);
        setError(err.message || "Failed to load video analysis");
      })
      .finally(() => setLoading(false));
  }, [id]);

  if (loading) {
    return (
      <div className="flex flex-col items-center justify-center py-28 space-y-4">
        <Loader2 className="size-10 animate-spin text-blue-600" />
        <p className="text-muted-foreground font-medium">Loading analysis results...</p>
      </div>
    );
  }

  if (error || !data || !data.summary) {
    return (
      <div className="text-center py-16 space-y-4 max-w-md mx-auto">
        <div className="mx-auto w-12 h-12 bg-red-100 rounded-full flex items-center justify-center">
          <AlertTriangle className="size-6 text-red-600" />
        </div>
        <h2 className="text-xl font-bold">Analysis Not Found or Server Error</h2>
        <p className="text-sm text-muted-foreground">
          {error || "Could not retrieve analysis for this video. Make sure backend servers are running."}
        </p>
        <div className="pt-2">
          <Button asChild>
            <Link to="/history">
              <ArrowLeft className="size-4 mr-2" />
              Back to History
            </Link>
          </Button>
        </div>
      </div>
    );
  }

  const { summary, events, series } = data;
  const editsDetected = summary.edits_detected || events.length || 0;
  const isClean = editsDetected === 0;

  const isEventLowConfidence = (
    ev?: { confidence?: number; evidence_z?: number } | number,
    zScore?: number
  ) => {
    if (ev === undefined || ev === null) return false;
    let conf: number;
    let z: number | undefined;
    if (typeof ev === "object") {
      conf = typeof ev.confidence === "number" ? ev.confidence : 1.0;
      z = typeof ev.evidence_z === "number" ? ev.evidence_z : undefined;
    } else {
      conf = ev;
      z = zScore;
    }

    if (conf > 1.0) conf = conf / 100.0;

    // Events with evidence_z >= 15 are high confidence
    if (z !== undefined && z >= 15) {
      return false;
    }

    // Call an event "low confidence" only if evidence_z < 15 AND confidence < 0.5
    if (z !== undefined) {
      return z < 15 && conf < 0.5;
    }

    return conf < 0.5;
  };

  const lowConfidenceEvents = events.filter((ev) => isEventLowConfidence(ev));
  const hasLowConfidence = lowConfidenceEvents.length > 0;

  const downloadReport = () => {
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `analysis-report-${summary.video_id}-${summary.filename}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const formatEditBadge = (
    type: string,
    evOrConf?: { confidence?: number; evidence_z?: number } | number,
    zScore?: number
  ) => {
    const isLow = isEventLowConfidence(evOrConf, zScore);
    if (isLow) {
      return (
        <Badge
          variant="outline"
          className="bg-slate-100 text-slate-600 border-slate-300 font-normal hover:bg-slate-200"
        >
          Possible {type} (low confidence)
        </Badge>
      );
    }

    switch (type.toLowerCase()) {
      case "deletion":
        return <Badge variant="destructive">Deletion Cut</Badge>;
      case "insertion":
        return <Badge className="bg-blue-600 hover:bg-blue-700">Segment Insertion</Badge>;
      case "duplication":
        return <Badge className="bg-purple-600 hover:bg-purple-700">Frame Duplication</Badge>;
      case "frame_drop":
        return <Badge className="bg-amber-600 hover:bg-amber-700">Frame Drop</Badge>;
      default:
        return <Badge variant="secondary">{type}</Badge>;
    }
  };

  const formatRoleLabel = (role: string) => {
    switch (role) {
      case "before":
        return "Before Cut (Pre-edit)";
      case "junction":
        return "Junction Frame (First Post-cut)";
      case "after":
        return "After Cut (Post-edit)";
      case "before_start":
        return "Before Start (Pre-edit)";
      case "start":
        return "Start Junction";
      case "after_start":
        return "After Start (Interior)";
      case "before_end":
        return "Before End (Interior)";
      case "end":
        return "End Junction";
      case "after_end":
        return "After End (Resumed)";
      default:
        return role;
    }
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-start justify-between gap-4">
        <div className="space-y-1">
          <div className="flex items-center gap-2">
            <Button asChild variant="ghost" size="sm">
              <Link to="/history">
                <ArrowLeft className="size-4 mr-2" />
                Back to History
              </Link>
            </Button>
          </div>
          <h1 className="text-3xl font-bold">Analysis Results</h1>
          <p className="text-muted-foreground font-mono text-sm">{summary.filename}</p>
        </div>
        <div className="flex items-center gap-2">
          <Button onClick={downloadReport} variant="outline">
            <Download className="size-4 mr-2" />
            Download JSON Report
          </Button>
        </div>
      </div>

      {/* Summary Cards */}
      <div className="grid grid-cols-2 md:grid-cols-6 gap-4">
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-xs font-medium text-muted-foreground">Duration</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="flex items-center gap-2">
              <Clock className="size-4 text-blue-600" />
              <span className="text-xl font-bold">{summary.duration}s</span>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-xs font-medium text-muted-foreground">Frame Rate</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="flex items-center gap-2">
              <Activity className="size-4 text-indigo-600" />
              <span className="text-xl font-bold">{summary.fps} fps</span>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-xs font-medium text-muted-foreground">Total Frames</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="flex items-center gap-2">
              <Film className="size-4 text-blue-600" />
              <span className="text-xl font-bold">{summary.total_frames}</span>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-xs font-medium text-muted-foreground">Edits Detected</CardTitle>
          </CardHeader>
          <CardContent className="space-y-1">
            <div className="flex items-center gap-2">
              <AlertTriangle className={`size-4 ${isClean ? "text-green-600" : "text-red-600"}`} />
              <span className={`text-xl font-bold ${isClean ? "text-green-600" : "text-red-600"}`}>
                {editsDetected}
              </span>
            </div>
            {hasLowConfidence && (
              <p className="text-[11px] text-amber-600 font-medium leading-tight">
                Low-confidence events need manual review
              </p>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-xs font-medium text-muted-foreground">Detection score</CardTitle>
          </CardHeader>
          <CardContent className="space-y-1">
            <div className="flex items-center gap-2">
              <Sparkles className="size-4 text-purple-600" />
              <span className="text-xl font-bold">{(summary.confidence_score || 0).toFixed(1)}%</span>
            </div>
            {hasLowConfidence && (
              <p className="text-[11px] text-amber-600 font-medium leading-tight">
                Low-confidence events need manual review
              </p>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-xs font-medium text-muted-foreground">Verdict</CardTitle>
          </CardHeader>
          <CardContent>
            <Badge variant={isClean ? "secondary" : "destructive"} className="text-xs">
              {isClean ? "No edits detected" : "Edited / Modified"}
            </Badge>
          </CardContent>
        </Card>
      </div>

      {/* Note under summary cards */}
      <div className="space-y-2">
        {hasLowConfidence && (
          <div className="flex items-center gap-2 text-xs text-amber-800 bg-amber-50 border border-amber-200 rounded-md p-2.5">
            <AlertTriangle className="size-4 text-amber-600 flex-shrink-0" />
            <span className="font-medium">
              Note: One or more detected events have a low detection score (evidence &lt; 15x and score &lt; 50%). Low-confidence events need manual review.
            </span>
          </div>
        )}
        <div className="flex items-center gap-2 text-xs text-muted-foreground bg-slate-50 border border-slate-200 rounded-md p-2.5">
          <Info className="size-4 text-slate-500 flex-shrink-0" />
          <span>
            Detection covers cuts, insertions, duplicated frames and dropped frames. Not detecting an edit does not guarantee the video is authentic.
          </span>
        </div>
      </div>

      {/* Tabs */}
      <Tabs defaultValue="events" className="space-y-4">
        <TabsList>
          <TabsTrigger value="events">Detected Edits ({events.length})</TabsTrigger>
          <TabsTrigger value="chart">Cut Probability Curve</TabsTrigger>
          <TabsTrigger value="frames">Junction Frames</TabsTrigger>
          <TabsTrigger value="details">Technical Details</TabsTrigger>
        </TabsList>

        {/* Events Tab */}
        <TabsContent value="events" className="space-y-4">
          {events.length === 0 ? (
            <Card className="border-green-200 bg-green-50/50">
              <CardContent className="py-12 text-center space-y-3">
                <div className="mx-auto w-12 h-12 bg-green-100 rounded-full flex items-center justify-center">
                  <CheckCircle className="size-6 text-green-600" />
                </div>
                <h3 className="text-lg font-semibold text-green-900">No Edits Detected</h3>
                <p className="text-sm text-green-700 max-w-md mx-auto">
                  The model did not find any deletion cuts, inserted segments, duplicated frames, or rate-change drops in this clip.
                </p>
              </CardContent>
            </Card>
          ) : (
            <div className="space-y-4">
              <Card>
                <CardHeader>
                  <CardTitle>Edit Event Timeline</CardTitle>
                  <CardDescription>
                    Temporal boundaries and classifications identified by the two-stage model
                  </CardDescription>
                </CardHeader>
                <CardContent className="p-0">
                  <div className="overflow-x-auto">
                    <table className="w-full text-sm">
                      <thead className="bg-slate-50 border-b text-muted-foreground">
                        <tr>
                          <th className="py-3 px-4 text-left">#</th>
                          <th className="py-3 px-4 text-left">Edit Type</th>
                          <th className="py-3 px-4 text-left">Frame Range</th>
                          <th className="py-3 px-4 text-left">Timestamp (sec)</th>
                          <th className="py-3 px-4 text-left">Detection score</th>
                          <th className="py-3 px-4 text-left">Frames Captured</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y">
                        {events.map((ev, idx) => {
                          const frameCount = ev.extracted_frames ? Object.keys(ev.extracted_frames).length : 0;
                          const isLow = isEventLowConfidence(ev);
                          return (
                            <tr
                              key={ev.segment_id || idx}
                              className={`hover:bg-slate-50/60 ${isLow ? "bg-amber-50/30" : ""}`}
                            >
                              <td className="py-3 px-4 font-mono font-medium">{idx + 1}</td>
                              <td className="py-3 px-4">{formatEditBadge(ev.edit_type, ev)}</td>
                              <td className="py-3 px-4 font-mono">
                                {ev.start_frame === ev.end_frame
                                  ? `Frame ${ev.start_frame}`
                                  : `Frames ${ev.start_frame} → ${ev.end_frame}`}
                              </td>
                              <td className="py-3 px-4 text-muted-foreground">
                                {ev.start_time === ev.end_time
                                  ? `${ev.start_time.toFixed(3)}s`
                                  : `${ev.start_time.toFixed(3)}s – ${ev.end_time.toFixed(3)}s`}
                              </td>
                              <td className="py-3 px-4 font-medium">
                                <span className={isLow ? "text-amber-700" : ""}>
                                  {(ev.confidence * 100).toFixed(1)}%
                                </span>
                                {ev.evidence_z != null && (
                                  <span className="block text-[11px] text-muted-foreground font-normal">
                                    Change vs normal: {ev.evidence_z.toFixed(1)}x
                                  </span>
                                )}
                                {isLow && (
                                  <span className="block text-[11px] text-amber-600 font-normal">
                                    Manual review needed
                                  </span>
                                )}
                              </td>
                              <td className="py-3 px-4 text-muted-foreground">
                                {frameCount > 0 ? (
                                  <span className="text-xs bg-slate-100 text-slate-700 px-2 py-0.5 rounded font-mono">
                                    {frameCount} frames
                                  </span>
                                ) : (
                                  <span className="text-xs text-muted-foreground">None</span>
                                )}
                              </td>
                            </tr>
                          );
                        })}
                      </tbody>
                    </table>
                  </div>
                </CardContent>
              </Card>

              {/* In-line Frame Preview for Events */}
              <div className="space-y-6 pt-2">
                <h3 className="text-lg font-bold flex items-center gap-2">
                  <Layers className="size-5 text-blue-600" />
                  Captured Junction Frame Evidence
                </h3>
                {events.map((ev, idx) => {
                  const frameKeys = ev.extracted_frames ? Object.keys(ev.extracted_frames) : [];
                  if (frameKeys.length === 0) return null;

                  return (
                    <Card key={idx} className="border-slate-200 shadow-sm">
                      <CardHeader className="bg-slate-50/70 border-b py-3 px-4">
                        <div className="flex flex-wrap items-center justify-between gap-2">
                          <div className="flex items-center gap-2">
                            <span className="font-semibold text-sm">Event {idx + 1}:</span>
                            {formatEditBadge(ev.edit_type, ev)}
                            <span className="text-xs text-muted-foreground font-mono">
                              (Frames {ev.start_frame} → {ev.end_frame} | {ev.start_time.toFixed(3)}s – {ev.end_time.toFixed(3)}s)
                            </span>
                          </div>
                          <div className="flex items-center gap-2">
                            {ev.evidence_z != null && (
                              <Badge variant="outline" className="text-xs border-slate-300 text-slate-700 bg-slate-50">
                                Change vs normal: {ev.evidence_z.toFixed(1)}x
                              </Badge>
                            )}
                            <Badge
                              variant="outline"
                              className={`text-xs ${
                                isEventLowConfidence(ev)
                                  ? "border-amber-300 text-amber-700 bg-amber-50"
                                  : ""
                              }`}
                            >
                              Detection score: {(ev.confidence * 100).toFixed(1)}%
                            </Badge>
                          </div>
                        </div>
                      </CardHeader>
                      <CardContent className="p-4">
                        <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-4">
                          {frameKeys.map((role) => {
                            const fInfo = ev.extracted_frames![role];
                            return (
                              <div
                                key={role}
                                className="border rounded-lg overflow-hidden bg-slate-900 flex flex-col"
                              >
                                <div className="p-2 bg-slate-800 text-slate-200 text-xs flex justify-between items-center border-b border-slate-700">
                                  <span className="font-medium">{formatRoleLabel(role)}</span>
                                  <span className="font-mono bg-slate-900 px-1.5 py-0.5 rounded text-[11px]">
                                    f{fInfo.frame}
                                  </span>
                                </div>
                                <div className="aspect-video bg-black flex items-center justify-center overflow-hidden">
                                  <FrameImage
                                    src={getFrameUrl(fInfo.url, summary.video_id)}
                                    alt={`${role} frame ${fInfo.frame}`}
                                    className="w-full h-full object-contain"
                                  />
                                </div>
                                <div className="p-1.5 bg-slate-800/80 text-[11px] text-slate-300 flex justify-between">
                                  <span>Time: {(fInfo.frame / summary.fps).toFixed(3)}s</span>
                                  <span className="text-slate-400 truncate max-w-[140px]">{fInfo.filename}</span>
                                </div>
                              </div>
                            );
                          })}
                        </div>
                      </CardContent>
                    </Card>
                  );
                })}
              </div>
            </div>
          )}
        </TabsContent>

        {/* Cut Probability Chart Tab */}
        <TabsContent value="chart" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Cut Probability Across Frame Pairs</CardTitle>
              <CardDescription>
                Downsampled prediction series from the Stage 1 Cut Detector with detected event boundaries
              </CardDescription>
            </CardHeader>
            <CardContent>
              {!series || series.length === 0 ? (
                <div className="py-16 text-center text-muted-foreground text-sm">
                  No probability series data available for this video.
                </div>
              ) : (
                <div className="space-y-4">
                  <div className="h-[360px] w-full">
                    <ResponsiveContainer width="100%" height="100%">
                      <LineChart data={series} margin={{ top: 15, right: 25, left: 0, bottom: 20 }}>
                        <CartesianGrid strokeDasharray="3 3" vertical={false} />
                        <XAxis
                          dataKey="frame"
                          label={{ value: "Frame Index", position: "insideBottom", offset: -10 }}
                        />
                        <YAxis
                          domain={[0, 1]}
                          label={{ value: "Cut Probability", angle: -90, position: "insideLeft" }}
                        />
                        <Tooltip
                          formatter={(value: any) => [`${(Number(value) * 100).toFixed(2)}%`, "Cut Probability"]}
                          labelFormatter={(label) => `Frame #${label}`}
                        />
                        <Legend verticalAlign="top" height={36} />
                        <Line
                          type="monotone"
                          dataKey="prob"
                          name="Cut Probability"
                          stroke="#2563eb"
                          strokeWidth={2}
                          dot={false}
                        />
                        {events.map((ev, i) => (
                          <ReferenceLine
                            key={`ref-s-${i}`}
                            x={ev.start_frame}
                            stroke="#ef4444"
                            strokeDasharray="4 4"
                            label={{
                              value: `${ev.edit_type} (f${ev.start_frame})`,
                              fill: "#dc2626",
                              fontSize: 11,
                              position: "top",
                            }}
                          />
                        ))}
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                  <div className="text-xs text-muted-foreground bg-slate-50 p-3 rounded-md flex items-center gap-2">
                    <Info className="size-4 text-blue-600 flex-shrink-0" />
                    <span>
                      Dashed red lines mark the exact frames where junction cuts or edit transitions were identified by the model.
                    </span>
                  </div>
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        {/* Frames Tab */}
        <TabsContent value="frames" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Saved Frame Evidence</CardTitle>
              <CardDescription>
                All extracted junction frames served through the API proxy
              </CardDescription>
            </CardHeader>
            <CardContent>
              {events.length === 0 ? (
                <p className="text-sm text-muted-foreground py-8 text-center">
                  No edit junctions were detected in this video, so no anomaly frames were saved.
                </p>
              ) : (
                <div className="space-y-6">
                  {events.map((ev, evIdx) => (
                    <div key={evIdx} className="space-y-3">
                      <div className="flex items-center gap-2">
                        <span className="font-semibold text-sm">Event {evIdx + 1}:</span>
                        {formatEditBadge(ev.edit_type, ev)}
                        <span className="text-xs text-muted-foreground font-mono">
                          Frames {ev.start_frame} → {ev.end_frame}
                        </span>
                      </div>
                      <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-4">
                        {ev.extracted_frames &&
                          Object.entries(ev.extracted_frames).map(([role, fInfo]) => (
                            <Card key={role} className="overflow-hidden border-slate-200 shadow-sm">
                              <div className="p-2.5 bg-slate-50 border-b flex justify-between items-center text-xs">
                                <span className="font-semibold text-slate-700">{formatRoleLabel(role)}</span>
                                <Badge variant="outline" className="font-mono text-[11px]">
                                  Frame {fInfo.frame}
                                </Badge>
                              </div>
                              <div className="aspect-video bg-black flex items-center justify-center overflow-hidden">
                                <FrameImage
                                  src={getFrameUrl(fInfo.url, summary.video_id)}
                                  alt={fInfo.filename}
                                  className="w-full h-full object-contain"
                                />
                              </div>
                              <div className="p-2 text-xs text-muted-foreground flex justify-between">
                                <span>{(fInfo.frame / summary.fps).toFixed(3)}s</span>
                                <span className="font-mono text-[10px] truncate max-w-[160px]">{fInfo.filename}</span>
                              </div>
                            </Card>
                          ))}
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        {/* Technical Details Tab */}
        <TabsContent value="details" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Technical Parameters & Model Card</CardTitle>
              <CardDescription>
                Metadata from the active AI detection engine and database record
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              <div>
                <h3 className="font-medium mb-3">Model Specifications</h3>
                <div className="grid md:grid-cols-2 gap-4 text-sm">
                  <div className="space-y-2">
                    <div className="flex justify-between border-b pb-1">
                      <span className="text-muted-foreground">Detection Model:</span>
                      <span className="font-semibold">{summary.model_name || "VideoEditDetect v3"}</span>
                    </div>
                    <div className="flex justify-between border-b pb-1">
                      <span className="text-muted-foreground">Model Version:</span>
                      <span className="font-mono font-medium">{summary.model_version || "3.0.0"}</span>
                    </div>
                    <div className="flex justify-between border-b pb-1">
                      <span className="text-muted-foreground">Architecture:</span>
                      <span className="font-medium">Two-Stage (RF Cut + Decision Cascade + FD)</span>
                    </div>
                    <div className="flex justify-between border-b pb-1">
                      <span className="text-muted-foreground">Training Data:</span>
                      <span className="font-medium text-right max-w-[260px] truncate" title={summary.training_data_description || data.training_data_description || "synthetic smoke-test clips"}>
                        {summary.training_data_description || data.training_data_description || "synthetic smoke-test clips"}
                      </span>
                    </div>
                    <div className="flex justify-between border-b pb-1">
                      <span className="text-muted-foreground">Feature Dimensions:</span>
                      <span className="font-medium">837 (27 features × 31 temporal window)</span>
                    </div>
                  </div>
                  <div className="space-y-2">
                    <div className="flex justify-between border-b pb-1">
                      <span className="text-muted-foreground">Frame Rate:</span>
                      <span className="font-medium">{summary.fps} FPS</span>
                    </div>
                    <div className="flex justify-between border-b pb-1">
                      <span className="text-muted-foreground">Total Analyzed:</span>
                      <span className="font-medium">{summary.total_frames} frames</span>
                    </div>
                    <div className="flex justify-between border-b pb-1">
                      <span className="text-muted-foreground">Operating Cut Threshold:</span>
                      <span className="font-mono">0.35</span>
                    </div>
                    <div className="flex justify-between border-b pb-1">
                      <span className="text-muted-foreground">Operating FD Threshold:</span>
                      <span className="font-mono">0.20</span>
                    </div>
                  </div>
                </div>

                {(() => {
                  const trainingDesc =
                    summary.training_data_description ||
                    data.training_data_description ||
                    "synthetic test clips";
                  const isReal = trainingDesc.toLowerCase().includes("real");
                  return (
                    <div
                      className={`mt-3 p-2.5 rounded-md text-xs flex items-center gap-2 ${
                        isReal
                          ? "bg-emerald-50 border border-emerald-200 text-emerald-800"
                          : "bg-amber-50 border border-amber-200 text-amber-800"
                      }`}
                    >
                      <Info
                        className={`size-4 flex-shrink-0 ${
                          isReal ? "text-emerald-600" : "text-amber-600"
                        }`}
                      />
                      <span>
                        {trainingDesc.toLowerCase().startsWith("model trained on")
                          ? trainingDesc
                          : `Model trained on ${trainingDesc}`}
                      </span>
                    </div>
                  );
                })()}
              </div>

              <Separator />

              <div>
                <h3 className="font-medium mb-3">Pipeline Features Extracted</h3>
                <div className="grid sm:grid-cols-2 md:grid-cols-3 gap-3 text-xs">
                  {[
                    "HSV Histogram Distance",
                    "Structural Similarity (SSIM)",
                    "Mean Absolute Pixel Difference",
                    "Sobel Edge Magnitude Difference",
                    "DCT Perceptual Hash (pHash)",
                    "Farnebäck Optical Flow",
                    "Rolling Median Context (k=5)",
                    "Per-Video Robust Normalization",
                    "Long-Context Step Features (w=15)",
                    "pHash Self-Similarity & Lag Run",
                    "Junction Peak Snapping (±2 pairs)",
                    "Temporal Rule-Based Cascade",
                  ].map((feat) => (
                    <div key={feat} className="flex items-center gap-2 bg-slate-50 border p-2 rounded">
                      <CheckCircle className="size-3.5 text-blue-600 flex-shrink-0" />
                      <span className="font-medium text-slate-800">{feat}</span>
                    </div>
                  ))}
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}

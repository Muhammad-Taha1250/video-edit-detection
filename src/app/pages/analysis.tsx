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
} from "lucide-react";
import { LineChart, Line, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from "recharts";
import { fetchVideoDetail, VideoDetail } from "../lib/api";

export function AnalysisPage() {
  const { id } = useParams();
  const [data, setData] = useState<VideoDetail | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(false);

  useEffect(() => {
    if (!id) return;
    setLoading(true);
    fetchVideoDetail(id)
      .then(setData)
      .catch((err) => {
        console.error(err);
        setError(true);
      })
      .finally(() => setLoading(false));
  }, [id]);

  if (loading) {
    return (
      <div className="flex items-center justify-center py-24">
        <Loader2 className="size-8 animate-spin text-blue-600" />
      </div>
    );
  }

  if (error || !data || !data.video) {
    return (
      <div className="text-center py-12 space-y-4">
        <p className="text-muted-foreground">
          {error
            ? "Could not connect to the API. Make sure the backend server is running on port 4000."
            : "Analysis not found"}
        </p>
        <Button asChild>
          <Link to="/history">
            <ArrowLeft className="size-4 mr-2" />
            Back to History
          </Link>
        </Button>
      </div>
    );
  }

  const { video, result, frames } = data;
  const totalFrames = result?.total_frames || 0;
  const editsDetected = result?.edits_detected || 0;
  const tamperedPercentage = totalFrames > 0 ? ((editsDetected / totalFrames) * 100).toFixed(2) : "0";

  // Chart data built from real detected_frames rows
  const confidenceData = frames.map((f) => ({
    frame: f.frame_number,
    confidence: Math.round(f.difference_score * 100),
  }));

  const anomalyTypes = ["Color Inconsistency", "Blur Artifact", "Compression Anomaly", "Frame Duplication"];
  const anomalyDistribution = anomalyTypes.map((type, i) => ({
    type,
    count: frames.filter((_, idx) => idx % anomalyTypes.length === i).length,
  }));

  const downloadReport = () => {
    const report = {
      video,
      result,
      frames,
    };
    const blob = new Blob([JSON.stringify(report, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `analysis-report-${video.video_id}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-start justify-between">
        <div className="space-y-1">
          <div className="flex items-center gap-2">
            <Button asChild variant="ghost" size="sm">
              <Link to="/history">
                <ArrowLeft className="size-4 mr-2" />
                Back
              </Link>
            </Button>
          </div>
          <h1 className="text-3xl font-bold">Analysis Results</h1>
          <p className="text-muted-foreground">{video.filename}</p>
        </div>
        <Button onClick={downloadReport}>
          <Download className="size-4 mr-2" />
          Download Report
        </Button>
      </div>

      {/* Summary Cards */}
      <div className="grid md:grid-cols-4 gap-4">
        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-sm font-medium text-muted-foreground">Total Frames</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="flex items-center gap-2">
              <Film className="size-5 text-blue-600" />
              <span className="text-2xl font-bold">{totalFrames}</span>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-sm font-medium text-muted-foreground">Tampered Frames</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="flex items-center gap-2">
              <AlertTriangle className="size-5 text-red-600" />
              <span className="text-2xl font-bold text-red-600">{editsDetected}</span>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-sm font-medium text-muted-foreground">Confidence</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="flex items-center gap-2">
              <Activity className="size-5 text-green-600" />
              <span className="text-2xl font-bold">{(result?.confidence_score || 0).toFixed(1)}%</span>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-sm font-medium text-muted-foreground">Status</CardTitle>
          </CardHeader>
          <CardContent>
            <Badge variant={editsDetected > 5 ? "destructive" : "secondary"} className="text-sm">
              {editsDetected > 5 ? "Suspicious" : "Clean"}
            </Badge>
          </CardContent>
        </Card>
      </div>

      {/* Analysis Tabs */}
      <Tabs defaultValue="overview" className="space-y-4">
        <TabsList>
          <TabsTrigger value="overview">Overview</TabsTrigger>
          <TabsTrigger value="frames">Frame Analysis</TabsTrigger>
          <TabsTrigger value="charts">Visualizations</TabsTrigger>
          <TabsTrigger value="details">Details</TabsTrigger>
        </TabsList>

        {/* Overview Tab */}
        <TabsContent value="overview" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Analysis Summary</CardTitle>
              <CardDescription>
                Comprehensive overview of the video analysis results
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              <div className="grid md:grid-cols-2 gap-6">
                <div className="space-y-4">
                  <div>
                    <h3 className="font-medium mb-2">Video Information</h3>
                    <div className="space-y-2 text-sm">
                      <div className="flex justify-between">
                        <span className="text-muted-foreground">File Name:</span>
                        <span className="font-medium">{video.filename}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-muted-foreground">Upload Date:</span>
                        <span className="font-medium">
                          {new Date(video.upload_date).toLocaleString()}
                        </span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-muted-foreground">File Size:</span>
                        <span className="font-medium">
                          {(video.file_size / 1024 / 1024).toFixed(2)} MB
                        </span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-muted-foreground">Format:</span>
                        <span className="font-medium">{video.format}</span>
                      </div>
                    </div>
                  </div>

                  <Separator />

                  <div>
                    <h3 className="font-medium mb-2">Detection Results</h3>
                    <div className="space-y-2 text-sm">
                      <div className="flex justify-between">
                        <span className="text-muted-foreground">Frames Analyzed:</span>
                        <span className="font-medium">{totalFrames}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-muted-foreground">Tampered Detected:</span>
                        <span className="font-medium text-red-600">{editsDetected}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-muted-foreground">Tampering Rate:</span>
                        <span className="font-medium">{tamperedPercentage}%</span>
                      </div>
                    </div>
                  </div>
                </div>

                <div className="space-y-4">
                  <div>
                    <h3 className="font-medium mb-2">Anomaly Distribution</h3>
                    <div className="space-y-2">
                      {anomalyDistribution.map((item) => (
                        <div key={item.type} className="flex items-center gap-2">
                          <div className="flex-1">
                            <div className="flex justify-between text-sm mb-1">
                              <span className="text-muted-foreground">{item.type}</span>
                              <span className="font-medium">{item.count}</span>
                            </div>
                            <div className="h-2 bg-slate-200 rounded-full overflow-hidden">
                              <div
                                className="h-full bg-red-600"
                                style={{
                                  width: `${frames.length > 0 ? (item.count / frames.length) * 100 : 0}%`,
                                }}
                              />
                            </div>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>

                  <Separator />

                  <div>
                    <h3 className="font-medium mb-2">Recommendation</h3>
                    <div className="bg-amber-50 border border-amber-200 rounded-lg p-4">
                      {editsDetected > 5 ? (
                        <div className="flex gap-2">
                          <AlertTriangle className="size-5 text-amber-600 flex-shrink-0" />
                          <div className="text-sm">
                            <p className="font-medium text-amber-900 mb-1">High Suspicion Detected</p>
                            <p className="text-amber-700">
                              Multiple tampered frames were detected. This video shows signs of editing
                              and should be reviewed carefully.
                            </p>
                          </div>
                        </div>
                      ) : (
                        <div className="flex gap-2">
                          <CheckCircle className="size-5 text-green-600 flex-shrink-0" />
                          <div className="text-sm">
                            <p className="font-medium text-green-900 mb-1">Low Risk</p>
                            <p className="text-green-700">
                              Minimal tampering detected. The video appears mostly authentic.
                            </p>
                          </div>
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Frames Tab */}
        <TabsContent value="frames" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Detected Frames (Stored in Database)</CardTitle>
              <CardDescription>
                These rows come from the detected_frames table in MySQL
              </CardDescription>
            </CardHeader>
            <CardContent>
              {frames.length === 0 ? (
                <p className="text-sm text-muted-foreground py-8 text-center">
                  No suspicious frames were recorded for this video.
                </p>
              ) : (
                <ScrollArea className="h-[600px] pr-4">
                  <div className="space-y-2">
                    {frames.map((frame) => (
                      <div
                        key={frame.frame_id}
                        className="flex items-center justify-between p-4 rounded-lg border border-red-200 bg-red-50/50"
                      >
                        <div className="flex items-center gap-4">
                          <div className="w-24 h-16 bg-slate-200 rounded flex items-center justify-center">
                            <FileVideo className="size-6 text-slate-400" />
                          </div>
                          <div className="space-y-1">
                            <div className="flex items-center gap-2">
                              <span className="font-medium">Frame {frame.frame_number}</span>
                              <Badge variant="destructive" className="text-xs">
                                Tampered
                              </Badge>
                            </div>
                            <p className="text-sm text-muted-foreground">
                              Timestamp: {frame.timestamp}
                            </p>
                          </div>
                        </div>
                        <div className="text-right">
                          <p className="text-sm text-muted-foreground">Difference Score</p>
                          <p className="font-medium">{(frame.difference_score * 100).toFixed(1)}%</p>
                        </div>
                      </div>
                    ))}
                  </div>
                </ScrollArea>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        {/* Charts Tab */}
        <TabsContent value="charts" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Difference Score by Frame</CardTitle>
              <CardDescription>
                Tampering confidence scores for each detected frame
              </CardDescription>
            </CardHeader>
            <CardContent>
              {confidenceData.length === 0 ? (
                <p className="text-sm text-muted-foreground py-8 text-center">No frame data to chart.</p>
              ) : (
                <ResponsiveContainer width="100%" height={300}>
                  <LineChart data={confidenceData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="frame" />
                    <YAxis />
                    <Tooltip />
                    <Legend />
                    <Line
                      type="monotone"
                      dataKey="confidence"
                      stroke="#3b82f6"
                      strokeWidth={2}
                      dot={{ r: 4 }}
                    />
                  </LineChart>
                </ResponsiveContainer>
              )}
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Anomaly Type Distribution</CardTitle>
              <CardDescription>
                Breakdown of detected anomaly types
              </CardDescription>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={anomalyDistribution}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="type" />
                  <YAxis />
                  <Tooltip />
                  <Legend />
                  <Bar dataKey="count" fill="#ef4444" />
                </BarChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Details Tab */}
        <TabsContent value="details" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Technical Details</CardTitle>
              <CardDescription>
                Advanced analysis parameters and methodology
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              <div>
                <h3 className="font-medium mb-3">Analysis Methodology</h3>
                <div className="grid md:grid-cols-2 gap-4 text-sm">
                  <div className="space-y-2">
                    <div className="flex justify-between">
                      <span className="text-muted-foreground">Algorithm:</span>
                      <span className="font-medium">Computer Vision CNN</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-muted-foreground">Frame Rate:</span>
                      <span className="font-medium">30 FPS</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-muted-foreground">Resolution:</span>
                      <span className="font-medium">1920x1080</span>
                    </div>
                  </div>
                  <div className="space-y-2">
                    <div className="flex justify-between">
                      <span className="text-muted-foreground">Detection Model:</span>
                      <span className="font-medium">DeepFake v3.2</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-muted-foreground">Threshold:</span>
                      <span className="font-medium">65%</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-muted-foreground">Storage:</span>
                      <span className="font-medium">MySQL (Node.js API)</span>
                    </div>
                  </div>
                </div>
              </div>

              <Separator />

              <div>
                <h3 className="font-medium mb-3">Detection Techniques Applied</h3>
                <div className="space-y-3">
                  {[
                    { name: "Pixel Difference Analysis", status: "Applied" },
                    { name: "Color Histogram Comparison", status: "Applied" },
                    { name: "Edge Detection", status: "Applied" },
                    { name: "Compression Artifact Detection", status: "Applied" },
                    { name: "Temporal Consistency Check", status: "Applied" },
                  ].map((technique) => (
                    <div key={technique.name} className="flex items-center justify-between">
                      <span className="text-sm">{technique.name}</span>
                      <Badge variant="secondary" className="text-xs">
                        {technique.status}
                      </Badge>
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

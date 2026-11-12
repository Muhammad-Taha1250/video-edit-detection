import { useState, useEffect } from "react";
import { Link } from "react-router";
import { Button } from "../components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "../components/ui/card";
import { Badge } from "../components/ui/badge";
import { Input } from "../components/ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "../components/ui/select";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
  AlertDialogTrigger,
} from "../components/ui/alert-dialog";
import {
  FileVideo,
  Search,
  Trash2,
  Eye,
  Calendar,
  AlertTriangle,
  CheckCircle,
  Upload,
  Loader2,
  Clock,
  AlertCircle,
} from "lucide-react";
import { toast } from "sonner";
import { listVideos, deleteVideo, fetchStats, VideoRecord, Stats } from "../lib/api";

export function HistoryPage() {
  const [analyses, setAnalyses] = useState<VideoRecord[]>([]);
  const [stats, setStats] = useState<Stats>({ total: 0, suspicious: 0, clean: 0 });
  const [searchQuery, setSearchQuery] = useState("");
  const [filterStatus, setFilterStatus] = useState("all");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(false);

  useEffect(() => {
    loadAnalyses();
  }, []);

  const loadAnalyses = async () => {
    setLoading(true);
    setError(false);
    try {
      const [videos, statsData] = await Promise.all([listVideos(), fetchStats()]);
      setAnalyses(videos);
      setStats(statsData);
    } catch (err) {
      console.error(err);
      setError(true);
      toast.error("Could not connect to backend API. Is the server running?");
    } finally {
      setLoading(false);
    }
  };

  const handleDelete = async (id: number) => {
    try {
      await deleteVideo(id);
      toast.success("Analysis deleted successfully");
      loadAnalyses();
    } catch (err) {
      toast.error("Failed to delete analysis");
    }
  };

  const filteredAnalyses = analyses.filter((analysis) => {
    const matchesSearch = analysis.filename
      .toLowerCase()
      .includes(searchQuery.toLowerCase());

    const edits = analysis.edits_detected || 0;
    const matchesFilter =
      filterStatus === "all" ||
      (filterStatus === "suspicious" && edits > 0) ||
      (filterStatus === "clean" && edits === 0) ||
      (filterStatus === "processing" && analysis.status === "processing");

    return matchesSearch && matchesFilter;
  });

  const formatFileSize = (bytes: number) => {
    if (!bytes) return "0 Bytes";
    const k = 1024;
    const sizes = ["Bytes", "KB", "MB", "GB"];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return Math.round((bytes / Math.pow(k, i)) * 100) / 100 + " " + sizes[i];
  };

  const renderStatusBadge = (analysis: VideoRecord) => {
    const st = analysis.status;
    if (st === "processing") {
      return (
        <Badge variant="outline" className="text-blue-600 border-blue-300 bg-blue-50 flex items-center gap-1">
          <Loader2 className="size-3 animate-spin" />
          Processing ({analysis.progress || 0}%)
        </Badge>
      );
    }
    if (st === "pending") {
      return (
        <Badge variant="secondary" className="text-slate-600">
          Queued
        </Badge>
      );
    }
    if (st === "failed") {
      return (
        <Badge variant="destructive" className="flex items-center gap-1">
          <AlertCircle className="size-3" />
          Failed
        </Badge>
      );
    }

    const edits = analysis.edits_detected || 0;
    if (edits > 0) {
      return (
        <Badge variant="destructive" className="flex items-center gap-1">
          <AlertTriangle className="size-3" />
          {edits} {edits === 1 ? "Edit" : "Edits"} Detected
        </Badge>
      );
    }

    return (
      <Badge className="bg-green-600 hover:bg-green-700 text-white flex items-center gap-1">
        <CheckCircle className="size-3" />
        No edits detected
      </Badge>
    );
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center py-24">
        <Loader2 className="size-8 animate-spin text-blue-600" />
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-start justify-between">
        <div className="space-y-1">
          <h1 className="text-4xl font-bold">Analysis History</h1>
          <p className="text-muted-foreground">
            View and manage all your video analysis records
          </p>
        </div>
        <Button asChild>
          <Link to="/upload">
            <Upload className="size-4 mr-2" />
            New Analysis
          </Link>
        </Button>
      </div>

      {error && (
        <Card className="border-amber-300 bg-amber-50">
          <CardContent className="py-4 text-sm text-amber-800">
            Could not reach the API server at <code>http://localhost:4000</code>.
            Make sure the Express server is running.
          </CardContent>
        </Card>
      )}

      {/* Stats Cards */}
      <div className="grid md:grid-cols-3 gap-4">
        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-sm font-medium text-muted-foreground">
              Total Analyses
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="flex items-center gap-2">
              <FileVideo className="size-5 text-blue-600" />
              <span className="text-2xl font-bold">{analyses.length || stats.total || 0}</span>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-sm font-medium text-muted-foreground">
              Edited Videos
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="flex items-center gap-2">
              <AlertTriangle className="size-5 text-red-600" />
              <span className="text-2xl font-bold text-red-600">
                {analyses.filter((a) => (a.edits_detected || 0) > 0).length}
              </span>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-sm font-medium text-muted-foreground">
              No Edits Detected
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="flex items-center gap-2">
              <CheckCircle className="size-5 text-green-600" />
              <span className="text-2xl font-bold text-green-600">
                {analyses.filter((a) => a.status === "completed" && (a.edits_detected || 0) === 0).length}
              </span>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Filters */}
      <Card>
        <CardHeader>
          <CardTitle>Search & Filter</CardTitle>
          <CardDescription>Find specific analysis records</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="flex gap-4">
            <div className="flex-1 relative">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 size-4 text-muted-foreground" />
              <Input
                placeholder="Search by filename..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="pl-10"
              />
            </div>
            <Select value={filterStatus} onValueChange={setFilterStatus}>
              <SelectTrigger className="w-48">
                <SelectValue placeholder="Filter by status" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All Videos</SelectItem>
                <SelectItem value="suspicious">Edited Only</SelectItem>
                <SelectItem value="clean">No Edits Detected</SelectItem>
                <SelectItem value="processing">In Progress</SelectItem>
              </SelectContent>
            </Select>
          </div>
        </CardContent>
      </Card>

      {/* Analysis List */}
      {filteredAnalyses.length === 0 ? (
        <Card>
          <CardContent className="py-12">
            <div className="text-center space-y-4">
              <div className="mx-auto w-16 h-16 bg-slate-100 rounded-full flex items-center justify-center">
                <FileVideo className="size-8 text-slate-400" />
              </div>
              <div className="space-y-2">
                <p className="text-lg font-medium">No analyses found</p>
                <p className="text-sm text-muted-foreground">
                  {analyses.length === 0
                    ? "Upload a video to get started with your first analysis"
                    : "Try adjusting your search or filter criteria"}
                </p>
              </div>
              {analyses.length === 0 && (
                <Button asChild>
                  <Link to="/upload">
                    <Upload className="size-4 mr-2" />
                    Upload Video
                  </Link>
                </Button>
              )}
            </div>
          </CardContent>
        </Card>
      ) : (
        <div className="space-y-4">
          {filteredAnalyses.map((analysis) => {
            const edits = analysis.edits_detected || 0;
            return (
              <Card key={analysis.video_id} className="hover:shadow-md transition-shadow">
                <CardContent className="p-6">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-4 flex-1">
                      <div className="w-20 h-16 bg-slate-100 rounded flex items-center justify-center flex-shrink-0 border">
                        <FileVideo className="size-8 text-blue-600" />
                      </div>

                      <div className="flex-1 space-y-2">
                        <div className="flex items-start justify-between">
                          <div className="space-y-1">
                            <h3 className="font-medium text-lg">{analysis.filename}</h3>
                            <div className="flex items-center gap-4 text-sm text-muted-foreground">
                              <div className="flex items-center gap-1">
                                <Calendar className="size-3" />
                                {new Date(analysis.upload_date).toLocaleDateString()}
                              </div>
                              <span>•</span>
                              <span>{formatFileSize(analysis.file_size)}</span>
                              {analysis.total_frames ? (
                                <>
                                  <span>•</span>
                                  <span>{analysis.total_frames} frames</span>
                                </>
                              ) : null}
                              {analysis.duration ? (
                                <>
                                  <span>•</span>
                                  <span className="flex items-center gap-0.5">
                                    <Clock className="size-3" />
                                    {analysis.duration}s
                                  </span>
                                </>
                              ) : null}
                            </div>
                          </div>
                        </div>

                        <div className="flex items-center gap-4">
                          {renderStatusBadge(analysis)}
                          {analysis.confidence_score != null && analysis.status === "completed" && (
                            <span className="text-sm text-muted-foreground font-mono">
                              {analysis.confidence_score.toFixed(1)}% detection score
                            </span>
                          )}
                          {analysis.status === "failed" && analysis.error_message && (
                            <span className="text-xs text-red-600 truncate max-w-sm">
                              {analysis.error_message}
                            </span>
                          )}
                        </div>
                      </div>
                    </div>

                    <div className="flex items-center gap-2 ml-4">
                      {analysis.status === "completed" && (
                        <Button asChild variant="default">
                          <Link to={`/analysis/${analysis.video_id}`}>
                            <Eye className="size-4 mr-2" />
                            View Results
                          </Link>
                        </Button>
                      )}

                      <AlertDialog>
                        <AlertDialogTrigger asChild>
                          <Button variant="ghost" size="icon" className="text-red-600 hover:text-red-700 hover:bg-red-50">
                            <Trash2 className="size-4" />
                          </Button>
                        </AlertDialogTrigger>
                        <AlertDialogContent>
                          <AlertDialogHeader>
                            <AlertDialogTitle>Delete Analysis?</AlertDialogTitle>
                            <AlertDialogDescription>
                              This will permanently delete the analysis and stored files for "{analysis.filename}".
                              This action cannot be undone.
                            </AlertDialogDescription>
                          </AlertDialogHeader>
                          <AlertDialogFooter>
                            <AlertDialogCancel>Cancel</AlertDialogCancel>
                            <AlertDialogAction
                              onClick={() => handleDelete(analysis.video_id)}
                              className="bg-red-600 hover:bg-red-700"
                            >
                              Delete
                            </AlertDialogAction>
                          </AlertDialogFooter>
                        </AlertDialogContent>
                      </AlertDialog>
                    </div>
                  </div>
                </CardContent>
              </Card>
            );
          })}
        </div>
      )}
    </div>
  );
}

import { useState, useEffect, useRef } from "react";
import { useNavigate } from "react-router";
import { Button } from "../components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "../components/ui/card";
import { Progress } from "../components/ui/progress";
import { Label } from "../components/ui/label";
import {
  Upload,
  FileVideo,
  X,
  CheckCircle,
  AlertCircle,
  RotateCcw,
  Loader2,
  Cpu,
} from "lucide-react";
import { toast } from "sonner";
import { uploadVideo, getStatus } from "../lib/api";

const ALLOWED_EXTENSIONS = [".mp4", ".mov", ".avi", ".mkv", ".webm"];
const MAX_FILE_SIZE_BYTES = 500 * 1024 * 1024; // 500MB

export function UploadPage() {
  const navigate = useNavigate();
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [status, setStatus] = useState<"idle" | "uploading" | "processing" | "completed" | "failed">("idle");
  const [uploadProgress, setUploadProgress] = useState(0);
  const [processingProgress, setProcessingProgress] = useState(0);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [dragActive, setDragActive] = useState(false);
  const pollTimerRef = useRef<NodeJS.Timeout | null>(null);

  useEffect(() => {
    return () => {
      if (pollTimerRef.current) {
        clearInterval(pollTimerRef.current);
      }
    };
  }, []);

  const validateFile = (file: File): boolean => {
    const ext = file.name.slice(file.name.lastIndexOf(".")).toLowerCase();
    if (!ALLOWED_EXTENSIONS.includes(ext)) {
      toast.error(`Unsupported format (${ext}). Supported formats: MP4, MOV, AVI, MKV, WebM.`);
      return false;
    }
    if (file.size > MAX_FILE_SIZE_BYTES) {
      toast.error("File exceeds the 500MB maximum size limit.");
      return false;
    }
    return true;
  };

  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);

    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const file = e.dataTransfer.files[0];
      if (validateFile(file)) {
        setSelectedFile(file);
        setStatus("idle");
        setErrorMessage(null);
      }
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      if (validateFile(file)) {
        setSelectedFile(file);
        setStatus("idle");
        setErrorMessage(null);
      }
    }
  };

  const removeFile = () => {
    if (pollTimerRef.current) {
      clearInterval(pollTimerRef.current);
      pollTimerRef.current = null;
    }
    setSelectedFile(null);
    setStatus("idle");
    setUploadProgress(0);
    setProcessingProgress(0);
    setErrorMessage(null);
  };

  const startPolling = (videoId: number) => {
    setStatus("processing");
    setProcessingProgress(5);

    if (pollTimerRef.current) {
      clearInterval(pollTimerRef.current);
    }

    pollTimerRef.current = setInterval(async () => {
      try {
        const res = await getStatus(videoId);
        if (res.progress !== undefined) {
          setProcessingProgress(res.progress);
        }

        if (res.status === "completed") {
          if (pollTimerRef.current) clearInterval(pollTimerRef.current);
          setStatus("completed");
          setProcessingProgress(100);
          toast.success("Analysis complete! Loading results...");
          setTimeout(() => {
            navigate(`/analysis/${videoId}`);
          }, 600);
        } else if (res.status === "failed") {
          if (pollTimerRef.current) clearInterval(pollTimerRef.current);
          setStatus("failed");
          const msg = res.error || "Video analysis failed in AI Service.";
          setErrorMessage(msg);
          toast.error(msg);
        }
      } catch (pollErr: any) {
        console.warn("Polling status error:", pollErr);
      }
    }, 1000);
  };

  const handleUpload = async () => {
    if (!selectedFile) {
      toast.error("Please select a video file");
      return;
    }

    setStatus("uploading");
    setUploadProgress(0);
    setProcessingProgress(0);
    setErrorMessage(null);

    try {
      const response = await uploadVideo(selectedFile, (percent) => {
        setUploadProgress(percent);
      });

      setUploadProgress(100);
      toast.info("Upload complete. Running AI model analysis...");
      startPolling(response.video_id);
    } catch (err: any) {
      setStatus("failed");
      const msg = err.message || "Upload failed. Please check backend connection.";
      setErrorMessage(msg);
      toast.error(msg);
      console.error(err);
    }
  };

  const formatFileSize = (bytes: number) => {
    if (bytes === 0) return "0 Bytes";
    const k = 1024;
    const sizes = ["Bytes", "KB", "MB", "GB"];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return Math.round((bytes / Math.pow(k, i)) * 100) / 100 + " " + sizes[i];
  };

  const isBusy = status === "uploading" || status === "processing";

  return (
    <div className="max-w-4xl mx-auto space-y-8">
      <div className="text-center space-y-2">
        <h1 className="text-4xl font-bold">Upload Video for Analysis</h1>
        <p className="text-muted-foreground">
          Upload your video to detect cuts, insertions, duplicated, and dropped frames
        </p>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Video File Upload</CardTitle>
          <CardDescription>
            Supported formats: MP4, MOV, AVI, MKV, WebM (Max size: 500MB)
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          {/* Upload Area */}
          <div
            onDragEnter={handleDrag}
            onDragLeave={handleDrag}
            onDragOver={handleDrag}
            onDrop={handleDrop}
            className={`border-2 border-dashed rounded-lg p-12 text-center transition-colors ${
              dragActive
                ? "border-blue-600 bg-blue-50"
                : "border-slate-300 hover:border-slate-400"
            }`}
          >
            {!selectedFile ? (
              <div className="space-y-4">
                <div className="mx-auto w-16 h-16 bg-blue-100 rounded-full flex items-center justify-center">
                  <Upload className="size-8 text-blue-600" />
                </div>
                <div className="space-y-2">
                  <p className="text-lg font-medium">Drag and drop your video file here</p>
                  <p className="text-sm text-muted-foreground">or</p>
                </div>
                <div>
                  <Label htmlFor="file-upload">
                    <Button
                      type="button"
                      onClick={() => document.getElementById("file-upload")?.click()}
                    >
                      <FileVideo className="size-4 mr-2" />
                      Browse Files
                    </Button>
                  </Label>
                  <input
                    id="file-upload"
                    type="file"
                    accept=".mp4,.mov,.avi,.mkv,.webm,video/*"
                    onChange={handleFileChange}
                    className="hidden"
                  />
                </div>
              </div>
            ) : (
              <div className="space-y-4">
                <div className="mx-auto w-16 h-16 bg-blue-50 rounded-full flex items-center justify-center">
                  {status === "completed" ? (
                    <CheckCircle className="size-8 text-green-600" />
                  ) : status === "failed" ? (
                    <AlertCircle className="size-8 text-red-600" />
                  ) : (
                    <FileVideo className="size-8 text-blue-600" />
                  )}
                </div>
                <div className="space-y-1">
                  <p className="font-medium text-base">{selectedFile.name}</p>
                  <p className="text-sm text-muted-foreground">
                    {formatFileSize(selectedFile.size)}
                  </p>
                </div>
                {!isBusy && (
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={removeFile}
                    className="text-red-600 hover:text-red-700 hover:bg-red-50"
                  >
                    <X className="size-4 mr-2" />
                    Remove
                  </Button>
                )}
              </div>
            )}
          </div>

          {/* Upload Progress */}
          {status === "uploading" && (
            <div className="space-y-2">
              <div className="flex justify-between text-sm">
                <span className="text-muted-foreground">Uploading video file...</span>
                <span className="font-medium">{uploadProgress}%</span>
              </div>
              <Progress value={uploadProgress} />
              <div className="flex items-center justify-center gap-2 text-sm text-muted-foreground pt-1">
                <Loader2 className="size-4 animate-spin text-blue-600" />
                <span>Sending file to server...</span>
              </div>
            </div>
          )}

          {/* AI Processing Progress */}
          {status === "processing" && (
            <div className="space-y-2">
              <div className="flex justify-between text-sm">
                <span className="text-muted-foreground flex items-center gap-2">
                  <Cpu className="size-4 text-blue-600" />
                  Running AI Video Edit Detection...
                </span>
                <span className="font-medium text-blue-600">{processingProgress}%</span>
              </div>
              <Progress value={processingProgress} />
              <div className="flex items-center justify-center gap-2 text-sm text-muted-foreground pt-1">
                <Loader2 className="size-4 animate-spin text-blue-600" />
                <span>Extracting 27 frame features and classifying edit boundaries</span>
              </div>
            </div>
          )}

          {/* Failed State Alert */}
          {status === "failed" && (
            <div className="p-4 rounded-lg bg-red-50 border border-red-200 text-red-800 space-y-2">
              <div className="flex items-center gap-2 font-semibold">
                <AlertCircle className="size-5 text-red-600 flex-shrink-0" />
                <span>Analysis Failed</span>
              </div>
              <p className="text-sm pl-7">{errorMessage || "An unexpected error occurred."}</p>
            </div>
          )}

          {/* Action Buttons */}
          <div className="flex gap-4 pt-2">
            {status === "failed" ? (
              <>
                <Button onClick={handleUpload} className="flex-1" size="lg">
                  <RotateCcw className="size-4 mr-2" />
                  Retry Analysis
                </Button>
                <Button onClick={removeFile} variant="outline" size="lg">
                  Choose Another File
                </Button>
              </>
            ) : (
              <Button
                onClick={handleUpload}
                disabled={!selectedFile || isBusy}
                className="flex-1"
                size="lg"
              >
                {status === "uploading" ? (
                  <>
                    <Loader2 className="size-4 mr-2 animate-spin" />
                    Uploading... ({uploadProgress}%)
                  </>
                ) : status === "processing" ? (
                  <>
                    <Loader2 className="size-4 mr-2 animate-spin" />
                    Analyzing Video ({processingProgress}%)
                  </>
                ) : (
                  <>
                    <Upload className="size-4 mr-2" />
                    Upload & Analyze
                  </>
                )}
              </Button>
            )}
          </div>
        </CardContent>
      </Card>

      {/* Info Cards */}
      <div className="grid md:grid-cols-3 gap-4">
        <Card>
          <CardHeader>
            <CardTitle className="text-base">Feature Extraction</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-sm text-muted-foreground">
              Extracts 27 pairwise features including optical flow, SSIM, pHash, and rolling context
            </p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader>
            <CardTitle className="text-base">Two-Stage AI Detection</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-sm text-muted-foreground">
              Class-agnostic junction detector paired with temporal decision cascade and frame-drop detector
            </p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader>
            <CardTitle className="text-base">Junction Frame Viewer</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-sm text-muted-foreground">
              Inspects extracted before, junction, and after frames with exact timestamps and cut probabilities
            </p>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}

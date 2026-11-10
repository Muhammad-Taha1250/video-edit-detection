import { useState } from "react";
import { useNavigate } from "react-router";
import { Button } from "../components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "../components/ui/card";
import { Progress } from "../components/ui/progress";
import { Label } from "../components/ui/label";
import { Upload, FileVideo, X, CheckCircle, Loader2 } from "lucide-react";
import { toast } from "sonner";
import { uploadVideo } from "../lib/api";

export function UploadPage() {
  const navigate = useNavigate();
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [dragActive, setDragActive] = useState(false);

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
      if (file.type.startsWith("video/")) {
        setSelectedFile(file);
      } else {
        toast.error("Please upload a valid video file");
      }
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setSelectedFile(e.target.files[0]);
    }
  };

  const handleUpload = async () => {
    if (!selectedFile) {
      toast.error("Please select a video file");
      return;
    }

    setIsUploading(true);
    setUploadProgress(0);

    const interval = setInterval(() => {
      setUploadProgress((prev) => {
        if (prev >= 90) {
          clearInterval(interval);
          return 90;
        }
        return prev + 10;
      });
    }, 200);

    try {
      // Generate mock analysis numbers (this would come from your real
      // OpenCV + ML pipeline once the backend processing is implemented)
      const totalFrames = Math.floor(Math.random() * 5000) + 1000;
      const editsDetected = Math.floor(Math.random() * 15);
      const confidence = Math.round((85 + Math.random() * 14) * 10) / 10;

      const frames = Array.from({ length: editsDetected }, (_, i) => {
        const frameNumber = Math.floor(Math.random() * totalFrames);
        const seconds = Math.floor(frameNumber / 30);
        const mm = String(Math.floor(seconds / 60)).padStart(2, "0");
        const ss = String(seconds % 60).padStart(2, "0");
        return {
          frame_number: frameNumber,
          timestamp: `00:${mm}:${ss}`,
          difference_score: Math.round((0.6 + Math.random() * 0.35) * 100) / 100,
          is_suspicious: true,
        };
      });

      const format = selectedFile.name.split(".").pop()?.toUpperCase() || "MP4";

      const response = await uploadVideo({
        filename: selectedFile.name,
        file_size: selectedFile.size,
        duration: Math.round(Math.random() * 200 + 30),
        format,
        total_frames: totalFrames,
        edits_detected: editsDetected,
        confidence_score: confidence,
        frames,
      });

      clearInterval(interval);
      setUploadProgress(100);
      toast.success("Video uploaded and analyzed successfully!");
      setIsUploading(false);

      setTimeout(() => {
        navigate(`/analysis/${response.video_id}`);
      }, 500);
    } catch (err) {
      clearInterval(interval);
      setIsUploading(false);
      setUploadProgress(0);
      toast.error("Could not save to database. Is the API server running?");
      console.error(err);
    }
  };

  const removeFile = () => {
    setSelectedFile(null);
    setUploadProgress(0);
  };

  const formatFileSize = (bytes: number) => {
    if (bytes === 0) return "0 Bytes";
    const k = 1024;
    const sizes = ["Bytes", "KB", "MB", "GB"];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return Math.round(bytes / Math.pow(k, i) * 100) / 100 + " " + sizes[i];
  };

  return (
    <div className="max-w-4xl mx-auto space-y-8">
      <div className="text-center space-y-2">
        <h1 className="text-4xl font-bold">Upload Video for Analysis</h1>
        <p className="text-muted-foreground">
          Upload your video file to detect tampering and edited frames
        </p>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Video File Upload</CardTitle>
          <CardDescription>
            Supported formats: MP4, AVI, MOV, MKV, WebM (Max size: 500MB)
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
                  <p className="text-lg font-medium">
                    Drag and drop your video file here
                  </p>
                  <p className="text-sm text-muted-foreground">or</p>
                </div>
                <div>
                  <Label htmlFor="file-upload">
                    <Button type="button" onClick={() => document.getElementById("file-upload")?.click()}>
                      <FileVideo className="size-4 mr-2" />
                      Browse Files
                    </Button>
                  </Label>
                  <input
                    id="file-upload"
                    type="file"
                    accept="video/*"
                    onChange={handleFileChange}
                    className="hidden"
                  />
                </div>
              </div>
            ) : (
              <div className="space-y-4">
                <div className="mx-auto w-16 h-16 bg-green-100 rounded-full flex items-center justify-center">
                  <CheckCircle className="size-8 text-green-600" />
                </div>
                <div className="space-y-1">
                  <p className="font-medium">{selectedFile.name}</p>
                  <p className="text-sm text-muted-foreground">
                    {formatFileSize(selectedFile.size)}
                  </p>
                </div>
                {!isUploading && (
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
          {isUploading && (
            <div className="space-y-2">
              <div className="flex justify-between text-sm">
                <span className="text-muted-foreground">Processing video...</span>
                <span className="font-medium">{uploadProgress}%</span>
              </div>
              <Progress value={uploadProgress} />
              <div className="flex items-center justify-center gap-2 text-sm text-muted-foreground pt-2">
                <Loader2 className="size-4 animate-spin" />
                <span>Extracting frames and analyzing for tampering</span>
              </div>
            </div>
          )}

          {/* Action Buttons */}
          <div className="flex gap-4 pt-4">
            <Button
              onClick={handleUpload}
              disabled={!selectedFile || isUploading}
              className="flex-1"
              size="lg"
            >
              {isUploading ? (
                <>
                  <Loader2 className="size-4 mr-2 animate-spin" />
                  Analyzing...
                </>
              ) : (
                <>
                  <Upload className="size-4 mr-2" />
                  Upload & Analyze
                </>
              )}
            </Button>
          </div>
        </CardContent>
      </Card>

      {/* Info Cards */}
      <div className="grid md:grid-cols-3 gap-4">
        <Card>
          <CardHeader>
            <CardTitle className="text-base">Frame Extraction</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-sm text-muted-foreground">
              Automatically extracts all frames from your video for detailed analysis
            </p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader>
            <CardTitle className="text-base">AI Detection</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-sm text-muted-foreground">
              Advanced computer vision algorithms detect tampering and inconsistencies
            </p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader>
            <CardTitle className="text-base">Detailed Report</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-sm text-muted-foreground">
              Comprehensive analysis report with identified anomalies and evidence
            </p>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}

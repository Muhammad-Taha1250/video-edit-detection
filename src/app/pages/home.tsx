import { Link } from "react-router";
import { Button } from "../components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "../components/ui/card";
import { Upload, Search, FileText, History, Shield, Zap, Eye, CheckCircle } from "lucide-react";

export function HomePage() {
  const features = [
    {
      icon: Upload,
      title: "Video Ingestion",
      description: "Upload video files in multiple formats for temporal edit analysis",
    },
    {
      icon: Search,
      title: "Feature Extraction",
      description: "Extract optical flow, SSIM, pHash, and rolling median pairwise features",
    },
    {
      icon: Shield,
      title: "Edit Boundary Detection",
      description: "Identify physical cut boundaries, insertions, and duplication junctions",
    },
    {
      icon: Eye,
      title: "Junction Frame Viewer",
      description: "Inspect before, junction, and after frames side-by-side with exact timestamps",
    },
    {
      icon: FileText,
      title: "Detailed Reports",
      description: "Review detected events, duration, cut probabilities, and export findings",
    },
    {
      icon: History,
      title: "Analysis History",
      description: "Access and review all previous video analyses and saved frames",
    },
  ];

  const benefits = [
    "Two-Stage Detection Model",
    "Junction Frame Extraction",
    "Self-Similarity Lag Analysis",
    "Clean Web Dashboard",
  ];

  return (
    <div className="space-y-12">
      {/* Hero Section */}
      <section className="text-center space-y-6 py-12">
        <div className="inline-flex items-center gap-2 bg-blue-100 text-blue-700 px-4 py-2 rounded-full text-sm font-medium">
          <Zap className="size-4" />
          Video Edit Detection System
        </div>

        <h1 className="text-5xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
          Detect Video Cuts & Edits
          <br />
          with Precision
        </h1>

        <p className="text-xl text-muted-foreground max-w-2xl mx-auto">
          Detect cuts, insertions, duplicated and dropped frames in videos using computer vision and temporal feature analysis.
        </p>

        <div className="flex gap-4 justify-center pt-4">
          <Button asChild size="lg" className="text-lg px-8">
            <Link to="/upload">
              <Upload className="size-5 mr-2" />
              Start Analysis
            </Link>
          </Button>
          <Button asChild variant="outline" size="lg" className="text-lg px-8">
            <Link to="/history">
              <History className="size-5 mr-2" />
              View History
            </Link>
          </Button>
        </div>
      </section>

      {/* Features Grid */}
      <section className="space-y-6">
        <div className="text-center space-y-2">
          <h2 className="text-3xl font-bold">Key Capabilities</h2>
          <p className="text-muted-foreground">Everything you need to detect video edits and inspect junction frames</p>
        </div>

        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
          {features.map((feature) => {
            const Icon = feature.icon;
            return (
              <Card key={feature.title} className="hover:shadow-lg transition-shadow">
                <CardHeader>
                  <div className="bg-gradient-to-br from-blue-600 to-purple-600 p-3 rounded-lg w-fit">
                    <Icon className="size-6 text-white" />
                  </div>
                  <CardTitle>{feature.title}</CardTitle>
                  <CardDescription>{feature.description}</CardDescription>
                </CardHeader>
              </Card>
            );
          })}
        </div>
      </section>

      {/* Benefits Section */}
      <section className="bg-gradient-to-br from-blue-600 to-purple-600 rounded-2xl p-12 text-white">
        <div className="max-w-3xl mx-auto text-center space-y-8">
          <h2 className="text-3xl font-bold">Why Choose VideoGuard?</h2>

          <div className="grid sm:grid-cols-2 gap-4">
            {benefits.map((benefit) => (
              <div key={benefit} className="flex items-center gap-3 bg-white/10 backdrop-blur-sm rounded-lg p-4">
                <CheckCircle className="size-5 flex-shrink-0" />
                <span className="font-medium">{benefit}</span>
              </div>
            ))}
          </div>

          <Button asChild size="lg" variant="secondary" className="text-lg px-8">
            <Link to="/upload">Get Started Now</Link>
          </Button>
        </div>
      </section>

      {/* How It Works */}
      <section className="space-y-6">
        <div className="text-center space-y-2">
          <h2 className="text-3xl font-bold">How It Works</h2>
          <p className="text-muted-foreground">Three simple steps to detect video tampering</p>
        </div>

        <div className="grid md:grid-cols-3 gap-6">
          {[
            { step: "01", title: "Upload Video", description: "Select and upload your video file for analysis" },
            { step: "02", title: "AI Analysis", description: "Our system extracts 27 temporal features and classifies boundaries" },
            { step: "03", title: "Get Report", description: "Inspect before/junction/after frames and probability charts" },
          ].map((item) => (
            <Card key={item.step} className="text-center">
              <CardHeader>
                <div className="text-4xl font-bold text-blue-600 mb-2">{item.step}</div>
                <CardTitle>{item.title}</CardTitle>
                <CardDescription>{item.description}</CardDescription>
              </CardHeader>
            </Card>
          ))}
        </div>
      </section>
    </div>
  );
}

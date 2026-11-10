import { Link } from "react-router";
import { Button } from "../components/ui/button";
import { Home, AlertCircle } from "lucide-react";

export function NotFound() {
  return (
    <div className="flex flex-col items-center justify-center min-h-[60vh] text-center space-y-6">
      <div className="w-20 h-20 bg-red-100 rounded-full flex items-center justify-center">
        <AlertCircle className="size-10 text-red-600" />
      </div>
      
      <div className="space-y-2">
        <h1 className="text-6xl font-bold">404</h1>
        <h2 className="text-2xl font-semibold">Page Not Found</h2>
        <p className="text-muted-foreground max-w-md">
          The page you're looking for doesn't exist or has been moved.
        </p>
      </div>

      <Button asChild size="lg">
        <Link to="/">
          <Home className="size-4 mr-2" />
          Back to Home
        </Link>
      </Button>
    </div>
  );
}

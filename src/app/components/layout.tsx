import { Outlet, Link, useLocation } from "react-router";
import { Video, Upload, History, Home, Database } from "lucide-react";
import { Button } from "./ui/button";

export function Layout() {
  const location = useLocation();

  const navItems = [
    { path: "/", label: "Home", icon: Home },
    { path: "/upload", label: "Upload", icon: Upload },
    { path: "/history", label: "History", icon: History },
    { path: "/database", label: "Database", icon: Database },
  ];

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100">
      <header className="border-b bg-white/80 backdrop-blur-sm sticky top-0 z-50">
        <div className="container mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <Link to="/" className="flex items-center gap-2">
              <div className="bg-gradient-to-br from-blue-600 to-purple-600 p-2 rounded-lg">
                <Video className="size-6 text-white" />
              </div>
              <div>
                <h1 className="font-bold text-xl">VideoGuard</h1>
                <p className="text-xs text-muted-foreground">Frame Detection System</p>
              </div>
            </Link>
            
            <nav className="flex gap-2">
              {navItems.map((item) => {
                const Icon = item.icon;
                const isActive = location.pathname === item.path;
                return (
                  <Button
                    key={item.path}
                    asChild
                    variant={isActive ? "default" : "ghost"}
                  >
                    <Link to={item.path}>
                      <Icon className="size-4 mr-2" />
                      {item.label}
                    </Link>
                  </Button>
                );
              })}
            </nav>
          </div>
        </div>
      </header>

      <main className="container mx-auto px-4 py-8">
        <Outlet />
      </main>

      <footer className="border-t bg-white/80 backdrop-blur-sm mt-auto py-6">
        <div className="container mx-auto px-4 text-center text-sm text-muted-foreground">
          <p>© 2026 VideoGuard. Advanced Video Tampering Detection System.</p>
        </div>
      </footer>
    </div>
  );
}

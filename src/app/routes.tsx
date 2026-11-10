import { createBrowserRouter } from "react-router";
import { Layout } from "./components/layout";
import { HomePage } from "./pages/home";
import { UploadPage } from "./pages/upload";
import { AnalysisPage } from "./pages/analysis";
import { HistoryPage } from "./pages/history";
import { DatabasePage } from "./pages/database";
import { NotFound } from "./pages/not-found";

export const router = createBrowserRouter([
  {
    path: "/",
    Component: Layout,
    children: [
      { index: true, Component: HomePage },
      { path: "upload", Component: UploadPage },
      { path: "analysis/:id", Component: AnalysisPage },
      { path: "history", Component: HistoryPage },
      { path: "database", Component: DatabasePage },
      { path: "*", Component: NotFound },
    ],
  },
]);

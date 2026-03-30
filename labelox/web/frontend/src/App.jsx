import { Routes, Route, Link, useLocation } from "react-router-dom";
import ProjectList from "./pages/ProjectList";
import ProjectDashboard from "./pages/ProjectDashboard";
import AnnotationView from "./pages/AnnotationView";
import ReviewPage from "./pages/ReviewPage";
import ExportPage from "./pages/ExportPage";

const NAV_ITEMS = [
  { path: "/", label: "Projects", icon: "📂" },
];

function Sidebar() {
  const location = useLocation();
  return (
    <aside className="w-56 bg-panel border-r border-border flex flex-col h-screen fixed left-0 top-0">
      <div className="p-4 border-b border-border">
        <h1 className="text-hi font-bold text-lg tracking-wide">LABELOX</h1>
        <p className="text-muted text-xs mt-1">AI Annotation Platform</p>
      </div>
      <nav className="flex-1 p-2 space-y-1">
        {NAV_ITEMS.map((item) => (
          <Link
            key={item.path}
            to={item.path}
            className={`flex items-center gap-2 px-3 py-2 rounded text-sm transition-colors ${
              location.pathname === item.path
                ? "bg-accent text-white"
                : "text-muted hover:bg-card hover:text-text"
            }`}
          >
            <span>{item.icon}</span>
            {item.label}
          </Link>
        ))}
      </nav>
    </aside>
  );
}

export default function App() {
  return (
    <div className="flex min-h-screen bg-bg">
      <Sidebar />
      <main className="ml-56 flex-1 p-6">
        <Routes>
          <Route path="/" element={<ProjectList />} />
          <Route path="/project/:projectId" element={<ProjectDashboard />} />
          <Route path="/project/:projectId/annotate" element={<AnnotationView />} />
          <Route path="/project/:projectId/review" element={<ReviewPage />} />
          <Route path="/project/:projectId/export" element={<ExportPage />} />
        </Routes>
      </main>
    </div>
  );
}

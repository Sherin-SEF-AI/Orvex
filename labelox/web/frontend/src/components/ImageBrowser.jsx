import { useState } from "react";

const STATUS_COLORS = {
  unlabeled: "#555566",
  in_progress: "#f5a623",
  annotated: "#f5a623",
  reviewed: "#4ecca3",
  rejected: "#e94560",
  skipped: "#888899",
};

export default function ImageBrowser({ images, currentImage, onSelect }) {
  const [search, setSearch] = useState("");
  const [statusFilter, setStatusFilter] = useState("all");

  const filtered = images.filter((img) => {
    if (search && !img.file_name.toLowerCase().includes(search.toLowerCase())) return false;
    if (statusFilter !== "all" && img.status !== statusFilter) return false;
    return true;
  });

  return (
    <div className="flex flex-col h-full">
      <div className="p-2 space-y-2 border-b border-border">
        <input
          type="text"
          className="w-full bg-bg border border-border rounded px-2 py-1 text-xs text-text"
          placeholder="Search..."
          value={search}
          onChange={(e) => setSearch(e.target.value)}
        />
        <select
          className="w-full bg-bg border border-border rounded px-2 py-1 text-xs text-text"
          value={statusFilter}
          onChange={(e) => setStatusFilter(e.target.value)}
        >
          <option value="all">All</option>
          <option value="unlabeled">Unlabeled</option>
          <option value="annotated">Annotated</option>
          <option value="reviewed">Reviewed</option>
          <option value="rejected">Rejected</option>
        </select>
        <p className="text-xs text-muted">{filtered.length} / {images.length} images</p>
      </div>

      <div className="flex-1 overflow-y-auto p-1 space-y-0.5">
        {filtered.map((img) => (
          <div
            key={img.id}
            onClick={() => onSelect(img)}
            className={`flex items-center gap-2 px-2 py-1.5 rounded cursor-pointer text-xs transition-colors ${
              currentImage?.id === img.id
                ? "bg-accent text-white"
                : "text-muted hover:bg-card hover:text-text"
            }`}
          >
            <span
              className="w-2 h-2 rounded-full flex-shrink-0"
              style={{ backgroundColor: STATUS_COLORS[img.status] || "#555" }}
            />
            <span className="truncate">{img.file_name}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

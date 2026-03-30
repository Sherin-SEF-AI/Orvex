const TOOLS = [
  { key: "bbox", label: "BBox", shortcut: "B" },
  { key: "mask", label: "Mask/SAM", shortcut: "M" },
  { key: "polygon", label: "Polygon", shortcut: "P" },
  { key: "polyline", label: "Polyline", shortcut: "L" },
  { key: "keypoint", label: "Keypoint", shortcut: "K" },
  { key: "cuboid", label: "Cuboid 3D", shortcut: "C" },
  { key: "classification", label: "Class", shortcut: "I" },
];

export default function Toolbar({ currentTool, onToolChange, onPrev, onNext, onSave }) {
  return (
    <div className="flex items-center gap-1 px-3 py-2 bg-panel border-b border-border">
      {/* Tool buttons */}
      {TOOLS.map((t) => (
        <button
          key={t.key}
          onClick={() => onToolChange(t.key)}
          className={`px-3 py-1.5 rounded text-xs font-medium transition-colors ${
            currentTool === t.key
              ? "bg-hi text-white"
              : "bg-card text-muted hover:bg-accent hover:text-text"
          }`}
          title={`${t.label} (${t.shortcut})`}
        >
          {t.label}
        </button>
      ))}

      <div className="w-px h-6 bg-border mx-2" />

      {/* Navigation */}
      <button onClick={onPrev} className="px-2 py-1.5 bg-card text-muted rounded text-xs hover:bg-accent" title="Previous (A)">
        Prev
      </button>
      <button onClick={onNext} className="px-2 py-1.5 bg-card text-muted rounded text-xs hover:bg-accent" title="Next (D)">
        Next
      </button>

      <div className="w-px h-6 bg-border mx-2" />

      <button onClick={onSave} className="px-3 py-1.5 bg-success text-black rounded text-xs font-medium hover:opacity-90" title="Save (S)">
        Save
      </button>
    </div>
  );
}

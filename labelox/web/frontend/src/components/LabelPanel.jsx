export default function LabelPanel({ classes, currentLabel, onLabelSelect, annotations }) {
  return (
    <div className="flex flex-col h-full">
      {/* Classes */}
      <div className="p-3 border-b border-border">
        <h4 className="text-xs font-semibold text-muted uppercase mb-2">Labels</h4>
        <div className="space-y-1">
          {classes.map((cls, i) => (
            <div
              key={cls.id || i}
              onClick={() => onLabelSelect(cls)}
              className={`flex items-center gap-2 px-2 py-1.5 rounded cursor-pointer text-sm transition-colors ${
                currentLabel?.id === cls.id
                  ? "bg-accent"
                  : "hover:bg-card"
              }`}
            >
              <span
                className="w-3 h-3 rounded-sm flex-shrink-0"
                style={{ backgroundColor: cls.color }}
              />
              <span className="flex-1 truncate" style={{ color: cls.color }}>
                {cls.hotkey && <span className="opacity-50 mr-1">[{cls.hotkey}]</span>}
                {cls.name}
              </span>
            </div>
          ))}
        </div>
      </div>

      {/* Annotations */}
      <div className="flex-1 overflow-y-auto p-3">
        <h4 className="text-xs font-semibold text-muted uppercase mb-2">
          Annotations ({annotations.length})
        </h4>
        <div className="space-y-1">
          {annotations.map((ann, i) => (
            <div
              key={ann.id || i}
              className="flex items-center gap-2 px-2 py-1.5 rounded bg-card text-xs"
            >
              <span className="font-medium">{ann.label_name}</span>
              <span className="text-muted">[{ann.annotation_type}]</span>
              {ann.confidence != null && (
                <span className="text-muted">({ann.confidence.toFixed(2)})</span>
              )}
              <span className="ml-auto">{ann.is_auto ? "AI" : "Manual"}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

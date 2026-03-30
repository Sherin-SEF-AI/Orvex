import { useState, useEffect } from "react";
import {
  getReviewFrames, saveFrameReview, exportCorrectedDataset, getReviewStats,
} from "../api/client";

const HI     = "#e94560";
const PANEL  = "#16213e";
const ACCENT = "#0f3460";
const TEXT   = "#e0e0e0";

const STATUS_COLOR = {
  pending:   "#888",
  accepted:  "#4caf50",
  corrected: "#ff9800",
  rejected:  "#f44336",
};

export default function AnnotationReview({ sessionId }) {
  const [frames, setFrames]   = useState([]);
  const [stats, setStats]     = useState(null);
  const [current, setCurrent] = useState(0);
  const [filter, setFilter]   = useState("all");
  const [status, setStatus]   = useState("Select a session to begin.");
  const [exporting, setExporting] = useState(false);

  useEffect(() => {
    if (sessionId) loadSession();
  }, [sessionId]);

  async function loadSession() {
    try {
      const [fr, st] = await Promise.all([
        getReviewFrames(sessionId),
        getReviewStats(sessionId),
      ]);
      setFrames(fr.data || []);
      setStats(st.data);
      setCurrent(0);
      setStatus(`Loaded ${(fr.data || []).length} reviewed frames.`);
    } catch (e) {
      setStatus("Error: " + e.message);
    }
  }

  async function doReview(newStatus) {
    if (!frames[current]) return;
    const updated = { ...frames[current], status: newStatus, reviewed_at: new Date().toISOString() };
    try {
      await saveFrameReview(sessionId, current, { review: updated });
      const next = [...frames];
      next[current] = updated;
      setFrames(next);
      // refresh stats
      const st = await getReviewStats(sessionId);
      setStats(st.data);
      // auto-advance
      const nextPending = next.findIndex((f, i) => i > current && f.status === "pending");
      if (nextPending >= 0) setCurrent(nextPending);
    } catch (e) {
      setStatus("Save failed: " + e.message);
    }
  }

  async function handleExport() {
    const outDir = prompt("Output directory for corrected dataset:");
    if (!outDir) return;
    setExporting(true);
    setStatus("Exporting…");
    try {
      const res = await exportCorrectedDataset(sessionId, outDir);
      setStatus(`Export complete: ${res.data.augmented_count} frames → ${res.data.output_dir}`);
    } catch (e) {
      setStatus("Export failed: " + e.message);
    } finally {
      setExporting(false);
    }
  }

  const visibleFrames = frames.filter(f =>
    filter === "all" || f.status === filter
  );

  const frame = frames[current];

  return (
    <div style={{ display: "flex", height: "100%", color: TEXT, fontFamily: "sans-serif" }}>

      {/* Left: stats + actions */}
      <div style={{ width: 200, background: PANEL, padding: 16, borderRight: "1px solid #333", flexShrink: 0 }}>
        <h3 style={{ color: HI, marginTop: 0 }}>✏️ Review</h3>
        {stats && (
          <div style={{ fontSize: 12, lineHeight: 2 }}>
            <div>Total: {stats.total_frames}</div>
            <div style={{ color: STATUS_COLOR.pending }}>Pending: {stats.pending}</div>
            <div style={{ color: STATUS_COLOR.accepted }}>Accepted: {stats.accepted}</div>
            <div style={{ color: STATUS_COLOR.corrected }}>Corrected: {stats.corrected}</div>
            <div style={{ color: STATUS_COLOR.rejected }}>Rejected: {stats.rejected}</div>
            <div>Coverage: {stats.coverage_percent}%</div>
          </div>
        )}
        <div style={{ marginTop: 16 }}>
          <button onClick={handleExport} disabled={exporting}
            style={btnStyle(ACCENT, exporting)}>
            Export dataset
          </button>
        </div>
        <div style={{ marginTop: 8 }}>
          <button onClick={loadSession} style={btnStyle("#444")}>↻ Refresh</button>
        </div>
      </div>

      {/* Center: image + controls */}
      <div style={{ flex: 1, padding: 16, display: "flex", flexDirection: "column" }}>
        <div style={{ flex: 1, background: "#0a0a1a", borderRadius: 6,
          display: "flex", alignItems: "center", justifyContent: "center",
          border: "1px solid #333", minHeight: 300 }}>
          {frame?.frame_path
            ? <p style={{ color: "#666", textAlign: "center" }}>
                {frame.frame_path.split("/").pop()}<br />
                <span style={{ fontSize: 11, color: "#555" }}>
                  (Image preview requires desktop app or local file server)
                </span>
              </p>
            : <p style={{ color: "#555" }}>Select a frame from the list →</p>
          }
        </div>

        {frame && (
          <div style={{ marginTop: 8, fontSize: 12, color: "#aaa" }}>
            Frame {current + 1}/{frames.length} — status: <span style={{ color: STATUS_COLOR[frame.status] || "#aaa" }}>
              {frame.status}
            </span>
            {" · "}{frame.original_detections?.length || 0} detection(s)
          </div>
        )}

        <div style={{ display: "flex", gap: 10, marginTop: 10 }}>
          <button onClick={() => doReview("accepted")}
            style={btnStyle("#4caf50")}>✓ Accept</button>
          <button onClick={() => doReview("corrected")}
            style={btnStyle("#ff9800")}>✏ Corrected</button>
          <button onClick={() => doReview("rejected")}
            style={btnStyle("#f44336")}>✗ Reject</button>
          <div style={{ flex: 1 }} />
          <button onClick={() => setCurrent(Math.max(0, current - 1))}
            disabled={current === 0} style={btnStyle(ACCENT, current === 0)}>◀ Prev</button>
          <button onClick={() => setCurrent(Math.min(frames.length - 1, current + 1))}
            disabled={current >= frames.length - 1}
            style={btnStyle(ACCENT, current >= frames.length - 1)}>Next ▶</button>
        </div>

        <div style={{ marginTop: 8, color: "#888", fontSize: 11 }}>{status}</div>
      </div>

      {/* Right: frame list */}
      <div style={{ width: 220, background: PANEL, padding: 8,
        borderLeft: "1px solid #333", display: "flex", flexDirection: "column" }}>
        <select value={filter} onChange={e => setFilter(e.target.value)}
          style={{ ...inputStyle, marginBottom: 8 }}>
          {["all", "pending", "accepted", "corrected", "rejected"].map(v => (
            <option key={v} value={v}>{v.charAt(0).toUpperCase() + v.slice(1)}</option>
          ))}
        </select>
        <div style={{ flex: 1, overflowY: "auto" }}>
          {visibleFrames.map((f, i) => {
            const globalIdx = frames.indexOf(f);
            return (
              <div key={i} onClick={() => setCurrent(globalIdx)}
                style={{
                  padding: "5px 8px", cursor: "pointer", fontSize: 11,
                  borderBottom: "1px solid #222",
                  background: globalIdx === current ? ACCENT : "transparent",
                  color: STATUS_COLOR[f.status] || TEXT,
                }}>
                {f.frame_path?.split("/").pop() || `Frame ${globalIdx}`}
              </div>
            );
          })}
          {visibleFrames.length === 0 && (
            <div style={{ color: "#555", padding: 8, textAlign: "center" }}>
              No frames for this filter
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

const inputStyle = {
  width: "100%", padding: "4px 8px", background: "#1a1a2e",
  color: "#e0e0e0", border: "1px solid #0f3460", borderRadius: 4,
  boxSizing: "border-box",
};

function btnStyle(bg, disabled = false) {
  return {
    background: disabled ? "#444" : bg, color: "#fff", border: "none",
    padding: "6px 14px", borderRadius: 4, cursor: disabled ? "not-allowed" : "pointer",
    fontSize: 12,
  };
}

import { useState, useEffect, useRef } from "react";
import {
  listModels, registerModel, activateModel, deleteModel,
  predictSingle, inferenceHealth,
} from "../api/client";

const ACCENT = "#0f3460";
const HI = "#e94560";
const PANEL = "#16213e";
const TEXT = "#e0e0e0";
const GREEN = "#4caf50";

export default function Inference() {
  const [models, setModels] = useState([]);
  const [health, setHealth] = useState(null);
  const [selectedId, setSelectedId] = useState(null);
  const [imagePath, setImagePath] = useState("");
  const [conf, setConf] = useState(0.25);
  const [iou, setIou] = useState(0.45);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [status, setStatus] = useState("Ready");

  useEffect(() => { refresh(); }, []);

  async function refresh() {
    try {
      const [m, h] = await Promise.all([listModels(), inferenceHealth()]);
      setModels(m.data || []);
      setHealth(h.data);
    } catch (e) {
      setStatus("Error loading registry: " + e.message);
    }
  }

  async function handleRegister() {
    const path = prompt("Weights file path (*.pt):");
    if (!path) return;
    const name = prompt("Model name:");
    if (!name) return;
    const variant = prompt("Model variant:", "yolov8n") || "yolov8n";
    try {
      await registerModel({ weights_path: path, name, model_variant: variant });
      await refresh();
      setStatus("Registered: " + name);
    } catch (e) { setStatus("Register failed: " + e.message); }
  }

  async function handleActivate() {
    if (!selectedId) { setStatus("Select a model first."); return; }
    try {
      await activateModel(selectedId);
      await refresh();
      setStatus("Activated: " + selectedId);
    } catch (e) { setStatus("Activate failed: " + e.message); }
  }

  async function handleDelete() {
    if (!selectedId) { setStatus("Select a model first."); return; }
    if (!confirm("Remove from registry? Weights file is NOT deleted.")) return;
    try {
      await deleteModel(selectedId);
      setSelectedId(null);
      await refresh();
    } catch (e) { setStatus("Delete failed: " + e.message); }
  }

  async function handlePredict() {
    if (!imagePath) { setStatus("Enter an image path."); return; }
    setLoading(true);
    setStatus("Running inference…");
    try {
      const res = await predictSingle({ image_path: imagePath, conf_threshold: conf, iou_threshold: iou });
      setResult(res.data);
      setStatus(`Done — ${res.data.detections.length} detection(s) | ${res.data.inference_time_ms.toFixed(1)} ms`);
    } catch (e) { setStatus("Inference failed: " + e.message); }
    finally { setLoading(false); }
  }

  return (
    <div style={{ padding: 24, color: TEXT, fontFamily: "sans-serif" }}>
      <h2 style={{ color: HI, marginTop: 0 }}>🎯 Inference</h2>

      {/* Health banner */}
      {health && (
        <div style={{ background: PANEL, borderRadius: 6, padding: "8px 14px", marginBottom: 16,
          border: `1px solid ${health.status === "ok" ? GREEN : "#888"}` }}>
          {health.status === "ok"
            ? <span style={{ color: GREEN }}>✓ Active: <b>{health.active_model}</b> ({health.variant})</span>
            : <span style={{ color: "#aaa" }}>No active model registered.</span>
          }
        </div>
      )}

      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 20 }}>

        {/* Left: registry */}
        <div>
          <h3 style={{ color: HI }}>Model Registry</h3>
          <table style={{ width: "100%", borderCollapse: "collapse", background: PANEL }}>
            <thead>
              <tr style={{ background: ACCENT }}>
                {["Name", "Variant", "mAP50", "Active"].map(h => (
                  <th key={h} style={{ padding: "6px 10px", textAlign: "left" }}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {models.map(m => (
                <tr key={m.model_id}
                  onClick={() => setSelectedId(m.model_id)}
                  style={{
                    cursor: "pointer",
                    background: selectedId === m.model_id ? ACCENT : "transparent",
                    borderBottom: "1px solid #333",
                  }}>
                  <td style={{ padding: "6px 10px", color: m.is_active ? GREEN : TEXT }}>{m.name}</td>
                  <td style={{ padding: "6px 10px" }}>{m.model_variant}</td>
                  <td style={{ padding: "6px 10px" }}>{(m.metrics?.map50 || 0).toFixed(3)}</td>
                  <td style={{ padding: "6px 10px", color: GREEN }}>{m.is_active ? "✓" : ""}</td>
                </tr>
              ))}
              {models.length === 0 && (
                <tr><td colSpan={4} style={{ padding: 12, textAlign: "center", color: "#888" }}>
                  No models registered yet
                </td></tr>
              )}
            </tbody>
          </table>
          <div style={{ display: "flex", gap: 8, marginTop: 10 }}>
            <button onClick={handleRegister} style={btnStyle(ACCENT)}>Register…</button>
            <button onClick={handleActivate} style={btnStyle(ACCENT)}>Set Active</button>
            <button onClick={handleDelete} style={btnStyle(HI)}>Delete</button>
            <button onClick={refresh} style={btnStyle("#444")}>↻</button>
          </div>
        </div>

        {/* Right: inference */}
        <div>
          <h3 style={{ color: HI }}>Run Inference</h3>
          <div style={{ marginBottom: 10 }}>
            <label style={{ display: "block", marginBottom: 4, fontSize: 12 }}>Image path</label>
            <input
              value={imagePath}
              onChange={e => setImagePath(e.target.value)}
              placeholder="/path/to/image.jpg"
              style={inputStyle}
            />
          </div>
          <div style={{ display: "flex", gap: 16, marginBottom: 10 }}>
            <label style={{ fontSize: 12 }}>Conf: {conf.toFixed(2)}
              <input type="range" min={0.01} max={1} step={0.01} value={conf}
                onChange={e => setConf(parseFloat(e.target.value))}
                style={{ display: "block", width: 120 }} />
            </label>
            <label style={{ fontSize: 12 }}>IoU: {iou.toFixed(2)}
              <input type="range" min={0.01} max={1} step={0.01} value={iou}
                onChange={e => setIou(parseFloat(e.target.value))}
                style={{ display: "block", width: 120 }} />
            </label>
          </div>
          <button onClick={handlePredict} disabled={loading} style={btnStyle(HI, loading)}>
            {loading ? "Running…" : "Run Inference"}
          </button>

          {result && (
            <div style={{ marginTop: 16 }}>
              <h4 style={{ marginBottom: 6, color: HI }}>Detections ({result.detections.length})</h4>
              <table style={{ width: "100%", borderCollapse: "collapse", background: PANEL, fontSize: 12 }}>
                <thead>
                  <tr style={{ background: ACCENT }}>
                    {["Class", "Conf", "x1,y1", "x2,y2"].map(h => (
                      <th key={h} style={{ padding: "4px 8px", textAlign: "left" }}>{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {result.detections.map((d, i) => (
                    <tr key={i} style={{ borderBottom: "1px solid #333" }}>
                      <td style={{ padding: "4px 8px" }}>{d.class_name}</td>
                      <td style={{ padding: "4px 8px" }}>{d.confidence.toFixed(3)}</td>
                      <td style={{ padding: "4px 8px" }}>{d.bbox_xyxy[0].toFixed(0)},{d.bbox_xyxy[1].toFixed(0)}</td>
                      <td style={{ padding: "4px 8px" }}>{d.bbox_xyxy[2].toFixed(0)},{d.bbox_xyxy[3].toFixed(0)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      </div>

      <div style={{ marginTop: 16, color: "#aaa", fontSize: 12 }}>{status}</div>
    </div>
  );
}

const inputStyle = {
  width: "100%", padding: "6px 10px", background: "#1a1a2e",
  color: "#e0e0e0", border: "1px solid #0f3460", borderRadius: 4, boxSizing: "border-box",
};

function btnStyle(bg, disabled = false) {
  return {
    background: disabled ? "#444" : bg, color: "#e0e0e0", border: "none",
    padding: "6px 14px", borderRadius: 4, cursor: disabled ? "not-allowed" : "pointer",
  };
}

import { useState, useEffect, useRef } from "react";
import {
  getLearningLog, checkLearningTrigger, triggerLearningCycle,
} from "../api/client";
import {
  LineChart, Line, XAxis, YAxis, Tooltip, CartesianGrid, ResponsiveContainer, Legend,
} from "recharts";

const HI    = "#e94560";
const PANEL = "#16213e";
const ACCENT= "#0f3460";
const TEXT  = "#e0e0e0";
const GREEN = "#4caf50";

export default function ContinuousLearning({ sessionId }) {
  const [log, setLog]           = useState([]);
  const [threshold, setThreshold] = useState(50);
  const [autoPromote, setAutoPromote] = useState(true);
  const [variant, setVariant]   = useState("yolov8n");
  const [epochs, setEpochs]     = useState(50);
  const [batch, setBatch]       = useState(16);
  const [lr, setLr]             = useState(0.01);
  const [multiplier, setMult]   = useState(3);
  const [running, setRunning]   = useState(false);
  const [status, setStatus]     = useState("Ready");
  const [wsLog, setWsLog]       = useState([]);
  const wsRef = useRef(null);

  useEffect(() => { loadLog(); }, []);

  async function loadLog() {
    try {
      const res = await getLearningLog();
      setLog(res.data || []);
    } catch (e) {
      setStatus("Error loading log: " + e.message);
    }
  }

  async function handleCheck() {
    if (!sessionId) { setStatus("Select a session first."); return; }
    try {
      const res = await checkLearningTrigger(sessionId, threshold);
      if (res.data.ready) {
        setStatus(`✓ Threshold met (${threshold} corrections). Ready to retrain!`);
      } else {
        setStatus(`Not yet — check API response for usable count. Threshold: ${threshold}`);
      }
    } catch (e) { setStatus("Check failed: " + e.message); }
  }

  async function handleRun() {
    if (!sessionId) { setStatus("Select a session first."); return; }
    setRunning(true);
    setWsLog([]);
    setStatus("Starting learning cycle…");

    try {
      const res = await triggerLearningCycle(sessionId, {
        trigger_type: "manual",
        threshold,
        auto_promote: autoPromote,
        training_config: {
          dataset_dir: "",
          model_variant: variant,
          pretrained_weights: variant + ".pt",
          epochs, batch_size: batch, learning_rate: lr,
          project_name: "rover_cl",
        },
        augmentation_config: { multiplier },
      });
      const taskId = res.data?.task_id;
      if (taskId) {
        // Connect WebSocket
        const ws = new WebSocket(`ws://localhost:8000/ws/tasks/${taskId}`);
        wsRef.current = ws;
        ws.onmessage = (ev) => {
          try {
            const msg = JSON.parse(ev.data);
            setWsLog(prev => [...prev, msg.message || msg.status]);
            if (msg.status === "done" || msg.status === "failed") {
              setRunning(false);
              setStatus(msg.status === "done" ? "Cycle complete!" : "Cycle failed.");
              ws.close();
              loadLog();
            }
          } catch {}
        };
        ws.onerror = () => { setRunning(false); setStatus("WebSocket error"); };
      }
    } catch (e) {
      setStatus("Trigger failed: " + e.message);
      setRunning(false);
    }
  }

  // Build chart data from learning log
  const chartData = log
    .filter(e => e.resulting_run_id)
    .map((e, i) => ({
      cycle: i + 1,
      date: e.triggered_at ? e.triggered_at.slice(0, 10) : `#${i + 1}`,
      corrections: e.corrections_count,
    }));

  return (
    <div style={{ padding: 24, color: TEXT, fontFamily: "sans-serif" }}>
      <h2 style={{ color: HI, marginTop: 0 }}>🔄 Continuous Learning</h2>

      <div style={{ display: "grid", gridTemplateColumns: "2fr 1fr", gap: 20 }}>

        {/* Left: config + log */}
        <div>
          {/* Config row */}
          <div style={{ display: "flex", gap: 16, marginBottom: 16, flexWrap: "wrap" }}>
            <label style={{ fontSize: 12 }}>Threshold
              <input type="number" value={threshold} min={1} max={9999}
                onChange={e => setThreshold(parseInt(e.target.value))}
                style={{ ...inputStyle, width: 70, display: "block" }} />
            </label>
            <label style={{ fontSize: 12 }}>Variant
              <select value={variant} onChange={e => setVariant(e.target.value)}
                style={{ ...inputStyle, display: "block" }}>
                {["yolov8n","yolov8s","yolov8m","yolov8l"].map(v => (
                  <option key={v}>{v}</option>
                ))}
              </select>
            </label>
            <label style={{ fontSize: 12 }}>Epochs
              <input type="number" value={epochs} min={1}
                onChange={e => setEpochs(parseInt(e.target.value))}
                style={{ ...inputStyle, width: 65, display: "block" }} />
            </label>
            <label style={{ fontSize: 12 }}>Batch
              <input type="number" value={batch} min={1}
                onChange={e => setBatch(parseInt(e.target.value))}
                style={{ ...inputStyle, width: 55, display: "block" }} />
            </label>
            <label style={{ fontSize: 12 }}>LR
              <input type="number" value={lr} min={0.00001} step={0.001}
                onChange={e => setLr(parseFloat(e.target.value))}
                style={{ ...inputStyle, width: 75, display: "block" }} />
            </label>
            <label style={{ fontSize: 12 }}>Aug mult.
              <input type="number" value={multiplier} min={1}
                onChange={e => setMult(parseInt(e.target.value))}
                style={{ ...inputStyle, width: 55, display: "block" }} />
            </label>
            <label style={{ fontSize: 12, display: "flex", alignItems: "center", gap: 6, marginTop: 14 }}>
              <input type="checkbox" checked={autoPromote}
                onChange={e => setAutoPromote(e.target.checked)} />
              Auto-promote
            </label>
          </div>

          {/* Learning history table */}
          <h3 style={{ color: HI }}>Learning History</h3>
          <table style={{ width: "100%", borderCollapse: "collapse", background: PANEL, fontSize: 12 }}>
            <thead>
              <tr style={{ background: ACCENT }}>
                {["Date", "Session", "Trigger", "Corrections", "Run ID"].map(h => (
                  <th key={h} style={{ padding: "6px 10px", textAlign: "left" }}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {log.map((e, i) => (
                <tr key={i} style={{ borderBottom: "1px solid #333" }}>
                  <td style={{ padding: "5px 10px" }}>{e.triggered_at?.slice(0, 16) || "—"}</td>
                  <td style={{ padding: "5px 10px" }}>{e.session_id?.slice(0, 8)}</td>
                  <td style={{ padding: "5px 10px" }}>{e.trigger_type}</td>
                  <td style={{ padding: "5px 10px" }}>{e.corrections_count}</td>
                  <td style={{ padding: "5px 10px", color: "#aaa" }}>{(e.resulting_run_id || "").slice(0, 12)}</td>
                </tr>
              ))}
              {log.length === 0 && (
                <tr><td colSpan={5} style={{ padding: 12, textAlign: "center", color: "#666" }}>
                  No learning cycles yet
                </td></tr>
              )}
            </tbody>
          </table>

          {/* Chart */}
          {chartData.length > 1 && (
            <div style={{ marginTop: 20 }}>
              <h4 style={{ color: HI }}>Corrections per Cycle</h4>
              <ResponsiveContainer width="100%" height={180}>
                <LineChart data={chartData}>
                  <CartesianGrid stroke="#333" />
                  <XAxis dataKey="date" tick={{ fill: "#aaa", fontSize: 10 }} />
                  <YAxis tick={{ fill: "#aaa", fontSize: 10 }} />
                  <Tooltip contentStyle={{ background: PANEL, border: "1px solid #444" }} />
                  <Line type="monotone" dataKey="corrections" stroke={HI} dot={{ r: 4 }} name="Corrections" />
                </LineChart>
              </ResponsiveContainer>
            </div>
          )}
        </div>

        {/* Right: run controls + live log */}
        <div>
          <div style={{ background: PANEL, borderRadius: 6, padding: 16, border: "1px solid #333" }}>
            <h4 style={{ color: HI, marginTop: 0 }}>Run</h4>
            <button onClick={handleRun} disabled={running}
              style={{ ...btnStyle(HI, running), width: "100%", marginBottom: 8 }}>
              {running ? "Running…" : "Run Learning Cycle"}
            </button>
            <button onClick={handleCheck} style={{ ...btnStyle(ACCENT), width: "100%", marginBottom: 8 }}>
              Check trigger now
            </button>
            <button onClick={loadLog} style={{ ...btnStyle("#444"), width: "100%" }}>
              ↻ Refresh log
            </button>

            <div style={{ marginTop: 12, fontSize: 11, color: "#aaa" }}>{status}</div>

            {wsLog.length > 0 && (
              <div style={{ marginTop: 12, background: "#0a0a1a", borderRadius: 4, padding: 8,
                maxHeight: 200, overflowY: "auto", fontSize: 11, fontFamily: "monospace" }}>
                {wsLog.map((line, i) => (
                  <div key={i} style={{ marginBottom: 2, color: TEXT }}>{line}</div>
                ))}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

const inputStyle = {
  padding: "4px 8px", background: "#1a1a2e", color: "#e0e0e0",
  border: "1px solid #0f3460", borderRadius: 4,
};

function btnStyle(bg, disabled = false) {
  return {
    background: disabled ? "#444" : bg, color: "#fff", border: "none",
    padding: "7px 14px", borderRadius: 4, cursor: disabled ? "not-allowed" : "pointer",
    fontSize: 13,
  };
}

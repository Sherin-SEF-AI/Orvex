import { useEffect, useState } from "react";

export default function ProgressOverlay({ taskId, onComplete }) {
  const [status, setStatus] = useState(null);

  useEffect(() => {
    if (!taskId) return;
    const ws = new WebSocket(`ws://${window.location.host}/ws/tasks/${taskId}`);
    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      setStatus(data);
      if (data.status === "success" || data.status === "failure") {
        ws.close();
        if (onComplete) onComplete(data);
      }
    };
    ws.onerror = () => ws.close();
    return () => ws.close();
  }, [taskId]);

  if (!status) return null;

  const pct = status.total > 0 ? Math.round((status.progress / status.total) * 100) : 0;

  return (
    <div className="fixed inset-0 bg-black/60 flex items-center justify-center z-50">
      <div className="bg-card border border-border rounded-lg p-6 w-96">
        <h3 className="font-semibold text-lg mb-3">Processing...</h3>
        <div className="w-full bg-bg rounded-full h-2.5 mb-3">
          <div
            className="bg-hi h-2.5 rounded-full transition-all"
            style={{ width: `${pct}%` }}
          />
        </div>
        <p className="text-sm text-muted">{status.message || `${pct}%`}</p>
        {status.error && <p className="text-sm text-hi mt-2">{status.error}</p>}
      </div>
    </div>
  );
}

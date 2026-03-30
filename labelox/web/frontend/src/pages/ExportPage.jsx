import { useState } from "react";
import { useParams } from "react-router-dom";
import { exportApi } from "../api/client";

const FORMATS = ["yolo", "coco", "cvat_xml", "pascal_voc", "csv"];

export default function ExportPage() {
  const { projectId } = useParams();
  const [format, setFormat] = useState("yolo");
  const [outputDir, setOutputDir] = useState("");
  const [includeImages, setIncludeImages] = useState(true);
  const [onlyReviewed, setOnlyReviewed] = useState(false);
  const [splitTrainVal, setSplitTrainVal] = useState(true);
  const [trainRatio, setTrainRatio] = useState(0.8);
  const [taskId, setTaskId] = useState(null);
  const [status, setStatus] = useState(null);
  const [polling, setPolling] = useState(false);

  async function handleExport() {
    if (!outputDir.trim()) {
      alert("Please enter an output directory path");
      return;
    }
    const result = await exportApi.start({
      project_id: projectId,
      format,
      output_dir: outputDir,
      include_images: includeImages,
      only_reviewed: onlyReviewed,
      split_train_val: splitTrainVal,
      train_ratio: trainRatio,
    });
    setTaskId(result.task_id);
    pollStatus(result.task_id);
  }

  async function pollStatus(tid) {
    setPolling(true);
    const interval = setInterval(async () => {
      try {
        const s = await exportApi.status(tid);
        setStatus(s);
        if (s.status === "success" || s.status === "failure") {
          clearInterval(interval);
          setPolling(false);
        }
      } catch {
        clearInterval(interval);
        setPolling(false);
      }
    }, 1000);
  }

  return (
    <div className="max-w-2xl">
      <h2 className="text-2xl font-bold mb-6">Export Annotations</h2>

      <div className="bg-card border border-border rounded-lg p-6 space-y-4">
        {/* Format */}
        <div>
          <label className="block text-sm font-medium mb-1">Format</label>
          <select
            value={format}
            onChange={(e) => setFormat(e.target.value)}
            className="w-full bg-bg border border-border rounded px-3 py-2 text-text"
          >
            {FORMATS.map((f) => (
              <option key={f} value={f}>{f.toUpperCase()}</option>
            ))}
          </select>
        </div>

        {/* Output dir */}
        <div>
          <label className="block text-sm font-medium mb-1">Output Directory</label>
          <input
            type="text"
            value={outputDir}
            onChange={(e) => setOutputDir(e.target.value)}
            className="w-full bg-bg border border-border rounded px-3 py-2 text-text"
            placeholder="/path/to/output"
          />
        </div>

        {/* Options */}
        <div className="space-y-2">
          <label className="flex items-center gap-2 text-sm">
            <input type="checkbox" checked={includeImages} onChange={(e) => setIncludeImages(e.target.checked)} />
            Copy images to output
          </label>
          <label className="flex items-center gap-2 text-sm">
            <input type="checkbox" checked={onlyReviewed} onChange={(e) => setOnlyReviewed(e.target.checked)} />
            Only reviewed images
          </label>
          <label className="flex items-center gap-2 text-sm">
            <input type="checkbox" checked={splitTrainVal} onChange={(e) => setSplitTrainVal(e.target.checked)} />
            Split train/val
          </label>
        </div>

        {splitTrainVal && (
          <div>
            <label className="block text-sm font-medium mb-1">Train Ratio: {trainRatio}</label>
            <input
              type="range"
              min="0.5"
              max="0.95"
              step="0.05"
              value={trainRatio}
              onChange={(e) => setTrainRatio(parseFloat(e.target.value))}
              className="w-full"
            />
          </div>
        )}

        <button
          onClick={handleExport}
          disabled={polling}
          className="bg-hi text-white px-6 py-2 rounded font-semibold hover:bg-red-600 disabled:opacity-50"
        >
          {polling ? "Exporting..." : "Export"}
        </button>

        {/* Status */}
        {status && (
          <div className={`mt-4 p-3 rounded text-sm ${
            status.status === "success" ? "bg-green-900/30 text-success" :
            status.status === "failure" ? "bg-red-900/30 text-hi" :
            "bg-blue-900/30 text-blue-300"
          }`}>
            <p>Status: {status.status}</p>
            {status.message && <p>{status.message}</p>}
            {status.error && <p>Error: {status.error}</p>}
            {status.progress > 0 && <p>Progress: {status.progress}/{status.total}</p>}
          </div>
        )}
      </div>
    </div>
  );
}

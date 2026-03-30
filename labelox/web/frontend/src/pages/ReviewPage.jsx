import { useEffect, useState } from "react";
import { useParams } from "react-router-dom";
import { reviewApi } from "../api/client";

export default function ReviewPage() {
  const { projectId } = useParams();
  const [queue, setQueue] = useState([]);
  const [selected, setSelected] = useState(null);
  const [comment, setComment] = useState("");

  useEffect(() => {
    reviewApi.queue(projectId).then(setQueue).catch(console.error);
  }, [projectId]);

  async function handleReview(decision) {
    if (!selected) return;
    await reviewApi.submit({
      image_id: selected.image_id,
      decision,
      comment: comment || null,
    });
    setComment("");
    const updated = await reviewApi.queue(projectId);
    setQueue(updated);
    setSelected(null);
  }

  return (
    <div>
      <h2 className="text-2xl font-bold mb-6">Review Queue</h2>
      <p className="text-muted mb-4">{queue.length} images awaiting review</p>

      <div className="flex gap-6">
        {/* Queue list */}
        <div className="w-80 space-y-2">
          {queue.map((item) => (
            <div
              key={item.image_id}
              onClick={() => setSelected(item)}
              className={`p-3 rounded cursor-pointer border transition-colors ${
                selected?.image_id === item.image_id
                  ? "bg-accent border-accent"
                  : "bg-card border-border hover:border-accent"
              }`}
            >
              <p className="text-sm font-medium truncate">{item.file_name}</p>
              <div className="flex gap-3 text-xs text-muted mt-1">
                <span>{item.annotation_count} annotations</span>
                {item.avg_confidence && (
                  <span>conf: {item.avg_confidence.toFixed(2)}</span>
                )}
              </div>
            </div>
          ))}
          {queue.length === 0 && (
            <p className="text-muted text-center py-8">No images to review</p>
          )}
        </div>

        {/* Review panel */}
        {selected && (
          <div className="flex-1 bg-card border border-border rounded-lg p-6">
            <h3 className="font-semibold text-lg mb-4">{selected.file_name}</h3>
            <p className="text-muted text-sm mb-4">
              {selected.annotation_count} annotations | Status: {selected.status}
            </p>

            <textarea
              className="w-full bg-bg border border-border rounded px-3 py-2 text-text mb-4"
              placeholder="Review comment (optional)..."
              rows={3}
              value={comment}
              onChange={(e) => setComment(e.target.value)}
            />

            <div className="flex gap-3">
              <button
                onClick={() => handleReview("approved")}
                className="bg-success text-black px-6 py-2 rounded font-semibold hover:opacity-90"
              >
                Approve
              </button>
              <button
                onClick={() => handleReview("needs_work")}
                className="bg-warning text-black px-6 py-2 rounded font-semibold hover:opacity-90"
              >
                Needs Work
              </button>
              <button
                onClick={() => handleReview("rejected")}
                className="bg-hi text-white px-6 py-2 rounded font-semibold hover:opacity-90"
              >
                Reject
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

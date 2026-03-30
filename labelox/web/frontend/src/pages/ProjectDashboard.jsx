import { useEffect, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { projectsApi, imagesApi, statsApi } from "../api/client";

export default function ProjectDashboard() {
  const { projectId } = useParams();
  const [project, setProject] = useState(null);
  const [stats, setStats] = useState(null);
  const [images, setImages] = useState([]);

  useEffect(() => {
    projectsApi.get(projectId).then(setProject).catch(console.error);
    statsApi.project(projectId).then(setStats).catch(() => {});
    imagesApi.list(projectId, { limit: 20 }).then(setImages).catch(() => {});
  }, [projectId]);

  async function handleUpload(e) {
    const files = Array.from(e.target.files);
    if (!files.length) return;
    await imagesApi.upload(projectId, files);
    const updated = await imagesApi.list(projectId);
    setImages(updated);
    const s = await statsApi.project(projectId);
    setStats(s);
  }

  if (!project) return <p className="text-muted">Loading...</p>;

  return (
    <div>
      <div className="flex items-center justify-between mb-6">
        <div>
          <h2 className="text-2xl font-bold">{project.name}</h2>
          <p className="text-muted text-sm">{project.description}</p>
        </div>
        <div className="flex gap-2">
          <Link
            to={`/project/${projectId}/annotate`}
            className="bg-hi text-white px-4 py-2 rounded font-semibold hover:bg-red-600"
          >
            Annotate
          </Link>
          <Link
            to={`/project/${projectId}/review`}
            className="bg-accent text-white px-4 py-2 rounded font-semibold hover:bg-blue-800"
          >
            Review
          </Link>
          <Link
            to={`/project/${projectId}/export`}
            className="bg-card border border-border text-text px-4 py-2 rounded font-semibold hover:bg-panel"
          >
            Export
          </Link>
        </div>
      </div>

      {/* Stats cards */}
      {stats && (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
          {[
            { label: "Images", value: stats.total_images || 0 },
            { label: "Annotated", value: stats.annotated_images || 0 },
            { label: "Reviewed", value: stats.reviewed_images || 0 },
            { label: "Annotations", value: stats.total_annotations || 0 },
          ].map((s) => (
            <div key={s.label} className="bg-card border border-border rounded-lg p-4 text-center">
              <p className="text-2xl font-bold text-text">{s.value}</p>
              <p className="text-muted text-sm">{s.label}</p>
            </div>
          ))}
        </div>
      )}

      {/* Upload */}
      <div className="bg-card border border-border rounded-lg p-6 mb-6">
        <h3 className="font-semibold mb-3">Import Images</h3>
        <label className="block cursor-pointer border-2 border-dashed border-border rounded-lg p-8 text-center hover:border-accent transition-colors">
          <input type="file" multiple accept="image/*" onChange={handleUpload} className="hidden" />
          <p className="text-muted">Click or drag images here to upload</p>
          <p className="text-xs text-muted mt-1">JPG, PNG, BMP, TIFF, WebP</p>
        </label>
      </div>

      {/* Image grid */}
      <h3 className="font-semibold mb-3">Images ({images.length})</h3>
      <div className="grid grid-cols-4 md:grid-cols-6 lg:grid-cols-8 gap-2">
        {images.map((img) => (
          <div
            key={img.id}
            className="bg-card border border-border rounded overflow-hidden aspect-square flex items-center justify-center"
            title={img.file_name}
          >
            <p className="text-xs text-muted truncate px-1">{img.file_name}</p>
          </div>
        ))}
      </div>

      {/* Label classes */}
      {project.label_classes && (
        <div className="mt-6">
          <h3 className="font-semibold mb-3">Label Classes</h3>
          <div className="flex flex-wrap gap-2">
            {project.label_classes.map((cls, i) => (
              <span
                key={i}
                className="px-3 py-1 rounded-full text-sm font-medium"
                style={{ backgroundColor: cls.color + "22", color: cls.color, border: `1px solid ${cls.color}` }}
              >
                {cls.hotkey && <span className="opacity-50 mr-1">[{cls.hotkey}]</span>}
                {cls.name}
              </span>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { projectsApi } from "../api/client";

export default function ProjectList() {
  const [projects, setProjects] = useState([]);
  const [showCreate, setShowCreate] = useState(false);
  const [name, setName] = useState("");
  const [desc, setDesc] = useState("");

  useEffect(() => {
    projectsApi.list().then(setProjects).catch(console.error);
  }, []);

  async function handleCreate(e) {
    e.preventDefault();
    if (!name.trim()) return;
    await projectsApi.create({
      name: name.trim(),
      description: desc.trim(),
      label_classes: [
        { name: "car", color: "#e94560", hotkey: "1", id: "cls_car" },
        { name: "person", color: "#4ecca3", hotkey: "2", id: "cls_person" },
        { name: "motorcycle", color: "#4a9eff", hotkey: "3", id: "cls_moto" },
      ],
    });
    setName("");
    setDesc("");
    setShowCreate(false);
    const updated = await projectsApi.list();
    setProjects(updated);
  }

  async function handleDelete(id) {
    if (!confirm("Delete this project?")) return;
    await projectsApi.delete(id);
    setProjects((p) => p.filter((x) => x.id !== id));
  }

  return (
    <div>
      <div className="flex items-center justify-between mb-6">
        <h2 className="text-2xl font-bold">Projects</h2>
        <button
          onClick={() => setShowCreate(true)}
          className="bg-hi text-white px-4 py-2 rounded font-semibold hover:bg-red-600 transition-colors"
        >
          + New Project
        </button>
      </div>

      {showCreate && (
        <form onSubmit={handleCreate} className="bg-card border border-border rounded-lg p-4 mb-6 space-y-3">
          <input
            className="w-full bg-bg border border-border rounded px-3 py-2 text-text"
            placeholder="Project name"
            value={name}
            onChange={(e) => setName(e.target.value)}
            autoFocus
          />
          <input
            className="w-full bg-bg border border-border rounded px-3 py-2 text-text"
            placeholder="Description (optional)"
            value={desc}
            onChange={(e) => setDesc(e.target.value)}
          />
          <div className="flex gap-2">
            <button type="submit" className="bg-hi text-white px-4 py-2 rounded font-semibold">
              Create
            </button>
            <button type="button" onClick={() => setShowCreate(false)} className="text-muted px-4 py-2">
              Cancel
            </button>
          </div>
        </form>
      )}

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {projects.map((p) => (
          <div key={p.id} className="bg-card border border-border rounded-lg p-4 hover:border-accent transition-colors">
            <Link to={`/project/${p.id}`} className="block">
              <h3 className="text-lg font-semibold text-text">{p.name}</h3>
              <p className="text-muted text-sm mt-1">{p.description || "No description"}</p>
              <div className="flex gap-4 mt-3 text-xs text-muted">
                <span>{p.image_count || 0} images</span>
                <span>{p.annotated_count || 0} annotated</span>
              </div>
            </Link>
            <button
              onClick={() => handleDelete(p.id)}
              className="text-hi text-xs mt-2 hover:underline"
            >
              Delete
            </button>
          </div>
        ))}
      </div>

      {projects.length === 0 && !showCreate && (
        <p className="text-muted text-center mt-12">No projects yet. Create one to get started.</p>
      )}
    </div>
  );
}

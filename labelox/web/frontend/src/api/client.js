import axios from "axios";

const api = axios.create({
  baseURL: "/api/v1",
  headers: { "Content-Type": "application/json" },
});

// Attach token if present
api.interceptors.request.use((config) => {
  const token = localStorage.getItem("labelox_token");
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

// Unwrap envelope
api.interceptors.response.use(
  (response) => {
    if (response.data && "data" in response.data) {
      return response.data.data;
    }
    return response.data;
  },
  (error) => {
    const msg =
      error.response?.data?.error || error.message || "Unknown error";
    return Promise.reject(new Error(msg));
  },
);

export default api;

// ─── Project API ──────────────────────────────────────────────────────────

export const projectsApi = {
  list: () => api.get("/projects"),
  get: (id) => api.get(`/projects/${id}`),
  create: (data) => api.post("/projects", data),
  update: (id, data) => api.put(`/projects/${id}`, data),
  delete: (id) => api.delete(`/projects/${id}`),
};

// ─── Images API ───────────────────────────────────────────────────────────

export const imagesApi = {
  list: (projectId, params = {}) => api.get(`/images/${projectId}`, { params }),
  get: (projectId, imageId) => api.get(`/images/${projectId}/${imageId}`),
  upload: (projectId, files) => {
    const form = new FormData();
    files.forEach((f) => form.append("files", f));
    return api.post(`/images/${projectId}/upload`, form, {
      headers: { "Content-Type": "multipart/form-data" },
    });
  },
  nextUnlabeled: (projectId) => api.get(`/images/${projectId}/next-unlabeled`),
};

// ─── Annotations API ──────────────────────────────────────────────────────

export const annotationsApi = {
  get: (imageId) => api.get(`/annotations/${imageId}`),
  save: (imageId, annotations) =>
    api.post(`/annotations/${imageId}`, { annotations }),
  delete: (imageId, annotationId) =>
    api.delete(`/annotations/${imageId}/${annotationId}`),
};

// ─── Auto-Annotate API ───────────────────────────────────────────────────

export const autoAnnotateApi = {
  start: (data) => api.post("/auto-annotate", data),
  status: (taskId) => api.get(`/auto-annotate/status/${taskId}`),
};

// ─── SAM API ──────────────────────────────────────────────────────────────

export const samApi = {
  point: (data) => api.post("/sam/point", data),
  box: (data) => api.post("/sam/box", data),
  points: (data) => api.post("/sam/points", data),
};

// ─── Export API ───────────────────────────────────────────────────────────

export const exportApi = {
  start: (data) => api.post("/export", data),
  status: (taskId) => api.get(`/export/status/${taskId}`),
};

// ─── Review API ───────────────────────────────────────────────────────────

export const reviewApi = {
  queue: (projectId) => api.get(`/review/queue/${projectId}`),
  submit: (data) => api.post("/review/submit", data),
  history: (projectId) => api.get(`/review/history/${projectId}`),
};

// ─── Stats API ────────────────────────────────────────────────────────────

export const statsApi = {
  project: (projectId) => api.get(`/stats/${projectId}`),
  classBalance: (projectId) => api.get(`/stats/${projectId}/class-balance`),
  quality: (projectId) => api.get(`/stats/${projectId}/quality`),
  daily: (projectId) => api.get(`/stats/${projectId}/daily`),
};

// ─── Users API ────────────────────────────────────────────────────────────

export const usersApi = {
  login: (name, role = "annotator") => api.post("/users/login", { name, role }),
  me: () => api.get("/users/me"),
};

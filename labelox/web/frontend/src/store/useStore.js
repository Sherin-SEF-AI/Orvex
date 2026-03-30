import { create } from "zustand";

const useStore = create((set, get) => ({
  // ─── Projects ──────────────────────────────
  projects: [],
  currentProject: null,
  setProjects: (projects) => set({ projects }),
  setCurrentProject: (project) => set({ currentProject: project }),

  // ─── Images ────────────────────────────────
  images: [],
  currentImage: null,
  setImages: (images) => set({ images }),
  setCurrentImage: (image) => set({ currentImage: image }),

  // ─── Annotations ──────────────────────────
  annotations: [],
  selectedAnnotation: null,
  setAnnotations: (annotations) => set({ annotations }),
  setSelectedAnnotation: (ann) => set({ selectedAnnotation: ann }),
  addAnnotation: (ann) =>
    set((s) => ({ annotations: [...s.annotations, ann] })),
  removeAnnotation: (id) =>
    set((s) => ({
      annotations: s.annotations.filter((a) => a.id !== id),
    })),

  // ─── Tool ──────────────────────────────────
  currentTool: "bbox",
  currentLabel: null,
  setCurrentTool: (tool) => set({ currentTool: tool }),
  setCurrentLabel: (label) => set({ currentLabel: label }),

  // ─── UI State ──────────────────────────────
  zoom: 1,
  setZoom: (zoom) => set({ zoom }),
}));

export default useStore;

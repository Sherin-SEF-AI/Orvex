import { useEffect, useState, useCallback } from "react";
import { useParams } from "react-router-dom";
import { projectsApi, imagesApi, annotationsApi } from "../api/client";
import useStore from "../store/useStore";
import AnnotationCanvas from "../components/AnnotationCanvas";
import ImageBrowser from "../components/ImageBrowser";
import LabelPanel from "../components/LabelPanel";
import Toolbar from "../components/Toolbar";

export default function AnnotationView() {
  const { projectId } = useParams();
  const [project, setProject] = useState(null);
  const {
    images, setImages,
    currentImage, setCurrentImage,
    annotations, setAnnotations,
    currentTool, setCurrentTool,
    currentLabel, setCurrentLabel,
  } = useStore();

  useEffect(() => {
    projectsApi.get(projectId).then((p) => {
      setProject(p);
      if (p.label_classes?.length) {
        setCurrentLabel(p.label_classes[0]);
      }
    });
    imagesApi.list(projectId, { limit: 9999 }).then(setImages);
  }, [projectId]);

  const loadAnnotations = useCallback(
    async (imageId) => {
      const anns = await annotationsApi.get(imageId);
      setAnnotations(anns || []);
    },
    [setAnnotations],
  );

  function handleImageSelect(image) {
    setCurrentImage(image);
    loadAnnotations(image.id);
  }

  async function handleSave() {
    if (!currentImage) return;
    await annotationsApi.save(currentImage.id, annotations);
  }

  function handlePrev() {
    if (!currentImage || !images.length) return;
    const idx = images.findIndex((i) => i.id === currentImage.id);
    if (idx > 0) handleImageSelect(images[idx - 1]);
  }

  function handleNext() {
    if (!currentImage || !images.length) return;
    const idx = images.findIndex((i) => i.id === currentImage.id);
    if (idx < images.length - 1) handleImageSelect(images[idx + 1]);
  }

  // Keyboard shortcuts
  useEffect(() => {
    function onKey(e) {
      if (e.target.tagName === "INPUT" || e.target.tagName === "TEXTAREA") return;
      const key = e.key.toLowerCase();
      const toolMap = { b: "bbox", m: "mask", p: "polygon", l: "polyline", k: "keypoint", c: "cuboid", i: "classification" };
      if (toolMap[key]) {
        setCurrentTool(toolMap[key]);
        return;
      }
      if (key === "a") handlePrev();
      if (key === "d") handleNext();
      if (key === "s" && !e.ctrlKey) handleSave();
      // Number keys for class selection
      if (/^[1-9]$/.test(key) && project?.label_classes) {
        const idx = parseInt(key) - 1;
        if (idx < project.label_classes.length) {
          setCurrentLabel(project.label_classes[idx]);
        }
      }
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [currentImage, images, project, annotations]);

  if (!project) return <p className="text-muted">Loading...</p>;

  return (
    <div className="flex flex-col h-[calc(100vh-3rem)] -m-6">
      {/* Toolbar */}
      <Toolbar
        currentTool={currentTool}
        onToolChange={setCurrentTool}
        onPrev={handlePrev}
        onNext={handleNext}
        onSave={handleSave}
      />

      <div className="flex flex-1 min-h-0">
        {/* Left: Image browser */}
        <div className="w-52 border-r border-border overflow-y-auto bg-panel">
          <ImageBrowser
            images={images}
            currentImage={currentImage}
            onSelect={handleImageSelect}
          />
        </div>

        {/* Center: Canvas */}
        <div className="flex-1 bg-canvas">
          <AnnotationCanvas
            image={currentImage}
            annotations={annotations}
            currentTool={currentTool}
            currentLabel={currentLabel}
          />
        </div>

        {/* Right: Label panel */}
        <div className="w-60 border-l border-border overflow-y-auto bg-panel">
          <LabelPanel
            classes={project.label_classes || []}
            currentLabel={currentLabel}
            onLabelSelect={setCurrentLabel}
            annotations={annotations}
          />
        </div>
      </div>
    </div>
  );
}

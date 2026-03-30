import { useRef, useState, useEffect, useCallback } from "react";
import { Stage, Layer, Image as KonvaImage, Rect, Line, Circle, Group } from "react-konva";
import useStore from "../store/useStore";

const ANNOTATION_COLORS = [
  "#e94560", "#4ecca3", "#4a9eff", "#f5a623", "#9b59b6",
  "#1abc9c", "#e67e22", "#3498db", "#e74c3c", "#2ecc71",
];

function useImage(src) {
  const [image, setImage] = useState(null);
  useEffect(() => {
    if (!src) { setImage(null); return; }
    const img = new window.Image();
    img.onload = () => setImage(img);
    img.src = src;
  }, [src]);
  return image;
}

export default function AnnotationCanvas({ image, annotations, currentTool, currentLabel }) {
  const containerRef = useRef(null);
  const [containerSize, setContainerSize] = useState({ width: 800, height: 600 });
  const [drawing, setDrawing] = useState(false);
  const [drawStart, setDrawStart] = useState(null);
  const [drawCurrent, setDrawCurrent] = useState(null);
  const [polygonPoints, setPolygonPoints] = useState([]);
  const { addAnnotation, zoom, setZoom } = useStore();

  // Image source — serve from backend or local file path
  const imageSrc = image?.file_path ? `/api/v1/images/file?path=${encodeURIComponent(image.file_path)}` : null;
  const loadedImage = useImage(imageSrc);

  // Fit container
  useEffect(() => {
    function resize() {
      if (containerRef.current) {
        setContainerSize({
          width: containerRef.current.offsetWidth,
          height: containerRef.current.offsetHeight,
        });
      }
    }
    resize();
    window.addEventListener("resize", resize);
    return () => window.removeEventListener("resize", resize);
  }, []);

  // Compute scale to fit image
  const imgW = image?.width || 800;
  const imgH = image?.height || 600;
  const scale = Math.min(
    containerSize.width / imgW,
    containerSize.height / imgH,
  ) * zoom;
  const offsetX = (containerSize.width - imgW * scale) / 2;
  const offsetY = (containerSize.height - imgH * scale) / 2;

  function toImageCoords(stageX, stageY) {
    return {
      x: (stageX - offsetX) / scale / imgW,
      y: (stageY - offsetY) / scale / imgH,
    };
  }

  function fromImageCoords(normX, normY) {
    return {
      x: normX * imgW * scale + offsetX,
      y: normY * imgH * scale + offsetY,
    };
  }

  function handleMouseDown(e) {
    const stage = e.target.getStage();
    const pos = stage.getPointerPosition();
    const coord = toImageCoords(pos.x, pos.y);

    if (currentTool === "bbox") {
      setDrawing(true);
      setDrawStart(coord);
      setDrawCurrent(coord);
    } else if (currentTool === "polygon" || currentTool === "polyline") {
      setPolygonPoints((prev) => [...prev, coord]);
    } else if (currentTool === "classification") {
      if (!currentLabel) return;
      const ann = {
        id: crypto.randomUUID(),
        image_id: image.id,
        label_id: currentLabel.id,
        label_name: currentLabel.name,
        annotation_type: "classification",
        classification: currentLabel.name,
        is_auto: false,
      };
      addAnnotation(ann);
    }
  }

  function handleMouseMove(e) {
    if (!drawing) return;
    const stage = e.target.getStage();
    const pos = stage.getPointerPosition();
    setDrawCurrent(toImageCoords(pos.x, pos.y));
  }

  function handleMouseUp() {
    if (!drawing || !drawStart || !drawCurrent) return;
    setDrawing(false);

    if (currentTool === "bbox" && currentLabel) {
      const x = Math.min(drawStart.x, drawCurrent.x);
      const y = Math.min(drawStart.y, drawCurrent.y);
      const w = Math.abs(drawCurrent.x - drawStart.x);
      const h = Math.abs(drawCurrent.y - drawStart.y);
      if (w > 0.005 && h > 0.005) {
        const ann = {
          id: crypto.randomUUID(),
          image_id: image.id,
          label_id: currentLabel.id,
          label_name: currentLabel.name,
          annotation_type: "bbox",
          bbox: { x, y, width: w, height: h },
          is_auto: false,
        };
        addAnnotation(ann);
      }
    }
    setDrawStart(null);
    setDrawCurrent(null);
  }

  function handleDblClick() {
    if ((currentTool === "polygon" || currentTool === "polyline") && polygonPoints.length >= 2) {
      if (!currentLabel) return;
      const ann = {
        id: crypto.randomUUID(),
        image_id: image.id,
        label_id: currentLabel.id,
        label_name: currentLabel.name,
        annotation_type: currentTool,
        polyline: {
          points: polygonPoints.map((p) => ({ x: p.x, y: p.y })),
          is_closed: currentTool === "polygon",
        },
        is_auto: false,
      };
      addAnnotation(ann);
      setPolygonPoints([]);
    }
  }

  function handleWheel(e) {
    e.evt.preventDefault();
    const delta = e.evt.deltaY > 0 ? 0.9 : 1.1;
    setZoom(Math.max(0.1, Math.min(10, zoom * delta)));
  }

  return (
    <div ref={containerRef} className="w-full h-full relative">
      <Stage
        width={containerSize.width}
        height={containerSize.height}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onDblClick={handleDblClick}
        onWheel={handleWheel}
      >
        <Layer>
          {/* Image */}
          {loadedImage && (
            <KonvaImage
              image={loadedImage}
              x={offsetX}
              y={offsetY}
              width={imgW * scale}
              height={imgH * scale}
            />
          )}

          {/* Existing annotations */}
          {annotations.map((ann, i) => {
            const color = ANNOTATION_COLORS[i % ANNOTATION_COLORS.length];
            if (ann.annotation_type === "bbox" && ann.bbox) {
              const b = ann.bbox;
              const pos = fromImageCoords(b.x, b.y);
              return (
                <Rect
                  key={ann.id}
                  x={pos.x}
                  y={pos.y}
                  width={b.width * imgW * scale}
                  height={b.height * imgH * scale}
                  stroke={color}
                  strokeWidth={2}
                  fill={color + "22"}
                />
              );
            }
            if ((ann.annotation_type === "polygon" || ann.annotation_type === "polyline") && ann.polyline?.points) {
              const pts = ann.polyline.points.flatMap((p) => {
                const s = fromImageCoords(p.x, p.y);
                return [s.x, s.y];
              });
              return (
                <Line
                  key={ann.id}
                  points={pts}
                  stroke={color}
                  strokeWidth={2}
                  closed={ann.polyline.is_closed}
                  fill={ann.polyline.is_closed ? color + "22" : undefined}
                />
              );
            }
            return null;
          })}

          {/* Drawing preview: bbox */}
          {drawing && drawStart && drawCurrent && currentTool === "bbox" && (
            <Rect
              x={fromImageCoords(Math.min(drawStart.x, drawCurrent.x), Math.min(drawStart.y, drawCurrent.y)).x}
              y={fromImageCoords(Math.min(drawStart.x, drawCurrent.x), Math.min(drawStart.y, drawCurrent.y)).y}
              width={Math.abs(drawCurrent.x - drawStart.x) * imgW * scale}
              height={Math.abs(drawCurrent.y - drawStart.y) * imgH * scale}
              stroke="#e94560"
              strokeWidth={2}
              dash={[6, 3]}
            />
          )}

          {/* Drawing preview: polygon/polyline */}
          {polygonPoints.length > 0 && (
            <>
              <Line
                points={polygonPoints.flatMap((p) => {
                  const s = fromImageCoords(p.x, p.y);
                  return [s.x, s.y];
                })}
                stroke="#4ecca3"
                strokeWidth={2}
                dash={[6, 3]}
              />
              {polygonPoints.map((p, i) => {
                const s = fromImageCoords(p.x, p.y);
                return <Circle key={i} x={s.x} y={s.y} radius={4} fill="#4ecca3" />;
              })}
            </>
          )}
        </Layer>
      </Stage>

      {/* No image placeholder */}
      {!image && (
        <div className="absolute inset-0 flex items-center justify-center">
          <p className="text-muted">Select an image to start annotating</p>
        </div>
      )}
    </div>
  );
}

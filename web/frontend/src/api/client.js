/**
 * src/api/client.js — Axios API client for RoverDataKit backend.
 *
 * All responses follow: { data: ..., error: string|null }
 * Long operations return: { data: { task_id: string }, error: null }
 *
 * WebSocket helper: connectTaskWS(taskId, onMessage, onClose)
 */
import axios from 'axios'

const http = axios.create({ baseURL: '/' })

function unwrap(res) {
  if (res.data.error) throw new Error(res.data.error)
  return res.data.data
}

// Sessions
export const getSessions    = ()         => http.get('/sessions').then(unwrap)
export const createSession  = (body)     => http.post('/sessions', body).then(unwrap)
export const getSession     = (id)       => http.get(`/sessions/${id}`).then(unwrap)
export const deleteSession  = (id)       => http.delete(`/sessions/${id}`).then(unwrap)
export const addFile        = (id, path) => http.post(`/sessions/${id}/files`, { file_path: path }).then(unwrap)

// Audit
export const runAudit        = (id)  => http.post(`/audit/${id}/run`).then(unwrap)
export const getAuditResults = (id)  => http.get(`/audit/${id}/results`).then(unwrap)

// Extraction
export const runExtraction       = (id, cfg) => http.post(`/extraction/${id}/run`, cfg).then(unwrap)
export const getExtractionStatus = (id)      => http.get(`/extraction/${id}/status`).then(unwrap)

// Calibration
export const createCalibration = (body)       => http.post('/calibration', body).then(unwrap)
export const getCalibration    = (id)         => http.get(`/calibration/${id}`).then(unwrap)
export const getCalibSteps     = (id)         => http.get(`/calibration/${id}/steps`).then(unwrap)
export const runCalibStep      = (id, body)   => http.post(`/calibration/${id}/run-step`, body).then(unwrap)

// Dataset
export const buildDataset = (body) => http.post('/dataset/build', body).then(unwrap)
export const getProfiles  = ()     => http.get('/dataset/profiles').then(unwrap)
export const saveProfile  = (body) => http.post('/dataset/profiles', body).then(unwrap)
export const exportCvat   = (body) => http.post('/dataset/export-cvat', body).then(unwrap)

// Calibration health
export const getCalibrationHealth = (id) => http.get(`/calibration/${id}/health`).then(unwrap)

// Auto-label
export const runAutolabel        = (id, cfg) => http.post(`/autolabel/${id}/run`, cfg).then(unwrap)
export const getAutolabelResults = (id)      => http.get(`/autolabel/${id}/results`).then(unwrap)
export const getAutolabelHealth  = ()        => http.get('/autolabel/health').then(unwrap)

// Depth estimation
export const runDepth        = (id, cfg) => http.post(`/depth/${id}/run`, cfg).then(unwrap)
export const getDepthResults = (id)      => http.get(`/depth/${id}/results`).then(unwrap)
export const getDepthHealth  = ()        => http.get('/depth/health').then(unwrap)

// SLAM
export const runSlam              = (id, cfg) => http.post(`/slam/${id}/run`, cfg).then(unwrap)
export const getSlamResults       = (id)      => http.get(`/slam/${id}/results`).then(unwrap)
export const getSlamInstallation  = ()        => http.get('/slam/installation').then(unwrap)
export const generateSlamConfig   = (id)      => http.post(`/slam/${id}/generate-config`).then(unwrap)

// Reconstruction
export const runReconstruction        = (id, cfg) => http.post(`/reconstruction/${id}/run`, cfg).then(unwrap)
export const getReconstructionResults = (id)      => http.get(`/reconstruction/${id}/results`).then(unwrap)
export const getColmapInstallation    = ()        => http.get('/reconstruction/installation').then(unwrap)

// Active learning
export const runActiveLearning        = (id, cfg) => http.post(`/active-learning/${id}/run`, cfg).then(unwrap)
export const getActiveLearningResults = (id)      => http.get(`/active-learning/${id}/results`).then(unwrap)

// Analytics
export const runAnalytics        = (id, cfg) => http.post(`/analytics/${id}/run`, cfg).then(unwrap)
export const getAnalyticsResults = (id)      => http.get(`/analytics/${id}/results`).then(unwrap)
export const generateReport      = (id)      => http.post(`/analytics/${id}/report`).then(unwrap)

// Augmentation
export const runAugmentation        = (id, cfg) => http.post(`/augmentation/${id}/run`, cfg).then(unwrap)
export const getAugmentationResults = (id)      => http.get(`/augmentation/${id}/results`).then(unwrap)
export const getAugmentationHealth  = ()        => http.get('/augmentation/health').then(unwrap)

// Training
export const runTraining    = (cfg)  => http.post('/training/run', cfg).then(unwrap)
export const cancelTraining = (body) => http.post('/training/cancel', body).then(unwrap)
export const exportModel    = (body) => http.post('/training/export-model', body).then(unwrap)
export const getTrainingHealth = ()  => http.get('/training/health').then(unwrap)

// ── Phase 3: Inference ──────────────────────────────────────────────────────
export const listModels       = ()        => http.get('/inference/models').then(unwrap)
export const registerModel    = (body)    => http.post('/inference/models/register', body).then(unwrap)
export const activateModel    = (id)      => http.post(`/inference/models/${id}/activate`).then(unwrap)
export const deleteModel      = (id)      => http.delete(`/inference/models/${id}`).then(unwrap)
export const predictSingle    = (req)     => http.post('/inference/predict', req).then(unwrap)
export const predictBatch     = (reqs)    => http.post('/inference/batch', { requests: reqs }).then(unwrap)
export const inferenceHealth  = ()        => http.get('/inference/health').then(unwrap)

// ── Phase 3: Annotation Review ──────────────────────────────────────────────
export const getReviewFrames      = (id)           => http.get(`/review/${id}/frames`).then(unwrap)
export const getReviewFrame       = (id, idx)      => http.get(`/review/${id}/frames/${idx}`).then(unwrap)
export const saveFrameReview      = (id, idx, body)=> http.post(`/review/${id}/frames/${idx}`, body).then(unwrap)
export const saveBulkReviews      = (id, body)     => http.post(`/review/${id}/bulk`, body).then(unwrap)
export const exportCorrectedDataset = (id, outDir) =>
  http.post(`/review/${id}/export`, null, { params: { output_dir: outDir } }).then(unwrap)
export const getReviewStats       = (id)           => http.get(`/review/${id}/stats`).then(unwrap)

// ── Phase 3: Continuous Learning ────────────────────────────────────────────
export const getLearningLog         = ()           => http.get('/learning/log').then(unwrap)
export const checkLearningTrigger   = (id, thresh) =>
  http.post(`/learning/${id}/check`, null, { params: { threshold: thresh } }).then(unwrap)
export const triggerLearningCycle   = (id, body)   => http.post(`/learning/${id}/trigger`, body).then(unwrap)
export const getLearningStatus      = (id)         => http.get(`/learning/${id}/status`).then(unwrap)

// ── Phase 3: Segmentation ───────────────────────────────────────────────────
export const runSegmentation        = (id, cfg) => http.post(`/segmentation/${id}/run`, cfg).then(unwrap)
export const getSegmentationResults = (id)      => http.get(`/segmentation/${id}/results`).then(unwrap)
export const getSegmentationHealth  = ()        => http.get('/segmentation/health').then(unwrap)

// ── Phase 3: Occupancy Grid ─────────────────────────────────────────────────
export const runOccupancy        = (id, cfg) => http.post(`/occupancy/${id}/run`, cfg).then(unwrap)
export const getOccupancyResults = (id)      => http.get(`/occupancy/${id}/results`).then(unwrap)

// ── Phase 3: Lane Detection ─────────────────────────────────────────────────
export const runLaneDetection = (id, cfg) => http.post(`/lanes/${id}/run`, cfg).then(unwrap)
export const getLaneResults   = (id)      => http.get(`/lanes/${id}/results`).then(unwrap)
export const getLaneHealth    = ()        => http.get('/lanes/health').then(unwrap)

// ── Phase 3: Tracking ───────────────────────────────────────────────────────
export const runTracking        = (id, cfg) => http.post(`/tracking/${id}/run`, cfg).then(unwrap)
export const getTrackingResults = (id)      => http.get(`/tracking/${id}/results`).then(unwrap)

// ── Phase 3: Versioning ─────────────────────────────────────────────────────
export const getVersioningHealth = ()           => http.get('/versioning/health').then(unwrap)
export const listVersions        = (dir)        => http.get('/versioning/versions', { params: { dataset_dir: dir } }).then(unwrap)
export const commitVersion       = (body)       => http.post('/versioning/versions', body).then(unwrap)
export const diffVersions        = (body)       => http.post('/versioning/diff', body).then(unwrap)
export const restoreVersion      = (dir, tag)   =>
  http.post('/versioning/restore', null, { params: { dataset_dir: dir, version_tag: tag } }).then(unwrap)

// ── Phase 3: Experiments ────────────────────────────────────────────────────
export const getExperimentsHealth = ()          => http.get('/experiments/health').then(unwrap)
export const listRuns             = (exp)       => http.get('/experiments/runs', { params: exp ? { experiment_name: exp } : {} }).then(unwrap)
export const getRun               = (id)        => http.get(`/experiments/runs/${id}`).then(unwrap)
export const compareRuns          = (runs)      => http.get('/experiments/compare', { params: { runs } }).then(unwrap)
export const launchMlflowUI       = (port)      => http.post('/experiments/ui/launch', null, { params: port ? { port } : {} }).then(unwrap)

// ── Phase 3: Edge Export ────────────────────────────────────────────────────
export const checkExportDependencies = ()       => http.get('/edge-export/dependencies').then(unwrap)
export const exportONNX              = (body)   => http.post('/edge-export/onnx', body).then(unwrap)
export const exportTRT               = (body)   => http.post('/edge-export/tensorrt', body).then(unwrap)
export const benchmarkModel          = (body)   => http.post('/edge-export/benchmark', body).then(unwrap)
export const buildJetsonPackage      = (body)   => http.post('/edge-export/package', body).then(unwrap)

// ── 360° Camera: Insta360 X4 ────────────────────────────────────────────────
export const scanINSV        = (body)       => http.post('/insta360/scan', body).then(unwrap)
export const validateINSV    = (body)       => http.post('/insta360/validate', body).then(unwrap)
export const processINSV     = (body)       => http.post('/insta360/process', body).then(unwrap)
export const getINSVTask     = (id)         => http.get(`/insta360/tasks/${id}`).then(unwrap)
export const getINSVGPS      = (sid)        => http.get(`/insta360/${sid}/gps`).then(unwrap)
export const getINSVIMU      = (sid, page)  => http.get(`/insta360/${sid}/imu`, { params: { page } }).then(unwrap)
export const getINSVFrames   = (sid, v, p)  => http.get(`/insta360/${sid}/frames/${v}`, { params: { page: p } }).then(unwrap)
export const getINSVManifest = (sid)        => http.get(`/insta360/${sid}/manifest`).then(unwrap)
export const getINSVHealth   = ()           => http.get('/insta360/health').then(unwrap)

// WebSocket — connect to task progress stream
export function connectTaskWS(taskId, onMessage, onClose) {
  const proto = window.location.protocol === 'https:' ? 'wss' : 'ws'
  const ws = new WebSocket(`${proto}://${window.location.host}/ws/tasks/${taskId}`)
  ws.onmessage = (e) => onMessage(JSON.parse(e.data))
  ws.onclose   = onClose || (() => {})
  ws.onerror   = (e) => console.error('WS error', e)
  return ws
}

import { useState } from 'react'
import { BrowserRouter, Routes, Route, NavLink, Navigate } from 'react-router-dom'
import Sessions       from './pages/Sessions'
import Audit          from './pages/Audit'
import Extraction     from './pages/Extraction'
import Calibration    from './pages/Calibration'
import Frames         from './pages/Frames'
import Dataset        from './pages/Dataset'
// Phase 2
import AutoLabel      from './pages/AutoLabel'
import ActiveLearning from './pages/ActiveLearning'
import Analytics      from './pages/Analytics'
import Augmentation   from './pages/Augmentation'
import Training       from './pages/Training'
import DepthEstimation from './pages/DepthEstimation'
import Reconstruction from './pages/Reconstruction'
import SlamValidation from './pages/SlamValidation'
// Phase 3
import Inference from './pages/Inference'
import AnnotationReview from './pages/AnnotationReview'
import ContinuousLearning from './pages/ContinuousLearning'

const NAV = [
  { path: '/sessions',    label: '📁 Sessions'        },
  { path: '/audit',       label: '🔍 Audit'           },
  { path: '/extract',     label: '⚙️ Extract'         },
  { path: '/calibrate',   label: '📐 Calibrate'       },
  { path: '/frames',      label: '🖼️ Frames'          },
  { path: '/dataset',     label: '💾 Dataset'         },
  { path: '/autolabel',   label: '🤖 Auto-Label'      },
  { path: '/active-learning', label: '🎯 Active Learning' },
  { path: '/analytics',   label: '📊 Analytics'       },
  { path: '/augment',     label: '🔀 Augment'         },
  { path: '/train',       label: '🏋️ Train'           },
  { path: '/depth',       label: '📏 Depth'           },
  { path: '/reconstruct', label: '🗺️ 3D Reconstruct'  },
  { path: '/slam',        label: '🛰️ SLAM Validate'   },
  // Phase 3
  { path: '/inference',  label: '🎯 Inference'        },
  { path: '/review',     label: '✏️ Review'            },
  { path: '/retrain',    label: '🔄 Auto-Retrain'      },
]

export default function App() {
  const [selectedSessionId, setSelectedSessionId] = useState(null)

  return (
    <BrowserRouter future={{ v7_startTransition: true, v7_relativeSplatPath: true }}>
      <div className="flex h-screen overflow-hidden bg-bg text-gray-200">
        {/* Sidebar nav */}
        <nav className="w-48 flex-shrink-0 bg-panel border-r border-accent flex flex-col overflow-y-auto">
          <div className="px-4 py-4 border-b border-accent">
            <h1 className="text-sm font-bold text-white">RoverDataKit</h1>
            <p className="text-xs text-muted mt-0.5">v3.0 · Full Pipeline</p>
          </div>
          <ul className="flex flex-col py-2">
            {NAV.map(({ path, label }) => (
              <li key={path}>
                <NavLink to={path}
                  className={({ isActive }) =>
                    `block px-3 py-2 text-xs transition-colors
                     ${isActive ? 'bg-highlight text-white' : 'text-muted hover:text-white hover:bg-accent/50'}`
                  }
                >
                  {label}
                </NavLink>
              </li>
            ))}
          </ul>
          {selectedSessionId && (
            <div className="mt-auto px-4 py-3 border-t border-accent">
              <p className="text-xs text-muted">Active session:</p>
              <p className="text-xs font-mono text-gray-300 truncate">{selectedSessionId.slice(0, 8)}…</p>
            </div>
          )}
        </nav>

        {/* Main content */}
        <main className="flex-1 overflow-y-auto p-6">
          <Routes>
            <Route path="/" element={<Navigate to="/sessions" replace />} />
            <Route path="/sessions"
              element={<Sessions onSelect={setSelectedSessionId} selectedId={selectedSessionId} />} />
            <Route path="/audit"
              element={<Audit sessionId={selectedSessionId} />} />
            <Route path="/extract"
              element={<Extraction sessionId={selectedSessionId} />} />
            <Route path="/calibrate"
              element={<Calibration />} />
            <Route path="/frames"
              element={<Frames sessionId={selectedSessionId} />} />
            <Route path="/dataset"
              element={<Dataset />} />
            {/* Phase 2 routes */}
            <Route path="/autolabel"
              element={<AutoLabel sessionId={selectedSessionId} />} />
            <Route path="/active-learning"
              element={<ActiveLearning sessionId={selectedSessionId} />} />
            <Route path="/analytics"
              element={<Analytics sessionId={selectedSessionId} />} />
            <Route path="/augment"
              element={<Augmentation sessionId={selectedSessionId} />} />
            <Route path="/train"
              element={<Training sessionId={selectedSessionId} />} />
            <Route path="/depth"
              element={<DepthEstimation sessionId={selectedSessionId} />} />
            <Route path="/reconstruct"
              element={<Reconstruction sessionId={selectedSessionId} />} />
            <Route path="/slam"
              element={<SlamValidation sessionId={selectedSessionId} />} />
            {/* Phase 3 routes */}
            <Route path="/inference"
              element={<Inference />} />
            <Route path="/review"
              element={<AnnotationReview sessionId={selectedSessionId} />} />
            <Route path="/retrain"
              element={<ContinuousLearning sessionId={selectedSessionId} />} />
          </Routes>
        </main>
      </div>
    </BrowserRouter>
  )
}

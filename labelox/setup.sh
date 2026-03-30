#!/usr/bin/env bash
set -euo pipefail

echo "=== LABELOX Setup ==="

# 1. Check Python
PYTHON=${PYTHON:-python3}
PYTHON_VERSION=$($PYTHON --version 2>&1)
echo "Python: $PYTHON_VERSION"
if ! $PYTHON -c "import sys; assert sys.version_info >= (3, 11), 'Need Python 3.11+'" 2>/dev/null; then
    echo "ERROR: Python 3.11+ required"
    exit 1
fi

# 2. Virtual environment
VENV_DIR="${VENV_DIR:-venv}"
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtualenv at $VENV_DIR..."
    $PYTHON -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"
echo "Activated: $(which python)"

# 3. Install dependencies
echo "Installing desktop dependencies..."
pip install -r labelox/requirements.txt

echo "Installing web dependencies..."
pip install -r labelox/requirements-web.txt

# 4. Optional: faiss-cpu
echo "Installing faiss-cpu (optional)..."
pip install faiss-cpu 2>/dev/null || echo "WARN: faiss-cpu not installed — similarity search will use sklearn fallback"

# 5. Check FFmpeg (used by some AI models internally)
if command -v ffmpeg &>/dev/null; then
    echo "FFmpeg: $(ffmpeg -version 2>&1 | head -1)"
else
    echo "WARN: FFmpeg not found — some features may not work"
fi

# 6. Create data directories
DATA_DIR="$HOME/.labelox"
mkdir -p "$DATA_DIR"/{projects,uploads,exports,models}
echo "Data directory: $DATA_DIR"

# 7. Initialize database
echo "Initializing database..."
PYTHONPATH=. python -c "
from labelox.core.database import init_db
init_db('sqlite:///$DATA_DIR/labelox.db')
print('Database initialized at $DATA_DIR/labelox.db')
"

echo ""
echo "=== Setup Complete ==="
echo "Desktop:  PYTHONPATH=. python -m labelox.desktop.main"
echo "Web API:  PYTHONPATH=. uvicorn labelox.web.backend.main:app --reload"
echo "Frontend: cd labelox/web/frontend && npm install && npm run dev"

#!/bin/bash
# Quick start script for RAG Teaching Assistant

echo "========================================"
echo "  RAG Teaching Assistant — Quick Start"
echo "========================================"
echo ""

# ── Check Python (need 3.9+) ──────────────────────────────────────────────────
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 not found. Install from https://python.org"
    exit 1
fi

PY_VER=$(python3 -c "import sys; print(sys.version_info.minor)")
if [ "$PY_VER" -lt 9 ]; then
    echo "❌ Python 3.9+ required. You have 3.$PY_VER"
    exit 1
fi
echo "✅ Python 3.$PY_VER found"

# ── Check ffmpeg ──────────────────────────────────────────────────────────────
if ! command -v ffmpeg &> /dev/null; then
    echo "❌ ffmpeg not found."
    echo "   Mac:   brew install ffmpeg"
    echo "   Linux: sudo apt install ffmpeg"
    exit 1
fi
echo "✅ ffmpeg found"

# ── Check API key ─────────────────────────────────────────────────────────────
if [ -z "$OPENAI_API_KEY" ]; then
    echo ""
    echo "⚠  OPENAI_API_KEY is not set!"
    echo "   Run this before starting:"
    echo "   export OPENAI_API_KEY=sk-your-key-here"
    echo ""
    read -p "   Enter your OpenAI API key now (or press Enter to skip): " KEY_INPUT
    if [ -n "$KEY_INPUT" ]; then
        export OPENAI_API_KEY="$KEY_INPUT"
        echo "✅ API key set for this session"
    else
        echo "⚠  Continuing without API key — ingestion will fail without it"
    fi
fi
echo ""

# ── Virtual environment ───────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

source venv/bin/activate

# Upgrade pip first (fixes Python 3.9 pkg_resources issue)
echo "📦 Upgrading pip..."
python3 -m pip install --upgrade pip setuptools wheel -q

echo "📦 Installing dependencies (this may take a few minutes on first run)..."
pip install -r requirements.txt -q

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ Dependency installation failed. Try running manually:"
    echo "   source venv/bin/activate"
    echo "   pip install -r requirements.txt"
    exit 1
fi

echo "✅ Dependencies installed"
echo ""
echo "🚀 Starting backend on http://localhost:8000"
echo "🌐 Open  frontend/index.html  in your browser"
echo ""
echo "Press Ctrl+C to stop"
echo ""

# Use venv's uvicorn directly (avoids 'command not found' on some setups)
cd backend
python3 -m uvicorn main:app --reload --host 0.0.0.0 --port 8000


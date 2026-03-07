#!/usr/bin/env bash
# Start both the FastAPI backend and Vite dev server

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "▸ Starting Bloomdow UI..."
echo ""

# Backend
cd "$SCRIPT_DIR/backend"
"$SCRIPT_DIR/.venv/bin/uvicorn" main:app --host 0.0.0.0 --port 8000 --reload &
BACKEND_PID=$!
echo "  Backend → http://localhost:8000  (pid $BACKEND_PID)"

# Frontend
cd "$SCRIPT_DIR/frontend"
npm run dev &
FRONTEND_PID=$!
echo "  Frontend → http://localhost:5173  (pid $FRONTEND_PID)"

echo ""
echo "  Press Ctrl-C to stop both servers."
echo ""

trap "kill $BACKEND_PID $FRONTEND_PID 2>/dev/null" EXIT INT TERM
wait

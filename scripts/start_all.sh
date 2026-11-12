#!/usr/bin/env bash
set -e
trap '' HUP

# Change directory to project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR"

echo "=== Starting All Services ==="

# 1. Kill anything on ports 5001, 4000, 5173
echo "Stopping any existing processes on ports 5001, 4000, 5173..."
for PORT in 5001 4000 5173; do
  PIDS=$(lsof -ti :$PORT 2>/dev/null || true)
  if [ -n "$PIDS" ]; then
    echo "  Killing process(es) on port $PORT: $PIDS"
    echo "$PIDS" | xargs kill -9 2>/dev/null || true
  fi
done
sleep 1

# 2. Create logs/ directory
mkdir -p logs

# 3. Start Flask, Express, and Vite in background with logs
echo "Starting Flask AI service..."
PYTHONPATH=ai-service ai-service/venv/bin/python ai-service/service/app.py > logs/flask.log 2>&1 &

echo "Starting Express backend..."
(cd server && npm start) > logs/express.log 2>&1 &

echo "Starting Vite frontend..."
npm run dev > logs/vite.log 2>&1 &

# 4. Wait 5 seconds
echo "Waiting 5 seconds for services to initialize..."
sleep 5

# 5. Curl health endpoints and print OK or FAILED
echo ""
echo "=== Health Checks ==="

if curl -s -f --connect-timeout 3 --max-time 5 http://localhost:5001/health > /dev/null 2>&1; then
  echo "Flask:   OK"
else
  echo "Flask:   FAILED"
fi

if curl -s -f --connect-timeout 3 --max-time 5 http://localhost:4000/api/health > /dev/null 2>&1; then
  echo "Express: OK"
else
  echo "Express: FAILED"
fi

# 6. Print the three URLs
echo ""
echo "=== Service URLs ==="
echo "  Flask AI Service:  http://localhost:5001"
echo "  Express API:       http://localhost:4000"
echo "  Vite Frontend:     http://localhost:5173"
echo "===================="

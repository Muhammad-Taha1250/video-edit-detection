#!/usr/bin/env bash
set -e

echo "=== Stopping All Services ==="

for PORT in 5001 4000 5173; do
  PIDS=$(lsof -ti :$PORT 2>/dev/null || true)
  if [ -n "$PIDS" ]; then
    echo "Killing process(es) on port $PORT: $PIDS"
    echo "$PIDS" | xargs kill -9 2>/dev/null || true
  else
    echo "No process running on port $PORT"
  fi
done

echo "All services stopped."

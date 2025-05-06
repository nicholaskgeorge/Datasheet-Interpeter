#!/usr/bin/env bash
# kill any process listening on port 8000
p=$(lsof -t -i tcp:8000)
if [ -n "$p" ]; then
  echo "Killing process(es) on port 8000: $p"
  kill -9 $p
fi

# launch FastAPI backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000

#!/bin/bash
# Start the NLI Span Labeler
cd "$(dirname "$0")"

# Create data directory if it doesn't exist
mkdir -p data/nli

echo "Starting NLI Span Labeler on http://localhost:8000"
echo "Press Ctrl+C to stop"
echo

uvicorn app:app --reload --port 8000 --host 0.0.0.0

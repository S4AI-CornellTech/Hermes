#!/bin/bash
# run_indices.sh - A script to create various indices

# Exit immediately if a command exits with a non-zero status,
# treat unset variables as errors, and propagate errors in pipelines.
set -euo pipefail

# Function to output log messages with timestamps
log() {
  echo "[INFO] $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

log "Starting index creation script..."

log "Creating monolithic index..."
python index/hermes_create_monolithic_index.py --index-size 100K

log "Creating split indices..."
python index/hermes_create_split_indices.py --dataset-size 100k --num-indices 10

log "Creating clustered indices..."
python index/hermes_create_clustered_indices.py --dataset-size 100k --num-indices 10

log "All index creation tasks completed successfully."

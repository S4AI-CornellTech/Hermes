#!/bin/bash
# run_power_measurements.sh - A script to run performance and power measurements

# Exit immediately if a command fails, treat unset variables as errors, and propagate errors in pipelines.
set -euo pipefail

# Function to log messages with a timestamp.
log() {
  echo "[INFO] $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

# List available power event sources.
log "Listing available power event sources:"
ls /sys/bus/event_source/devices/power/events || log "Warning: Unable to list power events."

echo "==============================="

# Run perf stat for a 3-second measurement.
log "Running perf stat for a 3-second measurement..."
sudo perf stat -e power/energy-pkg/ -e power/energy-ram/ sleep 3

echo "==============================="

# Run perf stat for a 1-second measurement.
log "Running perf stat for a 1-second measurement..."
sudo perf stat -e power/energy-pkg/ -e power/energy-ram/ sleep 1

echo "==============================="

# Run rapl-read with the -s option.
log "Running rapl-read with the -s option..."
sudo ./uarch-configure/rapl-read/rapl-read -s

echo "==============================="

# Run rapl-read with the -p option.
log "Running rapl-read with the -p option..."
sudo ./uarch-configure/rapl-read/rapl-read -p

echo "==============================="

# Run rapl-read with the -m option.
log "Running rapl-read with the -m option..."
sudo ./uarch-configure/rapl-read/rapl-read -m

echo "==============================="
log "Completed all power measurement commands."

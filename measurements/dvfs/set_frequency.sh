#!/bin/bash
# set_cpu_frequency.sh - A script to set the CPU frequency for all available CPUs

# Exit immediately if a command exits with a non-zero status,
# treat unset variables as errors, and propagate errors in pipelines.
set -euo pipefail

# Function to log messages with a timestamp
log() {
  echo "[INFO] $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

# Check that a frequency value is provided.
if [ -z "${1:-}" ]; then
    echo "Usage: $0 <frequency_in_hertz>"
    exit 1
fi

freq="$1"
log "Starting CPU frequency update: setting frequency to ${freq} Hz."

# Loop through each CPU directory that matches the pattern.
for cpu in /sys/devices/system/cpu/cpu[0-9]*; do
    # Check if the cpufreq directory exists to avoid errors.
    if [ -d "$cpu/cpufreq" ]; then
        log "Updating frequency for $cpu..."
        echo "$freq" | sudo tee "$cpu/cpufreq/scaling_min_freq" > /dev/null
        echo "$freq" | sudo tee "$cpu/cpufreq/scaling_max_freq" > /dev/null
        log "Frequency updated for $cpu."
    else
        log "Skipping $cpu: cpufreq directory not found."
    fi
done

log "CPU frequency update completed for all applicable CPUs."

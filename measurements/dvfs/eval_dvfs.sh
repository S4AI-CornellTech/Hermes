#!/bin/bash
# run_power_measurement.sh - A script to run a stress test and measure power consumption at various CPU frequencies

# Exit immediately if a command fails, treat unset variables as errors, and propagate errors in pipelines.
set -euo pipefail

# Function to log messages with a timestamp.
log() {
  echo "[INFO] $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

measure() {
  local type="$1"

  log "Extracting CPU frequency range using lscpu..."
  # Extract the min and max CPU frequencies (in MHz) from lscpu.
  local min_mhz max_mhz
  min_mhz=$(lscpu | awk '/^CPU min MHz:/ {print $4}')
  max_mhz=$(lscpu | awk '/^CPU max MHz:/ {print $4}')

  # Convert MHz to Hz (multiply by 1,000,000) and format as an integer.
  local min_hz max_hz
  min_hz=$(awk "BEGIN {printf \"%d\", $min_mhz * 1000000}")
  max_hz=$(awk "BEGIN {printf \"%d\", $max_mhz * 1000000}")

  log "CPU frequency range: ${min_hz} Hz to ${max_hz} Hz"

  # Calculate step size by dividing the Hz range into 100 segments.
  local step=$(( (max_hz - min_hz) / 100 ))
  log "Calculated step size: ${step} Hz"

  # Iterate from min_hz to max_hz in increments of 'step'.
  for (( freq = min_hz; freq <= max_hz; freq += step )); do
    log "Setting CPU frequency to ${freq} Hz..."
    bash measurements/dvfs/set_cpu_freq.sh "$freq"
    
    # Allow system to stabilize at the new frequency.
    sleep 3
    
    local log_file="_output_${freq}_${type}.log"
    log "Measuring power consumption at ${freq} Hz; logging output to ${log_file}"
    bash measurements/dvfs/measure_power.sh > "$log_file" 2>&1
  done
}

# Start the stress test in the background.
log "Starting the stress test..."
python measurements/dvfs/stress_ivf.py &
stress_pid=$!

# Allow time for the system to load.
log "Waiting 60 seconds to allow the system to load..."
sleep 60

# Define the measurement type and run the measurement.
measurement_type="busy"
log "Starting power measurement with type: $measurement_type"
measure "$measurement_type"

# Terminate the stress test.
log "Terminating the stress test..."
pkill -9 python || log "No python process found to kill."

log "Power measurement script completed."

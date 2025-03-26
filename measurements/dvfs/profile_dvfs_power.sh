#!/bin/bash


# Exit immediately if a command fails, treat unset variables as errors, and propagate errors in pipelines.

# Ensure that at least three arguments are provided.
if [ "$#" -lt 3 ]; then
  echo "Usage: $0 <folder> <nProbe> <queries>"
  exit 1
fi

# Get the folder name and nProbe value; the remaining arguments are the list of queries.
folder="$1"
nProbe="$2"
queries="$3"

# Function to log messages with a timestamp.
log() {
  echo "[INFO] $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

# Loop over each file in the provided folder.
log "Extracting CPU frequency range using lscpu..."
local min_mhz max_mhz
min_mhz=$(lscpu | awk '/^CPU min MHz:/ {print $4}')
max_mhz=$(lscpu | awk '/^CPU max MHz:/ {print $4}')

local min_hz max_hz
min_hz=$(awk "BEGIN {printf \"%d\", $min_mhz * 1000}")
max_hz=$(awk "BEGIN {printf \"%d\", $max_mhz * 1000}")

log "CPU frequency range: ${min_hz} KHz to ${max_hz} KHz"

# Iterate from min_hz to max_hz in increments of 'step'.
measure() { 
  for (( freq = min_hz; freq <= max_hz; freq += 100000 )); do
    # echo "Processing file: $index_file"
    # echo "CPU frequency: $freq"
    bash measurements/dvfs/set_cpu_freq.sh "$freq"
    sleep 3
    bash measurements/dvfs/measure_power.sh > output_$index_file_$freq.log
  done
}

for index_file in "$folder"/*; do
  python measurements/dvfs/stress_ivf.py --index $index_file --nprobe $nProbe --queries $queries &
  sleep 60 
  measure
  pkill -9 python
done
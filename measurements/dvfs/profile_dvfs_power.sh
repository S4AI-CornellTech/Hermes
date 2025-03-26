#!/bin/bash


# Exit immediately if a command fails, treat unset variables as errors, and propagate errors in pipelines.
set -eduo pipefail

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
min_mhz=$(lscpu | awk '/^CPU min MHz:/ {print $4}')
max_mhz=$(lscpu | awk '/^CPU max MHz:/ {print $4}')

min_hz=$(awk "BEGIN {printf \"%d\", $min_mhz * 1000}")
max_hz=$(awk "BEGIN {printf \"%d\", $max_mhz * 1000}")

min_hz=3400000
max_hz=3800000

log "CPU frequency range: ${min_hz} KHz to ${max_hz} KHz"
echo "Energy" > data/profiling/measurement.csv

# Iterate from min_hz to max_hz in increments of 'step'.
measure() { 
  for (( freq = min_hz; freq <= max_hz; freq += 100000 )); do
    # echo "Processing file: $index_file"
    # echo "CPU frequency: $freq"
    bash measurements/dvfs/set_frequency.sh "$freq"
    sleep 3
    output=$(bash measurements/dvfs/measure_power.sh)
    energy=$(echo "$output" | grep "Joules" | sed 's/^[ \t]*//' | awk '{print $1}')
    echo "$energy" >> data/profiling/measurement.csv
  done
}

for index_file in "$folder"/*; do
  python measurements/dvfs/stress_ivf.py --index "$index_file" --nprobe "$nProbe" --queries "$queries" > stress.log 2>&1 &

  # Loop until "Index Loaded" appears in the log file.
  while ! grep -q "Index Loaded" stress.log; do
    sleep 1
  done

  # Once "Index Loaded" is detected, run the measure command.
  measure

  # Terminate the Python process.
  kill -9 $pid
  rm stress.log
done

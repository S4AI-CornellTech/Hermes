#!/bin/bash
set -eduo pipefail

usage() {
  echo "Usage: $0 --folder <folder> --queries <query> --nprobe <nprobe1> [nprobe2 ...] --batch-size <size1> [size2 ...] --retrieved-docs <doc1> [doc2 ...] --num-threads <thread1> [thread2 ...]"
  exit 1
}

# Initialize variables and arrays.
folder=""
query=""
nprobes=()
batch_sizes=()
retrieved_docs=()
num_threads=()

# Parse command-line arguments.
while [[ $# -gt 0 ]]; do
  key="$1"
  case $key in
    --folder)
      folder="$2"
      shift 2
      ;;
    --queries)
      query="$2"
      shift 2
      ;;
    --nprobe)
      shift
      while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
        nprobes+=("$1")
        shift
      done
      ;;
    --batch-size)
      shift
      while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
        batch_sizes+=("$1")
        shift
      done
      ;;
    --retrieved-docs)
      shift
      while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
        retrieved_docs+=("$1")
        shift
      done
      ;;
    --num-threads)
      shift
      while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
        num_threads+=("$1")
        shift
      done
      ;;
    *)
      echo "Unknown parameter: $1"
      usage
      ;;
  esac
done

# Validate required parameters.
if [ -z "$folder" ] || [ -z "$query" ] || [ ${#nprobes[@]} -eq 0 ] || [ ${#batch_sizes[@]} -eq 0 ] || [ ${#retrieved_docs[@]} -eq 0 ] || [ ${#num_threads[@]} -eq 0 ]; then
  usage
fi

# Function to log messages with a timestamp.
log() {
  echo "[INFO] $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

log "Extracting CPU frequency range using lscpu..."
min_mhz=$(lscpu | awk '/^CPU min MHz:/ {print $4}')
max_mhz=$(lscpu | awk '/^CPU max MHz:/ {print $4}')

min_hz=$(awk "BEGIN {printf \"%d\", $min_mhz * 1000}")
max_hz=$(awk "BEGIN {printf \"%d\", $max_mhz * 1000}")

# Override frequencies if needed.
min_hz=5000000
max_hz=5400000

log "CPU frequency range: ${min_hz} KHz to ${max_hz} KHz"
echo "Energy" > data/profiling/measurement.csv

# Loop over the frequency values.
for (( freq = min_hz; freq <= max_hz; freq += 100000 )); do
  log "Setting CPU frequency to $freq KHz"
  bash measurements/dvfs/set_frequency.sh "$freq"
  sleep 3

  # Iterate over all combinations of nProbe, Batch Size, Retrieved Docs, and Num Threads.
  for nProbe in "${nprobes[@]}"; do
    for batch_size in "${batch_sizes[@]}"; do
      for doc in "${retrieved_docs[@]}"; do
        for thread in "${num_threads[@]}"; do
          log "Running with nProbe: $nProbe, Batch Size: $batch_size, Retrieved Docs: $doc, Num Threads: $thread, Query: $query"
          python measurements/dvfs/retrieval_dvfs_latency.py \
            --index-folder "$folder" \
            --nprobe "$nProbe" \
            --batch-size "$batch_size" \
            --queries "$query" \
            --retrieved-docs "$doc" \
            --num-threads "$thread" \
            --frequency "$freq"
        done
      done
    done
  done
done

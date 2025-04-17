#!/bin/bash
# run_measurements.sh - A script to run various latency and power measurements

# Exit immediately if a command exits with a non-zero status,
# treat unset variables as errors, and propagate errors in pipelines.
# Function to log messages with timestamps
log() {
  echo "[INFO] $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

log "Starting measurements script..."

log "Running retrieval monolithic latency measurement..."
python measurements/latency/retrieval_monolithic_latency.py \
    --index-name data/indices/monolithic_indices/hermes_index_monolithic_100k.faiss \
    --nprobe 256 \
    --batch-size 32 \
    --retrieved-docs 5 \
    --num-threads 32 \
    --queries triviaqa/triviaqa_encodings.npy

log "Running retrieval split latency measurement..."
python measurements/latency/retrieval_split_latency.py \
    --index-folder data/indices/split_indices \
    --nprobe 128 \
    --batch-size 32 \
    --retrieved-docs 5 \
    --num-threads 32 \
    --dataset-size 1000000 \
    --queries triviaqa/triviaqa_encodings.npy

log "Running retrieval hermes clusters latency measurement..."
python measurements/latency/retrieval_hermes_clusters_latency.py \
    --index-folder data/indices/hermes_clusters \
    --nprobe 8 128 \
    --batch-size 1 2 4 8 16 24 32 40 48 56 64 72 80 88 96 104 112 120 128 136 144 152 160 \
    --retrieved-docs 5 \
    --num-threads 32 \
    --queries triviaqa/triviaqa_encodings.npy

log "Running retrieval hermes sample and deep search latency measurement..."
python measurements/latency/retrieval_hermes_sample_deep_latency.py \
    --index-folder data/indices/hermes_clusters \
    --sample-nprobe 8 \
    --deep-nprobe 128 \
    --retrieved-docs 5 \
    --num-threads 32 \
    --queries triviaqa/triviaqa_encodings.npy

log "Running encoding latency measurement..."
python measurements/latency/encoding_latency.py \
    --model-name BAAI/bge-large-en \
    --batch-size 32 \
    --input-lengths 128

log "Running encoding power measurement..."
python measurements/power/encoding_power.py \
    --model-name BAAI/bge-large-en \
    --batch-size 32 \
    --input-lengths 128

log "Running inference latency measurement..."
python measurements/latency/inference_latency.py \
    --model-name "gpt2" \
    --num-gpus 1 \
    --batch-size 32 \
    --input-lengths 512 \
    --output-lengths 32

log "Running inference power measurement..."
python measurements/power/inference_power.py \
    --model-name "gpt2" \
    --num-gpus 1 \
    --batch-size 32 \
    --input-lengths 512 \
    --output-lengths 32

log "Profiling Hermes Clusters with DVFS Frequencies..."
source measurements/dvfs/profile_dvfs_latency.sh \
    --folder data/indices/hermes_clusters \
    --queries triviaqa/triviaqa_encodings.npy \
    --nprobe 128 \
    --batch-size 1 2 4 8 16 24 32 40 48 56 64 72 80 88 96 104 112 120 128 136 144 152 160 \
    --retrieved-docs 5 \
    --num-threads 32

log "Running Trace Generator..."
python modeling/trace_generator.py

log "All measurements completed successfully."

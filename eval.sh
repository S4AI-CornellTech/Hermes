log() {
  echo "[INFO] $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

log "Running Hermes Multi Node Aggregation Tool..."
python modeling/latency_sim.py \
    --latency-data data/profiling/hermes_cluster_latency.csv \
    --query-trace data/modeling/cluster_trace.csv \
    --retrieved-docs 5 \
    --batch-size 32 \
    --sample-nprobe 8 \
    --deep-nprobe 128 \
    --num-threads 32 

log "Evaluating Hermes Energy Usage with and Without DVFS..."
python modeling/dvfs_sim.py \
    --latency-frequency-data data/profiling/hermes_cluster_latency_dvfs.csv \
    --power-frequency-data data/profiling/hermes_cluster_latency_power.csv \
    --query-trace data/modeling/cluster_trace.csv \
    --inference-trace data/profiling/inference_latency.csv \
    --retrieved-docs 5 \
    --batch-size 32 \
    --sample-nprobe 8 \
    --deep-nprobe 128 \
    --num-threads 32 \
    --input-size 512 \
    --stride-length 16 

log "Evaluating Hermes Accuracy..."
python measurements/accuracy/evaluate_retrieval_accuracy.py \
    --flat-index data/indices/flat_indices/hermes_index_flat_100k.faiss \
    --monolithic-index data/indices/monolithic_indices/hermes_index_monolithic_100k.faiss \
    --split-index-folder data/indices/split_indices \
    --split-index-size 100000 \
    --cluster-index-folder data/indices/hermes_clusters/clusters \
    --cluster-index-indices-folder data/indices/hermes_clusters/cluster_indices \
    --monolithic-nprobe 256 \
    --deep-nprobe 128 \
    --sample-nprobe 8 \
    --retrieved-docs 5 \
    --queries triviaqa/triviaqa_encodings.npy

log "Finished Evaluation."


#!/usr/bin/env bash
# This script runs a series of Python scripts to generate figures.
# Each command prints a message before executing to make the progress clear.

log() {
  echo "[INFO] $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

log "Running Trace Generator..."
python modeling/trace_generator.py

log "Running Figure 11: Hermes Accuracy Comparison"
python figures/fig_11_hermes_accuracy_comparison.py --data-file 100m_data/hermes_100m_accuracy_analysis.csv

log "Running Figure 12: Hermes Nprobe DSE NDCG"
python figures/fig_12_hermes_nprobe_dse_ndcg.py --data-file 100m_data/hermes_100m_accuracy_analysis.csv

log "Running Figure 12: Hermes Nprobe DSE Latency"
python figures/fig_12_hermes_nprobe_dse_latency.py --data-file 100m_data/heremes_100m_sample_deep_analysis.csv

log "Running Figure 13: Cluster Size Frequency Analysis"
python figures/fig_13_cluster_size_frequency_analysis.py \
  --index-folder data/indices/hermes_clusters/clusters \
  --cluster-access-trace data/modeling/cluster_trace.csv \
  --clusters-searched 5

log "Running Figure 14: End-to-End Hermes Latency Comparison"
python figures/fig_14_end_to_end_hermes_latency_comparison.py \
  --input-size 512 \
  --output-size 128 \
  --stride-length 16 \
  --batch-size 32 \
  --sample-nprobe 8 \
  --deep-nprobe 128 \
  --retrieved-docs 5 \
  --clusters-searched 4 \
  --monolithic-retrieval-trace 100m_data/monolithic_retrieval_latency.csv \
  --hermes-retrieval-trace 100m_data/hermes_platinum_8380_100m_modeled_retrieval_latency.csv \
  --encoding-trace 100m_data/bge_large_latency.csv \
  --inference-trace 100m_data/gemma_2_9b_latency.csv

log "Running Figure 14: End-to-End Hermes Energy Comparison"
python figures/fig_14_end_to_end_hermes_energy_comparison.py \
  --input-size 512 \
  --output-size 128 \
  --stride-length 16 \
  --batch-size 32 \
  --sample-nprobe 8 \
  --deep-nprobe 128 \
  --retrieved-docs 5 \
  --clusters-searched 4 \
  --hermes-retrieval-trace 100m_data/hermes_platinum_8380_100m_modeled_retrieval_energy.csv \
  --monolithic-retrieval-trace 100m_data/monolithic_retrieval_latency.csv \
  --encoding-trace 100m_data/bge_large_latency.csv \
  --inference-trace 100m_data/gemma_2_9b_latency.csv \
  --monolithic-retrieval-trace-power 100m_data/monolithic_retrieval_power.csv \
  --encoding-trace-power 100m_data/bge_large_power.csv \
  --inference-trace-power 100m_data/gemma_2_9b_power.csv

log "Running Figure 16: TTFT Hermes Latency Comparison"
python figures/fig_16_ttft_hermes_latency_comparison.py \
  --input-size 512 \
  --stride-length 16 \
  --batch-size 32 \
  --sample-nprobe 8 \
  --deep-nprobe 128 \
  --retrieved-docs 5 \
  --clusters-searched 4 \
  --monolithic-retrieval-trace 100m_data/monolithic_retrieval_latency.csv \
  --hermes-retrieval-trace 100m_data/hermes_platinum_8380_100m_modeled_retrieval_latency.csv \
  --encoding-trace 100m_data/bge_large_latency.csv \
  --inference-trace 100m_data/gemma_2_9b_latency.csv

log "Running Figure 18: Hermes Energy Throughput Trend"
python figures/fig_18_hermes_energy_throuhgput_analysis.py \
  --sample-nprobe 8 \
  --deep-nprobe 128 \
  --retrieved-docs 5 \
  --batch-size 32 \
  --hermes-retrieval-trace 100m_data/hermes_platinum_8380_100m_modeled_retrieval_latency.csv \
  --hermes-energy-trace 100m_data/hermes_platinum_8380_100m_modeled_retrieval_energy.csv

log "Running Figure 20: Hermes Diff Hardware Comparison"
python figures/fig_20_hermes_diff_hardware_comparison.py \
  --sample-nprobe 8 \
  --deep-nprobe 128 \
  --retrieved-docs 5 \
  --batch-size 32 \
  --hermes-retrieval-traces 100m_data/hermes_neoverse_n1_100m_modeled_retrieval_latency.csv \
  100m_data/hermes_gold_6448y_100m_modeled_retrieval_latency.csv \
  100m_data/hermes_platinum_8380_100m_modeled_retrieval_latency.csv \
  100m_data/hermes_silver_4316_100m_modeled_retrieval_latency.csv

log "Running Figure 21: Hermes DVFS Energy Analysis"
python figures/fig_21_hermes_dvfs_analysis.py \
  --sample-nprobe 8 \
  --deep-nprobe 128 \
  --retrieved-docs 5 \
  --batch-size 32 \
  --data-file 100m_data/hermes_platinum_8380_100m_modeled_retrieval_energy.csv

log "All figures generated successfully."

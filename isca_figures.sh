python figures/fig_11_hermes_accuracy_comparison.py --data-file 100m_data/hermes_100m_accuracy_analysis.csv
python figures/fig_12_hermes_nprobe_dse_ndcg.py --data-file 100m_data/hermes_100m_accuracy_analysis.csv

python figures/fig_13_cluster_size_frequency_analysis.py \
    --index-folder data/indices/hermes_clusters/clusters \
    --cluster-access-trace data/modeling/cluster_trace.csv \ 
    --clusters-searched 5

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

python figures/fig_15_ttft_hermes_latency_comparison.py \
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
python figures/fig_11_hermes_accuracy_comparison.py --data-file 100m_data/hermes_100m_accuracy_analysis.csv
python figures/fig_12_hermes_nprobe_dse_ndcg.py --data-file 100m_data/hermes_100m_accuracy_analysis.csv

python figures/fig_13_cluster_size_frequency_analysis.py --index-folder data/indices/hermes_clusters/clusters --cluster-access-trace data/modeling/cluster_trace.csv --clusters-searched 5


import csv
import argparse
import ast
import os
from collections import Counter
from itertools import product
from tqdm import tqdm
import numpy as np

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Cluster Query Benchmark")
    parser.add_argument("--latency-frequency-data", type=str, required=True, help="Path to the profiled latency frequency data CSV file")
    parser.add_argument("--power-frequency-data", type=str, required=True, help="Path to the profiled power frequency data CSV file")
    parser.add_argument("--query-trace", type=str, required=True, help="Path to the cluster trace CSV file")
    parser.add_argument("--sample-nprobe", type=int, nargs='+', required=True, help="List of nprobe values for Hermes sampling search")
    parser.add_argument("--deep-nprobe", type=int, nargs='+', required=True, help="List of nprobe values for Hermes deep search")
    parser.add_argument("--retrieved-docs", type=int, nargs='+', required=True, help="List of numbers of docs retrieved per query")
    parser.add_argument("--batch-size", type=int, nargs='+', required=True, help="List of batch sizes for querying")
    parser.add_argument("--num-threads", type=int, nargs='+', required=True, help="Number of Threads")
    parser.add_argument("--output-dir", type=str, default="data/modeling/", help="Directory where the results will be saved")
    parser.add_argument("--slowed-latency", type=float, required=True, help="Latency to slow enhanced DCFS too")
    return parser.parse_args()

def load_csv_data(file_path):
    with open(file_path, "r", newline="") as csvfile:
        return list(csv.DictReader(csvfile))

def get_num_clusters(latency_data):
    return len({row["Cluster ID"] for row in latency_data})

def get_sampling_latency(latency_data, batch_size, sample_nprobe, retrieved_docs, num_threads):
    latencies = [
        float(row["Avg Retrieval Latency (s)"])
        for row in latency_data
        if int(row["Batch Size"]) == batch_size
           and int(row["nprobe"]) == sample_nprobe
           and int(row["Retrieved Docs"]) == retrieved_docs
           and int(row["Num Threads"]) == num_threads
    ]
    return max(latencies), latencies

def compute_deep_search_latency(latency_dict, cluster_counts, deep_nprobe, retrieved_docs, num_threads):
    deep_latencies = []
    for cluster, count in cluster_counts.items():
        updated_count = count
        key = (cluster, updated_count, deep_nprobe, retrieved_docs, num_threads)
        while key not in latency_dict:
            updated_count += 1
            key = (cluster, updated_count, deep_nprobe, retrieved_docs, num_threads)
        deep_latencies.append(latency_dict[key])
    return max(deep_latencies) if deep_latencies else None, deep_latencies

def get_sampling_energy(latency_data, sampling_latency, all_sampling_latencies, slowed_latency):
    sampling_energy_base, sampling_energy_dvfs, sampling_energy_dvfs_enhanced = 0, 0, 0
    for sampling_latency in all_sampling_latencies:
        sampling_energy_base = 
        print(sampling_latency)
    return sampling_energy_base, sampling_energy_dvfs, sampling_energy_dvfs_enhanced

def process_benchmark(args):
    # Load all CSV data into memory
    latency_data = load_csv_data(args.latency_frequency_data)
    queries = load_csv_data(args.query_trace)
    num_clusters = get_num_clusters(latency_data)
    
    # Precompute a lookup dictionary for deep search latency.
    latency_dict = {
        (int(row["Cluster ID"]), int(row["Batch Size"]), int(row["nprobe"]),
         int(row["Retrieved Docs"]), int(row["Num Threads"])): float(row["Avg Retrieval Latency (s)"])
        for row in latency_data
    }
    
    # Ensure output directory exists.
    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(args.output_dir, 'hermes_retrieval_energy.csv')
    fieldnames = [
        "Sample nprobe", "Deep nprobe", "Batch Size", "Retrieved Docs",
        "Clusters Searched", "Avg Hermes Retrieval Latency (s)", "Avg Hermes Throughput (QPS)"
    ]
    
    with open(output_file, "w", newline="") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        
        # Loop over all parameter combinations using itertools.product.
        param_combinations = list(product(args.sample_nprobe, args.deep_nprobe, args.num_threads, args.batch_size, args.retrieved_docs))
        for sample_nprobe, deep_nprobe, num_threads, batch_size, retrieved_docs in tqdm(param_combinations, desc="Parameter Combinations"):
            sampling_latency, all_sampling_latencies = get_sampling_latency(latency_data, batch_size, sample_nprobe, retrieved_docs, num_threads)
            sampling_energy_base, sampling_energy_dvfs, sampling_energy_dvfs_enhanced = get_sampling_energy(latency_data, sampling_latency, all_sampling_latencies, args.slowed_latency)

            # Vary the number of clusters searched.
            for clusters_searched in tqdm(range(1, num_clusters + 1), desc="Clusters Searched", leave=False):
                hermes_latencies = []
                cluster_counts = Counter()
                
                # Process each query from the preloaded trace.
                for idx, query in tqdm(enumerate(queries), total=len(queries), desc="Query Trace", leave=False):
                    ranked_clusters = ast.literal_eval(query['Ranked Clusters'])
                    # Consider only the top 'clusters_searched' clusters.
                    cluster_counts.update(ranked_clusters[:clusters_searched])
                    
                    if idx % batch_size == 0:
                        deep_latency, all_deep_latencies = compute_deep_search_latency(latency_dict, cluster_counts, deep_nprobe, retrieved_docs, num_threads)
                        if sampling_latency is not None and deep_latency is not None:
                            hermes_latencies.append(sampling_latency + deep_latency)
                        cluster_counts = Counter()
                
                avg_latency = np.mean(hermes_latencies) if hermes_latencies else None
                throughput = batch_size / avg_latency if avg_latency else None
                
                writer.writerow({
                    "Sample nprobe": sample_nprobe,
                    "Deep nprobe": deep_nprobe,
                    "Batch Size": batch_size,
                    "Retrieved Docs": retrieved_docs,
                    "Clusters Searched": clusters_searched,
                    "Avg Hermes Retrieval Latency (s)": avg_latency,
                    "Avg Hermes Throughput (QPS)": throughput
                })
                outfile.flush()

def main():
    args = parse_arguments()
    process_benchmark(args)

if __name__ == "__main__":
    main()


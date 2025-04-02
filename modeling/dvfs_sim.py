
import csv
import argparse
import ast
import os
from collections import Counter
from itertools import product
from tqdm import tqdm
import numpy as np
import pandas as pd

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
    parser.add_argument("--inference-trace", type=str, required=True, help="Path to the inference trace CSV file")
    parser.add_argument("--input-size", type=int, required=True, help="Input size")
    parser.add_argument("--stride-length", type=int, required=True, help="Stride length")

    return parser.parse_args()

def load_csv_data(file_path):
    with open(file_path, "r", newline="") as csvfile:
        return list(csv.DictReader(csvfile))

def get_num_clusters(latency_data):
    return len({row["Cluster ID"] for row in latency_data})

def get_sampling_latency(latency_data, batch_size, sample_nprobe, retrieved_docs, num_threads, total_clusters):
    
    unique_frequencies = sorted({int(row["Frequency"]) for row in latency_data})

    latencies = [
        float(row["Avg Retrieval Latency (s)"])
        for row in latency_data
        if int(row["Batch Size"]) == batch_size
           and int(row["nprobe"]) == sample_nprobe
           and int(row["Retrieved Docs"]) == retrieved_docs
           and int(row["Num Threads"]) == num_threads
           and int(row["Frequency"]) == unique_frequencies[-1]
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

def get_energy(latency_data, power_data, sorted_cluster_counts, nprobe, retrieved_docs, max_latency, all_latencies, slowed_latency, search_type="sampling"):
    unique_frequencies = sorted({int(row["Frequency"]) for row in latency_data})

    busy_max_power_value = power_data[(power_data['State'] == 'busy') & (power_data['Frequency (Khz)'] == unique_frequencies[-1])]['Power (W)'].iloc[0]
    idle_max_power_value = power_data[(power_data['State'] == 'idle') & (power_data['Frequency (Khz)'] == unique_frequencies[-1])]['Power (W)'].iloc[0]

    batch_sizes = list(sorted_cluster_counts.values())
    cluster_ids = list(sorted_cluster_counts)
    # print(sorted_cluster_counts)
    # print(batch_sizes)

    energy_base, energy_dvfs, energy_enhanced_dvfs = 0, 0, 0
    for num, latency in enumerate(all_latencies):
        energy_val = (latency * busy_max_power_value) + (idle_max_power_value * (max_latency - latency))
        energy_dvfs_max = energy_val
        energy_enhanced_dvfs_max = energy_val

        not_entered = True
        for frequency in unique_frequencies:   
            current_batch_size = batch_sizes[num]
            # Loop until a match is found
            while True:
                freq_latency = [
                    row for row in latency_data 
                    if int(row['Frequency']) == frequency and 
                    int(row['Batch Size']) == current_batch_size and 
                    int(row['nprobe']) == nprobe and 
                    int(row['Retrieved Docs']) == retrieved_docs and
                    int(row['Cluster ID']) == cluster_ids[num]
                ]
                if freq_latency:
                    # Once a match is found, update the batch size to the valid one
                    batch_sizes[num] = current_batch_size
                    if not_entered:
                        not_entered = False
                        # print(current_batch_size)
                    break
                current_batch_size += 1

            frequency_latency_value = float(freq_latency[0]["Avg Retrieval Latency (s)"])

            if frequency_latency_value < max_latency:
                busy_freq_power_value = power_data[(power_data['State'] == 'busy') & (power_data['Frequency (Khz)'] == frequency)]['Power (W)'].iloc[0]
                idle_freq_power_value = power_data[(power_data['State'] == 'idle') & (power_data['Frequency (Khz)'] == frequency)]['Power (W)'].iloc[0]
                energy_freq = frequency_latency_value * busy_freq_power_value + (max_latency - frequency_latency_value) * idle_freq_power_value
                if energy_freq < energy_dvfs_max:
                    energy_dvfs_max = energy_freq

            if frequency_latency_value < max(max_latency, slowed_latency):
                busy_freq_power_value = power_data[(power_data['State'] == 'busy') & (power_data['Frequency (Khz)'] == frequency)]['Power (W)'].iloc[0]
                idle_freq_power_value = power_data[(power_data['State'] == 'idle') & (power_data['Frequency (Khz)'] == frequency)]['Power (W)'].iloc[0]
                energy_enhanced_freq = frequency_latency_value * busy_freq_power_value + (max_latency - frequency_latency_value) * idle_freq_power_value
                if energy_enhanced_freq < energy_enhanced_dvfs_max:
                    energy_enhanced_dvfs_max = energy_enhanced_freq

        energy_base += energy_val
        energy_dvfs += energy_dvfs_max
        energy_enhanced_dvfs += energy_enhanced_dvfs_max

    if search_type == "sampling":
        energy_enhanced_dvfs = energy_dvfs

    return energy_base, energy_dvfs, energy_enhanced_dvfs

def process_benchmark(args):
    # Load all CSV data into memory
    latency_data = load_csv_data(args.latency_frequency_data)
    power_data = pd.read_csv(args.power_frequency_data)

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
        "Clusters Searched", "Avg Hermes Energy (J)", "Avg Hermes DVFS Energy (J)", "Avg Hermes Enhanced DVFS Energy (J)"
    ]
    
    with open(output_file, "w", newline="") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        
        # Loop over all parameter combinations using itertools.product.
        param_combinations = list(product(args.sample_nprobe, args.deep_nprobe, args.num_threads, args.batch_size, args.retrieved_docs))
        for sample_nprobe, deep_nprobe, num_threads, batch_size, retrieved_docs in tqdm(param_combinations, desc="Parameter Combinations"):
            unique_cluster_ids = len(set(row["Cluster ID"] for row in latency_data))
            
            prefill_time = 0
            decoding_time = 0

            with open(args.inference_trace, "r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if (int(row["Batch Size"]) == batch_size and 
                        int(row["Input Token Length"]) == args.input_size and 
                        int(row["Output Token Length"]) == args.stride_length):
                        prefill_time = float(row["Avg Prefill Time (s)"])
                        decoding_time = float(row["Avg Decode Time (s)"])
            
            cluster_counts = Counter({cluster_id: batch_size for cluster_id in range(unique_cluster_ids)})

            sampling_latency, all_sampling_latencies = get_sampling_latency(latency_data, batch_size, sample_nprobe, retrieved_docs, num_threads, unique_cluster_ids)
            sampling_energy_base, sampling_energy_dvfs, _ = get_energy(latency_data, power_data, cluster_counts, sample_nprobe, retrieved_docs, sampling_latency, all_sampling_latencies, 0)

            slowed_latency = prefill_time + decoding_time - sampling_latency

            hermes_energy = []
            hermes_dvfs_energy = []
            hermes_enhanced_dvfs_energy = []

            # Vary the number of clusters searched.
            for clusters_searched in tqdm(range(1, num_clusters + 1), desc="Clusters Searched", leave=False):
                cluster_counts = Counter()
                
                # Process each query from the preloaded trace.
                for idx, query in tqdm(enumerate(queries[:1000]), total=1000, desc="Query Trace", leave=False):
                    ranked_clusters = ast.literal_eval(query['Ranked Clusters'])
                    # Consider only the top 'clusters_searched' clusters.
                    cluster_counts.update(sorted(ranked_clusters[:clusters_searched]))
                    
                    if idx % batch_size == 0:
                        sorted_cluster_counts = {k: cluster_counts[k] for k in sorted(cluster_counts)}
                        deep_latency, all_deep_latencies = compute_deep_search_latency(latency_dict, sorted_cluster_counts, deep_nprobe, retrieved_docs, num_threads)
                        deep_energy_base, deep_energy_dvfs, deep_energy_enhanced_dvfs = get_energy(latency_data, power_data, sorted_cluster_counts, deep_nprobe, retrieved_docs, deep_latency, all_deep_latencies, slowed_latency, "deep")
                        cluster_counts = Counter()

                    hermes_energy.append(sampling_energy_base + deep_energy_base)
                    hermes_dvfs_energy.append(sampling_energy_dvfs + deep_energy_dvfs)  
                    hermes_enhanced_dvfs_energy.append(sampling_energy_dvfs + deep_energy_enhanced_dvfs)

                writer.writerow({
                    "Sample nprobe": sample_nprobe,
                    "Deep nprobe": deep_nprobe,
                    "Batch Size": batch_size,
                    "Retrieved Docs": retrieved_docs,
                    "Clusters Searched": clusters_searched,
                    "Avg Hermes Energy (J)": np.mean(hermes_energy),
                    "Avg Hermes DVFS Energy (J)": np.mean(hermes_dvfs_energy),
                    "Avg Hermes Enhanced DVFS Energy (J)": np.mean(hermes_enhanced_dvfs_energy),
                })
                outfile.flush()

def main():
    args = parse_arguments()
    process_benchmark(args)

if __name__ == "__main__":
    main()


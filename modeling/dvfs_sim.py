
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

def get_sampling_energy(latency_data, power_data, sampling_latency, all_sampling_latencies):
    unique_frequencies = sorted({int(row["Frequency"]) for row in latency_data})

    busy_max_power_value = power_data[(power_data['State'] == 'busy') & (power_data['Frequency (Khz)'] == unique_frequencies[-1])]['Power (W)'].iloc[0]
    idle_max_power_value = power_data[(power_data['State'] == 'idle') & (power_data['Frequency (Khz)'] == unique_frequencies[-1])]['Power (W)'].iloc[0]

    sampling_energy_base = 0
    for latency in all_sampling_latencies:
        sampling_energy_base += (latency * busy_max_power_value) + (idle_max_power_value * (sampling_latency - latency))

    return sampling_energy_base

def get_deep_energy(latency_data, power_data, batch_size, nprobe, retrieved_docs, deep_latency, all_deep_latencies, slowed_latency):
    unique_frequencies = sorted({int(row["Frequency"]) for row in latency_data})

    busy_max_power_value = power_data[(power_data['State'] == 'busy') & (power_data['Frequency (Khz)'] == unique_frequencies[-1])]['Power (W)'].iloc[0]
    idle_max_power_value = power_data[(power_data['State'] == 'idle') & (power_data['Frequency (Khz)'] == unique_frequencies[-1])]['Power (W)'].iloc[0]

    deep_energy_base, deep_energy_dvfs, deep_energy_enhanced_dvfs = 0, 0, 0
    for num, latency in enumerate(all_deep_latencies):
        energy_val = (latency * busy_max_power_value) + (idle_max_power_value * (deep_latency - latency))
        deep_energy_dvfs_max = energy_val
        deep_energy_enhanced_dvfs_max = energy_val

        for frequency in unique_frequencies:   
            freq_latency = [
                row for row in latency_data 
                if int(row['Frequency']) == frequency and 
                int(row['Batch Size']) == batch_size and 
                int(row['nprobe']) == nprobe and 
                int(row['Retrieved Docs']) == retrieved_docs and
                int(row['Cluster ID']) == num
            ]

            frequency_latency_value = float(freq_latency[0]["Avg Retrieval Latency (s)"])

            if frequency_latency_value < deep_latency:
                busy_freq_power_value = power_data[(power_data['State'] == 'busy') & (power_data['Frequency (Khz)'] == frequency)]['Power (W)'].iloc[0]
                idle_freq_power_value = power_data[(power_data['State'] == 'idle') & (power_data['Frequency (Khz)'] == frequency)]['Power (W)'].iloc[0]
                deep_energy_freq = frequency_latency_value * busy_freq_power_value + (deep_latency - frequency_latency_value) * idle_freq_power_value
                if deep_energy_freq < deep_energy_dvfs_max:
                    deep_energy_dvfs_max = deep_energy_freq

            if frequency_latency_value < slowed_latency:
                busy_freq_power_value = power_data[(power_data['State'] == 'busy') & (power_data['Frequency (Khz)'] == frequency)]['Power (W)'].iloc[0]
                idle_freq_power_value = power_data[(power_data['State'] == 'idle') & (power_data['Frequency (Khz)'] == frequency)]['Power (W)'].iloc[0]
                deep_energy_enhanced_freq = frequency_latency_value * busy_freq_power_value + (deep_latency - frequency_latency_value) * idle_freq_power_value
                if deep_energy_enhanced_freq < deep_energy_enhanced_dvfs_max:
                    deep_energy_enhanced_dvfs_max = deep_energy_enhanced_freq

        deep_energy_base += energy_val
        deep_energy_dvfs += deep_energy_dvfs_max
        deep_energy_enhanced_dvfs += deep_energy_enhanced_dvfs_max

    return deep_energy_base, deep_energy_dvfs, deep_energy_enhanced_dvfs

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
            sampling_latency, all_sampling_latencies = get_sampling_latency(latency_data, batch_size, sample_nprobe, retrieved_docs, num_threads)
            sampling_energy_base = get_sampling_energy(latency_data, power_data, sampling_latency, all_sampling_latencies)

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
                        
            slowed_latency = prefill_time + decoding_time

            hermes_energy = []
            hermes_dvfs_energy = []
            hermes_enhanced_dvfs_energy = []

            # Vary the number of clusters searched.
            for clusters_searched in tqdm(range(1, num_clusters + 1), desc="Clusters Searched", leave=False):
                cluster_counts = Counter()
                
                # Process each query from the preloaded trace.
                for idx, query in tqdm(enumerate(queries), total=len(queries), desc="Query Trace", leave=False):
                    ranked_clusters = ast.literal_eval(query['Ranked Clusters'])
                    # Consider only the top 'clusters_searched' clusters.
                    cluster_counts.update(sorted(ranked_clusters[:clusters_searched]))
                    
                    if idx % batch_size == 0:
                        sorted_cluster_counts = {k: cluster_counts[k] for k in sorted(cluster_counts)}
                        deep_latency, all_deep_latencies = compute_deep_search_latency(latency_dict, cluster_counts, deep_nprobe, retrieved_docs, num_threads)
                        deep_energy_base, deep_energy_dvfs, deep_energy_enhanced_dvfs = get_deep_energy(latency_data, power_data, batch_size, deep_nprobe, retrieved_docs, deep_latency, all_deep_latencies, slowed_latency)
                        cluster_counts = Counter()

                    hermes_energy.append(sampling_energy_base + deep_energy_base)
                    hermes_dvfs_energy.append(sampling_energy_base + deep_energy_dvfs)  
                    hermes_enhanced_dvfs_energy.append(sampling_energy_base + deep_energy_enhanced_dvfs)


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


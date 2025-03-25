import csv
import argparse
import ast
from collections import Counter
from tqdm import tqdm
import os
import numpy as np

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Cluster Query Benchmark")
    parser.add_argument("--latency-data", type=str, required=True, help="Path to the profiled latency data CSV file")
    parser.add_argument("--query-trace", type=str, required=True, help="Path to the cluster trace CSV file")
    parser.add_argument("--sample-nprobe", type=int, nargs='+', required=True, help="List of nprobe values for Hermes sampling search")
    parser.add_argument("--deep-nprobe", type=int, nargs='+', required=True, help="List of nprobe values for Hermes deep search")
    parser.add_argument("--retrieved-docs", type=int, nargs='+', required=True, help="List of numbers of docs retrieved per query")
    parser.add_argument("--batch-size", type=int, nargs='+', required=True, help="List of batch sizes for querying")
    parser.add_argument("--num-threads", type=int, nargs='+', required=True, help="Number of Threads")
    parser.add_argument("--output-dir", type=str, default="data/modeling/", help="Directory where the results will be saved")
    return parser.parse_args()

def load_csv_data(file_path):
    """Load CSV data from a file and return a list of rows."""
    with open(file_path, "r", newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        return list(reader)

def get_num_clusters(latency_lookup_table):
    """Determine the number of unique clusters in the latency data."""
    cluster_ids = {row["Cluster ID"] for row in latency_lookup_table}
    return len(cluster_ids)

def get_sampling_latency(latency_lookup_table, batch_size, sample_nprobe, retrieved_docs, num_threads):
    """
    Calculate the maximum sampling search latency from the latency rows that match
    the given batch size and sample nprobe value.
    """
    latencies = []
    for row in latency_lookup_table:
        if int(row["Batch Size"]) == batch_size and int(row["nprobe"]) == sample_nprobe and int(row["Retrieved Docs"]) == retrieved_docs and int(row["Num Threads"]) == num_threads:
            latencies.append(float(row["Avg Retrieval Latency (s)"]))
    return max(latencies) if latencies else None

def compute_deep_search_latency(latency_lookup_table, cluster_counts, deep_nprobe, retrieved_docs, num_threads):
    """
    For each cluster in the current counts, directly look up the matching row in the latency data
    (matching cluster id, batch size, and nprobe) and return the maximum latency.
    """
    # Create a lookup dictionary keyed by (Cluster ID, Batch Size, nprobe)
    latency_dict = {
        (int(row["Cluster ID"]), int(row["Batch Size"]), int(row["nprobe"]), int(row["Retrieved Docs"]), int(row["Num Threads"])): float(row["Avg Retrieval Latency (s)"])
        for row in latency_lookup_table
    }
    
    deep_latencies = []
    for cluster, count in cluster_counts.items():
        key = (cluster, count, deep_nprobe, retrieved_docs, num_threads)
        if key in latency_dict:
            deep_latencies.append(latency_dict[key])

    return max(deep_latencies) if deep_latencies else None


def process_benchmark(args):
    """Main processing function for the benchmark."""
    # Load the full latency data once.
    latency_lookup_table = load_csv_data(args.latency_data)
    num_clusters = get_num_clusters(latency_lookup_table)

    # Prepare output file using DictWriter for clarity.
    output_file = os.path.join(args.output_dir, 'hermes_retrieval.csv')
    fieldnames = ["Sample nprobe", "Deep nprobe", "Batch Size", "Retrieved Docs",
                  "Clusters Searched", "Avg Hermes Retrieval Latency (s)", "Avg Hermes Throughput (QPS)"]
    with open(output_file, "w", newline="") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()

        # Iterate over all parameter combinations.
        for sample_nprobe in tqdm(args.sample_nprobe, desc="Sample nprobe values", position=0):
            for deep_nprobe in tqdm(args.deep_nprobe, desc=f"Deep nprobe values (sample_nprobe={sample_nprobe})", position=1, leave=False):
                for num_threads in tqdm(args.num_threads, desc=f"Deep nprobe values (sample_nprobe={sample_nprobe})", position=2, leave=False):
                    for batch_size in tqdm(args.batch_size, desc="Batch sizes", position=3):
                        for retrieved_docs in tqdm(args.retrieved_docs, desc=f"Retrieved Docs (batch_size={batch_size})", position=4, leave=False):
                            # Compute the sampling latency once for these parameters.
                            sampling_latency = get_sampling_latency(latency_lookup_table, batch_size, sample_nprobe, retrieved_docs, num_threads)
                            hermes_latencies = []

                            # Loop over different numbers of clusters searched.
                            for clusters_searched in tqdm(range(1, num_clusters + 1), desc="Clusters Searched", position=5, leave=False):
                                # Initialize a counter for clusters (assume clusters 0-9).
                                cluster_counts = Counter({i: 0 for i in range(num_clusters)})

                                # Process the query trace file.
                                with open(args.query_trace, "r", newline="") as query_file:
                                    query_reader = csv.DictReader(query_file)
                                    total_lines = sum(1 for _ in open(args.query_trace, "r", newline=""))

                                    for idx, query in tqdm(enumerate(query_reader), total=total_lines, desc="Query Trace", position=6, leave=False):
                                        # Parse the ranked clusters list from the query.
                                        ranked_clusters = ast.literal_eval(query['Ranked Clusters'])
                                        # Only consider the top 'clusters_searched' clusters.
                                        searched_clusters = ranked_clusters[:clusters_searched]
                                        cluster_counts.update(searched_clusters)

                                        # At each batch boundary, perform the deep search latency calculation.
                                        if idx % batch_size == 0:
                                            deep_latency = compute_deep_search_latency(latency_lookup_table, cluster_counts, deep_nprobe, retrieved_docs, num_threads)
                                            if sampling_latency is not None and deep_latency is not None:
                                                combined_latency = sampling_latency + deep_latency
                                                hermes_latencies.append(combined_latency)
                                            # Reset the counter for the next batch.
                                            cluster_counts = Counter({i: 0 for i in range(num_clusters)})

                                # Calculate the average Hermes latency for this configuration.
                                avg_latency = np.mean(hermes_latencies) if hermes_latencies else None

                                # Write out the results.
                                writer.writerow({
                                    "Sample nprobe": sample_nprobe,
                                    "Deep nprobe": deep_nprobe,
                                    "Batch Size": batch_size,
                                    "Retrieved Docs": retrieved_docs,
                                    "Clusters Searched": clusters_searched,
                                    "Avg Hermes Retrieval Latency (s)": avg_latency,
                                    "Avg Hermes Throughput (QPS)": batch_size / avg_latency
                                })
                                outfile.flush()  # Write incrementally

def main():
    args = parse_arguments()
    process_benchmark(args)

if __name__ == "__main__":
    main()

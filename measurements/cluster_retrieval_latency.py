import argparse
import time
import csv
import os
import numpy as np
import faiss
from tqdm import tqdm

def parse_arguments():
    parser = argparse.ArgumentParser(description="Cluster Query Benchmark")
    parser.add_argument("--index-folder", type=str, required=True, help="Path to the clustered FAISS index file")
    parser.add_argument("--sample-nprobe", type=int, nargs='+', required=True, help="List of nprobe values for hermes sampling search")
    parser.add_argument("--deep-nprobe", type=int, nargs='+', required=True, help="List of nprobe values for hermes deep search")
    parser.add_argument("--queries", type=str, required=True, help="Path to the NumPy file containing embeddings")
    parser.add_argument("--retrieved-docs", type=int, nargs='+', required=True, help="List of numbers of docs retrieved per query")
    parser.add_argument("--num-threads", type=int, nargs='+', required=True, help="List of numbers of threads to run retrieval")
    parser.add_argument("--output-dir", type=str, default="data/profiling/", help="Directory where the results will be saved")
    return parser.parse_args()

def load_clustered_indices(indices_dir):
    indices_files = sorted([os.path.join(indices_dir, f) for f in os.listdir(indices_dir) if f.endswith(".faiss")])
    clustered_indices = [faiss.read_index(index_file) for index_file in indices_files]
    num_indices = len(indices_files)
    return clustered_indices, num_indices

def perform_queries(clustered_indices, num_clusters_searched, retrieved_docs, embeddings, sample_nprobe, deep_nprobe, clustered_index_files, max_queries=1000):
    for idx in tqdm(range(0, max_queries, 1),
                        desc="Querying batches",
                        leave=False,
                        position=4):

        sample_search_times, deep_search_times, aggregation_times = [], [], []

        query = embeddings[idx:idx + 1]

        sample_search_distances, sample_search = [], []
        for index_num, clustered_index in tqdm(enumerate(clustered_indices),
                                           desc="Sample Search",
                                           leave=False,
                                           total=len(clustered_indices),
                                           position=5):
            clustered_index.nprobe = sample_nprobe
            
            sample_search_start = time.time()
            cluster_distances, _ = clustered_index.search(query, retrieved_docs)
            sample_search_distances.append(sum(cluster_distances.flatten()) / len(cluster_distances.flatten()))
            sample_search_end = time.time()
            sample_search.append(sample_search_end - sample_search_start)

        sample_search_sort_start = time.time()
        sorted_indices = sorted(range(len(sample_search_distances)), key=lambda x: sample_search_distances[x], reverse=True)
        sample_search_sort_end = time.time()
        
        deep_search_distances, deep_search_docs, deep_search_times = [], [], []

        for i, index_id in tqdm(enumerate(sorted_indices[:num_clusters_searched]),
                                desc="Deep Search",
                                leave=False,
                                total=num_clusters_searched,
                                position=5):
            cluster_index = clustered_indices[index_id]
            cluster_index.nprobe = deep_nprobe
            cluster_values_path = os.path.join(clustered_index_files, f'cluster_indices/cluster_{index_id}_indices.npy')
            cluster_values = np.load(cluster_values_path)
            
            deep_search_start_time = time.time()
            cluster_distances, cluster_docs = cluster_index.search(query, retrieved_docs)
            deep_search_end_time = time.time()

            deep_search_times.append(deep_search_end_time - deep_search_start_time)
            deep_search_distances.extend(cluster_distances.flatten().tolist())
            deep_search_docs.extend([cluster_values[idx] for idx in cluster_docs.flatten()])

            deep_search_agg_start = time.time()
            sorted_kmeans_indices = np.argsort(deep_search_distances)[-1:-(retrieved_docs + 1):-1]
            nprobe_corresponding_docs = [deep_search_docs[i] for i in sorted_kmeans_indices]
            deep_search_agg_end = time.time()


        sample_search_times.append(max(sample_search))
        deep_search_times.append(max(deep_search_times))
        aggregation_times.append(sample_search_sort_end - sample_search_sort_start + deep_search_agg_end - deep_search_agg_start)

        if idx >= max_queries:
            break
        
    return np.mean(sample_search_times), np.mean(deep_search_times), np.mean(aggregation_times)

def main():
    args = parse_arguments()
    
    # Create the output directory if it doesn't exist and set the output file path.
    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(args.output_dir, "clustered_retrieval_latency.csv")
    
    # Initially load the index with a dummy nprobe; it will be updated later in the loop.
    indices, num_indices = load_clustered_indices(os.path.join(args.index_folder, "clusters"))
    # centroid_search_index = os.path.join(args.index_folder, "kmeans_centroids.npy")
    embeddings = np.load(args.queries)
    
    with open(output_file, mode='w', newline='') as file:
        fieldnames = ["Number of Searched Indices", "Sample nProbe", "Deep nProbe", "Batch Size", "Retrieved Docs", "Num Threads", "Avg Sample Search Time (s)", "Avg Deep Search Time (s)", "Avg Aggregation Time (s)"]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        
        for sample_nprobe in tqdm(args.sample_nprobe, desc="Sample nprobe values", position=0):
            for deep_nprobe in tqdm(args.deep_nprobe, desc=f"Deep nprobe values (sample_nprobe={sample_nprobe})", position=1, leave=False):
                for retrieved_docs in tqdm(args.retrieved_docs, desc=f"Retrieved Docs (sample_nprobe={sample_nprobe}, deep_nprobe={deep_nprobe})", position=2, leave=False):
                    for num_threads in tqdm(args.num_threads, desc=f"Num Threads (sample_nprobe={sample_nprobe}, deep_nprobe={deep_nprobe}, retrieved_docs={retrieved_docs})", position=3, leave=False):
                        for k in range(1, num_indices + 1):
                            faiss.omp_set_num_threads(num_threads)
                            avg_sample_search_time, avg_deep_search_time, avg_aggregation_time = perform_queries(indices, k, retrieved_docs, embeddings, sample_nprobe, deep_nprobe, args.index_folder)
                            writer.writerow({
                                "Number of Searched Indices": k,
                                "Sample nProbe": sample_nprobe,
                                "Deep nProbe": deep_nprobe,
                                "Retrieved Docs": retrieved_docs,
                                "Num Threads": num_threads,
                                "Avg Sample Search Time (s)": avg_sample_search_time,
                                "Avg Deep Search Time (s)": avg_deep_search_time, 
                                "Avg Aggregation Time (s)": avg_aggregation_time
                            })
                            file.flush()  # Ensure data is written incrementally

if __name__ == "__main__":
    main()

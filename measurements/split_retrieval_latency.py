import argparse
import time
import csv
import os
import numpy as np
import faiss
from tqdm import tqdm

def parse_arguments():
    parser = argparse.ArgumentParser(description="Split Retrieval Latency Benchmark")
    parser.add_argument("--indices-dir", type=str, required=True,
                        help="Directory containing split FAISS index files (ending with .faiss)")
    parser.add_argument("--queries", type=str, required=True,
                        help="Path to the NumPy file containing query embeddings")
    parser.add_argument("--batch-size", type=int, nargs='+', required=True,
                        help="List of batch sizes for querying")
    parser.add_argument("--retrieved-docs", type=int, nargs='+', required=True,
                        help="List of numbers of nearest neighbors to retrieve")
    parser.add_argument("--nprobe", type=int, nargs='+', required=True,
                        help="List of nprobe values to use for each index")
    parser.add_argument("--num-threads", type=int, nargs='+', required=True,
                        help="List of numbers of threads to run retrieval")
    parser.add_argument("--max-k", type=int, default=10,
                        help="Maximum number of clusters (split indices) to search")
    parser.add_argument("--output-dir", type=str, default="data/profiling/",
                        help="Directory where the results will be saved")
    return parser.parse_args()

def load_split_indices(indices_dir):
    indices_files = sorted([os.path.join(indices_dir, f)
                            for f in os.listdir(indices_dir) if f.endswith(".faiss")])
    split_indices = [faiss.read_index(index_file) for index_file in indices_files]
    return split_indices

def perform_split_query(query, split_indices, k, retrieved_docs, nprobe):
    """
    Performs a split query on the first k indices.
    
    For each index:
      - sets the nprobe value,
      - performs a search for the given number of retrieved_docs,
      - adjusts document ids to avoid collisions.
    
    Returns:
      - query_time: maximum search time across the k indices.
      - agg_time: time to sort and aggregate the results.
    """
    split_indices_distances = []
    split_indices_docs = []
    split_indices_times = []
    
    # Search on each of the first k split indices
    for i, index in enumerate(split_indices[:k]):
        index.nprobe = nprobe
        start_time = time.time()
        distances, docs = index.search(query, retrieved_docs)
        search_time = time.time() - start_time
        split_indices_times.append(search_time)
        # Adjust document ids to avoid collisions across indices
        adjusted_docs = docs.flatten() + i * 10_000_000
        split_indices_docs.extend(adjusted_docs.tolist())
        split_indices_distances.extend(distances.flatten().tolist())
    
    # Maximum time among indices is used as the query time
    query_time = max(split_indices_times)
    
    # Aggregation: sort all distances and select the top retrieved_docs
    agg_start = time.time()
    sorted_indices = np.argsort(split_indices_distances)[-retrieved_docs:][::-1]
    _ = [split_indices_docs[i] for i in sorted_indices]
    agg_time = time.time() - agg_start

    return query_time, agg_time

def main():
    args = parse_arguments()
    
    # Ensure the output directory exists and define the CSV output file
    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(args.output_dir, "split_retrieval_latency.csv")
    
    # Load split indices and query embeddings
    split_indices = load_split_indices(args.indices_dir)
    embeddings = np.load(args.queries)
    
    with open(output_file, mode='w', newline='') as csvfile:
        fieldnames = [
            "Max Clusters Searched", "nprobe", "Batch Size", "Retrieved Docs", 
            "Num Threads", "Avg Split Query Time (s)", "Avg Aggregation Time (s)"
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        # Loop over the number of clusters (k) to search
        for k in tqdm(range(1, args.max_k + 1), desc="Clusters (k)", position=0):
            for nprobe in tqdm(args.nprobe, desc=f"nprobe values (k={k})", position=1, leave=False):
                for batch_size in tqdm(args.batch_size, desc=f"Batch sizes (k={k}, nprobe={nprobe})", position=2, leave=False):
                    for retrieved_docs in tqdm(args.retrieved_docs, desc=f"Retrieved Docs (k={k}, nprobe={nprobe}, batch={batch_size})", position=3, leave=False):
                        for num_threads in tqdm(args.num_threads, desc=f"Num Threads (k={k}, nprobe={nprobe}, batch={batch_size}, retrieved_docs={retrieved_docs})", position=4, leave=False):
                            # Set the number of FAISS threads
                            faiss.omp_set_num_threads(num_threads)
                            
                            query_times = []
                            agg_times = []
                            
                            # Process queries in batches
                            for idx in range(0, len(embeddings), batch_size):
                                batch = embeddings[idx:idx + batch_size]
                                for query in batch:
                                    query = query.reshape(1, -1)
                                    qt, at = perform_split_query(query, split_indices, k, retrieved_docs, nprobe)
                                    query_times.append(qt)
                                    agg_times.append(at)
                            
                            avg_query_time = np.mean(query_times) if query_times else 0
                            avg_agg_time = np.mean(agg_times) if agg_times else 0
                            
                            writer.writerow({
                                "Max Clusters Searched": k,
                                "nprobe": nprobe,
                                "Batch Size": batch_size,
                                "Retrieved Docs": retrieved_docs,
                                "Num Threads": num_threads,
                                "Avg Split Query Time (s)": avg_query_time,
                                "Avg Aggregation Time (s)": avg_agg_time
                            })
                            csvfile.flush()

if __name__ == "__main__":
    main()

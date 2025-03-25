import argparse
import time
import csv
import os
import numpy as np
import faiss
from tqdm import tqdm

def parse_arguments():
    parser = argparse.ArgumentParser(description="Split Query Benchmark")
    parser.add_argument("--index-folder", type=str, required=True, help="Path to the split FAISS index file")
    parser.add_argument("--nprobe", type=int, nargs='+', required=True, help="List of nprobe values for FAISS search")
    parser.add_argument("--batch-size", type=int, nargs='+', required=True, help="List of batch sizes for querying")
    parser.add_argument("--queries", type=str, required=True, help="Path to the NumPy file containing embeddings")
    parser.add_argument("--retrieved-docs", type=int, nargs='+', required=True, help="List of numbers of docs retrieved per query")
    parser.add_argument("--num-threads", type=int, nargs='+', required=True, help="List of numbers of threads to run retrieval")
    parser.add_argument("--dataset-size", type=int, required=True, help="Total number of vectors across all indices (used to adjust document IDs)")
    parser.add_argument("--output-dir", type=str, default="data/profiling/", help="Directory where the results will be saved")
    return parser.parse_args()

def load_split_indices(indices_dir):
    indices_files = sorted([os.path.join(indices_dir, f) for f in os.listdir(indices_dir) if f.endswith(".faiss")])
    split_indices = [faiss.read_index(index_file) for index_file in indices_files]
    num_indices = len(indices_files)
    return split_indices, num_indices

def perform_queries(split_indices, retrieved_docs, embeddings, batch_size, nprobe, dataset_size, num_indices, max_batches=1000):
    retrieval_times = []
    aggregation_times = []

    # Progress bar for the query batches (position=4)

    for idx in tqdm(range(0, min(len(embeddings), batch_size * max_batches), batch_size),
                        desc="Querying batches",
                        leave=False,
                        position=4):
        split_indices_docs = []
        split_indices_distances = []
        retrieval_times = []

        batch = embeddings[idx:idx + batch_size]
        
        for index_num, split_index in tqdm(enumerate(split_indices),
                                           desc="Search Indices",
                                           leave=False,
                                           total=len(split_indices),
                                           position=5):
            split_index.nprobe = nprobe
            
            query_start = time.time()
            split_distances, split_docs = split_index.search(batch, retrieved_docs)
            query_end = time.time()
            retrieval_times.append(query_end - query_start)
            split_indices_docs.extend((split_docs.flatten() + index_num * (dataset_size / num_indices)).tolist())
            split_indices_distances.extend(split_distances.flatten().tolist())
        
        retrieval_times.append(max(retrieval_times))
        split_agg_start_time = time.time()
        sorted_split_indices = np.argsort(split_indices_distances)[-1:-(retrieved_docs + 1):-1]
        split_corresponding_docs = [split_indices_docs[i] for i in sorted_split_indices]
        split_agg_end_time = time.time()
        aggregation_times.append(split_agg_end_time - split_agg_start_time)

        if idx >= batch_size * max_batches:
            break
        
    return sum(retrieval_times) / len(retrieval_times) if retrieval_times else 0, sum(aggregation_times) / len(aggregation_times) if aggregation_times else 0   

def main():
    args = parse_arguments()
    
    # Create the output directory if it doesn't exist and set the output file path.
    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(args.output_dir, "retrieval_split_latency.csv")
    
    # Initially load the index with a dummy nprobe; it will be updated later in the loop.
    indices, num_indices = load_split_indices(args.index_folder)
    embeddings = np.load(args.queries)
    
    with open(output_file, mode='w', newline='') as file:
        fieldnames = ["nprobe", "Batch Size", "Retrieved Docs", "Num Threads", "Avg Retrieval Time (s)", "Avg Aggregation Time (s)"]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        
        for nprobe in tqdm(args.nprobe, desc="nprobe values", position=0):
            for batch_size in tqdm(args.batch_size, desc=f"Batch sizes (nprobe={nprobe})", position=1, leave=False):
                for retrieved_docs in tqdm(args.retrieved_docs, desc=f"Retrieved Docs (nprobe={nprobe}, batch_size={batch_size})", position=2, leave=False):
                    for num_threads in tqdm(args.num_threads, desc=f"Num Threads (nprobe={nprobe}, batch_size={batch_size}, retrieved_docs={retrieved_docs})", position=3, leave=False):
                        faiss.omp_set_num_threads(num_threads)
                        avg_query_time, avg_aggregation_time = perform_queries(indices, retrieved_docs, embeddings, batch_size, nprobe, args.dataset_size, num_indices)
                        writer.writerow({
                            "nprobe": nprobe,
                            "Batch Size": batch_size,
                            "Retrieved Docs": retrieved_docs,
                            "Num Threads": num_threads,
                            "Avg Retrieval Time (s)": avg_query_time,
                            "Avg Aggregation Time (s)": avg_aggregation_time
                        })
                        file.flush()  # Ensure data is written incrementally

if __name__ == "__main__":
    main()

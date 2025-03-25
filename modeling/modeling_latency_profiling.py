import argparse
import time
import csv
import os
import numpy as np
import faiss
from tqdm import tqdm

def parse_arguments():
    parser = argparse.ArgumentParser(description="FAISS Query Benchmark")
    parser.add_argument("--index-folder", type=str, required=True, help="Path to the clustered FAISS index file")
    parser.add_argument("--nprobe", type=int, nargs='+', required=True, help="List of nprobe values for FAISS search")
    parser.add_argument("--batch-size", type=int, nargs='+', required=True, help="List of batch sizes for querying")
    parser.add_argument("--queries", type=str, required=True, help="Path to the NumPy file containing embeddings")
    parser.add_argument("--retrieved-docs", type=int, nargs='+', required=True, help="List of numbers of docs retrieved per query")
    parser.add_argument("--num-threads", type=int, nargs='+', required=True, help="List of numbers of threads to run retrieval")
    parser.add_argument("--output-dir", type=str, default="data/modeling/", help="Directory where the results will be saved")
    return parser.parse_args()

def load_clustered_indices(indices_dir):
    indices_files = sorted([os.path.join(indices_dir, f) for f in os.listdir(indices_dir) if f.endswith(".faiss")])
    clustered_indices = [faiss.read_index(index_file) for index_file in indices_files]
    num_indices = len(indices_files)
    return clustered_indices, num_indices

def  perform_queries(clustered_index, batch_size, retrieved_docs, embeddings, max_batches=1000):
    query_times = []
    
    # Progress bar for the query batches (position=4)
    for idx in tqdm(range(0, min(len(embeddings), batch_size * max_batches), batch_size),
                    desc="Querying batches",
                    leave=False,
                    position=5):
        batch = embeddings[idx:idx + batch_size]
        query_start = time.time()
        _, _ = clustered_index.search(batch, retrieved_docs)
        query_end = time.time()
        query_times.append(query_end - query_start)
        
        if idx >= batch_size * max_batches:
            break
    
    return sum(query_times) / len(query_times) if query_times else 0

def main():
    args = parse_arguments()
    
    # Create the output directory if it doesn't exist and set the output file path.
    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(args.output_dir, "modeling_latency_profiling.csv")
    
    # Initially load the index with a dummy nprobe; it will be updated later in the loop.
    indices, num_indices = load_clustered_indices(os.path.join(args.index_folder, "clusters"))
    embeddings = np.load(args.queries)
    
    with open(output_file, mode='w', newline='') as file:
        fieldnames = ["Cluster ID", "Batch Size", "nprobe", "Retrieved Docs", "Num Threads", "Avg Retrieval Latency (s)"]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        
        for nprobe in tqdm(args.nprobe, desc="nprobe values", position=0):
            for batch_size in tqdm(args.batch_size, desc=f"Batch Size (nprobe={nprobe})", position=1, leave=False):
                for retrieved_docs in tqdm(args.retrieved_docs, desc=f"Retrieved Docs (Batch Size={batch_size}, nprobe={nprobe})", position=2, leave=False):
                    for num_threads in tqdm(args.num_threads, desc=f"Num Threads (Retrieved Docs={retrieved_docs}, Batch Size={batch_size}, nprobe={nprobe})", position=3, leave=False):
                        for index_num, clustered_index in tqdm(enumerate(indices), total=len(indices), desc=f"Index Num (Num Threads={num_threads}, Retrieved Docs={retrieved_docs}, Batch Size={batch_size}, nprobe={nprobe})", position=4, leave=False):
                            faiss.omp_set_num_threads(num_threads)
                            clustered_index.nprobe = nprobe
                            avg_retrieval_time = perform_queries(clustered_index, batch_size, retrieved_docs, embeddings)
                            writer.writerow({
                                "Cluster ID": index_num,
                                "Batch Size": batch_size,
                                "nprobe": nprobe,
                                "Retrieved Docs": retrieved_docs,
                                "Num Threads": num_threads,
                                "Avg Retrieval Latency (s)": avg_retrieval_time
                            })
                            file.flush()  # Ensure data is written incrementally


if __name__ == "__main__":
    main()

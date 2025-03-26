import argparse
import time
import csv
import os
import numpy as np
import faiss
from tqdm import tqdm

def parse_arguments():
    parser = argparse.ArgumentParser(description="FAISS Query Benchmark")
    parser.add_argument("--index-name", type=str, required=True, help="Path to the FAISS index file")
    parser.add_argument("--nprobe", type=int, nargs='+', required=True, help="List of nprobe values for FAISS search")
    parser.add_argument("--batch-size", type=int, nargs='+', required=True, help="List of batch sizes for querying")
    parser.add_argument("--queries", type=str, required=True, help="Path to the NumPy file containing embeddings")
    parser.add_argument("--retrieved-docs", type=int, nargs='+', required=True, help="List of numbers of docs retrieved per query")
    parser.add_argument("--num-threads", type=int, nargs='+', required=True, help="List of numbers of threads to run retrieval")
    parser.add_argument("--output-dir", type=str, default="data/profiling/", help="Directory where the results will be saved")
    return parser.parse_args()

def load_faiss_index(index_name, nprobe):
    index = faiss.read_index(index_name)
    index.nprobe = nprobe
    return index

def perform_queries(index, retrieved_docs, embeddings, batch_size, max_batches=1000):
    query_times = []
    
    # Progress bar for the query batches (position=4)
    for idx in tqdm(range(0, len(embeddings), batch_size),
                    desc="Querying batches",
                    leave=False,
                    position=4):
        batch = embeddings[idx:idx + batch_size]
        query_start = time.time()
        _, _ = index.search(batch, retrieved_docs)
        query_end = time.time()
        query_times.append(query_end - query_start)
        
        if idx >= batch_size * max_batches:
            break
    
    return sum(query_times) / len(query_times) if query_times else 0

def main():
    args = parse_arguments()
    
    # Create the output directory if it doesn't exist and set the output file path.
    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(args.output_dir, "retrieval_monolithic_latency.csv")
    
    # Initially load the index with a dummy nprobe; it will be updated later in the loop.
    index = load_faiss_index(args.index_name, args.nprobe[0])
    embeddings = np.load(args.queries)
    
    with open(output_file, mode='w', newline='') as file:
        fieldnames = ["Index Name", "nprobe", "Batch Size", "Retrieved Docs", "Num Threads", "Avg Retrieval Time (s)"]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        
        for nprobe in tqdm(args.nprobe, desc="nprobe values", position=0):
            index.nprobe = nprobe  # Update the index's nprobe
            for batch_size in tqdm(args.batch_size, desc=f"Batch sizes (nprobe={nprobe})", position=1, leave=False):
                for retrieved_docs in tqdm(args.retrieved_docs, desc=f"Retrieved Docs (nprobe={nprobe}, batch_size={batch_size})", position=2, leave=False):
                    for num_threads in tqdm(args.num_threads, desc=f"Num Threads (nprobe={nprobe}, batch_size={batch_size}, retrieved_docs={retrieved_docs})", position=3, leave=False):
                        # Set the FAISS thread count
                        faiss.omp_set_num_threads(num_threads)
                        # Measure the average query time for the current combination
                        avg_query_time = perform_queries(index, retrieved_docs, embeddings, batch_size)
                        writer.writerow({
                            "Index Name": args.index_name,
                            "nprobe": nprobe,
                            "Batch Size": batch_size,
                            "Retrieved Docs": retrieved_docs,
                            "Num Threads": num_threads,
                            "Avg Retrieval Time (s)": avg_query_time
                        })
                        file.flush()  # Ensure data is written incrementally

if __name__ == "__main__":
    main()

import argparse
import time
import numpy as np
import faiss


def parse_arguments():
    parser = argparse.ArgumentParser(description="FAISS Query Benchmark")
    parser.add_argument("--index-name", type=str, required=True, help="Path to the FAISS index file")
    parser.add_argument("--nprobe", type=int, required=True, help="Number of probes to use for the FAISS search")
    parser.add_argument("--batch-size", type=int, required=True, help="Batch size for querying")
    parser.add_argument("--queries", type=str, required=True, help="Path to the NumPy file containing embeddings")
    parser.add_argument("--retrieved-docs", type=int, required=True, help="Number of docs retrieved per query")
    parser.add_argument("--num-threads", type=int, required=True, help="Number of threads to run retrieval")
    return parser.parse_args()


def load_faiss_index(index_name, nprobe):
    index = faiss.read_index(index_name)
    index.nprobe = nprobe
    return index


def perform_queries(index, retrieved_docs, embeddings, batch_size, max_batches=10000):
    query_times = []
    
    for idx in range(0, len(embeddings), batch_size):
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
    index = load_faiss_index(args.index_name, args.nprobe)
    embeddings = np.load(args.queries)
    avg_query_time = perform_queries(index, args.retrieved_docs, embeddings, args.batch_size)
    
    faiss.omp_set_num_threads(args.num_threads)
    
    print(f"Index: {args.index_name}, nprobe: {args.nprobe}, Batch Size: {args.batch_size}, Query Time: {avg_query_time:.6f} sec")


if __name__ == "__main__":
    main()
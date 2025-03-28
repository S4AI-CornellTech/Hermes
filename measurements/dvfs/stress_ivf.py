import argparse
import numpy as np
import faiss
import time

def main(args):
    # Set number of threads for FAISS
    NUM_THREADS = 32
    faiss.omp_set_num_threads(NUM_THREADS)

    # Load query encodings from file
    query_encodings = np.load(args.queries)
    # print('Query encodings shape:', query_encodings.shape)

    # Load FAISS index from file
    # print('Loading index from:', args.index)
    index = faiss.read_index(args.index)
    print('Index Loaded')

    # print('Index total vectors:', index.ntotal)
    # num_clusters = index.invlists.nlist
    # print('Number of clusters (nlist):', num_clusters, 'Computed value:', (num_clusters / 4) ** 2)

    # Set number of nearest neighbors to retrieve
    k = 5
    
    # Set nprobe (number of clusters to search)
    index.nprobe = int(args.nprobe)
    # print('nprobe set to:', index.nprobe)

    # Run the search multiple times and time each run
    RUNS = 10
    times = []

    for run in range(RUNS):
        # print(f'RUN {run}')
        start_time = time.time()
        distances, indices = index.search(query_encodings, k)
        elapsed = time.time() - start_time
        times.append(elapsed)
        # print(f'Elapsed time: {elapsed:.4f} seconds')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FAISS Index Search Script")
    parser.add_argument("--queries", type=str, help="Path to the numpy file containing query encodings")
    parser.add_argument("--index", type=str, help="Path to the FAISS index file")
    parser.add_argument("--nprobe", type=str, help="nprobe value")
    args = parser.parse_args()
    main(args)
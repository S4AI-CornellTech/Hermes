#!/usr/bin/env python3
import argparse
import math
import numpy as np
import faiss
from tqdm import tqdm
import multiprocessing
import os

# Fixed number of vectors generated per batch.
NUM_VECTORS_PER_BATCH = 100_000

def parse_total_index_size(size_str):
    """
    Parse a string representing the total number of vectors.
    Accepts formats ending with:
      - 'k' for thousand,
      - 'm' for million,
      - 'b' for billion.
      
    Examples:
      "1k" -> 1,000
      "10m" -> 10,000,000
      "1b" -> 1,000,000,000
    """
    size_str = size_str.lower().strip()
    if size_str.endswith("k"):
        multiplier = 10**3
        number_part = size_str[:-1]
    elif size_str.endswith("m"):
        multiplier = 10**6
        number_part = size_str[:-1]
    elif size_str.endswith("b"):
        multiplier = 10**9
        number_part = size_str[:-1]
    else:
        raise ValueError("Index size must end with 'k' (thousand), 'm' (million) or 'b' (billion).")
    
    try:
        number = float(number_part)
    except ValueError:
        raise ValueError("Invalid number provided for the index size.")
    
    return int(number * multiplier)

def generate_vectors(num_vectors, dim, queue):
    """
    Generate a batch of random vectors and put them in a multiprocessing queue.
    
    Parameters:
      num_vectors (int): Number of vectors to generate.
      dim (int): Dimensionality of each vector.
      queue (multiprocessing.Queue): Queue to store the generated vectors.
    """
    # Generate random vectors uniformly distributed between -1 and 1.
    vectors = np.random.uniform(low=-1.0, high=1.0, size=(num_vectors, dim)).astype('float32')
    queue.put(vectors)

def create_faiss_index(total_vectors, dim, num_workers, num_vectors_per_batch):
    """
    Create and populate a FAISS index with randomly generated vectors.
    
    Parameters:
      total_vectors (int): Total number of vectors to add to the index.
      dim (int): Dimensionality of each vector.
      num_workers (int): Number of parallel worker processes for vector generation.
      num_vectors_per_batch (int): Number of vectors generated per batch.
      
    Returns:
      faiss.IndexIVFScalarQuantizer: The populated FAISS index.
    """
    # --- FAISS Index Configuration ---
    # Calculate the number of lists based on the total number of vectors.
    C = 1  
    nlists = C * int(np.sqrt(total_vectors))
    
    # Create the quantizer and the index.
    quantizer = faiss.IndexFlatIP(dim)
    index = faiss.IndexIVFScalarQuantizer(
        quantizer, dim, nlists, faiss.ScalarQuantizer.QT_8bit, faiss.METRIC_INNER_PRODUCT
    )
    
    # --- Index Training ---
    # Set the training size to the square root of the total number of vectors.
    initial_train_size = int(math.sqrt(total_vectors))
    print(f"Training the FAISS index with {initial_train_size} vectors (sqrt(total_vectors))...")
    initial_vectors = np.random.uniform(low=-1.0, high=1.0, size=(initial_train_size, dim)).astype('float32')
    index.train(initial_vectors)
    
    # --- Parallel Vector Generation and Insertion ---
    # Calculate the number of batches required.
    num_batches = math.ceil(total_vectors / num_vectors_per_batch)
    queue = multiprocessing.Queue(maxsize=num_workers)
    processes = []
    
    print("Adding vectors to the FAISS index in parallel batches...")
    with tqdm(total=num_batches, desc="Batches Processed", unit="batch") as pbar:
        for batch in range(num_batches):
            # If fewer than num_workers processes are running, start a new one.
            if len(processes) < num_workers:
                p = multiprocessing.Process(target=generate_vectors, args=(num_vectors_per_batch, dim, queue))
                p.start()
                processes.append(p)
            
            # Retrieve the generated vectors from the queue and add them to the index.
            vectors = queue.get()
            index.add(vectors)
            pbar.update(1)
            
            # Clean up finished processes.
            processes = [p for p in processes if p.is_alive()]
        
        # Ensure all remaining processes have finished.
        for p in processes:
            p.join()
    
    return index

def main():
    # --- Command Line Argument Parsing ---
    parser = argparse.ArgumentParser(description="FAISS Index Generation Script")
    parser.add_argument(
        "--index-size", type=str, required=True,
        help="Total number of vectors in the index (e.g., 1k, 1m, 10m, 100m, 1b, 10b)"
    )
    parser.add_argument(
        "--dim", type=int, required=True,
        help="Dimensionality of each vector (default: 768)"
    )
    parser.add_argument(
        "--threads", type=int, default=5,
        help="Number of FAISS threads to use (default: 5)"
    )
    parser.add_argument(
        "--output-dir", type=str, default="hermes/indices/synthetic_monolithic_indices",
        help="Directory where the indices will be saved (default: hermes/indices/synthetic_monolithic_indices)"
    )
    args = parser.parse_args()
    
    # Parse the total index size from the command line argument.
    total_vectors = parse_total_index_size(args.index_size)
    
    # Display the configuration.
    print("FAISS Index Generation Parameters:")
    print(f"  Total vectors       : {total_vectors}")
    print(f"  Vector dimension    : {args.dim}")
    print(f"  Vectors per batch   : {NUM_VECTORS_PER_BATCH}")
    print(f"  Number of workers   : {args.threads}")
    
    # Create and populate the FAISS index.
    index = create_faiss_index(total_vectors, args.dim, args.threads, NUM_VECTORS_PER_BATCH)
    
    # Define the index file path.
    index_filename = f"{args.output_dir}/synthetic_index_monolithic_{args.index_size}.faiss"
    
    # Ensure the output directory exists; if not, create it.
    output_dir = os.path.dirname(index_filename)
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the populated FAISS index to disk.
    faiss.write_index(index, index_filename)
    print(f"FAISS index saved to {index_filename}")


if __name__ == "__main__":
    main()

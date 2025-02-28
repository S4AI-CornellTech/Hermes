#!/usr/bin/env python3
import argparse
import math
import os
import numpy as np
import faiss
from datasets import load_dataset
from tqdm import tqdm

# Mapping from index size to dataset name and default limits.
DATASET_MAPPING = {
    "100k": "mohdumar/SPHERE_100K",
    "100m": "mohdumar/SPHERE_100M",
    "899m": "mohdumar/SPHERE_899M"
}
DEFAULT_LIMITS = {
    "100k": 100_000,
    "100m": 1_000_000,
    "899m": 899_000_000
}

def create_faiss_index_for_block(train_vectors, dim):
    """
    Create and train a FAISS index for a block using the provided training vectors.
    
    Parameters:
      train_vectors (list): A list of vectors (each a list or array) used for training.
      dim (int): Dimensionality of the vectors.
      
    Returns:
      faiss.IndexIVFScalarQuantizer: The trained FAISS index.
    """
    # Create a flat inner product quantizer.
    quantizer = faiss.IndexFlatIP(dim)
    # Initialize the IVF-SQ8 index with 8 lists (as in the original code).
    ivf_index = faiss.IndexIVFScalarQuantizer(
        quantizer, dim, 8, faiss.ScalarQuantizer.QT_8bit, faiss.METRIC_INNER_PRODUCT
    )
    ivf_index.nprobe = 64
    ivf_index.own_fields = True
    quantizer.this.disown()
    
    # Convert training vectors to numpy array and train the index.
    train_vecs = np.array(train_vectors, dtype="float32")
    print(f"Training index with {len(train_vecs)} vectors")
    ivf_index.train(train_vecs)
    return ivf_index

def create_indices(args):
    # Determine dataset name and total document limit.
    dataset_name = DATASET_MAPPING[args.index_size]
    block_size = 100_000
    batch_size = args.dataset_size / 100

    # Set FAISS number of threads.
    faiss.omp_set_num_threads(args.threads)

    print(f"Loading dataset '{dataset_name}' in streaming mode...")
    # Create two independent streaming iterators.
    train_dataset = load_dataset(dataset_name, split="train", streaming=True)
    build_dataset = load_dataset(dataset_name, split="train", streaming=True)
    train_iter = iter(train_dataset)
    build_iter = iter(build_dataset)

    # Collect training data for each block.
    print("Collecting training data for each block...")
    train_lists = []
    for block in range(num_blocks):
        train_list = []
        for _ in range(20_000):
            vec = next(train_iter)["vector"]
            train_list.append(vec)
        train_lists.append(train_list)
        # Skip the next 80K vectors for diversity.
        for _ in range(80_000):
            next(train_iter)

    # Process each block: build the index, add vectors in batches, and save.
    for block in range(num_blocks):
        start_idx = block * block_size
        end_idx = (block + 1) * block_size
        print(f"\nBuilding index for block {block+1} (documents {start_idx} to {end_idx})")
        
        # Create and train the index for this block.
        index = create_faiss_index_for_block(train_lists[block], 768)
        
        # Add vectors to the index in batches.
        print("Adding vectors to the index...")
        added_vecs = 0
        batch_vecs = []
        for i in range(start_idx, end_idx):
            if i >= limit:
                break
            vec = next(build_iter)[args.column]
            batch_vecs.append(vec)
            if len(batch_vecs) == batch_size or i == end_idx - 1:
                vecs = np.array(batch_vecs, dtype="float32")
                index.add(vecs)
                added_vecs += len(vecs)
                print(f"Added {added_vecs} vectors to the index")
                batch_vecs = []
        
        # Ensure the output directory exists.
        os.makedirs(args.output_dir, exist_ok=True)
        fpath = os.path.join(args.output_dir, f"ivf_sq8_{start_idx}_{end_idx}.faiss")
        print(f"Saving index to {fpath}")
        faiss.write_index(index, fpath)

def main():
    parser = argparse.ArgumentParser(
        description="Build FAISS IVF-SQ8 indices on Hugging Face streaming datasets."
    )
    parser.add_argument(
        "--dataset-size", type=str, required=True, choices=["100k", "100m", "899m"],
        help="Dataset to use. Choices: 100k, 100m, 899m"
    )
    parser.add_argument(
        "--threads", type=int, default=5,
        help="Number of FAISS threads to use (default: 5)"
    )
    parser.add_argument(
        "--output-dir", type=str, default="index/hermes_indices",
        help="Directory where the indices will be saved (default: index/hermes_indices/)"
    )
    parser.add
    args = parser.parse_args()

    create_indices(args)

if __name__ == "__main__":
    main()

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
DATASET_VALUES = {
    "100k": 100_000,
    "100m": 1_000_000,
    "899m": 899_000_000
}

def create_faiss_index_for_index(train_vectors, dim, index_size, train_size):
    """
    Create and train a FAISS index using the provided training vectors.
    
    Parameters:
      train_vectors (list): A list of vectors used for training.
      dim (int): Dimensionality of the vectors.
      
    Returns:
      faiss.IndexIVFScalarQuantizer: The trained FAISS index.
    """
    
    nlists = int(np.sqrt(index_size))
    
    quantizer = faiss.IndexFlatIP(dim)
    faiss_index = faiss.IndexIVFScalarQuantizer(
        quantizer, dim, nlists, faiss.ScalarQuantizer.QT_8bit, faiss.METRIC_INNER_PRODUCT
    )
    faiss_index.own_fields = True
    quantizer.this.disown()
    
    train_vecs = np.array(train_vectors, dtype="float32")
    faiss_index.train(train_vecs)
    return faiss_index

def create_indices(args):
    dataset_name = DATASET_MAPPING[args.dataset_size]
    index_size = int(DATASET_VALUES[args.dataset_size] / args.num_indices)
    batch_size = int(DATASET_VALUES[args.dataset_size] / 100)
    train_size = int(index_size / 10)

    print(f"Loading dataset '{dataset_name}'...")
    train_dataset = load_dataset(dataset_name, split="train", streaming=args.dataset_streaming)
    build_dataset = load_dataset(dataset_name, split="train", streaming=args.dataset_streaming)
    train_iter = iter(train_dataset)
    build_iter = iter(build_dataset)

    print("Collecting training data for each index...")
    train_lists = []
    for idx in tqdm(range(args.num_indices), desc="Collecting training data"):
        train_list = []
        for _ in range(train_size):
            vec = next(train_iter)["vector"]
            train_list.append(vec)
        train_lists.append(train_list)
        # Skip the remaining vectors for this index.
        for _ in range(index_size - (train_size)):
            next(train_iter)

    # Process each index: build the FAISS index, add vectors in batches, and save.
    for idx in range(args.num_indices):
        start_idx = idx * index_size
        end_idx = (idx + 1) * index_size

        # Create and train the FAISS index for this partition.
        faiss_index = create_faiss_index_for_index(train_lists[idx], 768, index_size, train_size)
        
        print(f"Trained index {idx} with {len(train_lists[idx])} vectors.")
        
        # Add vectors to the FAISS index using a tqdm progress bar.
        with tqdm(total=index_size, desc=f"Adding vectors to index {idx+1}") as pbar:
            batch_vecs = []
            for i in range(start_idx, end_idx):
                vec = next(build_iter)["vector"]
                batch_vecs.append(vec)
                if len(batch_vecs) == batch_size or i == end_idx - 1:
                    vecs = np.array(batch_vecs, dtype="float32")
                    faiss_index.add(vecs)
                    pbar.update(len(batch_vecs))
                    batch_vecs = []
        
        os.makedirs(args.output_dir, exist_ok=True)
        fpath = os.path.join(args.output_dir, f"hermes_index_split_{args.dataset_size}_{idx}.faiss")
        faiss.write_index(faiss_index, fpath)

def main():
    parser = argparse.ArgumentParser(
        description="Build FAISS IVF-SQ8 indices on Hugging Face streaming datasets."
    )
    parser.add_argument(
        "--dataset-size", type=str, required=True, choices=["100k", "100m", "899m"],
        help="Dataset to use. Choices: 100k, 100m, 899m"
    )
    parser.add_argument(
        "--output-dir", type=str, default="hermes/indices/split_indices",
        help="Directory where the indices will be saved (default: hermes/indices/split_indices)"
    )
    parser.add_argument(
        "--num-indices", type=int, default=10,
        help="Number of indices to create"
    )
    parser.add_argument(
        "--dataset-streaming", type=bool, default=False,
        help="Enable dataset streaming to avoid loading the entire dataset into memory (default: False)"
    )
    args = parser.parse_args()

    create_indices(args)

if __name__ == "__main__":
    main()

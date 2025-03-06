#!/usr/bin/env python3
import argparse
import math
import numpy as np
import faiss
from tqdm import tqdm
from datasets import load_dataset
import os

# Fixed number of vectors to add per batch.
NUM_VECTORS_PER_BATCH = 100_000

# Mapping of allowed index sizes to dataset names.
DATASET_MAPPING = {
    "100k": "mohdumar/SPHERE_100K",
    "100m": "mohdumar/SPHERE_100M",
    "899m": "mohdumar/SPHERE_899M",
}

def create_faiss_index_from_dataset(vectors, dim):
    """
    Create and populate a FAISS index using vectors from a Hugging Face dataset.
    
    Parameters:
      vectors (np.ndarray): Array of vectors with shape (total_vectors, dim).
      dim (int): Dimensionality of each vector.
      
    Returns:
      faiss.IndexIVFScalarQuantizer: The populated FAISS index.
    """
    total_vectors = vectors.shape[0]
    
    # --- FAISS Index Configuration ---
    # Number of lists is set to C * sqrt(total_vectors) (C is set to 1).
    C = 1  
    nlists = C * int(np.sqrt(total_vectors))
    quantizer = faiss.IndexFlatIP(dim)
    index = faiss.IndexIVFScalarQuantizer(
        quantizer, dim, nlists, faiss.ScalarQuantizer.QT_8bit, faiss.METRIC_INNER_PRODUCT
    )
    
    # --- Index Training ---
    # Use the first sqrt(total_vectors) vectors from the dataset to train.
    initial_train_size = int(total_vectors / 10)
    print(f"Training the FAISS index with {initial_train_size} vectors from the dataset...")
    index.train(vectors[:initial_train_size])
    
    # --- Adding Dataset Vectors ---
    print("Adding dataset vectors to the FAISS index in batches...")
    for i in tqdm(range(0, total_vectors, NUM_VECTORS_PER_BATCH), desc="Batches Processed", unit="batch"):
        batch = np.array(vectors[i : i + NUM_VECTORS_PER_BATCH])
        index.add(batch)
    return index

def main():
    parser = argparse.ArgumentParser(
        description="FAISS Index Generation Script using Hugging Face datasets"
    )
    parser.add_argument(
        "--index-size", type=str, required=True,
        help="Index size. Allowed values: 100k, 100m, 899m"
    )
    parser.add_argument(
        "--output-dir", type=str, default="index/indices/monolithic_indices",
        help="Directory where the indices will be saved (default: index/indices/monolithic_indices)"
    )
    parser.add_argument(
        "--dataset-streaming", type=bool, default=False,
        help="Enable dataset streaming to avoid loading the entire dataset into memory (default: False)"
    )
    args = parser.parse_args()

    index_size = args.index_size.lower()
    if index_size not in DATASET_MAPPING:
        raise ValueError("Invalid index size. Choose one of: 100k, 100m, 899m")
    
    dataset_name = DATASET_MAPPING[index_size]
    
    # --- Load the Hugging Face Dataset ---
    print(f"Loading Hugging Face dataset: {dataset_name} ...")
    dataset = load_dataset(dataset_name, split="train", streaming=args.dataset_streaming)
    
    # Check that the dataset contains the expected 'vector' column.
    if "vector" not in dataset.column_names:
        raise ValueError("The dataset does not contain an 'vector' column. Please verify the dataset fields.")
    
    # Convert the list of vectors to a numpy array.
    vectors = np.array(dataset["vector"], dtype="float32")
    total_vectors = vectors.shape[0]
    print(f"Dataset loaded. Total vectors: {total_vectors}")
    
    if vectors.shape[1] != 768:
        raise ValueError(f"Provided dimension {768} does not match vector dimension {vectors.shape[1]} in the dataset.")
    
    # --- Create and Populate the FAISS Index ---
    index = create_faiss_index_from_dataset(vectors, 768)
    
    # Define the index file path.
    index_filename = f"{args.output_dir}/hermes_index_monolithic_{index_size}.faiss"
    
    # Ensure the output directory exists.
    output_dir = os.path.dirname(index_filename)
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the populated FAISS index to disk.
    faiss.write_index(index, index_filename)
    print(f"FAISS index saved to {index_filename}")

if __name__ == "__main__":
    main()

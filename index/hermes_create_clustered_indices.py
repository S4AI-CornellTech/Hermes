#!/usr/bin/env python3
import argparse
import os
import math
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


def collect_vectors(args):
    """
    Load the dataset in streaming mode and collect the first `limit` vectors.
    """
    print(f"Loading dataset '{DATASET_MAPPING[args.dataset_size]}'...")
    dataset = load_dataset(DATASET_MAPPING[args.dataset_size], split="train", streaming=args.dataset_streaming)
    stream_iter = iter(dataset)
    all_vectors = []
    for _ in tqdm(range(DATASET_VALUES[args.dataset_size]), desc="Collecting vectors"):
        vec = next(stream_iter)["vector"]
        all_vectors.append(vec)
    all_vectors = np.array(all_vectors, dtype="float32")
    print(f"Collected {len(all_vectors)} vectors")
    return all_vectors

def perform_kmeans(vectors, dim, n_clusters, niter=20):
    """
    Perform KMeans clustering on the given vectors.
    """
    print(f"Performing KMeans clustering with {n_clusters} clusters and {niter} iterations")
    kmeans = faiss.Kmeans(dim, n_clusters, niter=niter, verbose=True)
    kmeans.train(vectors)
    return kmeans

def save_centroids(centroids, output_path):
    """
    Save the KMeans centroids to disk.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.save(output_path, centroids)
    print(f"Saved KMeans centroids to {output_path}")

def assign_clusters(kmeans, vectors):
    """
    Assign each vector to its nearest cluster.
    """
    distances, cluster_assignments = kmeans.index.search(vectors, 1)
    return cluster_assignments

def save_cluster_indices(cluster_assignments, n_clusters, output_dir):
    """
    Save the indices for each cluster into separate numpy files.
    """
    cluster_indices = {cluster_id: [] for cluster_id in range(n_clusters)}
    for idx, cluster_id in enumerate(cluster_assignments.flatten()):
        cluster_indices[cluster_id].append(idx)
    for cluster_id, indices in cluster_indices.items():
        if not indices:
            print(f"Skipping empty cluster {cluster_id}")
            continue
        indices_fpath = os.path.join(output_dir, f"cluster_{cluster_id}_indices.npy")
        os.makedirs(os.path.dirname(indices_fpath), exist_ok=True)
        np.save(indices_fpath, np.array(indices))
        print(f"Saved indices for cluster {cluster_id} to {indices_fpath}")

def build_faiss_index_for_cluster(cluster_id, cluster_vectors, dim, batch_size, output_dir):
    """
    Build a FAISS index for the given cluster.
    """
    print(f"Building index for cluster {cluster_id}")
    if len(cluster_vectors) == 0:
        print(f"Skipping empty cluster {cluster_id}")
        return

    nlists = int(math.sqrt(len(cluster_vectors)))
    train_size = len(cluster_vectors) // 10

    quantizer = faiss.IndexFlatIP(dim)
    ivf_index = faiss.IndexIVFScalarQuantizer(
        quantizer, dim, nlists, faiss.ScalarQuantizer.QT_8bit, faiss.METRIC_INNER_PRODUCT
    )
    ivf_index.own_fields = True
    quantizer.this.disown()
    index = ivf_index

    # Train the index on all cluster vectors (using all available vectors)
    print(f"Training index for cluster {cluster_id} with {train_size} vectors")
    index.train(cluster_vectors[:train_size])

    # Add vectors to the index in batches using a tqdm progress bar
    with tqdm(total=len(cluster_vectors), desc=f"Adding vectors to index for cluster {cluster_id}") as pbar:
        batch_vecs = []
        for i in range(len(cluster_vectors)):
            batch_vecs.append(cluster_vectors[i])
            if len(batch_vecs) == batch_size or i == len(cluster_vectors) - 1:
                vecs = np.array(batch_vecs, dtype="float32")
                index.add(vecs)
                pbar.update(len(batch_vecs))
                batch_vecs = []

    # Save the index to disk
    index_fpath = os.path.join(output_dir, f"ivf_sq8_cluster_{cluster_id}.faiss")
    os.makedirs(os.path.dirname(index_fpath), exist_ok=True)
    print(f"Saving index for cluster {cluster_id} to {index_fpath}")
    faiss.write_index(index, index_fpath)

def main():
    parser = argparse.ArgumentParser(
        description="Build FAISS indices using KMeans clustering on a streaming dataset."
    )
    parser.add_argument(
        "--dataset-size", type=str, required=True, choices=["100k", "100m", "899m"],
        help="Dataset to use. Choices: 100k, 100m, 899m"
    )
    parser.add_argument(
        "--num-indices", type=int, default=10,
        help="Number of KMeans clusters (default: 10)"
    )
    parser.add_argument(
        "--centroids-output", type=str,
        default="hermes/indices/hermes_clusters/kmeans_centroids.npy",
        help="Path to save KMeans centroids (default: hermes/indices/hermes_clusters/kmeans_centroids.npy)"
    )
    parser.add_argument(
        "--cluster-indices-dir", type=str,
        default="hermes/indices/hermes_clusters/cluster_indices",
        help="Directory to save cluster indices (default: hermes/indices/hermes_clusters/cluster_indices)"
    )
    parser.add_argument(
        "--clusters-output-dir", type=str,
        default="hermes/indices/hermes_clusters/clusters",
        help="Directory to save FAISS indices for clusters (default: hermes/indices/hermes_clusters/clusters)"
    )
    parser.add_argument(
        "--niter", type=int, default=20,
        help="Number of iterations for KMeans clustering (default: 20)"
    )
    parser.add_argument(
        "--dataset-streaming", type=bool, default=False,
        help="Enable dataset streaming to avoid loading the entire dataset into memory (default: False)"
    )
    args = parser.parse_args()

    # Step 1: Collect vectors from the dataset
    all_vectors = collect_vectors(args)

    # Step 2: Perform KMeans clustering
    dim = 768
    kmeans = perform_kmeans(all_vectors, dim, args.num_indices, niter=args.niter)

    # Step 3: Save KMeans centroids for later use
    save_centroids(kmeans.centroids, args.centroids_output)

    # Step 4: Assign each vector to its nearest cluster
    cluster_assignments = assign_clusters(kmeans, all_vectors)

    # Step 5: Save indices of vectors for each cluster
    save_cluster_indices(cluster_assignments, args.num_indices, args.cluster_indices_dir)

    # Step 6: Build a FAISS index for each cluster
    batch_size =  int(DATASET_VALUES[args.dataset_size] / 100)
    for cluster_id in range(args.num_indices):
        cluster_vectors = all_vectors[cluster_assignments.flatten() == cluster_id]
        build_faiss_index_for_cluster(cluster_id, cluster_vectors, dim, batch_size, args.clusters_output_dir)

if __name__ == "__main__":
    main()

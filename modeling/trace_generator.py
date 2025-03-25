import os
import csv
import argparse
import numpy as np
import torch
import faiss
from tqdm import tqdm

def load_faiss_indices(directory: str, num_clusters: int):
    """Load FAISS indices from the specified directory."""
    return [faiss.read_index(os.path.join(directory, f'ivf_sq8_cluster_{i}.faiss')) for i in range(num_clusters)]

def rank_clusters(query, cluster_indices, small_nprobe, small_ndr):
    """Rank clusters based on small_nprobe search distances."""
    distances = []
    
    for cluster_index in cluster_indices:
        cluster_index.nprobe = small_nprobe
        cluster_distances, _ = cluster_index.search(query, small_ndr)
        avg_distance = np.mean(cluster_distances.flatten())
        distances.append(avg_distance)
    
    # Rank clusters in descending order of average distance
    return sorted(range(len(distances)), key=lambda x: distances[x], reverse=True)

def main():
    parser = argparse.ArgumentParser(description="Cluster ranking script")
    parser.add_argument("--cluster-indices-dir", type=str, default='data/indices/hermes_clusters/clusters', help="Path to the directory containing FAISS cluster indices")
    parser.add_argument("--embeddings-path", type=str, default='triviaqa/triviaqa_encodings.npy', help="Path to the embeddings numpy file")
    parser.add_argument("--output-folder", type=str, default='data/modeling', help="Path to save the output CSV file")
    args = parser.parse_args()
    
    torch.set_grad_enabled(False)  # Disable gradients for efficiency
    
    # Dynamically determine the number of clusters by counting the FAISS files in the directory
    cluster_files = [f for f in os.listdir(args.cluster_indices_dir)
                     if f.startswith('ivf_sq8_cluster_') and f.endswith('.faiss')]
    num_clusters = len(cluster_files)
    if num_clusters == 0:
        raise ValueError("No FAISS index files found in the specified directory.")
    
    batch_size = 1
    small_nprobe = 8
    small_ndr = 1
    
    # Load FAISS indices
    cluster_indices = load_faiss_indices(args.cluster_indices_dir, num_clusters)
    
    # Load embeddings
    embeddings = np.load(args.embeddings_path)
    
    # Process queries and write results
    output_file = os.path.join(args.output_folder, 'cluster_trace.csv')
    with open(output_file, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['Query', 'Ranked Clusters'])
        
        for query_id, start_idx in enumerate(tqdm(range(0, len(embeddings), batch_size), desc="Processing queries")):
            query = embeddings[start_idx:start_idx + batch_size]
            ranked_clusters = rank_clusters(query, cluster_indices, small_nprobe, small_ndr)
            # Optionally, if you want to limit the output to a fixed number, you can slice the list (e.g., ranked_clusters[:10])
            writer.writerow([query_id, ranked_clusters])
            csv_file.flush()
    
if __name__ == "__main__":
    main()

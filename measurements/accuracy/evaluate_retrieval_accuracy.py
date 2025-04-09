from faiss import read_index
import numpy as np
import faiss
from tqdm import tqdm
import os
import csv
import argparse

def calculate_recall(retrieved_indices: np.ndarray, ground_truth_indices: np.ndarray) -> float:
    matched_elements = np.isin(retrieved_indices, ground_truth_indices)
    num_matches = np.sum(matched_elements)
    recall = float(num_matches / len(ground_truth_indices))  # Recall calculation
    return recall

def calculate_dcg(retrieved_indices: np.ndarray, ground_truth_indices: np.ndarray) -> float:
    dcg = 0.0
    for i, index in enumerate(retrieved_indices):
        if index in ground_truth_indices:
            rank = i + 1
            dcg += 1 / np.log2(rank + 1)
    return dcg

def calculate_ndcg(retrieved_indices: np.ndarray, ground_truth_indices: np.ndarray) -> float:
    dcg = calculate_dcg(retrieved_indices, ground_truth_indices)
    idcg = calculate_dcg(ground_truth_indices, ground_truth_indices)
    if idcg == 0:
        return 0.0
    return dcg / idcg

def main():
    parser = argparse.ArgumentParser(description="FAISS Query Benchmark")
    parser.add_argument("--flat-index", type=str, required=True, help="Path to the flat FAISS index file")
    parser.add_argument("--monolithic-index", type=str, required=True, help="Path to the monolithic FAISS index file")
    parser.add_argument("--split-index-folder", type=str, required=True, help="Path to the split FAISS index folder")
    parser.add_argument("--split-index-size", type=int, required=True, help="Total number of vectors across all indices (used to adjust document IDs)")
    parser.add_argument("--cluster-index-folder", type=str, required=True, help="Path to the clustered FAISS index folder")
    parser.add_argument("--cluster-index-indices-folder", type=str, required=True, help="Path to the clustered FAISS index folder")
    parser.add_argument("--monolithic-nprobe", type=int, required=True, help="nprobe value for monolithic index")
    parser.add_argument("--deep-nprobe", type=int, nargs='+', required=True, help="List of deep nprobe values for FAISS search")
    parser.add_argument("--sample-nprobe", type=int, nargs='+', required=True, help="List of nprobe values for FAISS search")
    parser.add_argument("--retrieved-docs", type=int, nargs='+', required=True, help="List of numbers of docs retrieved per query")
    parser.add_argument("--queries", type=str, required=True, help="Path to the NumPy file containing embeddings")
    parser.add_argument("--output-dir", type=str, default="data/", help="Directory where the results will be saved")
    args = parser.parse_args()

    flat_index = read_index(args.flat_index)

    monolithic_index = read_index(args.monolithic_index)
    
    split_indices_files = sorted([os.path.join(args.split_index_folder, f) for f in os.listdir(args.split_index_folder) if f.endswith(".faiss")])
    split_indices = [faiss.read_index(index_file) for index_file in split_indices_files]
    num_split_indices = len(split_indices_files)

    cluster_indices_files = sorted([os.path.join(args.cluster_index_folder, f) for f in os.listdir(args.cluster_index_folder) if f.endswith(".faiss")])
    cluster_indices = [faiss.read_index(index_file) for index_file in cluster_indices_files]
    num_cluster_indices = len(cluster_indices_files)

    embeddings_array = np.load(args.queries)

    with open(os.path.join(args.output_dir, 'accuracy_eval.csv'), mode='w', newline='') as output_file:
        fieldnames = [
            "Number of Clusters Searched",
            "Sample nProbe",
            "nProbe",
            "Monolithic Recall",
            "Split Recall",
            "Cluster Recall",
            "Monolithic NDCG",
            "Split NDCG",
            "Cluster NDCG",
        ]
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        
        for sample_nProbe in tqdm(args.sample_nprobe, desc=f"Sample nProbe", position=0, leave=False):
            for nProbe in tqdm(args.deep_nprobe, desc=f"nProbe (Sample nProbe={sample_nProbe})", position=1, leave=False):
                for retrieved_docs in tqdm(args.retrieved_docs, desc=f"Retrieved Docs (Sample nProbe={sample_nProbe}, nprobe={nProbe})", position=2, leave=False):
                    monolithic_index.nprobe = args.monolithic_nprobe
                    
                    monolithic_ndcgs, monolithic_recalls = [], []
                    split_ndcgs, split_recalls = [], []
                    cluster_ndcgs, cluster_recalls = [], []
                    
                    for clusters_searched in tqdm(range(1, num_cluster_indices + 1), desc=f"Clusters Searched (Sample nProbe={sample_nProbe}, nprobe={nProbe}, Retrieved Docs={retrieved_docs})", position=3, leave=False):
                        for emb_num in tqdm(range(0, 100, 1), desc=f"Queries (Sample nProbe={sample_nProbe}, nprobe={nProbe}, Retrieved Docs={retrieved_docs}, Clusters Searched={clusters_searched})", position=4, leave=False):
                            query = embeddings_array[emb_num:emb_num+1][0:1]

                            ground_truth_distances, ground_truth_docs = flat_index.search(query, retrieved_docs)

                            monolithic_distances, monolithic_docs = monolithic_index.search(query, retrieved_docs)
                            monolithic_ndcg = calculate_ndcg(monolithic_docs[0], ground_truth_docs[0])
                            monolithic_recall = calculate_recall(monolithic_docs[0], ground_truth_docs[0])

                            small_nprobe_searches = []
                            for i in range(num_cluster_indices):
                                small_nprobe_cluster_index = cluster_indices[i]
                                small_nprobe_cluster_index.nprobe = sample_nProbe
                                cluster_distances, _ = small_nprobe_cluster_index.search(query, retrieved_docs)
                                small_nprobe_searches.append(sum(cluster_distances.flatten()) / len(cluster_distances.flatten()))
                            
                            sorted_indices = sorted(range(len(small_nprobe_searches)), key=lambda x: small_nprobe_searches[x], reverse=True)
                        
                            split_indices_distances, split_indices_docs = [], []
                            for i, split_index in enumerate(split_indices[:clusters_searched]):
                                split_index.nprobe = nProbe
                                split_distances, split_docs = split_index.search(query, retrieved_docs)
                                split_indices_docs.extend((split_docs.flatten() + i * (args.split_index_size / num_split_indices)).tolist())
                                split_indices_distances.extend(split_distances.flatten().tolist())

                            sorted_split_indices = np.argsort(split_indices_distances)[-1:-(retrieved_docs + 1):-1]
                            split_corresponding_docs = [split_indices_docs[i] for i in sorted_split_indices]

                            split_ndcg = calculate_ndcg(split_corresponding_docs, ground_truth_docs[0])
                            split_recall = calculate_recall(split_corresponding_docs, ground_truth_docs[0])
                            
                            cluster_distances, nprobe_docs = [], []
                            nprobe_distances = []

                            for i, index_id in enumerate(sorted_indices[:clusters_searched]):
                                cluster_index = cluster_indices[index_id]
                                cluster_index.nprobe = nProbe
                                cluster_values_path = os.path.join(args.cluster_index_indices_folder, f'cluster_{index_id}_indices.npy')
                                cluster_values = np.load(cluster_values_path)
                                
                                cluster_distances, cluster_docs = cluster_index.search(query, retrieved_docs)

                                nprobe_distances.extend(cluster_distances.flatten().tolist())
                                nprobe_docs.extend([cluster_values[idx] for idx in cluster_docs.flatten()])

                                sorted_kmeans_indices = np.argsort(nprobe_distances)[-1:-(retrieved_docs + 1):-1]
                                cluster_corresponding_docs = [nprobe_docs[i] for i in sorted_kmeans_indices]

                            cluster_ndcg = calculate_ndcg(cluster_corresponding_docs, ground_truth_docs[0])
                            cluster_recall = calculate_recall(cluster_corresponding_docs, ground_truth_docs[0])

                            monolithic_ndcgs.append(monolithic_ndcg)
                            monolithic_recalls.append(monolithic_recall)
                            split_ndcgs.append(split_ndcg)
                            split_recalls.append(split_recall)
                            cluster_ndcgs.append(cluster_ndcg)
                            cluster_recalls.append(cluster_recall)

                        writer.writerow({
                            "Number of Clusters Searched": clusters_searched,
                            "Sample nProbe": sample_nProbe,
                            "Deep nProbe": nProbe,
                            "Monolithic Recall": np.mean(monolithic_recalls),
                            "Split Recall": np.mean(split_recalls),
                            "Cluster Recall": np.mean(cluster_recalls),
                            "Monolithic NDCG": np.mean(monolithic_ndcgs),
                            "Split NDCG": np.mean(split_ndcgs),
                            "Cluster NDCG": np.mean(cluster_ndcgs),
                        })
                        output_file.flush()  # Ensure data is written incrementally


if __name__ == "__main__":
    main()
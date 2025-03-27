#!/usr/bin/env python3
import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
import faiss
from tqdm import tqdm
import ast
import csv
from collections import Counter

def analyze_clusters(args):
    # Get a sorted list of .faiss index files
    index_files = sorted(
        [os.path.join(args.index_folder, f) for f in os.listdir(args.index_folder) if f.endswith(".faiss")]
    )
    # Read the FAISS index for each file
    clustered_indices = [faiss.read_index(index_file) for index_file in index_files]
    
    # Get document counts for each cluster from the FAISS indices (using the ntotal attribute)
    document_counts = [cluster.ntotal for cluster in clustered_indices]

    # Count access frequencies from the cluster access trace file using Counter.
    access_counts = Counter()
    
    # Open and process the CSV trace file.
    with open(args.cluster_access_trace, "r", newline="") as csvfile:
        csv_trace = csv.DictReader(csvfile)
        # Iterate over each query in the CSV file.
        for query in tqdm(csv_trace, desc="Query Trace"):
            # Parse the 'Ranked Clusters' field into a list.
            ranked_clusters = ast.literal_eval(query['Ranked Clusters'])
            # Update the counter with only the top 'clusters_searched' clusters.
            access_counts.update(ranked_clusters[:args.clusters_searched])
    
    # Create a DataFrame to combine the document and access counts for each cluster.
    # Assuming clusters are numbered sequentially starting at 0.
    df = pd.DataFrame({
         "cluster_id": list(range(len(document_counts))),
         "document_count": document_counts,
         "access_count": [access_counts.get(i, 0) for i in range(len(document_counts))]
    })
    
    # Plotting using two subplots as per the provided example.
    fig, axs = plt.subplots(1, 2, figsize=(4.5, 1.5), constrained_layout=True)
    x = df["cluster_id"].tolist()
    y1 = df["document_count"].tolist()
    y2 = df["access_count"].tolist()
    
    # First subplot: Bar chart for document counts.
    axs[0].bar(x, y1, color='skyblue', edgecolor='black')
    axs[0].set_xlabel("Cluster ID", fontsize=7, fontweight='bold')
    axs[0].set_ylabel("Size (Docs)", fontsize=7, fontweight='bold')
    axs[0].set_xticks(x)
    axs[0].grid(visible=True, axis='y', linestyle='--', alpha=0.6)
    axs[0].tick_params(axis='both', which='major', labelsize=6)
    axs[0].yaxis.get_offset_text().set_fontsize(6)
    
    # Second subplot: Bar chart for access frequencies.
    axs[1].bar(x, y2, color='lightgreen', edgecolor='black')
    axs[1].set_xlabel("Cluster ID", fontsize=7, fontweight='bold')
    axs[1].set_ylabel("Access Frequency", fontsize=7, fontweight='bold')
    axs[1].set_xticks(x)
    axs[1].grid(visible=True, axis='y', linestyle='--', alpha=0.6)
    axs[1].tick_params(axis='both', which='major', labelsize=6)
    
    # Ensure the output directory exists.
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, "fig_13_cluster_size_frequency_analysis.pdf")
    plt.savefig(output_path)

def main():
    parser = argparse.ArgumentParser(
        description="Count documents in each FAISS cluster and analyze query access frequencies."
    )
    parser.add_argument("--index-folder", type=str, required=True,help="Path to the folder containing the clustered FAISS index files")
    parser.add_argument("--cluster-access-trace", type=str, required=True, help="Path to the cluster access trace CSV file")
    parser.add_argument("--clusters-searched", type=int, required=True, help="Number of top clusters to consider per query")
    parser.add_argument("--output-dir", type=str, default="data/figures/", help="Directory to save the figure")
    args = parser.parse_args()

    analyze_clusters(args)

if __name__ == "__main__":
    main()

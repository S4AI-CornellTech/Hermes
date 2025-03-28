#!/usr/bin/env python3
import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt

def plot_accuracy(file_path: str, output_dir: str) -> None:
    """
    Load CSV data and create two subplots showing NDCG accuracy for both:
      - Sample nProbe sweep (rows with nProbe == 128)
      - nProbe sweep (rows with Sample nProbe == 8)
    """
    # Load the CSV file
    df = pd.read_csv(file_path)
    
    # Create a figure with 1 row and 2 columns
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    
    # Adjust tick label sizes for both subplots
    for ax in axes:
        ax.tick_params(axis='both', labelsize=8)
    
    # --- Plot 1: Sample nProbe Sweep (Accuracy) ---
    df_sample = df[df['Deep nProbe'] == 128]
    sample_group = df_sample.groupby(['Sample nProbe', 'Number of Searched Indices'])['Avg Sample Search Time (s)'].mean().reset_index()
    unique_samples = sample_group['Sample nProbe'].unique()
    
    # Create a color map for the sample sweep
    cmap_sample = plt.get_cmap('tab10')
    colors_sample = [cmap_sample(i) for i in range(len(unique_samples))]
    
    for i, sample in enumerate(unique_samples):
        sub_df = sample_group[sample_group['Sample nProbe'] == sample]
        axes[0].plot(
            sub_df['Number of Searched Indices'],
            sub_df['Avg Sample Search Time (s)'],
            label=f'Sample nProbe = {sample}',
            color=colors_sample[i],
            marker='o', markersize=4
        )
    axes[0].set_xlabel('Clusters Searched', fontsize=10, fontweight='bold')
    axes[0].set_ylabel('NDCG', fontsize=10, fontweight='bold')
    axes[0].set_title("Sample nProbe Sweep", fontsize=12, fontweight='bold')
    axes[0].grid(True)
    axes[0].legend(fontsize=8)
    
    # --- Plot 2: nProbe Sweep (Accuracy) ---
    df_nprobe = df[df['Sample nProbe'] == 8]
    nprobe_group = df_nprobe.groupby(['Deep nProbe', 'Number of Searched Indices'])['Avg Deep Search Time (s)'].mean().reset_index()
    unique_nprobes = nprobe_group['Deep nProbe'].unique()
    
    # Create a color map for the nProbe sweep
    cmap_nprobe = plt.get_cmap('tab10')
    colors_nprobe = [cmap_nprobe(i) for i in range(len(unique_nprobes))]
    
    for i, nprobe in enumerate(unique_nprobes):
        sub_df = nprobe_group[nprobe_group['Deep nProbe'] == nprobe]
        axes[1].plot(
            sub_df['Number of Searched Indices'],
            sub_df['Avg Deep Search Time (s)'],
            label=f'nProbe = {nprobe}',
            color=colors_nprobe[i],
            marker='o', markersize=4
        )
    axes[1].set_xlabel('Clusters Searched', fontsize=10, fontweight='bold')
    axes[1].set_ylabel('NDCG', fontsize=10, fontweight='bold')
    axes[1].set_title("nProbe Sweep", fontsize=12, fontweight='bold')
    axes[1].grid(True)
    axes[1].legend(fontsize=8)
    
    # Adjust layout and save the figure
    plt.tight_layout()
    output_path = os.path.join(output_dir, "fig_12_hermes_nprobe_dse_latency.pdf")
    plt.savefig(output_path)

def main():
    parser = argparse.ArgumentParser(
        description="Plot NDCG accuracy analysis from CSV data (latency plots removed)"
    )
    parser.add_argument('--data-file', type=str, help="Path to the CSV file containing the data")
    parser.add_argument("--output-dir", type=str, default="data/figures/", help="Directory to save the figure")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    plot_accuracy(args.data_file, args.output_dir)

if __name__ == "__main__":
    main()

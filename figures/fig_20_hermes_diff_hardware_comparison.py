#!/usr/bin/env python3
import argparse
import os
import csv
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

def main():
    # Set the plotting aesthetic (mimicking the previous style)
    plt.rcParams.update({
        'font.size': 6,
        'axes.titlesize': 8,
        'axes.labelsize': 7,
        'xtick.labelsize': 6,
        'ytick.labelsize': 6,
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'grid.color': 'gray',
        'grid.linestyle': '--',
        'grid.linewidth': 0.5,
    })
    
    parser = argparse.ArgumentParser(
        description="Script to compare end-to-end retrieval latency with Hermes"
    )
    
    # Required integer arguments
    parser.add_argument("--sample-nprobe", type=int, required=True, help="Sample nprobe")
    parser.add_argument("--deep-nprobe", type=int, required=True, help="Deep nprobe")
    parser.add_argument("--retrieved-docs", type=int, required=True, help="Number of documents retrieved")
    parser.add_argument("--batch-size", type=int, required=True, help="Number of Queries per batch")
    
    # Required file path arguments
    parser.add_argument("--hermes-retrieval-traces", nargs='+', type=str, required=True,
                        help="Path to the Hermes retrieval trace CSV file")
    
    parser.add_argument("--output-dir", type=str, default="data/figures/",
                        help="Directory to save the figure")
    
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Prepare a figure with two side-by-side subplots (4.25 x 2 inches)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(4.25, 2))
    
    # Marker style for all lines
    marker_style = dict(marker='o', markersize=4, linestyle='-', linewidth=1)
    
    # Define a list of colors (cycle through if more than four files)
    colors = ['#66BB6A', '#E57373', '#FFCA28', '#42A5F5']
    
    # List to store custom legend handles
    custom_handles = []
    
    # Loop over each provided CSV trace file and extract data
    for idx, file in enumerate(args.hermes_retrieval_traces):
        retrieval_latency = []
        retrieval_throughput = []
        retrieval_cluster = []
        
        with open(file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if (int(row["Sample nprobe"]) == args.sample_nprobe and
                    int(row["Deep nprobe"]) == args.deep_nprobe and
                    int(row["Retrieved Docs"]) == args.retrieved_docs and
                    int(row["Batch Size"]) == args.batch_size):
                    
                    retrieval_latency.append(float(row["Avg Hermes Retrieval Latency (s)"]))
                    retrieval_throughput.append(float(row["Avg Hermes Throughput (QPS)"]))
                    retrieval_cluster.append(int(row["Clusters Searched"]))
        
        # Determine color for this series
        color = colors[idx % len(colors)]
        # Use the base name of the file as a label
        label = os.path.basename(file)
        
        # Plot on the left: Clusters vs. Retrieval Latency
        ax1.plot(retrieval_cluster, retrieval_latency, color=color, label=label, **marker_style)
        
        # Plot on the right: Clusters vs. Retrieval Throughput
        ax2.plot(retrieval_cluster, retrieval_throughput, color=color, label=label, **marker_style)
        
        # Create a custom handle for the shared legend
        handle = Line2D([0], [0], marker='o', color=color, markersize=4, linestyle='-', linewidth=1, label=label)
        custom_handles.append(handle)
    
    # Customize left subplot (Retrieval Latency)
    ax1.set_xlabel("Clusters Searched", fontsize=7, fontweight='bold')
    ax1.set_ylabel("Retrieval Latency (s)", fontsize=7, fontweight='bold')
    ax1.grid(visible=True, linestyle='-', color='gray', which='major', axis='both', zorder=0)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    for tick_label in ax1.get_xticklabels():
        tick_label.set_fontweight('bold')
    
    # Customize right subplot (Retrieval Throughput)
    ax2.set_xlabel("Clusters Searched", fontsize=7, fontweight='bold')
    ax2.set_ylabel("Retrieval Throughput (QPS)", fontsize=7, fontweight='bold')
    ax2.grid(visible=True, linestyle='-', color='gray', which='major', axis='both', zorder=0)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    for tick_label in ax2.get_xticklabels():
        tick_label.set_fontweight('bold')
    
    # Add a shared legend at the top center of the figure
    plt.figlegend(handles=custom_handles, loc='upper center', bbox_to_anchor=(0.52, 0.95),
                  ncol=len(custom_handles), fontsize=5)
    
    plt.tight_layout(rect=[0, 0, 1, 0.85])
    output_file = os.path.join(args.output_dir, "fig_20_hermes_diff_hardware_comparison.pdf")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Figure saved to {output_file}")

if __name__ == "__main__":
    main()

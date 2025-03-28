import argparse
import os
import csv
import matplotlib.pyplot as plt
import numpy as np

def main():
    parser = argparse.ArgumentParser(
        description="Script to analyze throughput and energy with Hermes"
    )
    
    # Define required integer arguments
    parser.add_argument("--output-dir", type=str, default="data/figures/", help="Directory to save the figure")
    parser.add_argument("--batch-size", type=int, required=True, help="Batch size")
    parser.add_argument("--sample-nprobe", type=int, required=True, help="Sample nprobe")
    parser.add_argument("--deep-nprobe", type=int, required=True, help="Deep nprobe")
    parser.add_argument("--hermes-retrieval-trace", type=str, required=True, help="Path to the Hermes retrieval trace file")
    parser.add_argument("--hermes-energy-trace", type=str, required=True, help="Path to the Hermes energy file")
    parser.add_argument("--retrieved-docs", type=int, required=True, help="Number of documents retrieved")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    hermes_dvfs_enhanced_energy, hermes_throughputs = [], []

    with open(args.hermes_energy_trace, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if (int(row["Batch Size"]) == args.batch_size and 
                int(row["Deep nprobe"]) == args.deep_nprobe and 
                int(row["Retrieved Docs"]) == args.retrieved_docs and 
                int(row["Sample nprobe"]) == args.sample_nprobe):
                hermes_dvfs_enhanced_energy.append(float(row["Avg Hermes Enhanced DVFS Energy (J)"]))

    with open(args.hermes_retrieval_trace, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if (int(row["Batch Size"]) == args.batch_size and 
                int(row["Deep nprobe"]) == args.deep_nprobe and 
                int(row["Retrieved Docs"]) == args.retrieved_docs and 
                int(row["Sample nprobe"]) == args.sample_nprobe):
                hermes_throughputs.append(float(row["Avg Hermes Throughput (QPS)"]))

    # Convert lists to numpy arrays
    hermes_dvfs_enhanced_energy = np.array(hermes_dvfs_enhanced_energy)
    hermes_throughputs = np.array(hermes_throughputs)

    indices = range(1, len(hermes_throughputs) + 1)

    # Define colors matching your plotting style.
    data_color = "#4CAF50"      # Soft green for data points
    grid_color = "#B0BEC5"      # Light gray for gridlines

    # Create a figure with two subplots (throughput on top, energy on bottom)
    fig, ax = plt.subplots(2, 1, figsize=(4, 3))

    # Plot Throughput (upper subplot)
    ax[0].scatter(indices, hermes_throughputs, marker='o', color=data_color, s=20)
    ax[0].set_ylabel("Throughput (QPS)", fontsize=8, fontweight="bold")
    ax[0].tick_params(axis='both', labelsize=6)
    ax[0].grid(visible=True, linestyle='--', color=grid_color, alpha=0.6)
    ax[0].set_xticklabels([])  # Hide x-axis labels for the upper plot

    # Plot Energy (lower subplot)
    ax[1].scatter(indices, hermes_dvfs_enhanced_energy, marker='o', color=data_color, s=20)
    ax[1].set_xlabel("Data Point", fontsize=8, fontweight="bold")
    ax[1].set_ylabel("Energy (Joules)", fontsize=8, fontweight="bold")
    ax[1].tick_params(axis='both', labelsize=6)
    ax[1].grid(visible=True, linestyle='--', color=grid_color, alpha=0.6)

    # Adjust layout for readability
    fig.tight_layout(rect=[0, 0, 1, 0.9])

    # Save the figure to a PDF file and display it
    output_file = os.path.join(args.output_dir, "fig_18_hermes_energy_throughput_analysis.pdf")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Figure saved to {output_file}")

if __name__ == "__main__":
    main()

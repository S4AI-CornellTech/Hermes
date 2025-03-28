import argparse
import os
import csv
import matplotlib.pyplot as plt
import numpy as np

def main():
    parser = argparse.ArgumentParser(
        description="Script to compare DVFS energy with Hermes"
    )
    
    # Define required integer arguments
    parser.add_argument('--data-file', type=str, help="Path to the CSV file containing the data")    
    parser.add_argument("--output-dir", type=str, default="data/figures/", help="Directory to save the figure")
    parser.add_argument("--batch-size", type=int, required=True, help="Batch size")
    parser.add_argument("--sample-nprobe", type=int, required=True, help="Sample nprobe")
    parser.add_argument("--deep-nprobe", type=int, required=True, help="Deep nprobe")
    parser.add_argument("--retrieved-docs", type=int, required=True, help="Number of documents retrieved")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    hermes_energy = []
    hermes_dvfs_energy = []
    hermes_dvfs_enhanced_energy = []

    with open(args.data_file, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if (int(row["Batch Size"]) == args.batch_size and 
                int(row["Deep nprobe"]) == args.deep_nprobe and 
                int(row["Retrieved Docs"]) == args.retrieved_docs and 
                int(row["Sample nprobe"]) == args.sample_nprobe):
                hermes_energy.append(float(row["Avg Hermes Energy (J)"]))
                hermes_dvfs_energy.append(float(row["Avg Hermes DVFS Energy (J)"]))
                hermes_dvfs_enhanced_energy.append(float(row["Avg Hermes Enhanced DVFS Energy (J)"]))

    # Convert lists to numpy arrays
    hermes_energy = np.array(hermes_energy)
    hermes_dvfs_energy = np.array(hermes_dvfs_energy)
    hermes_dvfs_enhanced_energy = np.array(hermes_dvfs_enhanced_energy)

    # Normalize energy values relative to the baseline (hermes_energy)
    # This makes the baseline equal to 1 and scales the others accordingly
    norm_hermes_energy = hermes_energy / hermes_energy  # All ones (baseline)
    norm_hermes_dvfs_energy = hermes_dvfs_energy / hermes_energy
    norm_hermes_dvfs_enhanced_energy = hermes_dvfs_enhanced_energy / hermes_energy

    # Assume each row corresponds to a cluster; create cluster IDs for the x-axis
    clusters = np.arange(1, len(hermes_energy) + 1)

    # Update matplotlib parameters for a polished look
    plt.rcParams.update({
        "xtick.labelsize": 6,
        "legend.title_fontsize": 6,
        "axes.labelsize": 7,
        "axes.titlesize": 10,
        "legend.fontsize": 5,
        "ytick.labelsize": 6,
        "axes.labelweight": "bold",
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "--"
    })

    # Define a colorblind-friendly palette
    colors = {
        'hermes': '#4E79A7',       # Blue
        'dvfs': '#F28E2B',         # Orange
        'enhanced': '#E15759'      # Red-ish
    }

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(4, 1.75))

    # Bar width for three sets
    bar_width = 0.25

    # Calculate bar positions for a side-by-side layout
    r1 = clusters - bar_width
    r2 = clusters
    r3 = clusters + bar_width

    # Plot each data set as bars using normalized values
    ax.bar(r1, norm_hermes_energy, width=bar_width, color=colors['hermes'],
           edgecolor='black', linewidth=0.8, label='Hermes', alpha=0.85)

    ax.bar(r2, norm_hermes_dvfs_energy, width=bar_width, color=colors['dvfs'],
           edgecolor='black', linewidth=0.8, label='Hermes DVFS', alpha=0.85)

    ax.bar(r3, norm_hermes_dvfs_enhanced_energy, width=bar_width, color=colors['enhanced'],
           edgecolor='black', linewidth=0.8, label='Hermes DVFS Enhanced', alpha=0.85)

    # Set labels and ticks
    ax.set_xlabel("Clusters Searched", fontweight="bold")
    ax.set_ylabel("Norm. Energy", fontweight="bold")
    ax.set_xticks(clusters)
    ax.set_xticklabels(clusters, fontweight="bold", fontsize=6)
    # ax.set_title("Hermes Energy Comparison", fontweight="bold", fontsize=10)

    # Add legend
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.55, 0.85), ncol=3, fontsize=6)

    # Set y-axis limits to mirror the previous normalized plot
    ax.set_ylim(0.8, 1.02)
    ax.set_yticks([0.80, 0.85, 0.90, 0.95, 1.0])

    # Adjust layout and save the figure
    plt.tight_layout(rect=[0, 0, 1, 0.9])
    output_file = os.path.join(args.output_dir, "fig_21_hermes_dvfs_analysis.pdf")
    plt.savefig(output_file)
    # plt.show()

if __name__ == "__main__":
    main()

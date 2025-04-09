import argparse
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.patches as mpatches
import os

def plot_ndcg(args, file_path: str) -> None:
    """Load CSV data, filter rows, group by clusters, and plot NDCG values."""
    df = pd.read_csv(file_path)

    # Filter for Split Search based on sample and deep nprobe
    split_df = df[(df['Sample nProbe'] == args.sample_nprobe) & (df['nProbe'] == args.deep_nprobe)]

    # Filter for Monolithic Search based on monolithic nprobe
    monolithic_df = df[df['nProbe'] == args.monolithic_nprobe]

    # Group both datasets by 'Number of Clusters Searched'
    split_grouped = split_df.groupby('Number of Clusters Searched').mean()
    monolithic_grouped = monolithic_df.groupby('Number of Clusters Searched').mean()

    # Extract relevant columns for plotting
    clusters_searched = split_grouped.index.union(monolithic_grouped.index).sort_values()
    split_ndcg = split_grouped.reindex(clusters_searched)['Split NDCG']
    nprobe_search_ndcg = split_grouped.reindex(clusters_searched)['Cluster NDCG']
    monolithic_ndcg = monolithic_grouped.reindex(clusters_searched)['Monolithic NDCG']

    # Create the plot
    fig, ax1 = plt.subplots(figsize=(2.5, 2))

    # Define colors
    nprobe_color = '#276FBF'
    split_color = '#FFA500'
    hierarchical_color = 'green'

    # Plot the NDCG lines
    ax1.plot(clusters_searched, nprobe_search_ndcg, marker='.', color=nprobe_color,
             linewidth=1, markersize=4, label="Monolithic NDCG")
    ax1.plot(clusters_searched, split_ndcg, marker='.', color=split_color,
             linewidth=1, markersize=4, label="Split NDCG")
    ax1.plot(clusters_searched, monolithic_ndcg, marker='.', color=hierarchical_color,
             linewidth=1, markersize=4, label="Cluster NDCG")

    # Enhance labels, grid, and axes properties
    ax1.set_xlabel("Clusters Searched", fontsize=8, fontweight='bold')
    ax1.set_ylabel("NDCG", fontsize=8, fontweight='bold')
    ax1.grid(visible=True, which='both', linestyle='--', linewidth=0.3, color='gray')
    ax1.tick_params(axis='both', which='major', labelsize=6)
    ax1.set_ylim([0, 1])

    # Create a custom legend
    labels = ['Hermes', 'Split Search', 'Monolithic Search']
    colors = [nprobe_color, split_color, hierarchical_color]
    custom_handles = [mpatches.Patch(color=color, label=label) for label, color in zip(labels, colors)]
    fig.legend(handles=custom_handles, labels=labels, loc='upper center',
               bbox_to_anchor=(0.5, 0.975), ncol=3, fontsize=5)

    plt.tight_layout(rect=[0, 0, 1, 0.9])
    plt.savefig(os.path.join(args.output_dir, "fig_11_hermes_accuracy_comparison.pdf"))

def main():
    parser = argparse.ArgumentParser(description="Plot NDCG analysis from CSV data")
    parser.add_argument('--data-file', type=str, required=True, help="Path to the CSV file containing the data")
    parser.add_argument('--output-dir', type=str, default="data/figures/", help="Directory where the results will be saved")
    parser.add_argument('--sample-nprobe', type=int, required=True, help="Sample nProbe value for split search")
    parser.add_argument('--deep-nprobe', type=int, required=True, help="Deep nProbe value for split search")
    parser.add_argument('--monolithic-nprobe', type=int, required=True, help="nProbe value for monolithic search")

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    plot_ndcg(args, args.data_file)

if __name__ == "__main__":
    main()

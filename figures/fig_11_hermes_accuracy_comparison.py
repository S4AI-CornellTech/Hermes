import argparse
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.patches as mpatches
import os

def plot_ndcg(args, file_path: str) -> None:
    """Load CSV data, filter rows, group by clusters, and plot NDCG values."""
    # Load the CSV file
    df = pd.read_csv(file_path)

    # Filter rows where Sample nProbe is 8 and nProbe is 128
    filtered_df = df[(df['Sample nProbe'] == 8) & (df['nProbe'] == 128)]

    # Group by 'Number of Clusters Searched' and calculate the mean for each required column
    grouped_df = filtered_df.groupby('Number of Clusters Searched').mean()

    # Extract relevant columns for plotting
    clusters_searched = grouped_df.index
    nprobe_search_ndcg = grouped_df['Cluster NDCG']
    split_ndcg = grouped_df['Split NDCG']
    monolithic_ndcg = grouped_df['Monolithic NDCG']

    # Create a plot for NDCG vs. Number of Clusters Searched
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
    parser.add_argument('--data-file', type=str, help="Path to the CSV file containing the data")
    parser.add_argument("--output-dir", type=str, default="data/figures/", help="Directory where the results will be saved")

    args = parser.parse_args()

    # Ensure the output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    # If the data file doesn't exist, create a dummy file with sample data
    if not os.path.exists(args.data_file):
        print(f"Data file '{args.data_file}' does not exist. Creating a dummy file with sample data.")
        dummy_data = {
            'Sample nProbe': [8, 8, 8],
            'nProbe': [128, 128, 128],
            'Number of Clusters Searched': [1, 2, 3],
            'Cluster NDCG': [0.8, 0.85, 0.9],
            'Split NDCG': [0.75, 0.8, 0.85],
            'Monolithic NDCG': [0.7, 0.78, 0.82]
        }
        df_dummy = pd.DataFrame(dummy_data)
        df_dummy.to_csv(args.data_file, index=False)

    plot_ndcg(args, args.data_file)

if __name__ == "__main__":
    main()

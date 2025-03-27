import argparse
import os
import csv
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser(
        description="Script to compare end-to-end retrieval latency with Hermes"
    )
    
    # Define required integer arguments
    parser.add_argument("--input-size", type=int, required=True, help="Input size")
    parser.add_argument("--stride-length", type=int, required=True, help="Stride length")
    parser.add_argument("--batch-size", type=int, required=True, help="Batch size")
    parser.add_argument("--sample-nprobe", type=int, required=True, help="Sample nprobe")
    parser.add_argument("--deep-nprobe", type=int, required=True, help="Deep nprobe")
    parser.add_argument("--retrieved-docs", type=int, required=True, help="Number of documents retrieved")
    parser.add_argument("--clusters-searched", type=int, required=True, help="Number of clusters searched")

    # Define required file path arguments
    parser.add_argument("--monolithic-retrieval-trace", type=str, required=True, help="Path to the monolithic retrieval latency CSV file")
    parser.add_argument("--hermes-retrieval-trace", type=str, required=True, help="Path to the Hermes retrieval trace CSV file")
    parser.add_argument("--encoding-trace", type=str, required=True, help="Path to the encoding trace CSV file")
    parser.add_argument("--inference-trace", type=str, required=True, help="Path to the inference trace CSV file")
    
    parser.add_argument("--output-dir", type=str, default="data/figures/", help="Directory to save the figure")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    encoding_time, prefill_time, monolithic_retrieval_time = 0, 0, 0
    with open(args.monolithic_retrieval_trace, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if int(row["Batch Size"]) == args.batch_size and int(row["nprobe"]) == args.deep_nprobe:
                monolithic_retrieval_time = float(row["Avg Retrieval Time (s)"])

    with open(args.encoding_trace, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if int(row["Batch Size"]) == args.batch_size and int(row["Input Token Length"]) == args.input_size:
                encoding_time = float(row["Avg Latency (s)"])
    
    with open(args.inference_trace, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if (int(row["Batch Size"]) == args.batch_size and 
                int(row["Input Token Length"]) == args.input_size and 
                int(row["Output Token Length"]) == args.stride_length):
                prefill_time = float(row["Avg Prefill Time (s)"])

    with open(args.hermes_retrieval_trace, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if (int(row["Batch Size"]) == args.batch_size and 
                int(row["Deep nprobe"]) == args.deep_nprobe and 
                int(row["Retrieved Docs"]) == args.retrieved_docs and 
                int(row["Clusters Searched"]) == args.clusters_searched and 
                int(row["Sample nprobe"]) == args.sample_nprobe):
                hermes_retrieval_time = float(row["Avg Hermes Retrieval Latency (s)"])

    baseline_latency = (encoding_time + prefill_time + monolithic_retrieval_time)
    hermes = (encoding_time + prefill_time + hermes_retrieval_time)
    hermes_w_enhancements = (encoding_time + prefill_time + hermes_retrieval_time)

    bars = [baseline_latency, hermes, hermes_w_enhancements]
    labels = ["Baseline", "Hermes", "Hermes/PipeRAG/RAGCache"]

    # Create the bar plot using the same aesthetic as before
    fig, ax = plt.subplots(figsize=(6, 3))
    colors = ['#66BB6A', '#E57373', '#FFCA28']
    ax.bar(labels, bars, color=colors, edgecolor="black", width=0.7)
    
    # Set title and labels with custom font sizes and weight
    ax.set_title("TTFT Hermes Latency Comparison", fontsize=8, fontweight='bold')
    ax.set_ylabel("Latency (s)", fontsize=7, fontweight='bold')
    ax.set_xticklabels(labels, rotation=360, fontsize=6, fontweight='bold')
    ax.tick_params(axis='x', labelsize=6)
    ax.tick_params(axis='y', labelsize=6)
    
    # Add dashed gray gridlines for y-axis
    ax.grid(visible=True, linestyle='--', color='gray', which='major', axis='y', zorder=0)
    
    plt.tight_layout()
    output_path = os.path.join(args.output_dir, "fig_15_ttft_hermes_latency_comparison.pdf")
    plt.savefig(output_path)

if __name__ == "__main__":
    main()

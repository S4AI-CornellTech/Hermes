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
    parser.add_argument("--output-size", type=int, required=True, help="Output size")
    parser.add_argument("--stride-length", type=int, required=True, help="Stride length")
    parser.add_argument("--batch-size", type=int, required=True, help="Batch size")
    parser.add_argument("--monolithic-nprobe", type=int, required=True, help="Monolithic nprobe")
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

    encoding_time, prefill_time, decoding_time, monolithic_retrieval_time = 0, 0, 0, 0
    with open(args.monolithic_retrieval_trace, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if int(row["Batch Size"]) == args.batch_size and int(row["nprobe"]) == args.monolithic_nprobe:
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
                decoding_time = float(row["Avg Decode Time (s)"])

    with open(args.hermes_retrieval_trace, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if (int(row["Batch Size"]) == args.batch_size and 
                int(row["Deep nprobe"]) == args.deep_nprobe and 
                int(row["Retrieved Docs"]) == args.retrieved_docs and 
                int(row["Clusters Searched"]) == args.clusters_searched and 
                int(row["Sample nprobe"]) == args.sample_nprobe):
                hermes_retrieval_time = float(row["Avg Hermes Retrieval Latency (s)"])

    # print(monolithic_retrieval_time)
    # print(hermes_retrieval_time)

    num_strides = args.output_size // args.stride_length
    baseline_latency = (encoding_time + prefill_time + decoding_time + monolithic_retrieval_time) * num_strides 
    piperag = (monolithic_retrieval_time + encoding_time + prefill_time + decoding_time) + \
              (max(monolithic_retrieval_time, prefill_time + decoding_time) + encoding_time) * (num_strides - 1)
    ragcache = (monolithic_retrieval_time + encoding_time + decoding_time) * num_strides + prefill_time
    hermes = (encoding_time + prefill_time + decoding_time + hermes_retrieval_time) * num_strides
    hermes_w_enhancements = (hermes_retrieval_time + encoding_time + prefill_time + decoding_time) + \
                            (max(hermes_retrieval_time, decoding_time) + encoding_time) * (num_strides - 1)

    # Normalize to baseline
    bars = [
        baseline_latency / baseline_latency,
        piperag / baseline_latency,
        ragcache / baseline_latency,
        hermes / baseline_latency,
        hermes_w_enhancements / baseline_latency
    ]

    labels = ["Baseline", "Ragcache", "Piperag", "Hermes", "Hermes/PipeRAG/RAGCache"]

    # Create the bar plot using the same aesthetic as before
    fig, ax = plt.subplots(figsize=(6, 3))
    colors = ['#66BB6A', '#BA68C8', '#42A5F5', '#E57373', '#FFCA28']
    ax.bar(labels, bars, color=colors, edgecolor="black", width=0.7)
    
    # Set title and labels with custom font sizes and weight
    ax.set_title("Normalized End-to-End Retrieval Latency", fontsize=8, fontweight='bold')
    ax.set_ylabel("Normalized Latency (Relative to Baseline)", fontsize=7, fontweight='bold')
    ax.set_xticklabels(labels, rotation=360, fontsize=6, fontweight='bold')
    ax.tick_params(axis='x', labelsize=6)
    ax.tick_params(axis='y', labelsize=6)
    
    # Add dashed gray gridlines for y-axis
    ax.grid(visible=True, linestyle='--', color='gray', which='major', axis='y', zorder=0)
    
    plt.tight_layout()
    output_path = os.path.join(args.output_dir, "fig_14_normalized_latency_comparison.pdf")
    plt.savefig(output_path)

if __name__ == "__main__":
    main()

import argparse
import os
import csv
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser(
        description="Script to compare end-to-end retrieval energy with Hermes"
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
    
    # File path arguments
    parser.add_argument("--monolithic-retrieval-trace", type=str, required=True)
    parser.add_argument("--encoding-trace", type=str, required=True)
    parser.add_argument("--inference-trace", type=str, required=True)
    parser.add_argument("--monolithic-retrieval-trace-power", type=str, required=True)
    parser.add_argument("--encoding-trace-power", type=str, required=True)
    parser.add_argument("--inference-trace-power", type=str, required=True)
    parser.add_argument("--hermes-retrieval-trace-energy", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="data/figures/")

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # --- Load times ---
    encoding_time, prefill_time, decoding_time, monolithic_retrieval_time = 0, 0, 0, 0
    with open(args.monolithic_retrieval_trace, "r") as f:
        for row in csv.DictReader(f):
            if int(row["Batch Size"]) == args.batch_size and int(row["nprobe"]) == args.monolithic_nprobe:
                monolithic_retrieval_time = float(row["Avg Retrieval Time (s)"])

    with open(args.encoding_trace, "r") as f:
        for row in csv.DictReader(f):
            if int(row["Batch Size"]) == args.batch_size and int(row["Input Token Length"]) == args.input_size:
                encoding_time = float(row["Avg Latency (s)"])
    
    with open(args.inference_trace, "r") as f:
        for row in csv.DictReader(f):
            if (int(row["Batch Size"]) == args.batch_size and 
                int(row["Input Token Length"]) == args.input_size and 
                int(row["Output Token Length"]) == args.stride_length):
                prefill_time = float(row["Avg Prefill Time (s)"])
                decoding_time = float(row["Avg Decode Time (s)"])

    # --- Load power ---
    encoding_power, prefill_power, decoding_power, monolithic_retrieval_power = 0, 0, 0, 0
    with open(args.monolithic_retrieval_trace_power, "r") as f:
        for row in csv.DictReader(f):
            if int(row["Batch Size"]) == args.batch_size and int(row["nprobe"]) == args.monolithic_nprobe:
                monolithic_retrieval_power = float(row["Avg Retrieval Power (W)"])

    with open(args.encoding_trace_power, "r") as f:
        for row in csv.DictReader(f):
            if int(row["Batch Size"]) == args.batch_size and int(row["Input Token Length"]) == args.input_size:
                encoding_power = float(row["Avg Power (W)"])
    
    with open(args.inference_trace_power, "r") as f:
        for row in csv.DictReader(f):
            if (int(row["Batch Size"]) == args.batch_size and 
                int(row["Input Token Length"]) == args.input_size and 
                int(row["Output Token Length"]) == args.stride_length):
                prefill_power = float(row["Avg Prefill Power (W)"])
                decoding_power = float(row["Avg Decode Power (W)"])

    # --- Load Hermes energy directly ---
    hermes_energy = 0
    with open(args.hermes_retrieval_trace_energy, "r") as f:
        for row in csv.DictReader(f):
            if (int(row["Batch Size"]) == args.batch_size and 
                int(row["Deep nprobe"]) == args.deep_nprobe and 
                int(row["Retrieved Docs"]) == args.retrieved_docs and 
                int(row["Clusters Searched"]) == args.clusters_searched and 
                int(row["Sample nprobe"]) == args.sample_nprobe):
                hermes_energy = float(row["Avg Hermes Enhanced DVFS Energy (J)"])

    # --- Compute energy ---
    encoding_energy = encoding_power * encoding_time
    prefill_energy = prefill_power * prefill_time
    decoding_energy = decoding_power * decoding_time
    monolithic_retrieval_energy = monolithic_retrieval_power * monolithic_retrieval_time

    print(monolithic_retrieval_energy)
    print(hermes_energy)

    num_strides = args.output_size // args.stride_length

    baseline_energy = (encoding_energy + prefill_energy + decoding_energy + monolithic_retrieval_energy) * num_strides
    piperag = (encoding_energy + decoding_energy + monolithic_retrieval_energy + prefill_energy) + \
              (monolithic_retrieval_energy + decoding_energy + prefill_energy) * (num_strides - 1)
    ragcache = (monolithic_retrieval_energy + encoding_energy + decoding_energy) * num_strides + prefill_energy
    hermes = (encoding_energy + prefill_energy + decoding_energy + hermes_energy) * num_strides
    hermes_w_enhancements = (encoding_energy + decoding_energy + hermes_energy + prefill_energy) + \
                            (encoding_energy + decoding_energy + hermes_energy) * (num_strides - 1)

    print("Baseline Energy:", baseline_energy)
    print("Hermes Energy:", hermes)
    print("Hermes Energy with Enhancements:", hermes_w_enhancements)

    # --- Normalize to baseline energy ---
    bars = [
        baseline_energy / baseline_energy,
        piperag / baseline_energy,
        ragcache / baseline_energy,
        hermes / baseline_energy,
        hermes_w_enhancements / baseline_energy
    ]
    labels = ["Baseline", "Ragcache", "Piperag", "Hermes", "Hermes/PipeRAG/RAGCache"]

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(6, 3))
    colors = ['#66BB6A', '#BA68C8', '#42A5F5', '#E57373', '#FFCA28']
    ax.bar(labels, bars, color=colors, edgecolor="black", width=0.7)

    ax.set_title("Normalized End-to-End Retrieval Energy", fontsize=8, fontweight='bold')
    ax.set_ylabel("Normalized Energy (Relative to Baseline)", fontsize=7, fontweight='bold')
    ax.set_xticklabels(labels, rotation=360, fontsize=6, fontweight='bold')
    ax.tick_params(axis='x', labelsize=6)
    ax.tick_params(axis='y', labelsize=6)
    ax.grid(visible=True, linestyle='--', color='gray', which='major', axis='y', zorder=0)

    plt.tight_layout()
    output_path = os.path.join(args.output_dir, "fig_14_end_to_end_hermes_energy_comparison.pdf")
    plt.savefig(output_path)

if __name__ == "__main__":
    main()

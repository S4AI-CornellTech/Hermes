import argparse
import csv
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

BAR_COLORS = ['#4E79A7', '#E15759', '#FFD700', '#642CA9']
BAR_LABELS = ['Encoding', 'Retrieval', 'Prefill', 'Decoding']

def extract_shared_latencies(trace_file, filters, keys):
    with open(trace_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if all(int(row[k]) == v for k, v in filters.items()):
                return [float(row[k]) for k in keys]
    raise ValueError("Matching row not found.")

def extract_ttft_latencies(args):
    encoding = extract_shared_latencies(
        args.encoding_trace,
        {'Batch Size': args.batch_size, 'Input Token Length': args.input_size},
        ['Avg Latency (s)']
    )[0]

    prefill, decode = extract_shared_latencies(
        args.inference_trace,
        {'Batch Size': args.batch_size, 'Input Token Length': args.input_size, 'Output Token Length': args.stride_length},
        ['Avg Prefill Time (s)', 'Avg Decode Time (s)']
    )

    retrieval = extract_shared_latencies(
        args.monolithic_retrieval_trace,
        {'Batch Size': args.batch_size, 'nprobe': args.monolithic_nprobe},
        ['Avg Retrieval Time (s)']
    )[0]

    return [encoding, retrieval, prefill, decode / args.stride_length]

def extract_e2e_latencies(args):
    encoding, retrieval, prefill, decode = extract_ttft_latencies(args)
    scale = args.output_size // args.stride_length
    return [encoding * scale, retrieval * scale, prefill * scale, decode * args.output_size]

def plot_stacked_bar(ax, position, data, xlabel, ylabel):
    bottom = 0
    for i, val in enumerate(data):
        ax.bar(position, val, bottom=bottom, color=BAR_COLORS[i], width=0.2,
               linewidth=0.75, edgecolor='black', label=BAR_LABELS[i])
        bottom += val

    ax.set_xticks([position])
    ax.set_xticklabels([xlabel], fontsize=8, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=8, fontweight='bold')
    ax.tick_params(axis='both', labelsize=7)
    ax.grid(axis='y', linestyle=':', color='gray', alpha=0.6)
    ax.spines[['top', 'right']].set_visible(False)

def main():
    parser = argparse.ArgumentParser(description="Plot TTFT and E2E stacked bars using monolithic retrieval only.")
    parser.add_argument("--input-size", type=int, required=True)
    parser.add_argument("--output-size", type=int, required=True)
    parser.add_argument("--stride-length", type=int, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--monolithic-nprobe", type=int, required=True)
    parser.add_argument("--retrieved-docs", type=int, required=True)
    parser.add_argument("--monolithic-retrieval-trace", type=str, required=True)
    parser.add_argument("--encoding-trace", type=str, required=True)
    parser.add_argument("--inference-trace", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="data/figures/")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    ttft = extract_ttft_latencies(args)
    e2e = extract_e2e_latencies(args)

    fig = plt.figure(figsize=(3.6, 2))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1], wspace=0.5)

    ax_ttft = fig.add_subplot(gs[0])
    plot_stacked_bar(ax_ttft, 0, ttft, 'TTFT', 'TTFT Latency (s)')

    ax_e2e = fig.add_subplot(gs[1])
    plot_stacked_bar(ax_e2e, 0, e2e, 'E2E', 'E2E Latency (s)')

    handles, labels = ax_ttft.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.95), ncol=4, fontsize=7)

    fig.tight_layout(rect=[0, 0, 1, 1])
    fig.savefig(os.path.join(args.output_dir, "fig_06_rag_ttft_e2e_retrieval_overhead.pdf"), bbox_inches="tight")

if __name__ == "__main__":
    main()

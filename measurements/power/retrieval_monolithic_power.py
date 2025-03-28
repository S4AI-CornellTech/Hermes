import argparse
import time
import csv
import os
import numpy as np
import faiss
from tqdm import tqdm
import pyRAPL

# Setup pyRAPL
pyRAPL.setup()

def parse_arguments():
    parser = argparse.ArgumentParser(description="FAISS Query Benchmark with pyRAPL (Watts & Joules)")
    parser.add_argument("--index-name", type=str, required=True)
    parser.add_argument("--nprobe", type=int, required=True)
    parser.add_argument("--batch-size", type=int, nargs='+', required=True)
    parser.add_argument("--queries", type=str, required=True)
    parser.add_argument("--retrieved-docs", type=int, nargs='+', required=True)
    parser.add_argument("--num-threads", type=int, nargs='+', required=True)
    parser.add_argument("--output-dir", type=str, default="data/profiling/")
    return parser.parse_args()

def measure_search_power(index, queries, top_k):
    meter = pyRAPL.Measurement('query_power')
    meter.begin()
    start_time = time.time()
    _, _ = index.search(queries, top_k)
    end_time = time.time()
    meter.end()

    elapsed_time = end_time - start_time
    # Correct conversion: divide by 1e6 to convert µJ (microjoules) to J (joules)
    energy_joules = meter.result.pkg[0] / 1e6
    power_watts = energy_joules / elapsed_time if elapsed_time > 0 else 0

    return power_watts, energy_joules

def main():
    args = parse_arguments()
    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(args.output_dir, "retrieval_monolithic_energy_power.csv")

    embeddings = np.load(args.queries)
    index = faiss.read_index(args.index_name)
    index.nprobe = args.nprobe

    with open(output_file, mode='a', newline='') as f:
        writer = csv.writer(f)
        if f.tell() == 0:
            writer.writerow([
                "Index Name", "nprobe", "Batch Size", "Retrieved Docs", "Num Threads",
                "Avg Power (W)", "Avg Energy (J)"
            ])

        for batch_size in tqdm(args.batch_size, desc="Batch sizes"):
            for retrieved_docs in tqdm(args.retrieved_docs, desc=f"Retrieved Docs (batch_size={batch_size})", leave=False):
                for num_threads in tqdm(args.num_threads, desc=f"Threads (bs={batch_size}, k={retrieved_docs})", leave=False):
                    faiss.omp_set_num_threads(num_threads)

                    power_usages = []
                    energy_usages = []

                    for i in tqdm(range(0, min(len(embeddings), batch_size * 100), batch_size), total=(min((len(embeddings) // batch_size), 100)), desc="Processing batches", leave=False):
                        batch = embeddings[i:i + batch_size]
                        if len(batch) < batch_size:
                            break

                        power, energy = measure_search_power(index, batch, retrieved_docs)
                        power_usages.append(power)
                        energy_usages.append(energy)

                    avg_power = sum(power_usages) / len(power_usages) if power_usages else 0
                    avg_energy = sum(energy_usages) / len(energy_usages) if energy_usages else 0

                    writer.writerow([
                        args.index_name, args.nprobe, batch_size, retrieved_docs, num_threads,
                        avg_power, avg_energy
                    ])
                    f.flush()

if __name__ == "__main__":
    main()

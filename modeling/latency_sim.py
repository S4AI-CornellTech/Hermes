import csv
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description="Cluster Query Benchmark")
    parser.add_argument("--latency-data", type=str, required=True, help="Path to the profiled latency data csv file")
    parser.add_argument("--query-trace", type=str, required=True, help="Path to the cluster trace of accessed data")
    parser.add_argument("--retrieved-docs", type=int, required=True, help="List of numbers of docs retrieved per query")
    parser.add_argument("--batch-size", type=int, required=True, help="Batch size for querying")
    parser.add_argument("--num-threads", type=int, required=True, help="List of numbers of threads to run retrieval")
    parser.add_argument("--output-dir", type=str, default="data/modeling/", help="Directory where the results will be saved")
    return parser.parse_args()

def main():
    args = parse_arguments()
    
    # List to store latency values for rows matching the provided batch size.
    latencies = []
    
    with open(args.latency_data, "r", newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            try:
                # Check if the row's batch size matches the provided argument.
                if int(row["Batch Size"]) == args.batch_size:
                    # Convert the latency to a float and append it to the list.
                    latency = float(row["Avg Retrieval Latency (s)"])
                    latencies.append(latency)
            except (KeyError, ValueError):
                # Skip rows with missing or invalid data.
                continue

    if latencies:
        # Find the largest latency from the list.
        samping_latency = max(latencies)
        print(f"Largest latency (samping_latency): {samping_latency}")
    else:
        print("No latency data found for the specified batch size.")

if __name__ == "__main__":
    main()

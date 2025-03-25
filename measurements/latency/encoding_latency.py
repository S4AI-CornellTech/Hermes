import random
import time
import csv
import os
import argparse
from tqdm import tqdm
import torch
from transformers import AutoModel, AutoTokenizer

def get_gpu_type():
    """Returns the GPU type based on the available hardware."""
    if torch.cuda.is_available():
        return torch.cuda.get_device_name(0)
    return "CPU"

def generate_random_sequence(tokenizer, length):
    """
    Generates a random sequence of token IDs based on the tokenizer's vocabulary.
    """
    vocab_size = tokenizer.vocab_size
    return [random.randint(0, vocab_size - 1) for _ in range(length)]

def measure_latency(model, tokenizer, batch_size, input_length, iterations=1000):
    """
    Measures the average inference latency for a given batch size and token length.
    """
    latencies = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Use a nested progress bar for inference iterations (position=2)
    for _ in tqdm(range(iterations),
                  desc=f"Inference Iterations (input_length={input_length}, batch_size={batch_size})",
                  leave=False,
                  position=2):
        random_token_sequences = [generate_random_sequence(tokenizer, input_length) for _ in range(batch_size)]
        input_texts = [tokenizer.decode(seq, skip_special_tokens=True) for seq in random_token_sequences]
        inputs = tokenizer(input_texts, return_tensors="pt", padding=True, truncation=True).to(device)
        
        start_time = time.time()
        with torch.no_grad():
            model(**inputs)
        latencies.append(time.time() - start_time)
    
    # Sort and exclude the largest and smallest 2 values if enough iterations
    latencies.sort()
    trimmed_latencies = latencies[2:-2] if len(latencies) > 4 else latencies
    avg_latency = sum(trimmed_latencies) / len(trimmed_latencies)
    
    return avg_latency

def main():
    """
    Runs inference benchmarking and logs results to a CSV file.
    """
    parser = argparse.ArgumentParser(description="LLM Inference Benchmarking")
    parser.add_argument("--model-name", type=str, required=True, help="Model name to evaluate")
    parser.add_argument("--batch-sizes", type=int, nargs='+', required=True, help="List of batch sizes for inference")
    parser.add_argument("--input-lengths", type=int, nargs='+', required=True, help="List of input token lengths to test")
    parser.add_argument("--output-dir", type=str, default="data/profiling/", help="Directory to save the results")
    
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(args.output_dir, "encoding_latency.csv")
    
    model = AutoModel.from_pretrained(args.model_name).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    gpu_type = get_gpu_type()
    
    with open(output_file, mode='w', newline='') as file:
        fieldnames = ["Model Name", "GPU Type", "Batch Size", "Input Token Length", "Avg Latency (s)"]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        
        # Outer progress bar for input lengths (position=0)
        for input_length in tqdm(args.input_lengths, desc="Input Lengths", position=0):
            # Inner progress bar for batch sizes (position=1)
            for batch_size in tqdm(args.batch_sizes, desc=f"Batch Sizes (input_length={input_length})", leave=False, position=1):
                avg_latency = measure_latency(model, tokenizer, batch_size, input_length)
                writer.writerow({
                    "Model Name": args.model_name,
                    "GPU Type": gpu_type,
                    "Batch Size": batch_size,
                    "Input Token Length": input_length,
                    "Avg Latency (s)": avg_latency
                })

if __name__ == "__main__":
    main()

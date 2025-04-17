import random
import time
import csv
import os
import argparse
from tqdm import tqdm
import torch
from transformers import AutoModel, AutoTokenizer
import pynvml

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

def measure_power(model, tokenizer, batch_size, input_length, handle, iterations=1000):
    """
    Measures the average inference power (in watts) for a given batch size and token length.
    It queries the GPU power before and after inference and averages the two readings.
    """
    power_readings = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Nested progress bar for iterations
    for _ in tqdm(range(iterations),
                  desc=f"Inference Iterations (input_length={input_length}, batch_size={batch_size})",
                  leave=False,
                  position=2):
        # Generate a batch of random sequences and decode them to text
        random_token_sequences = [generate_random_sequence(tokenizer, input_length) for _ in range(batch_size)]
        input_texts = [tokenizer.decode(seq, skip_special_tokens=True) for seq in random_token_sequences]
        inputs = tokenizer(input_texts, return_tensors="pt", padding=True, truncation=True).to(device)
        
        # Measure power before and after inference (convert mW to W by dividing by 1000)
        power_start = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000
        with torch.no_grad():
            model(**inputs)
        power_end = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000
        
        # Average the two power readings
        average_power = (power_start + power_end) / 2
        power_readings.append(average_power)
    
    # Exclude the highest and lowest two readings if enough iterations are available
    power_readings.sort()
    trimmed_readings = power_readings[2:-2] if len(power_readings) > 4 else power_readings
    avg_power = sum(trimmed_readings) / len(trimmed_readings)
    
    return avg_power

def main():
    """
    Runs inference benchmarking for power consumption and logs results to a CSV file.
    """
    parser = argparse.ArgumentParser(description="LLM Inference Power Benchmarking")
    parser.add_argument("--model-name", type=str, required=True, help="Model name to evaluate")
    parser.add_argument("--batch-size", type=int, nargs='+', required=True, help="List of batch sizes for inference")
    parser.add_argument("--input-lengths", type=int, nargs='+', required=True, help="List of input token lengths to test")
    parser.add_argument("--output-dir", type=str, default="data/profiling/", help="Directory to save the results")
    
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(args.output_dir, "encoding_power.csv")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModel.from_pretrained(args.model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    gpu_type = get_gpu_type()
    
    # Initialize NVML and get the handle for the first GPU
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    
    with open(output_file, mode='w', newline='') as file:
        fieldnames = ["Model Name", "GPU Type", "Batch Size", "Input Token Length", "Avg Power (W)"]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        
        # Outer progress bar for input lengths
        for input_length in tqdm(args.input_lengths, desc="Input Lengths", position=0):
            # Inner progress bar for batch sizes
            for batch_size in tqdm(args.batch_size, desc=f"Batch Sizes (input_length={input_length})", leave=False, position=1):
                avg_power = measure_power(model, tokenizer, batch_size, input_length, handle)
                writer.writerow({
                    "Model Name": args.model_name,
                    "GPU Type": gpu_type,
                    "Batch Size": batch_size,
                    "Input Token Length": input_length,
                    "Avg Power (W)": avg_power,
                })

if __name__ == "__main__":
    main()

import random
import time
import csv
import os
import argparse
import torch
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from tqdm import tqdm

def get_gpu_type():
    """Returns the GPU type based on the available hardware."""
    if torch.cuda.is_available():
        return torch.cuda.get_device_name(0)
    return "CPU"

def generate_random_sequence(tokenizer, length):
    """
    Generates a random sequence of token IDs based on the tokenizer's vocabulary.
    """
    vocab_keys = list(tokenizer.get_vocab().keys())
    random_text = " ".join(random.choices(vocab_keys, k=length))
    tokenized_output = tokenizer.encode(random_text, truncation=True, max_length=length, add_special_tokens=False)
    
    while len(tokenized_output) < length:
        tokenized_output.append(tokenizer.convert_tokens_to_ids(random.choice(vocab_keys)))
    
    return tokenized_output[:length]

def measure_latency(llm, tokenizer, batch_size, input_length, output_length, iterations=30, warmup_runs=5):
    """
    Measures the average prefill and decode latency for a given batch size and token lengths.
    """
    prefill_times, decode_times = [], []
    
    for _ in tqdm(range(iterations),
                    desc=f"Iterations (batch_size={batch_size}, input_length={input_length}, output_length={output_length})",
                    leave=False, position=3):
        inputs = [tokenizer.decode(generate_random_sequence(tokenizer, input_length), skip_special_tokens=False)
                  for _ in range(batch_size)]
        
        # Warmup runs
        sampling_params_prefill = SamplingParams(min_tokens=0, max_tokens=1)
        for _ in range(warmup_runs):
            llm.generate(inputs, sampling_params_prefill)
        
        # Measure prefill time
        prefill_start = time.time()
        llm.generate(inputs, sampling_params_prefill)
        prefill_times.append(time.time() - prefill_start)
        
        # Measure decode time
        sampling_params_decode = SamplingParams(min_tokens=output_length-1, max_tokens=output_length)
        decode_start = time.time()
        llm.generate(inputs, sampling_params_decode)
        decode_times.append(time.time() - decode_start)
    
    avg_prefill = sum(prefill_times) / iterations
    avg_decode = sum(decode_times) / iterations
    return avg_prefill, avg_decode

def main():
    """
    Runs inference experiments and logs results to a CSV file.
    """
    parser = argparse.ArgumentParser(description="LLM Inference Benchmarking")
    parser.add_argument("--model-name", type=str, required=True, help="Model name to evaluate")
    parser.add_argument("--num-gpus", type=int, required=True, help="Number of GPUs to use")
    parser.add_argument("--batch-size", type=int, nargs='+', required=True, help="List of batch sizes for inference")
    parser.add_argument("--input-lengths", type=int, nargs='+', required=True, help="List of input token lengths to test")
    parser.add_argument("--output-lengths", type=int, nargs='+', required=True, help="List of output token lengths to test")
    parser.add_argument("--output-dir", type=str, default="data/profiling/", help="Directory where the results will be saved")
    
    args = parser.parse_args()
    
    os.environ["VLLM_ATTENTION_BACKEND"] = "FLASH_ATTN"
    gpu_type = get_gpu_type()
    
    # Ensure the output directory exists
    output_dir = os.path.dirname(args.output_dir)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    output_file = os.path.join(args.output_dir, "inference_latency.csv")
    with open(output_file, mode='w', newline='') as file:
        fieldnames = ["Model Name", "GPU Type", "Num GPUs", "Batch Size", "Input Token Length", 
                      "Output Token Length", "Avg Prefill Time (s)", "Avg Decode Time (s)"]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        
        llm = LLM(
            model=args.model_name, tensor_parallel_size=args.num_gpus, dtype="float16",
            kv_cache_dtype="auto", gpu_memory_utilization=0.9,
            enable_prefix_caching=True, trust_remote_code=True
        )
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        
        for input_length in tqdm(args.input_lengths, desc="Input Lengths", position=0):
            for output_length in tqdm(args.output_lengths, desc=f"Output Lengths (input_length={input_length})", position=1, leave=False):
                for batch_size in tqdm(args.batch_size, desc=f"Batch Sizes (input_length={input_length}, output_length={output_length})", position=2, leave=False):
                    avg_prefill, avg_decode = measure_latency(llm, tokenizer, batch_size, input_length, output_length)
                    
                    writer.writerow({
                        "Model Name": args.model_name,
                        "GPU Type": gpu_type,
                        "Num GPUs": args.num_gpus,
                        "Batch Size": batch_size,
                        "Input Token Length": input_length,
                        "Output Token Length": output_length,
                        "Avg Prefill Time (s)": avg_prefill,
                        "Avg Decode Time (s)": avg_decode - avg_prefill  # Isolated decode time
                    })

if __name__ == "__main__":
    main()

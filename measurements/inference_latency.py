import random
import time
import csv
import os
import argparse
import torch
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

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
    Measures the average prefill and decode latency for given batch size and token lengths.
    """
    prefill_times, decode_times = [], []
    
    for _ in range(iterations):
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
    
    return sum(prefill_times) / iterations, sum(decode_times) / iterations

def main():
    """
    Runs inference experiments and logs results to a CSV file.
    """
    parser = argparse.ArgumentParser(description="LLM Inference Benchmarking")
    parser.add_argument("--model_name", type=str, required=True, help="Model name to evaluate")
    parser.add_argument("--num_gpus", type=int, required=True, help="Number of GPUs to use")
    parser.add_argument("--batch_size", type=int, required=True, help="Batch size for inference")
    parser.add_argument("--input_lengths", type=int, nargs='+', required=True, help="List of input token lengths to test")
    parser.add_argument("--output_lengths", type=int, nargs='+', required=True, help="List of output token lengths to test")
    parser.add_argument("--output_file", type=str, required=True, help="CSV file path to store results")
    
    args = parser.parse_args()
    
    os.environ["VLLM_ATTENTION_BACKEND"] = "FLASH_ATTN"
    gpu_type = get_gpu_type()
    
    with open(args.output_file, mode='w', newline='') as file:
        fieldnames = ["Model Name", "GPU Type", "Num GPUs", "Batch Size", "Input Token Length", 
                      "Output Token Length", "Avg Prefill Time (s)", "Avg Decode Time (s)"]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        
        llm = LLM(
            model=args.model_name, tensor_parallel_size=args.num_gpus, dtype="float16",
            kv_cache_dtype="auto", gpu_memory_utilization=0.9,
            max_num_batched_tokens=2048, max_model_len=2048,
            enable_prefix_caching=True, trust_remote_code=True
        )
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        
        for input_length in args.input_lengths:
            for output_length in args.output_lengths:
                avg_prefill, avg_decode = measure_latency(llm, tokenizer, args.batch_size, input_length, output_length)
                
                writer.writerow({
                    "Model Name": args.model_name, "GPU Type": gpu_type, "Num GPUs": args.num_gpus,
                    "Batch Size": args.batch_size, "Input Token Length": input_length,
                    "Output Token Length": output_length,
                    "Avg Prefill Time (s)": avg_prefill,
                    "Avg Decode Time (s)": avg_decode - avg_prefill  # Isolated decode time
                })

if __name__ == "__main__":
    main()
# Hermes

Hermes is a public, open-source evaluation framework implementing the methodology described in our paper:

## **"Hermes: Algorithm-System Co-design for Efficient Retrieval Augmented Generation At-Scale"**

<img src="images/Hermes.png" alt="Hermes" style="width:100%;">

---

## Overview

Hermes is an algorithm-system co-design framework that intelligently distributes search clusters across multiple machines, employing hierarchical search and Dynamic Voltage and Frequency Scaling (DVFS) to optimize retrieval latency and energy consumption. Built on open-source LLMs and retrieval indices from publicly available datasets, running on commodity hardware, Hermes achieves:

✅ **10x reduction in latency**  
✅ **2x improvement in energy efficiency**

<img src="images/HermesArchitecture.png" alt="Hermes" style="width:100%;">

📖 **Read our full paper:** [here](https://anonymous.com)

🔗 **If you use Hermes in your research, please cite us:**  
```Citation```

---

## Open Source Datasets and Models

Hermes leverages publicly available datasets:

📂 **Datasets**
- [SPHERE_899M](https://huggingface.co/datasets/mohdumar/SPHERE_899M) – BERT Encoded 899M Subset of Common Crawl
- [SPHERE_100M](https://huggingface.co/datasets/mohdumar/SPHERE_100M) – BERT Encoded 100M Subset of Common Crawl
- [SPHERE_100K](https://huggingface.co/datasets/mohdumar/SPHERE_100K) – BERT Encoded 100K Subset of Common Crawl
- [TriviaQA](https://nlp.cs.washington.edu/triviaqa/) – Open-domain QA dataset

🧠 **Models**
- [GEMMA-2 9B](https://huggingface.co/google/gemma-2-9b)
- [BGE-Large](https://huggingface.co/BAAI/bge-large-en)
- [BERT-Base-Uncased](https://huggingface.co/google-bert/bert-base-uncased)

---

## Quickstart Guide

1. **[Environment Setup](#setup)**
2. **[Create Monolithic, Split, and Hermes Cluster Search Indices](#datastore-creation)**
3. **[Profile Search Latencies, Recall, and Energy](#retrieval)**
4. **[Profile Latency and Energy of Encoding & Inference](#inference)**
5. **[Generate Cluster Access Traces](#trace-generator)**
6. **[Run Multi-Node Aggregation Analysis](#multi-node-aggregation)**

---

## Setup

### Clone the Repository

Hermes uses Git Large File Storage (LFS). Install LFS [here](https://git-lfs.com/) and clone the repository:

```bash
git clone https://github.com/Michaeltshen/Hermes.git
cd Hermes
git lfs pull
```

Alternatively, manually download the required files and place them in the corresponding folders:
- `triviaqa_encodings.npy` → `triviaqa/`

### Development Environment

1. **Create a Conda Environment:**
    ```bash
    conda create -n hermes python=3.11
    conda activate hermes
    ```
2. **Install Dependencies:**
    ```bash
    conda install -c pytorch -c nvidia faiss-gpu=1.8.0 pytorch=*=*cuda* pytorch-cuda=11 numpy
    pip install transformers vllm datasets
    ```
3. **Torchvision Dependency Corrections:**

    If you encounter CUDA version mismatches between PyTorch and torchvision, run the following command to automatically detect and install the correct torchvision version for your setup:
    ```bash
    source setup/torchvision_version_fix.sh
    ```
    This script ensures that torchvision matches your installed PyTorch version and CUDA compatibility, preventing runtime errors.

---

## Datastore Creation

### SPHERE Index Creation

Building large-scale indices (e.g., 1B–10B vectors) can take days or weeks. These indices can also reach hundreds of gigabytes in size. Choose an approach based on your requirements:

🔹 **Monolithic Index**
```bash
python index/hermes_create_monolithic_index.py --index-size 100K
```
🔹 **Evenly Split Indices**
```bash
python index/hermes_create_split_indices.py --dataset-size 100k --num-indices 10
```
🔹 **Clustered Hermes Indices**
```bash
python index/hermes_create_clustered_indices.py --dataset-size 100k --num-indices 10
```

📌 **Custom Datasets:** Modify the dataset loading logic in the index creation files.

### Synthetic Indices

For benchmarking, create synthetic indices:
```bash
python index/synthetic_create_monolithic_index.py --index-size 1m --dim 768 --threads 32
```

---

## Hardware Profiling

Measure retrieval latency and energy performance of Hermes.

📌 **Pre-profiled results available at:** [Website]

### **Retrieval Profiling**

Run retrieval latency tests:
```bash
python measurements/retrieval_latency.py \
    --index-name index/indices/monolithic_indices/hermes_index_monolithic_100k.faiss \
    --nprobe 128 --batch-size 32 --queries triviaqa/triviaqa_encodings.npy \
    --retrieved-docs 5 --num-threads 32
```

### **Encoding & Inference Profiling**

⚡ **Measure Latency and Power Usage** for different encoding and inference models.

---

## Multi Node Aggregation Tool

<img src="images/MultiNodeAggregation.png" alt="MultiNodeAggregation" style="width:100%;">

This tool models and aggregates data across multiple nodes for system performance optimization.

📌 **Includes:**
- **Trace Generator** – Generate cluster access traces.
- **Multi-Node Aggregation** – Analyze RAG inference latency and energy usage.

---

## License
This project is licensed under the **MIT License**. See the LICENSE file for full details.


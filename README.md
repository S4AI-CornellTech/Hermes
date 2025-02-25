# Hermes

Hermes is a public, open‐source evaluation framework implementation for the methodology described in the paper: **"Hermes: Algorithm-System Co-design for Efficient Retrieval Augmented Generation At-Scale"**

---

## Paper

Read the full paper [here](https://anonymous.com).

If you use Hermes in your research, please cite us:

```Citation```

---

## Open Source Datasets and Models

This framework leverages several publicly available datasets:

- **SPHERE_899M**: [Explore on Hugging Face](https://huggingface.co/datasets/mohdumar/SPHERE_899M)
- **SPHERE_100M**: [Explore on Hugging Face](https://huggingface.co/datasets/mohdumar/SPHERE_100M)
- **SPHERE_100K**: [Explore on Hugging Face](https://huggingface.co/datasets/mohdumar/SPHERE_100K)

---

## Overview

Hermes is an algorithm-system co-design framework that intelligently distributes search clusters across multiple machines, employing hierarchical search along with Dynamic Voltage and Frequency Scaling (DVFS) to significantly reduce retrieval latency and energy consumption. Using open-source LLMs and retrieval indices built on publicly available datasets running on commodity hardware platforms, Hermes achieves:

- **10x reduction in latency**
- **2x improvement in energy efficiency**

<img src="images/Hermes.png" alt="Hermes">

---

## Setup

### Environment

To set up your development environment:

1. **Create a Conda Environment:**

   ```bash
   conda create -n hermes python=3.11
   conda activate hermes

## Latency Profiling
This section is dedicated to measuring and profiling the retrieval latency performance of Hermes. Detailed instructions and scripts will be provided to help you analyze and optimize the system's latency.

## Multi Node Aggregation Tool (Modeling Tool)

<img src="images/MultiNodeAggregation.png" alt="MultiNodeAggregation">

This component focuses on modeling and aggregating data across multiple nodes to enhance system performance and resource utilization. Documentation and usage instructions for the multi-node aggregation tool will be provided here.

## License
This project is licensed under the MIT License. See the LICENSE file for full details.
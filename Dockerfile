FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Install basic utilities and sudo
RUN apt-get update && \
    apt-get install -y wget git sudo build-essential libglib2.0-0 libsm6 libxrender1 libxext6 && \
    rm -rf /var/lib/apt/lists/*

# Install Miniconda
ENV CONDA_DIR /opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    bash ~/miniconda.sh -b -p $CONDA_DIR && \
    rm ~/miniconda.sh

ENV PATH=$CONDA_DIR/bin:$PATH
SHELL ["/bin/bash", "-c"]

# Create the environment and init shell support
RUN conda create -n hermes python=3.11 -y && \
    conda init bash && \
    echo "conda activate hermes" >> ~/.bashrc

# Clone Hermes repo
RUN git clone --recurse-submodules https://github.com/Michaeltshen/Hermes.git

# Copy the file into the repo
COPY triviaqa/triviaqa_encodings.npy /Hermes/triviaqa/triviaqa_encodings.npy

# Set working directory
WORKDIR /Hermes

# Install dependencies inside the hermes env
RUN conda run -n hermes conda install -c pytorch -c nvidia faiss-gpu=1.8.0 pytorch=*=*cuda* pytorch-cuda=11 numpy -y && \
    conda run -n hermes conda install -c conda-forge gcc_linux-64 gxx_linux-64 -y && \
    conda run -n hermes pip install transformers vllm datasets pynvml matplotlib pyRAPL pymongo

# Fix torchvision version
RUN conda run -n hermes bash setup/torchvision_version_fix.sh

# Enable reading energy values
RUN sudo chmod -R a+r /sys/class/powercap/intel-rapl/ || true && \
    sudo chmod a+r /sys/class/powercap/intel-rapl/intel-rapl:0/energy_uj || true

# Compile energy measurement tool
WORKDIR /Hermes/uarch-configure/rapl-read
RUN make

# Final working directory
WORKDIR /Hermes


#!/bin/bash

# Detect PyTorch and CUDA version
PYTORCH_VERSION=$(python -c "import torch; print(torch.__version__)")
CUDA_VERSION=$(python -c "import torch; print(torch.version.cuda)")

# Convert CUDA version from format "11.8" to "cu118"
CUDA_VERSION_SHORT="cu$(echo $CUDA_VERSION | tr -d '.')"

echo "Detected PyTorch Version: $PYTORCH_VERSION"
echo "Detected CUDA Version: $CUDA_VERSION (short format: $CUDA_VERSION_SHORT)"

# Uninstall incorrect torchvision version
echo "Uninstalling torchvision..."
pip uninstall -y torchvision

# Install the correct torchvision version
echo "Installing torchvision for PyTorch version $PYTORCH_VERSION and CUDA $CUDA_VERSION..."
pip install torchvision --no-cache-dir --index-url "https://download.pytorch.org/whl/$CUDA_VERSION_SHORT"

# Verify installation
python -c "import torchvision; print('torchvision version:', torchvision.__version__)" && echo "Installation successful!" || echo "Installation failed. Check for errors."

#!/bin/bash
#
# Setup script for micro-TCN training on PACE ICE cluster
# Run this once to create the conda environment with all dependencies
#
# Usage: bash setup_environment.sh
#

set -e  # Exit on error

echo "======================================================"
echo "micro-TCN PACE Environment Setup"
echo "======================================================"
echo ""

# Configuration
ENV_NAME="micro-tcn-env"
PYTHON_VERSION="3.10"

echo "Step 1: Loading required modules..."
module load anaconda3/2023.03
module load cuda/11.8

echo ""
echo "Step 2: Creating conda environment '$ENV_NAME'..."
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "Environment '$ENV_NAME' already exists."
    read -p "Do you want to remove and recreate it? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        conda env remove -n $ENV_NAME -y
        echo "Removed existing environment."
    else
        echo "Keeping existing environment. Skipping creation."
        exit 0
    fi
fi

conda create -n $ENV_NAME python=$PYTHON_VERSION -y

echo ""
echo "Step 3: Activating environment..."
source activate $ENV_NAME

echo ""
echo "Step 4: Installing PyTorch with CUDA 11.8 support..."
pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu118

echo ""
echo "Step 5: Installing project dependencies..."
pip install -r requirements.txt

echo ""
echo "Step 6: Installing project in development mode..."
pip install -e .

echo ""
echo "Step 7: Verifying installation..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}')"
python -c "import pytorch_lightning as pl; print(f'PyTorch Lightning version: {pl.__version__}')"

echo ""
echo "======================================================"
echo "Setup Complete!"
echo "======================================================"
echo ""
echo "Environment '$ENV_NAME' is ready for training."
echo ""
echo "To activate this environment, run:"
echo "  module load anaconda3/2023.03"
echo "  module load cuda/11.8"
echo "  conda activate $ENV_NAME"
echo ""
echo "Next steps:"
echo "  1. Review train_single_model.sbatch"
echo "  2. Run: bash submit_all_models.sh"
echo ""

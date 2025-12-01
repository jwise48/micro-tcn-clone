#!/bin/bash
# Run this ONCE before submitting jobs
echo "Removing conda env: micro-tcn-env..."
echo ""

rm -rf ~/.conda/envs/micro-tcn-env

echo "Loading modules..."
echo ""

module load anaconda3/2023.03
module load cuda/11.8

ENV_NAME="micro-tcn-env"
PYTHON_VERSION="3.10"

# Create environment if it doesn't exist
if ! conda env list | grep -q "^${ENV_NAME} "; then
    echo "Creating conda environment..."
    conda create -n $ENV_NAME python=$PYTHON_VERSION -y
    conda activate $ENV_NAME
    
    echo "Installing PyTorch..."
    pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 \
        --index-url https://download.pytorch.org/whl/cu118
    
    echo "Installing dependencies..."
    pip install -r requirements.txt
    
    echo "Environment setup complete!"
else
    echo "Environment already exists!"
fi


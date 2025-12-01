#!/bin/bash
#SBATCH -J micro-tcn-postprocess        # Job name
#SBATCH --account=musi                  # PACE account
#SBATCH --qos=pace-ice                  # quality of service
#SBATCH --gres=gpu:H100:1               # Request 1 H100 GPU for speed tests
#SBATCH --mem=128GB                     # Reduced memory for post-processing
#SBATCH --ntasks-per-node=1             # Number of tasks per node
#SBATCH --cpus-per-task=16              # CPU cores (for data loading)
#SBATCH --time=2:00:00                  # Wall time (2 hours should be sufficient)
#SBATCH --mail-type=BEGIN,END,FAIL      # Email notifications
#SBATCH --mail-user=jwise48@gatech.edu  # Email address
#SBATCH -o logs/postprocess-speed-report-%j.out
#SBATCH -e logs/postprocess-error-%j.err

set -e  # Exit on error

echo "======================================================"
echo "micro-TCN Post-Processing: Test, Speed, Plot"
echo "======================================================"
echo ""

# Configuration
#ENV_NAME="micro-tcn-env"

#echo "Step 1: Loading required modules..."
#module load anaconda3/2023.03
#module load cuda/11.8

#echo ""
#echo "Step 2: Activating environment..."
#conda activate $ENV_NAME


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


echo ""
echo "Step 3: Displaying GPU information..."
nvidia-smi

# ============================================================================
# Dataset Configuration
# ============================================================================
DATASET_PATH="/home/hice1/jwise48/scratch/data/micro-tcn/SignalTrain_LA2A_Dataset_1.1"
MODEL_DIR="./lightning_logs/bulk"
SAVE_DIR="evaluations"

echo ""
echo "Checking paths..."
if [ ! -d "$DATASET_PATH" ]; then
    echo "ERROR: Dataset not found at $DATASET_PATH"
    exit 1
fi
echo "Dataset found: $DATASET_PATH"

if [ ! -d "$MODEL_DIR" ]; then
    echo "ERROR: Model directory not found at $MODEL_DIR"
    exit 1
fi
echo "Model directory found: $MODEL_DIR"

# Create evaluation directory
mkdir -p $SAVE_DIR

# ============================================================================
# Testing Phase
# ============================================================================
echo ""
echo "======================================================"
echo "Running test.py to evaluate all trained models"
echo "======================================================"
echo ""

#srun python test.py \
#   --root_dir ${DATASET_PATH} \
#   --model_dir ${MODEL_DIR} \
#   --save_dir ${SAVE_DIR} \
#   --num_workers 14

echo ""
echo "======================================================"
echo "Running speed.py for performance benchmarks"
echo "======================================================"
echo ""

srun python speed.py --plot --rf

echo ""
echo "======================================================"
echo "Running plot.py to generate visualizations"
echo "======================================================"
echo ""

srun python plot.py

echo ""
echo "======================================================"
echo "Post-processing completed successfully!"
echo "======================================================"
echo ""
echo "Results saved to: ${SAVE_DIR}"
echo "Test results: test_results_val.p"
echo ""

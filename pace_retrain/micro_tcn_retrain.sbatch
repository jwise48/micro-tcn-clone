#!/bin/bash
#SBATCH -J micro-tcn-model              # Job name with array task ID
#SBATCH --account=musi                  # PACE account
#SBATCH --qos=pace-ice                  # quality of service
#SBATCH --gres=gpu:H100:1               # Request 1 H100 GPU
#SBATCH --mem=224GB                     # Request all memory on the node
#SBATCH --ntasks-per-node=1             # Number of tasks per node
#SBATCH --cpus-per-task=12              # CPU cores (for data loading)
#SBATCH --time=4:00:00                  # Wall time (6 hours per model, conservative)
#SBATCH --mail-type=BEGIN,END,FAIL      # Email notifications
#SBATCH --mail-user=jwise48@gatech.edu  # Email address
#SBATCH -o logs/model-%a-report-%j.out  # Output file (array task ID and job ID)
#SBATCH -e logs/model-%a-error-%j.err   # Error file (array task ID and job ID)

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
else
    echo "creating conda environment"	
    conda create -n $ENV_NAME python=$PYTHON_VERSION
fi

echo ""
echo "Step 3: Activating environment..."
source activate $ENV_NAME

echo ""
echo "Step 4: Installing PyTorch with CUDA 11.8 support..."
pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu118

echo ""
echo "Step 5: Installing project dependencies..."
pip install -r requirements.txt

MODEL_NAME=${1:-"uTCN-100-C"}  # Default to uTCN-100-C if not provided

echo "Model to train: $MODEL_NAME"
echo ""

echo ""
echo "Step 4: Displaying GPU information..."
nvidia-smi

# ============================================================================
# Dataset Configuration
# ============================================================================
DATASET_PATH="/home/hice1/jwise48/scratch/data/micro-tcn/SignalTrain_LA2A_Dataset_1.1"
EXPORT_DIR="./models_trained"

echo ""
echo "Step 5: Checking dataset..."
if [ ! -d "$DATASET_PATH" ]; then
    echo "ERROR: Dataset not found at $DATASET_PATH"
    exit 1
fi
echo "Dataset found: $DATASET_PATH"

# Create export directory if it doesn't exist
mkdir -p $EXPORT_DIR

# ============================================================================
# Training Configuration
# Optimized for H100-80GB GPU
# ============================================================================
BATCH_SIZE=384         # Optimized for 80GB memory
NUM_WORKERS=1         # Match cpus-per-task
MAX_EPOCHS=60          # Standard training length
PRECISION="16-mixed"   # FP16 mixed precision for Tensor Cores

echo ""
echo "Training Configuration:"
echo "  Batch Size:    $BATCH_SIZE"
echo "  Num Workers:   $NUM_WORKERS"
echo "  Max Epochs:    $MAX_EPOCHS"
echo "  Precision:     $PRECISION"
echo ""


# Run the training script
srun python train.py \
    --model_name "${MODEL_NAME}" \
    --root_dir ${DATASET_PATH} \
    --batch_size ${BATCH_SIZE} \
    --num_workers ${NUM_WORKERS} \
    --max_epochs ${MAX_EPOCHS} \
    --precision ${PRECISION} \
    --accelerator gpu \
    --devices 1 \
    --export \
    --save_dir ${EXPORT_DIR}

srun python test.py \
    --root_dir ${DATASET_PATH} \
    --model_dir ${EXPORT_DIR} \
    --save_dir "evaluations" \
    --num_workers ${NUM_WORKERS}

srun python speed.py --plot --full --gpu --rf

srun python plot.py

#!/bin/bash
#SBATCH -J micro-tcn-model              # Job name with array task ID
#SBATCH --account=musi                  # PACE account
#SBATCH --qos=pace-ice                  # quality of service
#SBATCH --gres=gpu:H100:1               # Request 1 H100 GPU
#SBATCH --mem=256GB                     # Optimized memory for batch_size=512
#SBATCH --ntasks-per-node=1             # Number of tasks per node
#SBATCH --cpus-per-task=16              # CPU cores (for data loading)
#SBATCH --time=6:30:00                  # Wall time (6 hours per model, conservative)
#SBATCH --mail-type=BEGIN,END,FAIL      # Email notifications
#SBATCH --mail-user=jwise48@gatech.edu  # Email address
#SBATCH -o logs/model-%a-report-%j.out  # Output file (array task ID and job ID)
#SBATCH -e logs/model-%a-error-%j.err   # Error file (array task ID and job ID)

set -e  # Exit on error
echo "======================================================"
echo "micro-TCN PACE Environment Setup"
echo "======================================================"
echo ""

# Show available memory before training
echo "Available memory before training:"
free -h
echo ""
echo "---"
echo ""
echo "---"
echo ""

# Configuration
ENV_NAME="micro-tcn-env"
PYTHON_VERSION="3.10"

echo "Step 1: Loading required modules..."
module load anaconda3/2023.03
module load cuda/11.8

echo ""
echo "Step 2: Activating environment..."
conda activate $ENV_NAME

MODEL_NAME=${1:-"uTCN-100-C"}  # Default to uTCN-100-C if not provided

echo "Model to train: $MODEL_NAME"
echo ""

echo ""
echo "Step 3: Displaying GPU information..."
nvidia-smi

# ============================================================================
# Dataset Configuration
# ============================================================================
DATASET_PATH="/home/hice1/jwise48/scratch/data/micro-tcn/SignalTrain_LA2A_Dataset_1.1"
EXPORT_DIR="./models_trained"

echo ""
echo "Step 4: Checking dataset..."
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
BATCH_SIZE=512         # Optimized for H100-80GB GPU
NUM_WORKERS=14         # Optimized for 16 CPUs (leave 2 for system)
MAX_EPOCHS=120          # Standard training length
PRECISION="16-mixed"   # FP16 mixed precision for Tensor Cores

echo ""
echo "Training Configuration:"
echo "  Batch Size:    $BATCH_SIZE"
echo "  Num Workers:   $NUM_WORKERS"
echo "  Max Epochs:    $MAX_EPOCHS"
echo "  Precision:     $PRECISION"
echo ""

# Show available memory before training
echo "Available memory before training:"
free -h
echo "---"
echo ""
echo "---"
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
   --preload \
   --devices 1 \
   --eval_length 65536 \
   --export \
   --save_dir ${EXPORT_DIR}

echo "======================================================"
echo "Training completed for ${MODEL_NAME}"
echo "======================================================"
echo ""
echo "Note: Run pace_retrain/post_process.sbatch after all models finish training"
echo "to execute test.py, speed.py, and plot.py on all trained models."
echo ""

#!/bin/bash
#SBATCH -J micro-tcn-model              # Job name with array task ID
#SBATCH --account=musi                  # PACE account
#SBATCH --qos=coe-ice                   # Quality of service
#SBATCH --partition=coe-gpu             # GPU partition
#SBATCH --gres=gpu:H100:1               # Request 1 H100 GPU
#SBATCH --mem-per-gpu=224GB             # Memory per GPU (224GB for A100-80GB nodes)
#SBATCH --ntasks-per-node=1             # Number of tasks per node
#SBATCH --cpus-per-task=12              # CPU cores (for data loading)
#SBATCH --time=6:00:00                  # Wall time (6 hours per model, conservative)
#SBATCH --mail-type=BEGIN,END,FAIL      # Email notifications
#SBATCH --mail-user=jwise48@gatech.edu  # Email address
#SBATCH -o logs/model-%a-report-%j.out  # Output file (array task ID and job ID)
#SBATCH -e logs/model-%a-error-%j.err   # Error file (array task ID and job ID)

set -e  # Exit on error

echo "Step 1: Loading modules..."
module load anaconda3/2023.03
module load cuda/11.8
module list

echo ""
echo "Step 2: Activating conda environment..."
source activate micro-tcn-env

echo ""
echo "Step 3: Verifying CUDA and PyTorch..."
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"

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
NUM_WORKERS=12         # Match cpus-per-task
MAX_EPOCHS=60          # Standard training length
PRECISION="16-mixed"   # FP16 mixed precision for Tensor Cores

echo ""
echo "Training Configuration:"
echo "  Batch Size:    $BATCH_SIZE"
echo "  Num Workers:   $NUM_WORKERS"
echo "  Max Epochs:    $MAX_EPOCHS"
echo "  Precision:     $PRECISION"
echo "  Preload Data:  Yes (entire dataset in RAM)"
echo ""


# Run the training script
python train.py \
    --model_name "uTCN-100-C" \
    --root_dir ${DATASET_PATH} \
    --preload \
    --batch_size ${BATCH_SIZE} \
    --num_workers ${NUM_WORKERS} \
    --max_epochs ${MAX_EPOCHS} \
    --precision ${PRECISION} \
    --accelerator gpu \
    --devices 1 \
    --export \
    --save_dir ${EXPORT_DIR}


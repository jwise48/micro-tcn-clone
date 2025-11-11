#!/bin/bash
#SBATCH -J micro-tcn-model-%a           # Job name with array task ID
#SBATCH --account=musi                  # PACE account
#SBATCH --qos=coe-ice                   # Quality of service
#SBATCH --partition=coe-gpu             # GPU partition
#SBATCH --gres=gpu:A100:1               # Request 1 A100 GPU
#SBATCH -C A100-80GB                    # Specifically request A100-80GB variant
#SBATCH --mem-per-gpu=224GB             # Memory per GPU (224GB for A100-80GB nodes)
#SBATCH --ntasks-per-node=1             # Number of tasks per node
#SBATCH --cpus-per-task=12              # CPU cores (for data loading)
#SBATCH --time=6:00:00                  # Wall time (6 hours per model, conservative)
#SBATCH --mail-type=BEGIN,END,FAIL      # Email notifications
#SBATCH --mail-user=jwise48@gatech.edu  # Email address
#SBATCH -o logs/model-%a-report-%j.out  # Output file (array task ID and job ID)
#SBATCH -e logs/model-%a-error-%j.err   # Error file (array task ID and job ID)

#
# PACE SLURM Training Script for micro-TCN
# Trains a single model configuration with optimized A100-80GB settings
#
# Usage: sbatch --array=1-7 train_single_model.sbatch
#        sbatch --array=3 train_single_model.sbatch     # Train only model 3
#

set -e  # Exit on error

# ============================================================================
# SLURM Job Information
# ============================================================================
echo "======================================================"
echo "micro-TCN Model Training - PACE ICE Cluster"
echo "======================================================"
echo ""
echo "---------------------------------------"
echo "Begin Slurm Prolog: $(date)"
echo "Job ID:        ${SLURM_JOB_ID}"
echo "Array Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "User ID:       ${USER}"
echo "Account:       ${SLURM_JOB_ACCOUNT}"
echo "Partition:     ${SLURM_JOB_PARTITION}"
echo "Node:          ${SLURMD_NODENAME}"
echo "---------------------------------------"
echo ""

# ============================================================================
# Environment Setup
# ============================================================================
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
# Model Selection
# ============================================================================
# The array task ID determines which model to train
MODEL_INDEX=${SLURM_ARRAY_TASK_ID}

echo ""
echo "======================================================"
echo "Training Model $MODEL_INDEX of 7"
echo "======================================================"

# ============================================================================
# Training Configuration
# Optimized for A100-80GB GPU
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

# ============================================================================
# Modified Training Script
# Uses a custom approach to train only the selected model
# ============================================================================
cd $SLURM_SUBMIT_DIR

echo "Working directory: $(pwd)"
echo ""
echo "Starting training at $(date)..."
echo ""

# Create a temporary Python script to train only the selected model
cat > train_model_${MODEL_INDEX}.py << 'EOPYTHON'
import os
import sys
import torch
import torchsummary
import pytorch_lightning as pl
from argparse import ArgumentParser

from microtcn.tcn import TCNModel
from microtcn.lstm import LSTMModel
from microtcn.data import SignalTrainLA2ADataset
from export import export_model

torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('medium')

# Model configurations (same as train.py)
train_configs = [
    {"name" : "uTCN-100-C",
     "model_type" : "tcn",
     "nblocks" : 4,
     "dilation_growth" : 10,
     "kernel_size" : 5,
     "causal" : True,
     "train_fraction" : 1.00
    },
    {"name" : "uTCN-300-C",
     "model_type" : "tcn",
     "nblocks" : 4,
     "dilation_growth" : 10,
     "kernel_size" : 13,
     "causal" : True,
     "train_fraction" : 1.00
    },
    {"name" : "uTCN-1000-C",
     "model_type" : "tcn",
     "nblocks" : 5,
     "dilation_growth" : 10,
     "kernel_size" : 5,
     "causal" : True,
     "train_fraction" : 1.00
    },
    {"name" : "uTCN-100-N",
     "model_type" : "tcn",
     "nblocks" : 4,
     "dilation_growth" : 10,
     "kernel_size" : 5,
     "causal" : False,
     "train_fraction" : 1.00
    },
    {"name" : "uTCN-300-N",
     "model_type" : "tcn",
     "nblocks" : 4,
     "dilation_growth" : 10,
     "kernel_size" : 13,
     "causal" : False,
     "train_fraction" : 1.00
    },
    {"name" : "uTCN-1000-N",
     "model_type" : "tcn",
     "nblocks" : 5,
     "dilation_growth" : 10,
     "kernel_size" : 5,
     "causal" : False,
     "train_fraction" : 1.00
    },
    {"name" : "uTCN-324-N",
     "model_type" : "tcn",
     "nblocks" : 10,
     "dilation_growth" : 2,
     "kernel_size" : 15,
     "causal" : False,
     "train_fraction" : 1.0,
     "max_epochs" : 60,
     "channel_width" : 16,
    },
]

if __name__ == '__main__':
    # Get model index from command line
    model_idx = int(sys.argv[1]) - 1  # Convert to 0-indexed
    
    if model_idx < 0 or model_idx >= len(train_configs):
        print(f"ERROR: Invalid model index {model_idx + 1}. Must be 1-{len(train_configs)}")
        sys.exit(1)
    
    tconf = train_configs[model_idx]
    
    print(f"Training Model {model_idx + 1}/{len(train_configs)}: {tconf['name']}")
    print(f"Configuration: {tconf}")
    print("")
    
    parser = ArgumentParser()
    
    # Add PROGRAM level args
    parser.add_argument('--model_type', type=str, default='tcn')
    parser.add_argument('--root_dir', type=str, required=True)
    parser.add_argument('--preload', action="store_true")
    parser.add_argument('--sample_rate', type=int, default=44100)
    parser.add_argument('--shuffle', type=bool, default=True)
    parser.add_argument('--train_subset', type=str, default='train')
    parser.add_argument('--val_subset', type=str, default='val')
    parser.add_argument('--train_length', type=int, default=65536)
    parser.add_argument('--train_fraction', type=float, default=1.0)
    parser.add_argument('--eval_length', type=int, default=131072)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=12)
    parser.add_argument('--max_epochs', type=int, default=60)
    parser.add_argument('--precision', type=str, default='16-mixed')
    parser.add_argument('--accelerator', type=str, default='auto')
    parser.add_argument('--devices', type=int, default=1)
    parser.add_argument('--default_root_dir', type=str, default='./lightning_logs')
    parser.add_argument('--export', action='store_true')
    parser.add_argument('--save_dir', type=str, default='./models')
    
    # Let the model add what it wants
    if tconf['model_type'] == 'tcn':
        parser = TCNModel.add_model_specific_args(parser)
    elif tconf['model_type'] == 'lstm':
        parser = LSTMModel.add_model_specific_args(parser)
    
    args = parser.parse_args()
    
    # Set seed
    pl.seed_everything(42)
    
    # Create model specifier
    if tconf["model_type"] == 'tcn':
        specifier = f"{model_idx + 1}-{tconf['name']}"
        specifier += "__causal" if tconf['causal'] else "__noncausal"
        specifier += f"__{tconf['nblocks']}-{tconf['dilation_growth']}-{tconf['kernel_size']}"
        specifier += f"__fraction-{tconf['train_fraction']}-bs{args.batch_size}"
    elif tconf["model_type"] == 'lstm':
        specifier = f"{model_idx + 1}-{tconf['name']}"
        specifier += f"__{tconf['num_layers']}-{tconf['hidden_size']}"
        specifier += f"__fraction-{tconf['train_fraction']}-bs{args.batch_size}"
    
    args.default_root_dir = os.path.join("lightning_logs", "bulk", specifier)
    print(f"Checkpoint directory: {args.default_root_dir}")
    print("")
    
    # Create trainer
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        precision=args.precision,
        accelerator=args.accelerator,
        devices=args.devices,
        default_root_dir=args.default_root_dir,
    )
    
    # Setup dataloaders
    use_half = args.precision in ["16-mixed", "16-true", "16", 16]
    
    train_dataset = SignalTrainLA2ADataset(
        args.root_dir,
        subset=args.train_subset,
        fraction=tconf["train_fraction"],
        half=use_half,
        preload=args.preload,
        length=args.train_length
    )
    
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=args.shuffle,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    val_dataset = SignalTrainLA2ADataset(
        args.root_dir,
        preload=args.preload,
        half=use_half,
        subset=args.val_subset,
        length=args.eval_length
    )
    
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        shuffle=False,
        batch_size=args.batch_size // 2,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Create model
    dict_args = vars(args)
    dict_args["nparams"] = 2
    
    if tconf["model_type"] == 'tcn':
        dict_args["nblocks"] = tconf["nblocks"]
        dict_args["dilation_growth"] = tconf["dilation_growth"]
        dict_args["kernel_size"] = tconf["kernel_size"]
        dict_args["causal"] = tconf["causal"]
        if "channel_width" in tconf:
            dict_args["channel_width"] = tconf["channel_width"]
        model = TCNModel(**dict_args)
    elif tconf["model_type"] == 'lstm':
        dict_args["num_layers"] = tconf["num_layers"]
        dict_args["hidden_size"] = tconf["hidden_size"]
        model = LSTMModel(**dict_args)
    
    # Model summary
    print("="*60)
    print("Model Architecture:")
    print("="*60)
    torchsummary.summary(model, [(1, 65536), (1, 2)], device="cpu")
    print("")
    
    # Train
    print("="*60)
    print("Starting Training...")
    print("="*60)
    trainer.fit(model, train_dataloader, val_dataloader)
    
    # Export
    if args.export:
        print("")
        print("="*60)
        print(f"Exporting trained model: {specifier}")
        print("="*60)
        model_dir = args.default_root_dir
        export_model(model_dir, args.save_dir)
        print(f"Export complete! Model saved to {args.save_dir}")
    
    print("")
    print("="*60)
    print("Training Complete!")
    print("="*60)
EOPYTHON

# Run the training script
python train_model_${MODEL_INDEX}.py ${MODEL_INDEX} \
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

TRAINING_EXIT_CODE=$?

# Cleanup temporary script
rm -f train_model_${MODEL_INDEX}.py

echo ""
echo "======================================================"
echo "Training finished at $(date)"
echo "Exit code: $TRAINING_EXIT_CODE"
echo "======================================================"
echo ""

# ============================================================================
# Final GPU Stats
# ============================================================================
echo "Final GPU state:"
nvidia-smi

echo ""
echo "---------------------------------------"
echo "End Slurm Epilog: $(date)"
echo "Job ID:        ${SLURM_JOB_ID}"
echo "Array Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Exit Code:     ${TRAINING_EXIT_CODE}"
echo "---------------------------------------"

exit $TRAINING_EXIT_CODE

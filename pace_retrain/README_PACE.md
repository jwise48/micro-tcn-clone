# micro-TCN Training on PACE ICE Cluster

Complete guide for training micro-TCN models on Georgia Tech's PACE ICE supercomputing cluster with A100-80GB GPUs.

## 📋 Table of Contents

1. [Quick Start](#quick-start)
2. [First-Time Setup](#first-time-setup)
3. [Training Workflow](#training-workflow)
4. [Monitoring Jobs](#monitoring-jobs)
5. [Understanding Output](#understanding-output)
6. [Troubleshooting](#troubleshooting)
7. [Advanced Usage](#advanced-usage)

---

## 🚀 Quick Start

If you're in a hurry and have everything set up:

```bash
# Submit all 7 models for training
bash submit_all_models.sh

# Monitor progress
bash check_jobs.sh
```

---

## 🔧 First-Time Setup

### 1. Clone Repository and Navigate

```bash
cd ~/
git clone <your-micro-tcn-repo>
cd micro-tcn-clone
```

### 2. Verify Dataset Location

Ensure the SignalTrain LA2A dataset exists at:
```
/home/hice1/jwise48/scratch/data/SignalTrain_LA2A_Dataset_1.1
```

Check with:
```bash
ls -lh /home/hice1/jwise48/scratch/data/SignalTrain_LA2A_Dataset_1.1
```

### 3. Run Environment Setup

This creates a conda environment with all dependencies:

```bash
bash setup_environment.sh
```

**What this does:**
- Creates conda environment named `micro-tcn-env`
- Installs PyTorch 2.1.0 with CUDA 11.8 support
- Installs all project dependencies
- Verifies CUDA and GPU availability

**Expected duration:** ~10-15 minutes

### 4. Verify Installation

```bash
module load anaconda3/2023.03
module load cuda/11.8
conda activate micro-tcn-env
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

You should see: `CUDA available: True`

---

## 🎯 Training Workflow

### Submit All Models

```bash
bash submit_all_models.sh
```

This submits 7 independent training jobs:
1. **uTCN-100-C** - Causal, 100-sample receptive field
2. **uTCN-300-C** - Causal, 300-sample receptive field
3. **uTCN-1000-C** - Causal, 1000-sample receptive field
4. **uTCN-100-N** - Non-causal, 100-sample receptive field
5. **uTCN-300-N** - Non-causal, 300-sample receptive field
6. **uTCN-1000-N** - Non-causal, 1000-sample receptive field
7. **uTCN-324-N** - Non-causal, 16 channels (vs 32)

### Submit Specific Models

```bash
# Train only models 2, 4, and 6
bash submit_all_models.sh --models "2 4 6"
```

### Dry Run (Preview Without Submitting)

```bash
bash submit_all_models.sh --dry-run
```

---

## 📊 Monitoring Jobs

### Continuous Monitoring (Recommended)

```bash
bash check_jobs.sh
```

This provides:
- Real-time job status updates every 10 seconds
- Count of running/pending jobs
- List of completed models
- Recent error notifications
- Estimated time remaining

Press `Ctrl+C` to exit.

### Single Status Check

```bash
# Quick one-time status check
bash check_jobs.sh --once

# Or use native SLURM commands
squeue -u $USER
```

### Useful SLURM Commands

```bash
# View your job queue
squeue -u $USER

# View detailed job info
scontrol show job <JOB_ID>

# Cancel a specific job
scancel <JOB_ID>

# Cancel all your jobs
scancel -u $USER

# View job history
sacct -u $USER --format=JobID,JobName,State,Start,End,Elapsed
```

---

## 📁 Understanding Output

### Directory Structure After Training

```
micro-tcn-clone/
├── logs/                                    # SLURM job logs
│   ├── model-1-report-<jobid>.out          # Training output
│   └── model-1-error-<jobid>.err           # Error logs (if any)
├── lightning_logs/bulk/                     # PyTorch Lightning logs
│   ├── 1-uTCN-100-C__causal__4-10-5.../    # Model 1 checkpoints
│   ├── 2-uTCN-300-C__causal__4-10-13.../   # Model 2 checkpoints
│   └── ...
└── models_trained/                          # Exported TorchScript models
    ├── 1-uTCN-100-C__causal__4-10-5...pt   # Ready for C++ plugin
    └── ...
```

### Log Files

**SLURM Output (`logs/model-X-report-<jobid>.out`):**
- Job information (ID, node, resources)
- Training progress (epochs, loss, metrics)
- Model architecture summary
- Export confirmation
- Resource usage statistics

**Error Logs (`logs/model-X-error-<jobid>.err`):**
- Only contains content if errors occurred
- Check these if a job fails

### Checkpoints

Located in `lightning_logs/bulk/<model-specifier>/`:
- `version_0/checkpoints/*.ckpt` - PyTorch Lightning checkpoints
- `events.out.tfevents.*` - TensorBoard logs
- `hparams.yaml` - Hyperparameters

### Exported Models

Located in `models_trained/`:
- `*.pt` files - TorchScript models for C++ plugin deployment
- Ready to use in real-time audio plugin

---

## 🐛 Troubleshooting

### Common Issues

#### 1. Environment Not Found

**Error:** `conda: command not found` or `micro-tcn-env not found`

**Solution:**
```bash
module load anaconda3/2023.03
conda activate micro-tcn-env
```

#### 2. Dataset Not Found

**Error:** `ERROR: Dataset not found at /home/hice1/jwise48/scratch/data/SignalTrain_LA2A_Dataset_1.1`

**Solution:**
- Verify dataset path exists
- Download dataset if missing
- Update `DATASET_PATH` in `train_single_model.sbatch`

#### 3. Out of Memory (OOM)

**Error:** `CUDA out of memory`

**Solution:**
Reduce batch size in `train_single_model.sbatch`:
```bash
BATCH_SIZE=256  # Instead of 384
```

#### 4. Job Pending Too Long

**Check queue:**
```bash
squeue -u $USER
```

**Possible reasons:**
- No A100-80GB GPUs available (REASON: Resources)
- Account limits reached (REASON: QOSMaxJobsPerUserLimit)
- Partition down (REASON: PartitionDown)

**Solutions:**
- Wait for resources to become available
- Request different GPU type (edit `train_single_model.sbatch`)
- Check PACE status page for maintenance

#### 5. Import Errors During Training

**Error:** `ModuleNotFoundError: No module named 'microtcn'`

**Solution:**
```bash
conda activate micro-tcn-env
pip install -e .
```

### Viewing Real-Time Training Progress

To see live training output for a running job:

```bash
# Find your job's output file
ls -lt logs/

# Watch it in real-time
tail -f logs/model-1-report-<jobid>.out
```

### Debugging Failed Jobs

```bash
# Check error log
cat logs/model-X-error-<jobid>.err

# Check full output
cat logs/model-X-report-<jobid>.out

# View SLURM job details
scontrol show job <JOB_ID>

# Check job efficiency
seff <JOB_ID>
```

---

## 🔬 Advanced Usage

### Modifying Training Configuration

Edit `train_single_model.sbatch`:

```bash
# GPU Settings
#SBATCH --gres=gpu:A100:1           # Change GPU type
#SBATCH -C A100-80GB                # Change GPU variant

# Memory and CPU
#SBATCH --mem-per-gpu=224GB         # Adjust memory
#SBATCH --cpus-per-task=12          # Adjust CPU cores

# Time Limit
#SBATCH --time=6:00:00              # Adjust wall time

# Training Parameters
BATCH_SIZE=384                      # Adjust batch size
NUM_WORKERS=12                      # Adjust data loading workers
MAX_EPOCHS=60                       # Adjust training epochs
```

### Using Different GPU Types

Available GPU options on PACE:
- A100-80GB (recommended for this project)
- A100-40GB
- H100
- H200
- A40
- V100

Example for H100:
```bash
#SBATCH --gres=gpu:H100:1
#SBATCH -C H100
#SBATCH --mem-per-gpu=80GB
```

### Interactive GPU Session (Testing)

For debugging or testing:

```bash
# Request interactive A100-80GB GPU
salloc --account=musi --qos=coe-ice --partition=coe-gpu \
  --gres=gpu:A100:1 -C A100-80GB --mem-per-gpu=224GB \
  --cpus-per-task=12 --time=2:00:00

# Once allocated, load environment
module load anaconda3/2023.03
module load cuda/11.8
conda activate micro-tcn-env

# Test GPU
nvidia-smi

# Run single model training
python train.py --root_dir /home/hice1/jwise48/scratch/data/SignalTrain_LA2A_Dataset_1.1 \
  --preload --batch_size 384 --num_workers 12

# Exit when done
exit
```

### Monitoring GPU Usage Live

For running jobs, connect to the compute node:

```bash
# Find node where your job is running
squeue -u $USER

# SSH to that node
ssh <node-name>

# Watch GPU usage
watch -n 1 nvidia-smi

# Exit
exit
```

### Adjusting Resource Requests

For faster iteration during development:

```bash
# Smaller model, less time
#SBATCH --time=2:00:00
BATCH_SIZE=128
MAX_EPOCHS=30
```

For production runs:

```bash
# Maximum resources
#SBATCH --time=8:00:00
BATCH_SIZE=512
MAX_EPOCHS=100
```

---

## 📈 Performance Expectations

### A100-80GB Optimized Settings

- **Batch Size:** 384
- **Precision:** FP16 mixed precision
- **Dataset:** Preloaded into RAM
- **Expected Training Time:** ~3-4 hours per model
- **GPU Utilization:** 80-95%
- **Memory Usage:** ~40-50GB

### Parallel Training Scenarios

| Scenario | Available GPUs | Total Time |
|----------|----------------|------------|
| Sequential | 1 GPU | ~21-28 hours |
| Partial Parallel | 3 GPUs | ~9-12 hours |
| Full Parallel | 7+ GPUs | ~3-4 hours |

---

## 📧 Support

### PACE Support

- **Documentation:** https://docs.pace.gatech.edu/
- **Help Desk:** pace-support@oit.gatech.edu
- **Office Hours:** Check PACE website

### Project Issues

For micro-TCN specific issues:
- Check GitHub Issues
- Review training logs
- Verify environment setup

---

## ✅ Checklist

Before submitting jobs:
- [ ] Environment setup completed (`bash setup_environment.sh`)
- [ ] Dataset path verified
- [ ] Test import: `python -c "from microtcn import TCNModel"`
- [ ] Review `train_single_model.sbatch` settings
- [ ] Enough disk space for outputs (~5GB per model)

During training:
- [ ] Monitor with `bash check_jobs.sh`
- [ ] Check for errors in logs
- [ ] Verify GPU utilization is high

After completion:
- [ ] All 7 models trained successfully
- [ ] Checkpoints exist in `lightning_logs/bulk/`
- [ ] Exported models in `models_trained/`
- [ ] Review training metrics in TensorBoard

---

## 🎓 Additional Resources

- [PyTorch Lightning Documentation](https://lightning.ai/docs/pytorch/stable/)
- [PACE ICE User Guide](https://docs.pace.gatech.edu/)
- [SLURM Documentation](https://slurm.schedmd.com/)
- [Original micro-TCN Paper](https://arxiv.org/abs/2010.04237)

---

**Last Updated:** November 2025  
**Maintained by:** jwise48@gatech.edu

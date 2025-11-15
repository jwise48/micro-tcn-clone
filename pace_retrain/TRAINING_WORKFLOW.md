# Optimized Training Workflow for micro-TCN Models

## Overview
This document describes the optimized workflow for training all 6 micro-TCN model configurations on PACE's H100 GPUs.

## Changes Made

### 1. Fixed Critical Issues
- ✅ Fixed syntax error in `micro_tcn_retrain.sbatch` (missing backslash after `--preload`)
- ✅ Removed redundant test/speed/plot execution from individual training jobs
- ✅ Created dedicated post-processing script

### 2. Optimizations
- ✅ Reduced memory allocation from 512GB to 256GB (sufficient for batch_size=512)
- ✅ Increased num_workers from 12 to 14 (better CPU utilization)
- ✅ Separated training from evaluation for efficiency

## Training Workflow

### Step 1: Submit Training Jobs
Run all 6 models in parallel (or sequentially):

```bash
# Create logs directory if it doesn't exist
mkdir -p logs

# Submit all training jobs
sbatch pace_retrain/micro_tcn_retrain.sbatch uTCN-100-C
sbatch pace_retrain/micro_tcn_retrain.sbatch uTCN-300-C
sbatch pace_retrain/micro_tcn_retrain.sbatch uTCN-1000-C
sbatch pace_retrain/micro_tcn_retrain.sbatch uTCN-100-N
sbatch pace_retrain/micro_tcn_retrain.sbatch uTCN-300-N
sbatch pace_retrain/micro_tcn_retrain.sbatch uTCN-1000-N
```

Or use the compound command:
```bash
sbatch pace_retrain/micro_tcn_retrain.sbatch uTCN-100-C && \
sbatch pace_retrain/micro_tcn_retrain.sbatch uTCN-300-C && \
sbatch pace_retrain/micro_tcn_retrain.sbatch uTCN-1000-C && \
sbatch pace_retrain/micro_tcn_retrain.sbatch uTCN-100-N && \
sbatch pace_retrain/micro_tcn_retrain.sbatch uTCN-300-N && \
sbatch pace_retrain/micro_tcn_retrain.sbatch uTCN-1000-N
```

### Step 2: Monitor Training Progress
```bash
# Check job status
squeue -u $USER

# Check running jobs
bash pace_retrain/check_jobs.sh

# View logs in real-time
tail -f logs/model-*-report-*.out
```

### Step 3: Run Post-Processing (After All Models Complete)
Once all 6 training jobs have finished successfully:

```bash
sbatch pace_retrain/post_process.sbatch
```

This will:
- Evaluate all trained models with `test.py`
- Run performance benchmarks with `speed.py`
- Generate visualizations with `plot.py`

## Resource Allocation per Job

| Resource | Allocation | Notes |
|----------|------------|-------|
| GPU | 1x H100-80GB | Sufficient for batch_size=512 |
| Memory | 256GB | Optimized for data preloading |
| CPUs | 16 cores | 14 for data loading, 2 for system |
| Time | 4 hours | Conservative estimate per model |

## Expected Timeline

- **Training per model**: ~2-3 hours (70 epochs)
- **Total training time** (parallel): ~3 hours
- **Total training time** (sequential): ~18 hours
- **Post-processing**: ~1-2 hours

## Output Structure

```
micro-tcn-clone/
├── lightning_logs/bulk/          # Training logs and checkpoints
│   ├── 1-uTCN-100-C__causal__4-10-5__fraction-1.0-bs512/
│   ├── 2-uTCN-300-C__causal__4-10-13__fraction-1.0-bs512/
│   ├── 3-uTCN-1000-C__causal__5-10-5__fraction-1.0-bs512/
│   ├── 4-uTCN-100-N__noncausal__4-10-5__fraction-1.0-bs512/
│   ├── 5-uTCN-300-N__noncausal__4-10-13__fraction-1.0-bs512/
│   └── 6-uTCN-1000-N__noncausal__5-10-5__fraction-1.0-bs512/
├── models_trained/               # Exported TorchScript models
│   ├── uTCN-100-C.pt
│   ├── uTCN-300-C.pt
│   ├── uTCN-1000-C.pt
│   ├── uTCN-100-N.pt
│   ├── uTCN-300-N.pt
│   └── uTCN-1000-N.pt
├── evaluations/                  # Test outputs
│   └── [audio files and metrics]
├── test_results_val.p            # Pickled test results
└── logs/                         # SLURM logs
    ├── model-*-report-*.out
    └── model-*-error-*.err
```

## Model Configurations

| Model | Causal | Blocks | Dilation | Kernel | Receptive Field |
|-------|--------|--------|----------|--------|-----------------|
| uTCN-100-C | Yes | 4 | 10 | 5 | ~100 samples |
| uTCN-300-C | Yes | 4 | 10 | 13 | ~300 samples |
| uTCN-1000-C | Yes | 5 | 10 | 5 | ~1000 samples |
| uTCN-100-N | No | 4 | 10 | 5 | ~100 samples |
| uTCN-300-N | No | 4 | 10 | 13 | ~300 samples |
| uTCN-1000-N | No | 5 | 10 | 5 | ~1000 samples |

## Training Configuration

- **Batch Size**: 512 (optimized for H100)
- **Precision**: 16-mixed (FP16 for speed)
- **Max Epochs**: 70
- **Data Preloading**: Enabled (faster training)
- **Optimizer**: AdamW (from model defaults)
- **Learning Rate**: From model defaults
- **Checkpoint**: Best model based on validation loss

## Troubleshooting

### Job Fails Immediately
- Check logs in `logs/model-*-error-*.err`
- Verify dataset path: `/home/hice1/jwise48/scratch/data/micro-tcn/SignalTrain_LA2A_Dataset_1.1`
- Ensure conda environment can be created

### Out of Memory
- Current allocation (256GB) should be sufficient
- If issues persist, increase to 384GB in sbatch file

### Slow Training
- Verify GPU is being used: check `nvidia-smi` output in logs
- Ensure data preloading is working
- Check num_workers isn't causing bottleneck

### Post-Processing Fails
- Verify all 6 model directories exist in `lightning_logs/bulk/`
- Check that checkpoint files were saved properly
- Ensure test dataset is accessible

## Additional Notes

### Environment Recreation
The current workflow recreates the conda environment for each job. This adds ~5-10 minutes per job but ensures a clean environment. If you want to optimize further:

1. Create a persistent environment once
2. Load it in each job instead of recreating
3. See `pace_retrain/setup_environment.sh` for reference

### Parallel vs Sequential
- **Parallel**: Submit all 6 jobs at once (faster if resources available)
- **Sequential**: Use `&&` to chain jobs (more reliable, uses fewer resources)

### Monitoring Best Practices
```bash
# Watch all jobs
watch -n 30 'squeue -u $USER'

# Check GPU utilization
squeue -u $USER -o "%.18i %.9P %.30j %.8T %.10M %.6D %R %b"

# Monitor memory usage in logs
grep "Available memory" logs/model-*-report-*.out
```
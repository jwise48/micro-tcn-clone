#!/bin/bash
#
# Master orchestration script for submitting all micro-TCN training jobs
# Submits 7 independent jobs that can run in parallel on PACE
#
# Usage: bash submit_all_models.sh [options]
#
# Options:
#   --models "1 3 5"    Submit only specific models (default: all)
#   --dry-run           Show what would be submitted without actually submitting
#

set -e

echo "======================================================"
echo "micro-TCN Parallel Training Job Submission"
echo "======================================================"
echo ""

# Parse command line arguments
MODELS_TO_SUBMIT="1 2 3 4 5 6 7"
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --models)
            MODELS_TO_SUBMIT="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: bash submit_all_models.sh [--models \"1 3 5\"] [--dry-run]"
            exit 1
            ;;
    esac
done

# Create logs directory if it doesn't exist
mkdir -p logs

# Model names for reference
declare -a MODEL_NAMES=(
    "uTCN-100-C (causal, 100 samples RF)"
    "uTCN-300-C (causal, 300 samples RF)"
    "uTCN-1000-C (causal, 1000 samples RF)"
    "uTCN-100-N (non-causal)"
    "uTCN-300-N (non-causal)"
    "uTCN-1000-N (non-causal)"
    "uTCN-324-N (16 channels)"
)

echo "Models to submit: $MODELS_TO_SUBMIT"
echo ""

# Check if SBATCH script exists
if [ ! -f "train_single_model.sbatch" ]; then
    echo "ERROR: train_single_model.sbatch not found!"
    echo "Make sure you're running this script from the project root directory."
    exit 1
fi

# Submit jobs
SUBMITTED_JOBS=()

for model_id in $MODELS_TO_SUBMIT; do
    # Validate model ID
    if [ $model_id -lt 1 ] || [ $model_id -gt 7 ]; then
        echo "WARNING: Skipping invalid model ID: $model_id (must be 1-7)"
        continue
    fi
    
    model_idx=$((model_id - 1))
    model_name="${MODEL_NAMES[$model_idx]}"
    
    echo "Submitting Model $model_id: $model_name"
    
    if [ "$DRY_RUN" = true ]; then
        echo "  [DRY RUN] Would execute: sbatch --array=$model_id train_single_model.sbatch"
    else
        # Submit the job and capture the job ID
        JOB_OUTPUT=$(sbatch --array=$model_id train_single_model.sbatch)
        JOB_ID=$(echo $JOB_OUTPUT | grep -oP '\d+')
        SUBMITTED_JOBS+=("$JOB_ID")
        echo "  Job ID: $JOB_ID"
    fi
    echo ""
done

echo "======================================================"
echo "Submission Complete!"
echo "======================================================"
echo ""

if [ "$DRY_RUN" = true ]; then
    echo "DRY RUN MODE - No jobs were actually submitted"
    echo ""
    echo "To submit for real, run:"
    echo "  bash submit_all_models.sh"
else
    echo "Submitted ${#SUBMITTED_JOBS[@]} job(s)"
    echo "Job IDs: ${SUBMITTED_JOBS[*]}"
    echo ""
    echo "Monitoring Commands:"
    echo "-------------------"
    echo "  View job queue:           squeue -u $USER"
    echo "  Watch queue continuously: bash check_jobs.sh"
    echo "  Cancel all jobs:          scancel -u $USER"
    echo "  Cancel specific job:      scancel <JOB_ID>"
    echo ""
    echo "Output Locations:"
    echo "----------------"
    echo "  SLURM logs:      logs/model-<array_id>-report-<job_id>.out"
    echo "  Training logs:   lightning_logs/bulk/<model-specifier>/"
    echo "  Trained models:  models_trained/"
    echo ""
    echo "Estimated Completion:"
    echo "--------------------"
    echo "  Per model:  ~3-4 hours on A100-80GB"
    echo "  All models: ~3-4 hours (if 7 GPUs available in parallel)"
    echo "             ~21-28 hours (if running sequentially)"
    echo ""
    echo "Check status with: squeue -u $USER"
fi

# Save job IDs to file for reference
if [ "$DRY_RUN" = false ] && [ ${#SUBMITTED_JOBS[@]} -gt 0 ]; then
    echo "${SUBMITTED_JOBS[*]}" > .submitted_jobs.txt
    echo "Job IDs saved to .submitted_jobs.txt"
fi

echo ""
echo "Happy training! 🚀"
echo ""

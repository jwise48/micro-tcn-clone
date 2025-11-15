#!/bin/bash

# Submit jobs and collect IDs
JOBS=()
for model in uTCN-100-C uTCN-300-C uTCN-1000-C uTCN-100-N uTCN-300-N uTCN-1000-N; do
    JOB_ID=$(sbatch --parsable pace_retrain/micro_tcn_retrain.sbatch $model)
    JOBS+=($JOB_ID)
    echo "Submitted $model: Job $JOB_ID"
done

# Wait for all jobs to complete
echo "Waiting for jobs to complete: ${JOBS[*]}"
while true; do
    # Check if any jobs are still running
    RUNNING=$(squeue -j $(IFS=,; echo "${JOBS[*]}") -h | wc -l)
    if [ $RUNNING -eq 0 ]; then
        break
    fi
    echo "Still running: $RUNNING jobs"
    bash pace_retrain/check_jobs.sh --once
    sleep 60
done

echo "All training jobs completed. Starting post-processing..."
sbatch pace_retrain/post_process.sbatch

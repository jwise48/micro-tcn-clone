#!/bin/bash
#
# Real-time monitoring utility for PACE training jobs
# Displays job status, GPU utilization, and progress tracking
#
# Usage: bash check_jobs.sh
#

set -e

echo "======================================================"
echo "micro-TCN Training Monitor - PACE ICE Cluster"
echo "======================================================"
echo ""

# Colors for output (if terminal supports it)
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to display job status
show_jobs() {
    echo "-----------------------------------------------------"
    echo "Current Job Status ($(date))"
    echo "-----------------------------------------------------"
    
    # Check if user has any jobs
    if ! squeue -u $USER &> /dev/null; then
        echo "No jobs found for user: $USER"
        return
    fi
    
    # Display detailed job information
    squeue -u $USER --format="%.10i %.12j %.10P %.8T %.10M %.6D %.20R %.10l" || echo "No active jobs"
    
    echo ""
    
    # Count jobs by status
    RUNNING=$(squeue -u $USER -t RUNNING -h | wc -l)
    PENDING=$(squeue -u $USER -t PENDING -h | wc -l)
    TOTAL=$(squeue -u $USER -h | wc -l)
    
    echo "Summary:"
    echo "  Running:  $RUNNING"
    echo "  Pending:  $PENDING"
    echo "  Total:    $TOTAL"
}

show_job_progress() {
    echo ""
    echo "-----------------------------------------------------"
    echo "Last 10 Lines of Running Jobs"
    echo "-----------------------------------------------------"

    # Get running job IDs
    RUNNING_JOBS=$(squeue -u $USER -t RUNNING -h -o "%i %j")

    if [ -z "$RUNNING_JOBS" ]; then
        echo "  No jobs currently running"
        return
    fi

    # For each running job, find its log file and show last 10 lines
    while IFS= read -r job_line; do
        JOB_ID=$(echo $job_line | awk '{print $1}')
        JOB_NAME=$(echo $job_line | awk '{print $2}')

        # Find the output file for this job
        LOG_FILE=$(find logs -name "*-report-${JOB_ID}.out" 2>/dev/null | head -1)

        if [ -n "$LOG_FILE" ] && [ -f "$LOG_FILE" ]; then
            echo ""
            echo "  Job ID: $JOB_ID | Name: $JOB_NAME"
            echo "  Log: $LOG_FILE"
            echo "  ----------------------------------------"
            tail -10 "$LOG_FILE" | sed 's/^/    /'
        fi
    done <<< "$RUNNING_JOBS"
}

# Function to check completed models
check_completed() {
    echo ""
    echo "-----------------------------------------------------"
    echo "Completed Models"
    echo "-----------------------------------------------------"
    
    if [ -d "lightning_logs/bulk" ]; then
        COMPLETED_COUNT=0
        for dir in lightning_logs/bulk/*/; do
            if [ -d "$dir" ]; then
                model_name=$(basename "$dir")
                # Check if checkpoint exists
                if ls "$dir"/**/checkpoints/*.ckpt &> /dev/null; then
                    echo "  ✓ $model_name"
                    COMPLETED_COUNT=$((COMPLETED_COUNT + 1))
                fi
            fi
        done
        
        if [ $COMPLETED_COUNT -eq 0 ]; then
            echo "  No completed models yet"
        else
            echo ""
            echo "  Total completed: $COMPLETED_COUNT / 7"
        fi
    else
        echo "  No training logs directory found"
    fi
}

# Function to show recent errors
check_errors() {
    echo ""
    echo "-----------------------------------------------------"
    echo "Recent Errors (if any)"
    echo "-----------------------------------------------------"
    
    if [ -d "logs" ]; then
        ERROR_FILES=$(find logs -name "model-*-error-*.err" -type f -mmin -60 2>/dev/null)
        
        if [ -z "$ERROR_FILES" ]; then
            echo "  No recent errors found"
        else
            for error_file in $ERROR_FILES; do
                if [ -s "$error_file" ]; then
                    echo "  ERROR in: $(basename $error_file)"
                    echo "  Last 15 lines:"
                    tail -15 "$error_file" | sed 's/^/    /'
                    echo ""
                fi
            done
        fi
    else
        echo "  No logs directory found"
    fi
}

# Function to estimate time remaining
estimate_time() {
    echo ""
    echo "-----------------------------------------------------"
    echo "Estimated Time Remaining"
    echo "-----------------------------------------------------"
    
    RUNNING=$(squeue -u $USER -t RUNNING -h | wc -l)
    PENDING=$(squeue -u $USER -t PENDING -h | wc -l)
    
    if [ $RUNNING -eq 0 ] && [ $PENDING -eq 0 ]; then
        echo "  All jobs complete or no jobs running"
    else
        echo "  Running jobs: $RUNNING"
        echo "  Pending jobs: $PENDING"
        echo ""
        echo "  Estimated time per model: ~3-4 hours"
        
        if [ $RUNNING -gt 0 ]; then
            echo "  Running jobs will complete in: ~3-4 hours"
        fi
        
        if [ $PENDING -gt 0 ]; then
            echo "  Pending jobs (if sequential): ~$((PENDING * 3))-$((PENDING * 4)) hours"
            echo "  Pending jobs (if parallel): ~3-4 hours (if enough GPUs available)"
        fi
    fi
}

# Function to show GPU usage for running jobs
show_gpu_usage() {
    echo ""
    echo "-----------------------------------------------------"
    echo "GPU Usage for Running Jobs"
    echo "-----------------------------------------------------"
    
    RUNNING_JOBS=$(squeue -u $USER -t RUNNING -h -o "%i|%j|%N")
    
    if [ -z "$RUNNING_JOBS" ]; then
        echo "  No jobs currently running on GPUs"
        return
    fi
    
    while IFS='|' read -r JOB_ID JOB_NAME NODE; do
        echo "  Job: $JOB_NAME (ID: $JOB_ID) on $NODE"
        
        # Try SSH first (fast), fall back to srun with timeout
        GPU_INFO=$(ssh -o ConnectTimeout=30 -o StrictHostKeyChecking=no $NODE \
            "nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits" 2>/dev/null)
        
	echo "getting more gpu info"
        if [ -z "$GPU_INFO" ]; then
            # SSH failed, try srun with timeout
            GPU_INFO=$(timeout 30 srun --jobid=$JOB_ID \
                nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits 2>/dev/null)
        fi

	echo "done srunning for gpu info"

        if [ -n "$GPU_INFO" ]; then
            while IFS=',' read -r idx gpu_util mem_used mem_total temp; do
                printf "    GPU %s: %s%% util | %s/%sMB | %s°C\n" \
                    "$(echo $idx | xargs)" "$(echo $gpu_util | xargs)" \
                    "$(echo $mem_used | xargs)" "$(echo $mem_total | xargs)" \
                    "$(echo $temp | xargs)"
            done <<< "$GPU_INFO"
        else
            echo "    ⚠ GPU query timed out or unavailable"
        fi
	echo ""
	echo "finito!"
        echo ""
    done <<< "$RUNNING_JOBS"
}

# Main monitoring function
monitor_once() {
    clear
    echo "======================================================"
    echo "micro-TCN Training Monitor"
    echo "======================================================"
    echo ""
    
    show_jobs
    check_completed
    estimate_time
    show_gpu_usage
    show_job_progress
    check_errors
    
    echo ""
    echo "======================================================"
    echo "Press Ctrl+C to exit auto-refresh mode"
    echo "Refreshing every 10 seconds..."
    echo "======================================================"
}

# Check if we should run in watch mode or single display
if [ "$1" == "--once" ]; then
    # Single display mode
    show_jobs
    check_completed
    estimate_time
    show_job_progress
    check_errors
else
    # Continuous monitoring mode
    echo "Starting continuous monitoring (updates every 10 seconds)..."
    echo "Press Ctrl+C to stop"
    echo ""
    sleep 2
    
    while true; do
        monitor_once
        sleep 20
    done
fi

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
                    echo "  Last 5 lines:"
                    tail -5 "$error_file" | sed 's/^/    /'
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
    
    # Get list of nodes where user's jobs are running
    NODES=$(squeue -u $USER -t RUNNING -h -o "%N" | sort -u)
    
    if [ -z "$NODES" ]; then
        echo "  No jobs currently running on GPUs"
    else
        echo "  Jobs running on nodes: $NODES"
        echo "  (Connect to nodes with: ssh <node> to run nvidia-smi)"
    fi
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
    check_errors
else
    # Continuous monitoring mode
    echo "Starting continuous monitoring (updates every 10 seconds)..."
    echo "Press Ctrl+C to stop"
    echo ""
    sleep 2
    
    while true; do
        monitor_once
        sleep 10
    done
fi

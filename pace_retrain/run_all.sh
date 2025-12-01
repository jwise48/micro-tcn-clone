#!/bin/bash

echo "======================================================"
echo "micro-TCN Complete Training Pipeline"
echo "======================================================"

# Step 0: Ensure environment exists (runs on login node)
echo "Step 0: Checking environment setup..."
if ! conda env list | grep -q "^micro-tcn-env "; then
    echo "Environment not found. Running setup..."
    bash pace_retrain/setup_environment.sh
else
    echo "Environment already exists. Skipping setup."
fi

echo ""
echo "======================================================"
echo "Running test.py to evaluate all trained models"
echo "======================================================"
echo ""

srun python test.py \
   --root_dir ${DATASET_PATH} \
   --model_dir ${MODEL_DIR} \
   --save_dir ${SAVE_DIR} \
   --num_workers 14

echo ""
echo "======================================================"
echo "Running speed.py for performance benchmarks"
echo "======================================================"
echo ""

python speed.py --plot --gpu --rf

echo ""
echo "======================================================"
echo "Running plot.py to generate visualizations"
echo "======================================================"
echo ""

python plot.py

echo ""
echo "======================================================"
echo "Post-processing completed successfully!"
echo "======================================================"
echo ""
echo "Results saved to: ${SAVE_DIR}"
echo "Test results: test_results_val.p"
echo ""

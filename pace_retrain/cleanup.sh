#!/bin/bash
# pace_retrain/cleanup_environments.sh
# Removes all micro-tcn conda environments

echo "Loading conda..."
module load anaconda3/2023.03

echo ""
echo "Current conda environments:"
conda env list

echo ""
echo "Removing micro-tcn-env..."
conda env remove -n micro-tcn-env -y

echo ""
echo "Final environment list:"
conda env list

echo ""
echo "Checking conda environment directories..."
find ~/.conda/envs -name "*micro-tcn*" -type d 2>/dev/null

echo ""
echo "To remove any remaining directories manually:"
echo "  rm -rf ~/.conda/envs/micro-tcn-env*"


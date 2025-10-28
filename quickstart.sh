#!/usr/bin/env bash
set -euo pipefail

#==============================================================================
# CSA Paper Quickstart Script
#==============================================================================
# This script provides a one-command way to reproduce all results from the
# CSA paper. It handles dependency checking and runs the complete 7-stage
# postprocessing pipeline.
#
# Prerequisites:
#   - Download test_preds.zip from Zenodo and unzip to ./test_preds/
#   - (Optional) Create conda environment from environment.yml
#
# Usage: ./quickstart.sh
#==============================================================================

# 1) (Optional) Create conda environment
# Uncomment the lines below if you want to automatically create/activate
# the conda environment defined in environment.yml
# conda env create -f environment.yml || true
# conda activate csa-paper

# 2) Check for required data dependency
# The pipeline requires model predictions from Zenodo to be present
if [ ! -d "test_preds" ]; then
  echo "ERROR: test_preds/ directory not found."
  echo "Please download test_preds.zip from Zenodo and unzip it to ./test_preds/"
  echo "Then run this script again."
  exit 1
fi

# 3) Execute the complete postprocessing pipeline
# This runs all 7 stages: aggregation, cleaning, G matrices, figures, stats, etc.
echo "Starting CSA paper postprocessing pipeline..."
./run_postprocessing.sh

echo "Quickstart complete! Check outputs/ directory for results."

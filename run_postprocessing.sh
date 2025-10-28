#!/usr/bin/env bash
set -euo pipefail

#==============================================================================
# CSA Paper Postprocessing Pipeline
#==============================================================================
# This script runs the complete 7-stage postprocessing pipeline for the CSA paper.
# It processes model predictions through aggregation, cleaning, statistical
# analysis, and generates all figures, tables, and derived results.
#
# Pipeline Stages:
#   1. Aggregate: Combine raw prediction data
#   2. Clean: Data cleaning and preprocessing
#   3. G Matrices: Compute geometric/statistical matrices
#   4. Ga/Gn/Gna: Compute specific matrix variants
#   5. Figures: Generate publication figures and tables
#   6. Statistics: Wilcoxon tests and statistical analysis
#   7. Coverage: Overlap and coverage analysis
#
# Outputs: All results saved to outputs/ directory
# Logs: Execution logs saved to outputs/logs/
#==============================================================================

# Load optional configuration file
# Users can customize paths by creating configs/paths.yaml
PATHS_FILE="configs/paths.yaml"
if [ ! -f "$PATHS_FILE" ]; then
  echo "[info] $PATHS_FILE not found, using defaults from scripts."
fi

# Create output directory structure for organized results
# This matches the camera-ready-repro branch structure
mkdir -p outputs/figures outputs/tables outputs/derived outputs/logs

# Utility function for timestamped logging
log() { echo "[$(date +'%F %T')] $*"; }

# Helper function to execute Jupyter notebooks if they exist
# Converts notebooks to executed versions for reproducibility
run_notebook () {
  local nb="$1"
  if [ -f "$nb" ]; then
    log "Executing notebook: $nb"
    jupyter nbconvert --to notebook --execute "$nb" --output "${nb%.ipynb}.executed.ipynb"
  fi
}

# Validate required input data exists
# The pipeline requires model predictions from Zenodo
if [ ! -d "test_preds" ]; then
  log "ERROR: test_preds/ not found. Please download/unzip Zenodo predictions into ./test_preds"
  exit 1
fi

#==============================================================================
# MAIN PIPELINE EXECUTION
#==============================================================================
# Run each stage sequentially with error handling and logging
# Each stage logs to outputs/logs/ for debugging and reproducibility

log "Running Stage 1: Data Aggregation"
python stage1_aggregate.py 2>&1 | tee outputs/logs/stage1.log || true

log "Running Stage 2: Data Cleaning"
python stage2_clean.py 2>&1 | tee outputs/logs/stage2.log || true

log "Running Stage 3: Compute G Matrices"
python stage3_compute_G_matrices.py 2>&1 | tee outputs/logs/stage3.log || true

log "Running Stage 4: Compute Ga/Gn/Gna Variants"
python stage4_compute_Ga_Gn_Gna.py 2>&1 | tee outputs/logs/stage4.log || true

log "Running Stage 5: Generate Figures and Tables"
python stage5_figures.py 2>&1 | tee outputs/logs/stage5.log || true
run_notebook stage5_figures.ipynb || true

log "Running Stage 6: Statistical Analysis (Wilcoxon Tests)"
python stage6_stats_wilcoxon.py 2>&1 | tee outputs/logs/stage6.log || true

log "Running Stage 7: Coverage Analysis"
python stage7_overlap_coverage.py 2>&1 | tee outputs/logs/stage7.log || true

# Pipeline completion message
log "Pipeline complete! All artifacts saved to outputs/ directory:"
log "  - Figures: outputs/figures/"
log "  - Tables: outputs/tables/"
log "  - Derived data: outputs/derived/"
log "  - Execution logs: outputs/logs/"

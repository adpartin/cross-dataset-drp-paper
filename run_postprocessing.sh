#!/usr/bin/env bash
set -euo pipefail

#==============================================================================
# CSA Paper Postprocessing Pipeline
#==============================================================================
# This script runs the complete 6-step postprocessing pipeline for the CSA paper.
# It processes model predictions through score computation, matrix generation,
# statistical analysis, and generates all figures, tables, and derived results.
#
# Pipeline Steps:
#   1. Compute Scores: Calculate performance metrics from predictions
#   2. G Matrices: Compute source×target performance matrices
#   3. Ga/Gn/Gna: Compute normalized and aggregated matrix variants
#   4. Statistics: Wilcoxon tests for pairwise model comparisons
#   5. Figures: Generate publication figures and tables (notebooks)
#   6. Coverage: Overlap and coverage analysis correlating with performance
#
# Note: Step 0 (s0_aggregate.py) is optional and generally for internal use only.
# Most users should download pre-computed predictions from Zenodo.
#
# Outputs: All results saved to outputs/ directory
# Logs: Execution logs saved to logs/ directory
#==============================================================================

# Create logs directory for execution logs
mkdir -p logs

# Utility function for timestamped logging
log() { echo "[$(date +'%F %T')] $*"; }

# Helper function to execute Jupyter notebooks if they exist
# Converts notebooks to executed versions for reproducibility
run_notebook () {
  local nb="$1"
  if [ ! -f "$nb" ]; then
    log "WARNING: Notebook not found: $nb (skipping)"
    return 0
  fi
  
  # Check if nbconvert is available
  if ! command -v jupyter >/dev/null 2>&1 || ! jupyter nbconvert --help >/dev/null 2>&1; then
    log "ERROR: jupyter nbconvert not found. Cannot execute notebook: $nb"
    log "Install nbconvert: conda install -c conda-forge nbconvert"
    log "Or install via pip: pip install nbconvert"
    return 1
  fi
  
  log "Executing notebook: $nb"
  if jupyter nbconvert --to notebook --execute "$nb" --output "${nb%.ipynb}.executed.ipynb" 2>&1 | tee -a logs/step5.log; then
    log "Successfully executed: $nb"
  else
    log "ERROR: Failed to execute notebook: $nb (check logs/step5.log for details)"
    return 1
  fi
}

# Validate required input data exists, download if needed
# The pipeline requires model predictions from Zenodo
if [ ! -d "test_preds" ] || [ ! -f "test_preds/.complete" ]; then
  log "test_preds/ not found or incomplete. Downloading from Zenodo..."
  if [ -f "fetch_test_preds.sh" ]; then
    ./fetch_test_preds.sh
  else
    log "ERROR: test_preds/ not found and fetch_test_preds.sh not available."
    log "Please download test_preds.zip from Zenodo and extract to ./test_preds/"
    exit 1
  fi
fi

#==============================================================================
# MAIN PIPELINE EXECUTION
#==============================================================================
# Run each step sequentially with error handling and logging
# Note: Python scripts write detailed logs to logs/sX_*.log
# The tee commands below capture stdout/stderr including bash messages

log "Running Step 1: Compute Performance Scores"
python s1_compute_scores.py 2>&1 | tee logs/step1.log || true

log "Running Step 2: Compute G Matrices"
python s2_compute_G_matrices.py 2>&1 | tee logs/step2.log || true

log "Running Step 3: Compute Ga/Gn/Gna Variants"
python s3_compute_Gn_Ga_Gna.py 2>&1 | tee logs/step3.log || true

log "Running Step 4: Statistical Analysis (Wilcoxon Tests)"
python s4_stats_wilcoxon.py 2>&1 | tee logs/step4.log || true

log "Running Step 5: Generate Figures and Tables"
# Execute figure generation notebooks if they exist
# Note: These notebooks require nbconvert to execute programmatically
if run_notebook s3_compute_and_plot_Gn_Ga_Gna.ipynb && \
   run_notebook s4_wilcoxon_and_bubble_plots.ipynb && \
   run_notebook s5_shap.ipynb; then
  log "All notebooks executed successfully"
else
  log "WARNING: Some notebooks failed to execute. Check logs/step5.log for details."
  log "You may need to install nbconvert: conda install -c conda-forge nbconvert"
fi

log "Running Step 6: Coverage Analysis"
python s6_overlap.py 2>&1 | tee logs/step6.log || true

# Pipeline completion message
log "Pipeline complete! All artifacts saved to outputs/ directory:"
log "  - Scores: outputs/s1_scores/"
log "  - G Matrices: outputs/s2_G_matrices/"
log "  - Ga/Gn/Gna: outputs/s3_GaGnGna/"
log "  - Stats: outputs/s4_stats/ (includes figures in subdirectories)"
log "  - SHAP: outputs/s5_shap/ (includes figures)"
log "  - Coverage: outputs/s6_overlap/ (includes figures)"
log "  - Execution logs: logs/ (stepX.log = orchestration logs, sX_*.log = detailed script logs)"

"""
Step 1: Compute Performance Scores from Predictions
====================================================

This step computes performance scores (R², MSE, etc.) from raw prediction CSV files
by averaging scores across data splits for each model, source dataset, and target dataset.

PURPOSE:
--------
- Reads prediction CSV files from test_preds/ directory
- Computes regression metrics (R², MSE, RMSE, MAE, etc.) for each split
- Averages scores across splits for each (model, source, target) combination
- Saves computed scores for downstream analysis stages

INPUT:
------
- Prediction CSV files from test_preds/ directory
- Files follow pattern: <SRC>_<TRG>_split_<ID>_<MODEL>.csv
- Each file contains true and predicted values for a specific split

OUTPUT:
-------
- Creates s1_scores/ directory with score CSV files
- Per-model scores: <MODEL>_scores.csv
- Combined scores: all_models_scores.csv
- All scores saved to outputs/s1_scores/

USAGE:
------
    python s1_compute_scores.py [--outdir <output_directory>]

RUNTIME:
--------
- Typical runtime: ~4 minutes
"""

import argparse
import logging
import os
import time
from pathlib import Path
from postprocess_utils import (
    compute_scores_from_averaged_splits,
    setup_logging
)


def main():
    start_time = time.time()

    parser = argparse.ArgumentParser(
        description='Step 1: Compute performance scores from prediction CSV files.',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '--outdir',
        default='./outputs',
        type=str,
        help='Output directory root (default: ./outputs)'
    )
    args = parser.parse_args()
 
    # Create output directory structure
    outdir = Path(args.outdir) / 's1_scores'
    outdir.mkdir(parents=True, exist_ok=True)
    
    # Create logs directory before setting up logging
    # logs_dir = Path('outputs') / 'logs'
    logs_dir =  Path('./logs')
    logs_dir.mkdir(parents=True, exist_ok=True)

    setup_logging(log_file=str(logs_dir / 's1_compute_scores.log'))

    logging.info("=" * 50)
    logging.info("Step 1: Compute Performance Scores from Predictions")
    logging.info("=" * 50)
    logging.info(f"Input: test_preds/ directory")
    logging.info(f"Output: {outdir}")
    
    # Validate input directory exists
    preds_dir = Path('test_preds')
    if not preds_dir.exists():
        logging.error(f"test_preds/ directory not found!")
        logging.error("Please download predictions using: ./fetch_test_preds.sh")
        raise FileNotFoundError(f"test_preds/ directory not found: {preds_dir}")
    
    pred_files = list(preds_dir.glob('*.csv'))
    logging.info(f"Found {len(pred_files)} prediction CSV files in test_preds/")
    
    # Compute scores from averaged splits
    compute_scores_from_averaged_splits(
        outdir=outdir,
        models=None,
        sources=None,
        targets=None
    )

    runtime = (time.time() - start_time) / 60
    logging.info(f"Runtime: {runtime:.2f} minutes")
    logging.info(f"Step 1 complete. Scores saved to {outdir}")
    
    print(f'\n✅ Finished {Path(__file__).name}!')


if __name__ == "__main__":
    main()

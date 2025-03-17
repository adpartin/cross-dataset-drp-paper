# stage1_collect_raw_predictions.py
"""
Stage 1: Collect raw prediction data from model runs.
Note: This script requires access to the full model runs (4TB) and generates ~2.4GB of data.
This stage is primarily for internal use by the original authors.
"""

import argparse
import logging
import time
from pathlib import Path
from postprocess_utils import collect_and_save_raw_preds, setup_logging


def main():
    start_time = time.time()

    parser = argparse.ArgumentParser(
        description='Collect raw prediction data from model runs.')
    parser.add_argument(
        '--outdir',
        default='.',
        type=str,
        help='Output directory')
    parser.add_argument(
        '--runs-dir',
        default='../run/v1.1',
        type=str, 
        help='Directory containing model runs')
    args = parser.parse_args()
    
    outdir = Path(args.outdir)
    setup_logging(log_file='stage1_collect_raw_predictions.log') # stage1 log file
    
    # Get model paths
    main_models_path = Path(args.runs_dir)
    models_paths_list = sorted(p for p in main_models_path.glob('*') 
                             if (p/'improve_output').exists())
    
    # Collect predictions
    collect_and_save_raw_preds(
        models_paths_list, outdir, subset_type='val', stage='models'
    )
    collect_and_save_raw_preds(
        models_paths_list, outdir, subset_type='test', stage='infer'
    )

    runtime = (time.time() - start_time) / 60
    logging.info(f'Runtime: {runtime:.2f} minutes')
    logging.info(f'Finished stage 1 (collect raw prediction data from model runs).')


if __name__ == "__main__":
    main()
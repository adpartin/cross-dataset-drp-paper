# stage2_compute_scores.py
"""
Stage 2: Compute scores from raw predictions.
Note: This script requires the output from stage1 (~2.4GB).
This stage takes about 70 minutes to run.
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

    parser = argparse.ArgumentParser(description='Compute scores from predictions.')
    parser.add_argument(
        '--outdir',
        default='.',
        type=str,
        help='Output directory'
    )
    args = parser.parse_args()
    
    outdir = Path(args.outdir) / 'splits_averaged'
    os.makedirs(outdir, exist_ok=True)
    setup_logging(log_file='stage2_compute_scores.log') # stage2 log file
    
    # Compute scores
    compute_scores_from_averaged_splits(
        outdir=outdir,
        models=None,
        sources=None,
        targets=None
    )

    runtime = (time.time() - start_time) / 60
    logging.info(f'Runtime: {runtime:.2f} minutes')
    logging.info(f'Finished stage 2 (compute scores from prediction data).')


if __name__ == "__main__":
    main()
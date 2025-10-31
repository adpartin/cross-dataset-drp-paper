# stage3_generate_paper_data.py
"""
Stage 3: Generate final tables for the paper.
This script generates the data required by postprocess_plot.ipynb.
The output of this stage will be shared with the community.
"""

import argparse
import logging
import time
from pathlib import Path
from postprocess_utils import (
    compute_csa_tables_from_averaged_splits,
    setup_logging
)


def main():
    start_time = time.time()

    parser = argparse.ArgumentParser(
        description='Generate CSA tables required by postprocess_plot.ipynb'
    )
    parser.add_argument(
        '--input_dir',
        default='./splits_averaged', 
        type=str,
        help='Input directory containing scores'
    )
    parser.add_argument(
        '--outdir',
        default='./splits_averaged', 
        type=str,
        help='Output directory'
    )
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    outdir = Path(args.outdir)
    setup_logging(log_file='stage3_generate_csa_tables.log') # stage3 log file
    
    # Generate CSA tables
    compute_csa_tables_from_averaged_splits(
        input_dir=input_dir,
        outdir=outdir
    )

    runtime = (time.time() - start_time) / 60
    logging.info(f'Runtime: {runtime:.2f} minutes')
    logging.info(f'Finished stage 3 (compute scores from prediction data).')


if __name__ == "__main__":
    main()
"""
Step 0: Aggregate Raw Predictions from CSA Workflow Outputs
================================================================

This stage collects raw prediction data from the complete CSA workflow model runs
and aggregates them into a structured format for post-processing analysis.

PURPOSE:
--------
- Collects test set predictions from all 7 DRP models across all datasets and splits
- Aggregates predictions from the 4TB CSA workflow outputs into organized CSV files
- Generates the test_preds/ directory

INPUT:
------
- Requires access to full CSA workflow outputs (~4TB)
- Default location: --runs-dir ../run/v1.1 (contains model run directories)
    - The full path is: /nfs/ml_lab/projects/improve/data/experiments/run/v1.1

OUTPUT:
-------
- Creates test_preds/ directory with prediction CSV files
- Files follow pattern: <SRC>_<TRG>_split_<ID>_<MODEL>.csv
- Output size: ~2.8GB when extracted, ~1.6GB zipped

USAGE:
------
    python s0_aggregate.py --runs-dir <path_to_model_runs> --outdir <output_directory>

NOTE:
-----
- This stage is OPTIONAL and primarily for internal use by the original authors
- Most users should skip this stage and download pre-computed predictions from Zenodo instead:
  https://zenodo.org/records/15851723
- Requires access to the complete 4TB model run outputs (not publicly available)

RUNTIME:
--------
- Typical runtime: ~10 minutes
"""

import argparse
import logging
import time
from pathlib import Path
from postprocess_utils import collect_and_save_raw_preds, setup_logging


def main():
    start_time = time.time()

    parser = argparse.ArgumentParser(
        description='Stage 0: Aggregate raw prediction data from CSA workflow model runs.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        This stage is optional and requires access to the full 4TB CSA workflow outputs.
        Most users should download pre-computed predictions from Zenodo instead.
        Zenodo link: https://zenodo.org/records/15851723/files/test_preds.zip?download=1
        """)
    parser.add_argument(
        '--outdir',
        default='.',
        type=str,
        help='Output directory for aggregated predictions (default: current directory)')
    parser.add_argument(
        '--runs-dir',
        default='../run/v1.1',
        type=str, 
        help='Directory containing model run outputs (default: ../run/v1.1)')
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    setup_logging(log_file='s0_aggregate.log')

    # logging.info("=" * 50)
    logging.info(f"\n{'=' * 50}")
    logging.info("Stage 0: Aggregate Raw Predictions from CSA Workflow")
    logging.info("=" * 50)
    logging.info(f"Input (runs directory): {args.runs_dir}")
    logging.info(f"Output directory: {outdir}")

    # Get model paths
    main_models_path = Path(args.runs_dir)
    if not main_models_path.exists():
        logging.error(f"Runs directory not found: {main_models_path}")
        logging.error("This stage requires access to the full CSA workflow outputs.")
        logging.error("Most users should download pre-computed predictions from Zenodo:")
        logging.error("  https://zenodo.org/records/15851723")
        raise FileNotFoundError(f"Runs directory not found: {main_models_path}")

    models_paths_list = sorted(p for p in main_models_path.glob('*') 
                             if (p/'improve_output').exists())

    if not models_paths_list:
        logging.warning(f"No model runs found in {main_models_path}")
        logging.warning("Expected subdirectories with 'improve_output' folders")
        return

    logging.info(f"Found {len(models_paths_list)} model run directories")

    # Collect predictions from validation and test sets
    logging.info("Collecting validation set predictions...")
    collect_and_save_raw_preds(
        models_paths_list, outdir, subset_type='val', stage='models'
    )

    logging.info("Collecting test set predictions...")
    collect_and_save_raw_preds(
        models_paths_list, outdir, subset_type='test', stage='infer'
    )

    runtime = (time.time() - start_time) / 60
    logging.info(f"Runtime: {runtime:.2f} minutes")
    logging.info(f"Stage 0 complete. Predictions saved to {outdir}")
    
    print(f'\n✅ Finished {Path(__file__).name}!')


if __name__ == "__main__":
    main()

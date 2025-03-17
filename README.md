# Data Generation Pipeline

This repository contains four stages of data processing:

## Stage 1: Collect Raw Predictions from Workflow Runs (Internal Use)
- Script: `stage1_collect_raw_predictions.py`
- Input: Access to full CSA model runs (~4TB)
- Output: Raw prediction data (~2.4GB) (stored in the `./test_preds/`, not available in this repo)
- Runtime: ~10 minutes
- Note: This stage is primarily for internal use by the original authors

## Stage 2: Compute Scores (Optional)
- Script: `stage2_compute_scores.py`
- Input: Test set predictions from Stage 1 (`./test_preds/`)
- Output: The computed scores are stored in the [`splits_averaged/`](./splits_averaged/) directory
- Runtime: ~70 minutes

## Stage 3: Paper Data Generation (Public Use)
- Script: `stage3_generate_csa_tables.py`
- Input: Scores from Stage 2 (provided in [`splits_averaged/`](./splits_averaged/))
- Output: CSA tables stored in [`splits_averaged/`](./splits_averaged/)
- Runtime: few seconds
- Note: This is the main entry point for reproducing paper results

## Stage 4: Generate Paper Plots
- Notebook: `stage4_generate_paper_plots.ipynb`
- Input: Scores (from Stage 2) and CSA tables (from Stage 3) all located in  [`splits_averaged/`](./splits_averaged/)
- Output: Plots for the paper saved in [`results_for_paper/`](./results_for_paper/)


## For Users
To reproduce the paper's results:
1. Clone this repository
2. Create computational environment
3. Run `stage3_generate_csa_tables.py` with the available data
4. Use `stage4_generate_paper_plots.ipynb` to generate plots


## For Advanced Users
If you want to recompute scores from raw predictions:
1. Contact authors for access to raw prediction data
2. Run `stage2_compute_scores.py`
3. Proceed with Stage 3


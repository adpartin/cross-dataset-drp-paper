# Reproduce results from the paper

A. Partin and P. Vasanthakumari et al. "Benchmarking community drug response prediction models: datasets, models, tools, and metrics for cross-dataset generalization analysis"

## Paper abstract:<br> 
Deep learning (DL) and machine learning (ML) models have shown promise in drug response prediction (DRP), yet their ability to generalize across datasets remains an open question, raising concerns about their real-world applicability. Due to the lack of standardized benchmarking approaches, model evaluations and comparisons often rely on inconsistent datasets and evaluation criteria, making it difficult to assess true predictive capabilities. In this work, we introduce a benchmarking framework for evaluating cross-dataset prediction generalization in DRP models. Our framework incorporates five publicly available drug screening datasets, six standardized DRP models, and a scalable workflow for systematic evaluation. To assess model generalization, we introduce a set of evaluation metrics that quantify both absolute performance (e.g., predictive accuracy across datasets) and relative performance (e.g., performance drop compared to within-dataset results), enabling a more comprehensive assessment of model transferability. Our results reveal substantial performance drops when models are tested on unseen datasets, underscoring the importance of rigorous generalization assessments. While several models demonstrate relatively strong cross-dataset generalization, no single model consistently outperforms across all datasets. Furthermore, we identify CTRPv2 as the most effective source dataset for training, yielding higher generalization scores across target datasets. By sharing this standardized evaluation framework with the community, our study aims to establish a rigorous foundation for model comparison, and accelerate the development of robust DRP models for real-world applications

## For Users
To reproduce the paper's results:
1. Clone this repository
2. Create computational environment
    ```bash
    bash ./setup_improve.sh
    conda env create -f environment.yml
    conda activate drp-benchmark
    ```
4. Run `stage3_generate_csa_tables.py` with the available data
5. Use `stage4_generate_paper_plots.ipynb` to generate plots


## For Advanced Users
If you want to recompute scores from raw predictions:
1. Contact authors for access to raw prediction data
2. Run `stage2_compute_scores.py`
3. Proceed with Stage 3


## Processing steps:

### Stage 1: Collect Raw Predictions from Workflow Runs (Internal Use)
- Script: `stage1_collect_raw_predictions.py`
- Input: Access to full CSA model runs (~4TB)
- Output: Raw prediction data (~2.4GB) (stored in the `./test_preds/`, not available in this repo)
- Runtime: ~10 minutes
- Note: This stage is primarily for internal use by the original authors

### Stage 2: Compute Scores (Optional)
- Script: `stage2_compute_scores.py`
- Input: Test set predictions from Stage 1 (`./test_preds/`)
- Output: The computed scores are stored in the [`splits_averaged`](./splits_averaged/) directory
- Runtime: ~70 minutes

### Stage 3: Paper Data Generation (Public Use)
- Script: `stage3_generate_csa_tables.py`
- Input: Scores from Stage 2 (provided in [`splits_averaged`](./splits_averaged/))
- Output: CSA tables stored in [`splits_averaged`](./splits_averaged/)
- Runtime: few seconds
- Note: This is the main entry point for reproducing paper results

### Stage 4: Generate Paper Plots
- Notebook: `stage4_generate_paper_plots.ipynb`
- Input: Scores (from Stage 2) and CSA tables (from Stage 3) all located in [`splits_averaged`](./splits_averaged/)
- Output: Plots for the paper saved in [`results_for_paper`](./results_for_paper/)


#### Location
The data is located in `/nfs/ml_lab/projects/improve/data/experiments` on Lambda (Argonne's GPU cluster)

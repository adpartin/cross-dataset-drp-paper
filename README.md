# Benchmarking Study of Model Generalization in Drug Response Prediction Across Datasets

**A. Partin and P. Vasanthakumari et al.**  
*"Benchmarking community drug response prediction models: datasets, models, tools, and metrics for cross-dataset generalization analysis"*

## Abstract

Deep learning (DL) and machine learning (ML) models have shown promise in drug response prediction (DRP), yet their ability to generalize across datasets remains an open question, raising concerns about their real-world applicability. Due to the lack of standardized benchmarking approaches, model evaluations and comparisons often rely on inconsistent datasets and evaluation criteria, making it difficult to assess true predictive capabilities. In this work, we introduce a benchmarking framework for evaluating cross-dataset prediction generalization in DRP models. Our framework incorporates five publicly available drug screening datasets, seven standardized DRP models, and a scalable workflow for systematic evaluation. To assess model generalization, we introduce a set of evaluation metrics that quantify both absolute performance (e.g., predictive accuracy across datasets) and relative performance (e.g., performance drop compared to within-dataset results), enabling a more comprehensive assessment of model transferability. Our results reveal substantial performance drops when models are tested on unseen datasets, underscoring the importance of rigorous generalization assessments. While several models demonstrate relatively strong cross-dataset generalization, no single model consistently outperforms across all datasets. Furthermore, we identify CTRPv2 as the most effective source dataset for training, yielding higher generalization scores across target datasets. By sharing this standardized evaluation framework with the community, our study aims to establish a rigorous foundation for model comparison, and accelerate the development of robust DRP models for real-world applications.

## Repository Overview

This repository serves as the main landing page for reproducing the results from the benchmarking study. It provides comprehensive instructions, scripts, and resources for:

1. **Running the cross-study analysis (CSA) workflow** for all 7 DRP models
2. **Reproducing paper results** using pre-computed model predictions
3. **Accessing all datasets, models, and experimental results**

### Research Questions Addressed

This study addresses several key questions about cross-dataset generalization in DRP models:

1. **Generalization Performance**: Do DRP models maintain performance when tested on datasets different from training?
2. **Dataset Transferability**: Which source datasets provide the best foundation for training generalizable models?
3. **Performance Degradation**: How much does performance drop when transferring across datasets vs. within-dataset performance?
4. **Coverage vs. Performance**: Is there a correlation between dataset overlap (drug/cell coverage) and cross-dataset performance?

## Quick Start (Recommended)

To reproduce the results without re-running computationally intensive model training:

```bash
# 1. Clone this repository
git clone https://github.com/adpartin/cross-dataset-drp-paper.git
cd cross-dataset-drp-paper

# 2. Set up environment
conda env create -f environment.yml
conda activate csa-paper-2025

# 3. Run complete postprocessing pipeline
# This will automatically download predictions if needed and run all 6 steps
./run_postprocessing.sh
```

**Results will be generated in the `outputs/` directory:**
- `outputs/s1_scores/` - Performance scores computed from predictions
- `outputs/s2_G_matrices/` - Source × target performance matrices
- `outputs/s3_GaGnGna/` - Normalized and aggregated variants of matrix G (includes figures)
- `outputs/s4_stats/` - Statistical test results with figures in subdirectories:
  - `reviewer2_comment8/figures/` - Bubble plots and bar plots
  - `reviewer3_comment1/figures/` - Wilcoxon test visualizations
- `outputs/s5_shap/` - SHAP analysis with figures in `figures/` subdirectory
- `outputs/s6_overlap/` - Coverage and overlap analysis with figures in `figures/` subdirectory
- `logs/` - Execution logs for debugging

## Complete Pipeline Overview

The complete pipeline has two phases:

1. **Phase 1: CSA Workflow Execution** (Computationally Intensive - **OPTIONAL**)
   - Run the CSA workflow for each of the 7 DRP models to generate raw predictions
   - **Most users can SKIP this phase** - pre-computed predictions are available on Zenodo
   
2. **Phase 2: Post-Processing Analysis** (6-Step Pipeline - **RECOMMENDED STARTING POINT**)
   - Process raw predictions through a 6-step analysis pipeline
   - Generates all paper results: scores, matrices, statistics, figures, and coverage analysis

**Quick Start:** If you want to reproduce paper results without re-running model training, start directly with **Phase 2** by downloading pre-computed predictions (see Quick Start section above).

---

### Phase 1: CSA Workflow Execution (OPTIONAL - Computationally Intensive)

**Note:** This phase is OPTIONAL. Most users should skip this and use pre-computed predictions from Zenodo (automatically downloaded in Phase 2).

Run the CSA workflow for each of the 7 DRP models to generate raw predictions:

**Prerequisites:**
- Access to computational resources (GPU recommended)
- Benchmark datasets from Zenodo: https://zenodo.org/records/15258883
- IMPROVE CSA workflow (tag `v0.1.0`): https://github.com/JDACS4C-IMPROVE/IMPROVE/tree/v0.1.0/workflows/csa/parsl

**Models and Versions:**
All models are tagged with `v0.1.0` to ensure reproducibility:

1. **DeepCDR**: https://github.com/JDACS4C-IMPROVE/DeepCDR/tree/branch-v0.1.0
2. **DeepTTC**: https://github.com/JDACS4C-IMPROVE/DeepTTC/tree/v0.1.0
3. **GraphDRP**: https://github.com/JDACS4C-IMPROVE/GraphDRP/tree/v0.1.0
4. **HiDRA**: https://github.com/JDACS4C-IMPROVE/HiDRA/tree/v0.1.0
5. **LGBM**: https://github.com/JDACS4C-IMPROVE/LGBM/tree/v0.1.0
6. **tCNNS**: https://github.com/JDACS4C-IMPROVE/tCNNS-Project/tree/v0.1.0
7. **UNO**: https://github.com/JDACS4C-IMPROVE/UNO/tree/v0.1.0

### Phase 2: Post-Processing Analysis (6-Step Pipeline)

**This is the main workflow for reproducing paper results.**

Process raw predictions through a 6-step analysis pipeline to generate all paper results.

**Naming Convention:** Scripts use the prefix `s1`, `s2`, `s3`, etc. to indicate their step in the pipeline. Some steps include both Python scripts (`.py`) and Jupyter notebooks (`.ipynb`) with matching script IDs (e.g., `s3_compute_Gn_Ga_Gna.py` and `s3_compute_and_plot_Gn_Ga_Gna.ipynb`).

**Pipeline Flow:**
```
Raw Predictions (test_preds/)
    ↓
Step 1: Compute Performance Scores → outputs/s1_scores/
    ↓
Step 2: Compute G Matrices → outputs/s2_G_matrices/
    ↓
Step 3: Compute Ga/Gn/Gna Variants → outputs/s3_GaGnGna/
    ↓
Step 4: Statistical Analysis (Wilcoxon) → outputs/s4_stats/
    ↓
Step 5: Figure Generation (Notebooks) → outputs/s3_GaGnGna/figures/
                                          outputs/s4_stats/*/figures/
                                          outputs/s5_shap/figures/
    ↓
Step 6: Coverage Analysis → outputs/s6_overlap/ (+ figures/)
```

```bash
# Run complete 6-step pipeline (RECOMMENDED)
./run_postprocessing.sh

# Or run individual steps manually:
# Step 1: Compute performance scores from predictions
python s1_compute_scores.py

# Step 2: Compute G matrices from scores  
python s2_compute_G_matrices.py

# Step 3: Compute Ga/Gn/Gna variants
python s3_compute_Gn_Ga_Gna.py

# Step 4: Statistical analysis (Wilcoxon tests)
python s4_stats_wilcoxon.py

# Step 5: Figure generation (executes 3 notebooks)
# - s3_compute_and_plot_Gn_Ga_Gna.ipynb
# - s4_wilcoxon_and_bubble_plots.ipynb  
# - s5_shap.ipynb
# These are executed automatically by run_postprocessing.sh
# Or execute manually: jupyter nbconvert --execute <notebook>.ipynb

# Step 6: Coverage/overlap analysis
python s6_overlap.py

# Note: s0_aggregate.py is optional (requires 4TB CSA workflow outputs)
# Most users should use pre-computed predictions from Zenodo instead
```

## Resources and Data Access

### Datasets
- **Benchmark Dataset**: https://zenodo.org/records/15258883
  - DOI: [10.5281/zenodo.15258883](https://doi.org/10.5281/zenodo.15258883)
  - Contains 5 publicly available drug screening datasets
  - Standardized format for cross-dataset evaluation

### Experimental Results
- **Pre-computed Predictions**: https://zenodo.org/records/15258742
  - DOI: [10.5281/zenodo.15258742](https://doi.org/10.5281/zenodo.15258742)
  - Contains `test_preds.zip` (~1.6GB) with raw model predictions
  - Extracts to ~2.8GB of prediction data
  - Enables Step 2 analysis without re-running computationally intensive Step 1
  - Automatically downloaded by `./fetch_test_preds.sh`

### Software Framework
- **IMPROVE v0.1.0**: https://github.com/JDACS4C-IMPROVE/IMPROVE/tree/v0.1.0
  - Core framework for DRP model training and evaluation
  - CSA workflow implementation
  - All model implementations depend on this version

## Detailed Pipeline Steps

### Step 0: Aggregate Raw Predictions from CSA Workflow (`s0_aggregate.py`) - OPTIONAL
- **Purpose**: Collect and aggregate raw prediction data from complete CSA workflow outputs
- **Input**: Full CSA workflow model run outputs (~4TB, not publicly available)
- **Output**: `test_preds/` directory with organized prediction CSV files (~2.8GB extracted)
- **Runtime**: ~10 minutes
- **Note**: This step is **OPTIONAL** and for internal use only. Most users should skip this step and download pre-computed predictions from Zenodo (automatically done by `./fetch_test_preds.sh`)

### Step 1: Compute Performance Scores (`s1_compute_scores.py`)
- **Purpose**: Compute performance metrics (R², MSE, RMSE, MAE, etc.) from prediction files
- **Input**: Prediction CSV files from `test_preds/` directory
- **Output**: Computed scores averaged across splits, saved to `outputs/s1_scores/`
- **Runtime**: ~70 minutes

### Step 2: Compute G Matrices (`s2_compute_G_matrices.py`)
- **Purpose**: Generate source×target performance matrices (G matrices) for cross-study analysis
- **Input**: Score CSV files from `outputs/s1_scores/` (generated by Step 1)
- **Output**: G matrices (mean and std) for each model and metric, saved to `outputs/s2_G_matrices/`
- **Runtime**: < 1 minute

### Step 3: Compute Ga/Gn/Gna Variants (`s3_compute_Gn_Ga_Gna.py`)
- **Purpose**: Compute normalized and aggregated matrix variants for generalization analysis
  - **Gn**: Row-wise normalized G matrices (normalized by within-study performance)
  - **Ga**: Aggregated G across targets per source (mean excluding diagonal)
  - **Gna**: Aggregated normalized G (mean of normalized scores excluding diagonal)
- **Input**: G matrices from `outputs/s2_G_matrices/`, scores from `outputs/s1_scores/`
- **Output**: Ga/Gn/Gna matrices and aggregated tables, saved to `outputs/s3_GaGnGna/`
- **Runtime**: ~5 minutes

### Step 4: Statistical Analysis (`s4_stats_wilcoxon.py`)
- **Purpose**: Perform Wilcoxon signed-rank tests for pairwise model comparisons
- **Input**: Score CSV files from `outputs/s1_scores/`
- **Output**: Statistical test results for all model pairs per (source, target), saved to `outputs/s4_stats/`
- **Runtime**: ~2 minutes

### Step 5: Figure Generation (Jupyter Notebooks)
- **Purpose**: Generate all publication figures and tables
- **Notebooks** (executed automatically by `run_postprocessing.sh`): 
  - `s3_compute_and_plot_Gn_Ga_Gna.ipynb` - Visualizes Ga/Gn/Gna matrices computed in Step 3
  - `s4_wilcoxon_and_bubble_plots.ipynb` - Visualizes statistical tests from Step 4 (Wilcoxon boxplots, bubble heatmaps)
  - `s5_shap.ipynb` - SHAP feature importance analysis (standalone visualization)
- **Naming Convention**: Notebook script IDs (s3, s4, s5) indicate they visualize outputs from corresponding pipeline steps, not their own step number. Step 5 encompasses all three notebooks.
- **Input**: All computed matrices from Steps 1-4
- **Output**: Publication-ready figures saved to step-specific subdirectories:
  - `outputs/s3_GaGnGna/figures/` - Ga/Gn/Gna plots (from s3 notebook)
  - `outputs/s4_stats/reviewer2_comment8/figures/` - Bubble heatmaps and bar plots (from s4 notebook)
  - `outputs/s4_stats/reviewer3_comment1/figures/` - Wilcoxon test boxplots (from s4 notebook)
  - `outputs/s5_shap/figures/` - SHAP plots (from s5 notebook)
- **Runtime**: ~20 minutes

### Step 6: Coverage Analysis (`s6_overlap.py`)
- **Purpose**: Analyze dataset overlap (drug/cell coverage) and correlate with cross-dataset performance
- **Input**: Prediction CSV files from `test_preds/`, G matrices from `outputs/s2_G_matrices/`
- **Output**: Overlap metrics, coverage matrices (CSV files), and visualizations, saved to:
  - `outputs/s6_overlap/` - Coverage CSV files
  - `outputs/s6_overlap/figures/` - Heatmaps and scatter plots
- **Runtime**: ~10 minutes

## Environment Setup

### Required Software
- Python 3.8+
- Conda package manager
- Jupyter nbconvert (for programmatic notebook execution)

### Installation
```bash
# Create conda environment (includes nbconvert for notebook execution)
conda env create -f environment.yml
conda activate csa-paper-2025

# Verify installation
python -c "import pandas, numpy, matplotlib, seaborn, scipy; print('All packages installed successfully')"

# Verify nbconvert is installed (required for Step 5)
jupyter nbconvert --version

# Note: Step 5 requires nbconvert to execute notebooks programmatically
# If nbconvert is missing, install it: conda install -c conda-forge nbconvert
```

### Output Structure

All scripts use hardcoded default paths (no configuration file needed). Results are organized as follows:

```
outputs/
├── s1_scores/          # Performance scores from Step 1
├── s2_G_matrices/      # G matrices from Step 2
├── s3_GaGnGna/         # Ga/Gn/Gna variants from Step 3 (+ figures from s3 notebook)
├── s4_stats/           # Statistical test results from Step 4 (+ figures from s4 notebook)
├── s5_shap/            # SHAP analysis results and figures from s5 notebook
└── s6_overlap/         # Coverage analysis from Step 6 (+ figures)

logs/                   # Execution logs (at repository root, not in outputs/)
```

## Troubleshooting

### Common Issues

**1. Missing test_preds directory**
```bash
# Error: test_preds/ not found
# Solution: Download and extract test_preds.zip automatically
./fetch_test_preds.sh
```

**2. Environment issues**
```bash
# Error: Package not found
# Solution: Recreate environment
conda env remove -n csa-paper-2025
conda env create -f environment.yml
conda activate csa-paper-2025
```

**3. Permission issues**
```bash
# Error: Permission denied
# Solution: Make scripts executable
chmod +x quickstart.sh run_postprocessing.sh
```

## Citation

If you use this work, please cite:

<!-- ```bibtex
@article{partin2025benchmarking,
  title={Benchmarking community drug response prediction models: datasets, models, tools, and metrics for cross-dataset generalization analysis},
  author={Partin, A. and Vasanthakumari, P. and others},
  journal={[Journal Name]},
  year={2024},
  doi={[DOI]}
}
``` -->

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---

<!-- **Last Updated**: [Current Date]  
**Repository Version**: v1.2   -->
**IMPROVE Framework Version**: `v0.1.0`
**All Model Versions**: `v0.1.0`

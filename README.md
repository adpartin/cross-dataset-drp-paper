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

For most users who want to reproduce the paper's results without re-running computationally intensive model training:

```bash
# 1. Clone this repository
git clone https://github.com/adpartin/cross-dataset-drp-paper.git
cd cross-dataset-drp-paper

# 2. Download and extract pre-computed predictions
./fetch_test_preds.sh

# 3. Set up environment
conda env create -f environment.yml
conda activate csa-paper

# 4. Run complete postprocessing pipeline
./quickstart.sh
```

**Results will be generated in the `outputs/` directory:**
- `outputs/s1_scores/` - Performance scores computed from predictions
- `outputs/s2_G_matrices/` - Source×target performance matrices
- `outputs/s3_GaGnGna/` - Normalized and aggregated matrix variants
- `outputs/s4_stats/` - Statistical test results (Wilcoxon)
- `outputs/figures/` - All publication figures
- `outputs/tables/` - All publication tables  
- `outputs/s6_overlap/` - Coverage and overlap analysis
- `logs/` - Execution logs for debugging

## Complete Workflow (Two-Step Process)

### Step 1: CSA Workflow Execution (Computationally Intensive)

Run the CSA workflow for each of the 7 DRP models to generate raw predictions:

**Prerequisites:**
- Access to computational resources (GPU recommended)
- IMPROVE software framework v0.1.0
- Benchmark datasets from Zenodo

**Models and Versions:**
All models are tagged with v0.1.0 to ensure reproducibility:

1. **DeepCDR**: https://github.com/JDACS4C-IMPROVE/DeepCDR/tree/branch-v0.1.0
2. **DeepTTC**: https://github.com/JDACS4C-IMPROVE/DeepTTC/tree/v0.1.0
3. **GraphDRP**: https://github.com/JDACS4C-IMPROVE/GraphDRP/tree/v0.1.0
4. **HiDRA**: https://github.com/JDACS4C-IMPROVE/HiDRA/tree/v0.1.0
5. **LGBM**: https://github.com/JDACS4C-IMPROVE/LGBM/tree/v0.1.0
6. **tCNNS**: https://github.com/JDACS4C-IMPROVE/tCNNS-Project/tree/v0.1.0
7. **UNO**: https://github.com/JDACS4C-IMPROVE/UNO/tree/v0.1.0

**CSA Workflow:**
- **IMPROVE Framework**: https://github.com/JDACS4C-IMPROVE/IMPROVE/tree/v0.1.0
- **CSA Workflow**: https://github.com/JDACS4C-IMPROVE/IMPROVE/tree/v0.1.0/workflows/csa/parsl
- **Benchmark Datasets**: https://zenodo.org/records/15258883

### Step 2: Post-Processing Analysis (6 Steps)

Process raw predictions through our 6-step analysis pipeline to generate all paper results:

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
Step 5: Figure Generation → outputs/figures/
    ↓
Step 6: Coverage Analysis → outputs/s6_overlap/
```

```bash
# Run complete pipeline
./run_postprocessing.sh

# Or run individual steps
# Note: Step 0 (s0_aggregate.py) is optional - requires 4TB CSA workflow outputs
python s1_compute_scores.py      # Step 1: Compute performance scores from predictions
python s2_compute_G_matrices.py  # Step 2: Compute G matrices from scores
python s3_compute_Gn_Ga_Gna.py   # Step 3: Compute Ga/Gn/Gna variants
python s4_stats_wilcoxon.py      # Step 4: Statistical analysis (Wilcoxon tests)
# Step 5: Figure generation (notebooks: stage4_generate_paper_plots.ipynb)
python s6_overlap.py             # Step 6: Coverage/overlap analysis
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

### Step 5: Figure Generation (Notebooks)
- **Purpose**: Generate all publication figures and tables
- **Scripts/Notebooks**: `stage4_generate_paper_plots.ipynb`, `stage5_revision1.ipynb`
- **Input**: All computed matrices from previous steps
- **Output**: Publication-ready figures and tables, saved to `outputs/figures/` and `outputs/tables/`
- **Runtime**: ~20 minutes

### Step 6: Coverage Analysis (`s6_overlap.py`)
- **Purpose**: Analyze dataset overlap (drug/cell coverage) and correlate with cross-dataset performance
- **Input**: Prediction CSV files from `test_preds/`, G matrices from `outputs/s2_G_matrices/`
- **Output**: Overlap metrics, coverage matrices, correlation analysis, and visualizations, saved to `outputs/s6_overlap/`
- **Runtime**: ~10 minutes

## Environment Setup

### Required Software
- Python 3.8+
- Conda package manager
- Jupyter notebook (for interactive analysis)

### Installation
```bash
# Create conda environment
conda env create -f environment.yml
conda activate csa-paper-2025

# Verify installation
python -c "import pandas, numpy, matplotlib, seaborn, scipy; print('All packages installed successfully')"

# Note: Some steps require Jupyter notebook support for figure generation
# The pipeline will automatically execute notebooks if available
```

### Optional Configuration
Create `configs/paths.yaml` to customize file paths:
```yaml
# Location of precomputed predictions
preds_dir: "./test_preds"

# Output directories
outputs_root: "./outputs"
derived_dir: "./outputs/derived"
figures_dir: "./outputs/figures"
tables_dir: "./outputs/tables"
logs_dir: "./outputs/logs"
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
conda env remove -n csa-paper
conda env create -f environment.yml
conda activate csa-paper
```

**3. Permission issues**
```bash
# Error: Permission denied
# Solution: Make scripts executable
chmod +x quickstart.sh run_postprocessing.sh
```

**4. Memory issues**
- Ensure sufficient RAM (8GB+ recommended)
- Monitor memory usage during execution
- Consider running stages individually if needed

### Getting Help
- Check execution logs in `logs/` for detailed error messages
- Review individual step scripts for specific requirements
- Contact authors for additional support

## Citation

If you use this work, please cite:

```bibtex
@article{partin2024benchmarking,
  title={Benchmarking community drug response prediction models: datasets, models, tools, and metrics for cross-dataset generalization analysis},
  author={Partin, A. and Vasanthakumari, P. and others},
  journal={[Journal Name]},
  year={2024},
  doi={[DOI]}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

For questions about this repository or the CSA benchmarking study:
- **Issues**: Use GitHub Issues for bug reports and questions
- **Email**: [Contact information]
- **Paper**: [Paper link when published]

---

**Last Updated**: [Current Date]  
**Repository Version**: v1.2  
**IMPROVE Framework Version**: v0.1.0  
**All Model Versions**: v0.1.0

"""
Ensemble analysis utilities for drug response prediction models.
This script provides functions for:
1. Building ensemble weights from validation scores
2. Averaging predictions across splits
3. Computing ensemble performance metrics
4. Analyzing dataset overlaps
"""

import argparse
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from improvelib.metrics import compute_metrics

# Constants
CANC_COL_NAME = 'improve_sample_id'
DRUG_COL_NAME = 'improve_chem_id'
Y_COL_NAME = 'auc'

def setup_logging(log_file: str = 'ensemble_analysis.log') -> None:
    """Configure logging for both file and console output."""
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(
        logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    )
    logging.getLogger().addHandler(console_handler)

def build_weights_dfs(val_scores: pd.DataFrame,
                     outdir: Union[Path, str]='.') -> None:
    """Build weights DataFrames for ensemble predictions.
    
    Args:
        val_scores: DataFrame containing validation scores
        outdir: Output directory for weight files
    """
    os.makedirs(outdir, exist_ok=True)
    metrics = val_scores.columns[4:].tolist()
    models = sorted(val_scores['model'].unique())
    
    for model in models:
        df = val_scores[val_scores['model'] == model]
        for source in sorted(val_scores['source'].unique()):
            jj = df[df['source'] == source].copy()
            for met in metrics:
                jj[f'{met}_weight'] = jj[met] / jj[met].sum()
                jj.to_csv(outdir / f'val_scores_and_weights_{model}_{source}.csv',
                         index=False)

def prediction_averaging(model: str, source: str, target: str,
                       outdir: Union[Path, str], met: str = 'r2') -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Average predictions across splits with optional weighting.
    
    Args:
        model: Model name
        source: Source dataset name
        target: Target dataset name
        outdir: Output directory
        met: Metric to use for weighting
    
    Returns:
        Tuple of (raw_predictions_df, weighted_predictions_df)
    """
    os.makedirs(outdir, exist_ok=True)
    
    # Load weights
    weights_dir = Path(outdir) / 'weights'
    preds_dir = Path(outdir) / 'test_preds'
    
    wdf = pd.read_csv(weights_dir / f'val_scores_and_weights_{model}_{source}.csv')
    weights = {f'split_{s}': w for s, w in zip(wdf['split'], wdf[f'{met}_weight'])}

    raw_preds = {}
    weighted_preds = {}

    for split, weight in weights.items():
        fname = f'{source}_{target}_{split}_{model}.csv'
        pdf = pd.read_csv(preds_dir / fname)
        raw_preds[split] = pdf['auc_pred'].values
        weighted_preds[split] = weight * pdf['auc_pred'].values

    # Process raw predictions
    raw_df = pd.DataFrame(raw_preds)
    raw_df['ens_pred'] = raw_df.mean(axis=1)

    # Process weighted predictions
    weighted_df = pd.DataFrame(weighted_preds)
    weighted_df['ens_pred'] = weighted_df.sum(axis=1)

    # Add metadata
    meta_cols = ['model', 'src', 'trg', CANC_COL_NAME, DRUG_COL_NAME, 'auc_true']
    meta_df = pdf[meta_cols]
    
    raw_df = pd.concat([meta_df, raw_df], axis=1)
    weighted_df = pd.concat([meta_df, weighted_df], axis=1)
    
    # Save results
    raw_df.to_csv(outdir / f'{model}_{source}_{target}_mean_preds.csv', index=False)
    weighted_df.to_csv(outdir / f'{model}_{source}_{target}_weighted_preds.csv', index=False)
    
    return raw_df, weighted_df

def compute_overlap_metrics(src_data: pd.DataFrame, 
                          trg_data: pd.DataFrame) -> Dict[str, float]:
    """Compute overlap metrics between source and target datasets.
    
    Args:
        src_data: Source dataset
        trg_data: Target dataset
    
    Returns:
        Dictionary containing overlap metrics
    """
    # Compute drug overlaps
    src_drugs = set(src_data[DRUG_COL_NAME])
    trg_drugs = set(trg_data[DRUG_COL_NAME])
    overlapping_drugs = trg_drugs.intersection(src_drugs)
    
    # Compute cell overlaps
    src_cells = set(src_data[CANC_COL_NAME])
    trg_cells = set(trg_data[CANC_COL_NAME])
    overlapping_cells = trg_cells.intersection(src_cells)
    
    # Compute overlap counts
    trg_drug_overlap_cnt = trg_data[trg_data[DRUG_COL_NAME].isin(overlapping_drugs)].shape[0]
    trg_cell_overlap_cnt = trg_data[trg_data[CANC_COL_NAME].isin(overlapping_cells)].shape[0]
    tot_trg_samples = trg_data.shape[0]
    
    return {
        "trg_drug_overlap_ratio": trg_drug_overlap_cnt / tot_trg_samples,
        "trg_cell_overlap_ratio": trg_cell_overlap_cnt / tot_trg_samples,
        "trg_drug_overlap_cnt": trg_drug_overlap_cnt,
        "trg_cell_overlap_cnt": trg_cell_overlap_cnt,
        "total_trg_samples": tot_trg_samples
    }

def analyze_model_overlaps(model_name: str, 
                         scores_dir: Path,
                         data_dir: Path) -> pd.DataFrame:
    """Analyze overlaps for a specific model.
    
    Args:
        model_name: Name of the model to analyze
        scores_dir: Directory containing score files
        data_dir: Directory containing data files
    
    Returns:
        DataFrame with overlap analysis results
    """
    perf_data = pd.read_csv(scores_dir / f'{model_name}_scores.csv')
    columns_to_load = [CANC_COL_NAME, DRUG_COL_NAME, Y_COL_NAME]
    dfs = []

    for (src, trg, split), group_df in perf_data.groupby(["src", "trg", "split"]):
        logging.info(f'Processing {src}-{trg}; split {split}')
        group_df = group_df.reset_index(drop=True)
        
        # Load source data
        src_data = pd.read_csv(
            data_dir / f'{model_name}_{src}_split_{split}_train.csv',
            usecols=columns_to_load
        )
        
        # Load target data
        if src == trg:
            trg_data = pd.read_csv(
                data_dir / f'{model_name}_{src}_split_{split}_test.csv',
                usecols=columns_to_load
            )
        else:
            trg_data = pd.read_csv(
                data_dir / f'{model_name}_{trg}_all.csv',
                usecols=columns_to_load
            )

        # Compute overlaps
        overlaps = compute_overlap_metrics(src_data, trg_data)
        for k, v in overlaps.items():
            group_df[k] = v

        group_df['src_samples'] = len(src_data)
        group_df['trg_samples'] = len(trg_data)
        dfs.append(group_df)

    return pd.concat(dfs, axis=0)

def main():
    parser = argparse.ArgumentParser(description='Ensemble analysis.')
    parser.add_argument('--outdir', default='.', type=str, help='Output directory')
    parser.add_argument('--model', default='graphdrp', type=str, help='Model to analyze')
    parser.add_argument('--data-dir', default='./y_data', type=str, help='Data directory')
    args = parser.parse_args()
    
    outdir = Path(args.outdir)
    setup_logging()
    
    # Analyze overlaps
    overlap_results = analyze_model_overlaps(
        args.model,
        scores_dir=outdir / 'splits_averaged',
        data_dir=Path(args.data_dir)
    )
    
    # Save results
    overlap_results.to_csv(
        outdir / 'splits_averaged' / f'{args.model}_scores_with_cell_drug_overlaps.csv',
        index=False
    )
    
    # Analyze correlations
    met = 'r2'
    df = overlap_results[
        (overlap_results['met'] == met) &
        (overlap_results['model'] == args.model)
    ].reset_index(drop=True)

    for col in ["trg_drug_overlap_ratio", "trg_cell_overlap_ratio"]:
        if col in df.columns:
            corr, _ = pearsonr(df[col], df['value'])
            logging.info(f"Correlation between {col} and metric_value: {corr}")

if __name__ == "__main__":
    main()

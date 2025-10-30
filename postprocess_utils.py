"""
Utility functions for generating data required by postprocess_plot.ipynb:
- all_models_scores.csv
- {model}_{metric}_mean_csa_table.csv
- {model}_{metric}_std_csa_table.csv
"""

import logging
import os
import sys
from pathlib import Path
from tqdm import tqdm
from typing import List, Optional, Dict, Any

import numpy as np
import pandas as pd
import sklearn
from math import sqrt
from scipy.stats import pearsonr, spearmanr

# Import sklearn metrics functions
# Handle different sklearn versions for root_mean_squared_error
try:
    from sklearn.metrics import (
        r2_score, mean_squared_error, root_mean_squared_error,
        accuracy_score, balanced_accuracy_score, f1_score,
        precision_score, recall_score, roc_auc_score, average_precision_score
    )
    HAS_RMSE_FUNC = True
except ImportError:
    # Fallback for older sklearn versions
    from sklearn.metrics import (
        r2_score, mean_squared_error,
        accuracy_score, balanced_accuracy_score, f1_score,
        precision_score, recall_score, roc_auc_score, average_precision_score
    )
    HAS_RMSE_FUNC = False

# Column names used throughout the analysis
CANC_COL_NAME = 'improve_sample_id'
DRUG_COL_NAME = 'improve_chem_id'
Y_COL_NAME = 'auc'


# ------------------------------------------------------------
# From improvelib.metrics import compute_metrics
# ------------------------------------------------------------
def mse(y_true, y_pred):
    return mean_squared_error(y_true, y_pred)


def rmse(y_true, y_pred):
    if HAS_RMSE_FUNC:
        return root_mean_squared_error(y_true, y_pred)
    else:
        # For older sklearn versions, compute manually
        try:
            # Try squared=False parameter (available in sklearn >= 0.22.0)
            return mean_squared_error(y_true, y_pred, squared=False)
        except TypeError:
            # Fallback for very old sklearn versions
            return sqrt(mean_squared_error(y_true, y_pred))


def pearson(y_true, y_pred):
    return pearsonr(y_true, y_pred)[0]


def spearman(y_true, y_pred):
    return spearmanr(y_true, y_pred)[0]


def r_square(y_true, y_pred):
    return r2_score(y_true, y_pred)


def acc(y_true, y_pred):
    return accuracy_score(y_true, y_pred)


def bacc(y_true, y_pred):
    return balanced_accuracy_score(y_true, y_pred)


def f1(y_true, y_pred):
    return f1_score(y_true, y_pred)


def precision(y_true, y_pred):
    return precision_score(y_true, y_pred)


def recall(y_true, y_pred):
    return recall_score(y_true, y_pred)


def roc_auc(y_true, y_pred):
    return roc_auc_score(y_true, y_pred)


def aupr(y_true, y_pred):
    return average_precision_score(y_true, y_pred)


def kappa(y_true, y_pred):
    from sklearn.metrics import cohen_kappa_score
    return cohen_kappa_score(y_true, y_pred)


def str2Class(str) -> Any:
    """Convert a string to a class reference.

    Args:
        class_name (str): The name of the class to retrieve.

    Returns:
        Any: The class reference corresponding to the class name.
    """
    return getattr(sys.modules[__name__], str)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_type: str,
    y_prob = None
    ) -> Dict[str, float]:
    """Compute the specified set of metrics.

    Args:
        y_true (np.ndarray): True values to predict.
        y_pred (np.ndarray): Predictions made by the model.
        metric_type (str): Type of metrics to compute ('classification' or 'regression').
        y_prob (np.ndaprray): Target scores made by the classification model. Optional, defaults to None.

    Returns:
        dict: A dictionary of evaluated metrics.
    """
    scores = {}

    if metric_type == "classification":
        metrics = ["acc", "recall", "precision", "f1", "kappa", "bacc"]
    elif metric_type == "regression":
        metrics = ["mse", "rmse", "pcc", "scc", "r2"]
    else:
        raise ValueError(f"Invalid metric_type provided: {metric_type}. \
                         Choose 'classification' or 'regression'.")

    for mtstr in metrics:
        mapstr = mtstr
        if mapstr == "pcc":
            mapstr = "pearson"
        elif mapstr == "scc":
            mapstr = "spearman"
        elif mapstr == "r2":
            mapstr = "r_square"
        scores[mtstr] = str2Class(mapstr)(y_true, y_pred)

    if metric_type == "classification":
        if y_prob is not None:
            scores["roc_auc"] = roc_auc(y_true, y_prob)
            scores["aupr"] = aupr(y_true, y_prob)

    scores = {k: float(v) for k, v in scores.items()}
    return scores
# ------------------------------------------------------------


def setup_logging(log_file: str = 'paper_data_generation.log') -> None:
    """Configure logging for both file and console output.
    
    Args:
        log_file: Name of the log file
    """
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Add console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(
        logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    )
    logging.getLogger().addHandler(console_handler)


def collect_and_save_raw_preds(
    models_paths_list: List[Path],
    outdir: Path,
    subset_type: str,
    stage: str) -> None:
    """Collect and save inference predictions for all models.
    
    Args:
        models_paths_list: List of paths to model directories
        outdir: Output directory for saving predictions
        subset_type: Type of subset ('val' or 'test')
        stage: Processing stage ('models' or 'infer')
    """
    assert subset_type in ['val', 'test'], f"Invalid 'subset_type' ({subset_type})"
    assert stage in ['models', 'infer'], f"Invalid 'stage' ({stage})"

    preds_fname = f'{subset_type}_y_data_predicted.csv'
    out_preds_dir = outdir / f'{subset_type}_preds'
    os.makedirs(out_preds_dir, exist_ok=True)

    cols = [CANC_COL_NAME, DRUG_COL_NAME, Y_COL_NAME,
            f'{Y_COL_NAME}_true', f'{Y_COL_NAME}_pred']
    extra_cols = ['model', 'src', 'trg', 'set', 'split']

    logging.info(f'Collect and save raw predictions into new files ({subset_type} data).')
    missing_files = []

    for model_dir in models_paths_list:
        model_name = model_dir.name.lower()
        logging.info(f'Processing model: {model_name}')
        stage_path = model_dir / 'improve_output' / f'{stage}'
        exps = sorted(stage_path.glob('*'))

        for exp_path in exps:
            src = str(exp_path.name).split("-")[0]
            trg = str(exp_path.name).split("-")[1] if '-' in str(exp_path.name) else None
            splits = sorted(exp_path.glob('*'))

            for split_path in splits:
                sp = split_path.name.split('split_')[1]
                try:
                    df = pd.read_csv(split_path / preds_fname)
                    if not all(c in df.columns for c in cols):
                        continue
                        
                    df = df[cols]
                    df['model'] = model_name
                    df['src'] = src
                    df['trg'] = trg
                    df['set'] = 'val' if trg is None else 'test'
                    df['split'] = sp
                    
                    fname = (f'{src}_{trg}_split_{sp}_{model_name}.csv' 
                           if trg is not None 
                           else f'{src}_split_{sp}_{model_name}.csv')
                    
                    df = df[extra_cols + cols]
                    df.to_csv(out_preds_dir / fname, index=False)
                    
                except FileNotFoundError:
                    logging.warning(f'File not found! {split_path / preds_fname}')
                    missing_files.append(split_path / preds_fname)

    with open(out_preds_dir / 'missing_files.csv', 'w') as f:
        for path in missing_files:
            f.write(f'{path}\n')

    return True


def compute_scores_from_averaged_splits(
    outdir: Path,
    models: Optional[List[str]]=None,
    sources: Optional[List[str]]=None,
    targets: Optional[List[str]]=None,
    filtering: Optional[str]=None
    ) -> pd.DataFrame:
    """Compute scores from averaged splits for each model.
    
    Args:
        outdir: Output directory for saving scores
        models: List of model names to process
        sources: List of source datasets
        targets: List of target datasets
        filtering: Type of filtering to apply
    
    Returns:
        DataFrame containing computed scores
    """
    assert filtering in ['drug_blind', 'cell_blind', 'disjoint', None]
    
    os.makedirs(outdir, exist_ok=True)
    
    # Glob pred files
    preds_dirname = 'test_preds'
    preds_dir = Path('.') / preds_dirname
    pred_files = sorted(preds_dir.glob('*'))

    def valid_fname(fname):
        if len(fname.stem.split('_')) < 3:
            return False
        return True if fname.stem.split('_')[2] == 'split' else False

    # Extract unique sources and model names
    if models is None:
        models = sorted(set([f.stem.split('_')[4] for f in pred_files if valid_fname(f)]))
    if sources is None:
        sources = sorted(set([f.stem.split('_')[0] for f in pred_files if valid_fname(f)]))
    if targets is None:
        targets = sorted(set([f.stem.split('_')[1] for f in pred_files if valid_fname(f)]))

    logging.info('Computing scores from averaged splits.')
    logging.info(f'Models: {models}')
    logging.info(f'Sources: {sources}')
    logging.info(f'Targets: {targets}')

    dfs = []
    # for model in models:
    for model in tqdm(models, desc='Processing models'):
        model_dfs = []
        scores_dict = {}

        # for src in sources:
        for src in tqdm(sources, desc=f'Processing sources for {model}', leave=False):
            # for trg in targets:
            for trg in tqdm(targets, desc=f'Processing targets for {src}', leave=False):

                files = sorted(preds_dir.glob(f'{src}*{trg}*{model}*'))
                
                for fname in files:
                    split = int(fname.stem.split('split_')[1].split('_')[0])
                    df = pd.read_csv(fname)
                    
                    if not df.empty:
                        scores = compute_metrics(
                            df[f'{Y_COL_NAME}_true'].values,
                            df[f'{Y_COL_NAME}_pred'].values,
                            metric_type='regression'
                        )
                        scores_dict[split] = scores

                if scores_dict:
                    scores_df = pd.DataFrame(scores_dict).stack().reset_index()
                    scores_df.columns = ['met', 'split', 'value']
                    scores_df['src'] = src
                    scores_df['trg'] = trg
                    scores_df['model'] = model
                    model_dfs.append(scores_df)

        if model_dfs:
            model_scores = pd.concat(model_dfs, axis=0).reset_index(drop=True)
            fname = f"{model}_scores_{filtering}.csv" if filtering else f"{model}_scores.csv"
            model_scores.to_csv(outdir / fname, index=False)
            dfs.append(model_scores)

    if dfs:
        scores = pd.concat(dfs, axis=0)
        fname = f'all_models_scores_{filtering}.csv' if filtering else 'all_models_scores.csv'
        scores.to_csv(outdir / fname, index=False)
        return scores

    return pd.DataFrame()


def compute_csa_tables_from_averaged_splits(
    input_dir: Path,
    outdir: Path,
    models: Optional[List[str]]=None,
    sources: Optional[List[str]]=None,
    targets: Optional[List[str]]=None,
    filtering: Optional[str]=None
    ) -> None:
    """Generate cross-study analysis (CSA) tables from averaged splits.
    
    Args:
        input_dir: Directory containing input scores
        outdir: Output directory for CSA tables
        models: List of model names to process
        sources: List of source datasets
        targets: List of target datasets
        filtering: Type of filtering to apply
    """
    os.makedirs(outdir, exist_ok=True)
    assert filtering in ['drug_blind', 'cell_blind', 'disjoint', None]

    if models is None:
        all_models_scores = pd.read_csv(input_dir / 'all_models_scores.csv')
        models = sorted(all_models_scores['model'].unique().tolist())

    logging.info('Computing CSA tables.')
    logging.info(f'Models: {models}')

    # for model_name in models:
    for model_name in tqdm(models, desc='Processing models'):
        scores_file = (f"{model_name}_scores_{filtering}.csv" 
                      if filtering else f'{model_name}_scores.csv')
        scores = pd.read_csv(input_dir / scores_file)

        # Generate CSA table for each metric
        for met in scores.met.unique():
            df = scores[scores.met == met]
            
            # Compute mean across splits
            mean = df.groupby(["src", "trg"])["value"].mean()
            mean = mean.unstack()
            mean = apply_decimal_to_dataframe(mean)
            
            # Compute standard deviation across splits
            std = df.groupby(["src", "trg"])["value"].std()
            std = std.unstack()
            
            # Save tables
            file_suffix = f"_{filtering}" if filtering else ""
            mean.to_csv(outdir / f"{model_name}_{met}_mean_csa_table{file_suffix}.csv")
            std.to_csv(outdir / f"{model_name}_{met}_std_csa_table{file_suffix}.csv")


def apply_decimal_to_dataframe(
    df: pd.DataFrame,
    decimal_places: int = 4) -> pd.DataFrame:
    """Apply specified decimal places to numeric columns in a DataFrame.
    
    Args:
        df: Input DataFrame
        decimal_places: Number of decimal places to round to
    
    Returns:
        DataFrame with rounded numeric values
    """
    for col in df.select_dtypes(include='number').columns:
        try:
            df[col] = df[col].round(decimal_places)
        except Exception as e:
            logging.warning(f"Error formatting column '{col}': {e}")
    return df
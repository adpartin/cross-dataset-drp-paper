""" Compute ensemble of model predictions. """

import argparse
import logging
import os
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import pandas as pd

# import seaborn as sns
# import matplotlib.pyplot as plt

from improvelib.metrics import compute_metrics

filepath = Path(__file__).parent
# filepath = Path(os.path.abspath(''))  # ipynb
print(filepath)

# Arg parser
parser = argparse.ArgumentParser(description='Ensemble.')
# parser.add_argument('--datadir',
#                     default='./agg_results',
#                     type=str,
#                     help='Dir containing the aggregated results from all models.')
# parser.add_argument('--filename',
#                     default='./all_model_all_scores.csv',
#                     type=str,
#                     help='File name containing aggregated results from all models.')
# parser.add_argument('--models_dir',
#                     default='../alex/models',
#                     type=str,
#                     help='Dir with raw data results.')
# ----------------------------------------------------------------
# parser.add_argument('--preds_dirname',
#                     default='agg_results/preds',
#                     type=str,
#                     help='Dir containing raw prediction files.')
# parser.add_argument('--scores_filename',
#                     default='agg_results/all_models_all_scores.csv',
#                     type=str,
#                     help='File name containing the aggregated performance scores.')
parser.add_argument('--outdir',
                    default='.',
                    type=str,
                    help='Output dir.')
args = parser.parse_args()
# filename = Path(args.models_dir)
outdir = Path(args.outdir)
# preds_dirname = Path(args.preds_dirname)
# scores_filename = Path(args.scores_filename)
os.makedirs(outdir, exist_ok=True)

weights_dir = outdir / 'weights'
# weighted_preds_dir = outdir / 'weighted_preds'
averaged_preds_dir = outdir / 'preds_averaged'
averaged_splits_dir = outdir / 'splits_averaged'

# Cell and drug col names
canc_col_name = 'improve_sample_id'
drug_col_name = 'improve_chem_id'
y_col_name = 'auc'

# Set up logging
logging.basicConfig(
    filename='meta_process.log', # Log file name
    level=logging.INFO, # Log level
    format='%(asctime)s - %(levelname)s - %(message)s' # Log format
)

# Console logging handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logging.getLogger().addHandler(console_handler)


# -----------------------------------------------
# Compute model weights for ensemble predictions -- weight based on val performance
# -----------------------------------------------

def agg_val_scores(outdir: Union[Path, str]='.',
                   models: Optional[List]=None,
                   sources: Optional[List]=None):
    """ Compute and aggregate val scores for multiple models.

    Load raw predictions computed on val data from all [model, source, split]
    combinations and compute performance scores. Combine results from multiple
    models. Save the resulting DataFrame, which will be used to compute weights
    for weighted prediction averaging (weighted ensemble predictions).

    Args:
    """
    breakpoint()

    # Glob pred files
    preds_dirname = 'val_preds'
    preds_dir = filepath / preds_dirname
    pred_files = sorted(preds_dir.glob('*'))

    # Extract unique sources and model names
    # File pattenr: <SOURCE>_split_<#>_<MODEL>.csv
    if models is None:
        models = sorted(set([f.stem.split('_')[-1] for f in pred_files]))
    if sources is None:
        sources = sorted(set([f.stem.split('_')[0] for f in pred_files]))

    rr = []  # list of dicts containing [model, source, split]
    for model in models:  # model

        for source in sources:  # source
            files = sorted(preds_dir.glob(f'{source}_split_*_{model}*'))

            for fname in files:  # filename for all [source, model] combos
                split = fname.stem.split('split_')[1].split('_')[0]
                print(f'{model}; {source}; {split}')
                pdf = pd.read_csv(fname)  # load raw preds
                scores = compute_metrics(
                    y_true=pdf['auc_true'], y_pred=pdf['auc_pred'],
                    metric_type='regression')
                metrics = list(scores.keys())
                scores['model'] = model
                scores['source'] = source
                scores['split'] = split
                rr.append(scores)  # append dict

    # Aggregate data into a DataFrame
    df = pd.DataFrame(rr)
    df['set'] = 'val'  # indicate that scores computed for val data
    extra_cols = [c for c in df.columns if c not in metrics]
    df = df[extra_cols + metrics]
    df = df.sort_values(['model', 'source', 'split'], ascending=True)

    df.to_csv(outdir / 'val_scores_agg.csv', index=False)
    return df


# Compute and aggregated val scores for multiple models (load if file already exists)
# breakpoint()
val_scores_fpath = outdir / 'val_scores_agg.csv'
if val_scores_fpath.exists():
    val_scores = pd.read_csv(val_scores_fpath)
else:
    val_scores = agg_val_scores(outdir)


def build_weights_dfs(val_scores: pd.DataFrame,
                      outdir: Union[Path, str]='.'):
    """ Build weights DataFrames. """
    breakpoint()

    os.makedirs(outdir, exist_ok=True)

    # Extract unique model names and metrics
    metrics = val_scores.columns[4:].tolist()  # TODO. Hardcoded 1st col id of metrics
    models = sorted(val_scores['model'].unique())

    for model in models:  # model
        df = val_scores[val_scores['model'] == model]

        for source in sorted(val_scores['source'].unique()):  # source
            jj = df[df['source'] == source].copy()

            for met in metrics:  # metric
                jj[f'{met}_weight'] = jj[met] / jj[met].sum()
                # print(jj.sum(axis=0))  # weights should sum up to 1.0
                jj.to_csv(outdir / f'val_scores_and_weights_{model}_{source}.csv',
                          index=False)

    return None


# breakpoint()
if not weights_dir.exists():
    build_weights_dfs(val_scores, outdir=weights_dir)


# ---------------------------
# Raw test model predictions
# ---------------------------


# def assign_weights_to_raw_preds_and_save_df(
#     outdir: Union[Path, str]='test_preds_weighted',
#     models: Optional[List]=None,
#     sources: Optional[List]=None,
#     targets: Optional[List]=None):
#     """ Load raw predictions dfs on test data, and weights dataframes and
#     assign weights to each test set prediction. Then save th prediction files.
#     """

#     outdir = Path(outdir)
#     os.makedirs(outdir, exist_ok=True)

#     # Glob pred files
#     preds_dirname = 'test_preds'
#     preds_dir = filepath / preds_dirname
#     pred_files = sorted(preds_dir.glob('*'))

#     # Extract unique sources and model names
#     # File pattenr: <SOURCE>_split_<#>_<MODEL>.csv
#     if models is None:
#         models = sorted(set([f.stem.split('_')[-1] for f in pred_files]))
#     if sources is None:
#         sources = sorted(set([f.stem.split('_')[0] for f in pred_files]))
#     if targets is None:
#         targets = sorted(set([f.stem.split('_')[1] for f in pred_files]))

#     # Cols to retain from raw pred files
#     cols = ['model', 'src', 'trg', 'split',
#             canc_col_name, drug_col_name, 'auc_true', 'auc_pred']

#     # breakpoint()
#     for model in models:  # model

#         for source in sources:  # source
#             # Load weights info
#             wdf = pd.read_csv(outdir / f'val_scores_and_weights_{model}_{source}.csv')

#             for target in targets:  # target
#                 files = sorted(preds_dir.glob(f'{source}*{target}*{model}*'))

#                 for fname in files:  # filename for all [source, model] combos
#                     split = int(fname.stem.split('split_')[1].split('_')[0])
#                     pdf = pd.read_csv(fname)  # preds dataframe
#                     if not all(True if c in pdf.columns else False for c in cols):
#                         print(fr"Don't include {fname} (not all columns present in the DataFrame)")
#                         continue
#                     print(f'Include {fname}')
#                     assert pdf['model'].unique()[0] == model, 'model name in a \
#                         file name is not consistent with the data'
#                     pdf = pdf[cols]

#                     # for met in metrics:  # met  # TODO. finish this!
#                     # breakpoint()
#                     met = 'r2'  # weighting metric
#                     weight = wdf[(wdf['split'] == split)][f'{met}_weight'].values
#                     pdf[f'auc_pred_w_{met}'] = weight * pdf['auc_pred']

#                     out_fname = fname.stem + '_weighted.csv'
#                     pdf.to_csv(outdir / out_fname, index=False)
#     return None


# breakpoint()
# if not (filepath / 'test_preds_weighted').exists():
#     assign_weights_to_raw_preds_and_save_df()


# ---------------------------------------------
# # breakpoint()
# # Example data: predictions from 10 models for the entire trg dataset
# predictions_trg = {
#     "model_0": [0.3, 0.5, 0.7],  # Predictions on trg by model trained on split 0
#     "model_1": [0.4, 0.6, 0.8],  # Predictions on trg by model trained on split 1
#     "model_2": [0.2, 0.4, 0.6],  # Predictions on trg by model trained on split 2
#     # Add predictions for all 10 models...
# }
# # Validation performance scores for each model (on src's val sets)
# validation_scores = {
#     "model_0": 0.85,
#     "model_1": 0.90,
#     "model_2": 0.88,
#     # Add scores for all 10 models...
# }

# # Step 1: Normalize the weights
# total_score = sum(validation_scores.values())
# weights = {model: score / total_score for model, score in validation_scores.items()}
# print("Normalized Weights:", weights)

# # Step 2: Compute weighted predictions for each sample in trg
# # Assuming all models predict on the same samples in trg
# num_samples = len(next(iter(predictions_trg.values())))  # Number of samples in trg
# ensemble_predictions = []

# for sample_idx in range(num_samples):
#     weighted_sum = 0
#     for model, preds in predictions_trg.items():
#         weighted_sum += weights[model] * preds[sample_idx]
#     ensemble_predictions.append(weighted_sum)

# # Step 3: Output ensemble predictions
# print("Ensemble Predictions for trg:", ensemble_predictions)

# # breakpoint()
# agg_preds = {}
# for model, weight in weights.items():
#     pp = weight * np.array(predictions_trg[model])
#     agg_preds[model] = pp
# df = pd.DataFrame(agg_preds)
# ens_preds = df.sum(axis=1)
# print("Ensemble Predictions for trg:", ens_preds.values)
# ---------------------------------------------


def prediction_averaging(model: str, source: str, target: str,
                         outdir=Union[Path,str], met='r2'):
    """ The prediction-averaging approach, averages the predictions across
    splits before computing the performance metrics.

    Load weights dataframes and dataframes with raw predictions computed on
    test datasets. Assign weights to each prediction sample (cell-drug pair).
    Perform both weighted prediction averaging (weighted by performance on val
    data) and mean prediction averaging. Then save the prediction files.

    Note! This is valid when src != trg

    Args:
        met (str): val set model performance metric is used to compute weight 
    """
    # breakpoint()

    os.makedirs(outdir, exist_ok=True)

    # Raw preds path
    preds_dirname = 'test_preds'
    preds_dir = filepath / preds_dirname

    # Load weights
    wdf = pd.read_csv(weights_dir / f'val_scores_and_weights_{model}_{source}.csv')
    weights = {f'split_{s}': w for s, w in zip(wdf['split'], wdf[f'{met}_weight'])}

    raw_preds = {}
    weighted_preds = {}

    for split, weight in weights.items():
        fname = f'{source}_{target}_{split}_{model}.csv'  # preds fname
        pdf = pd.read_csv(preds_dir / fname)  # preds df
        raw_preds[split] = pdf['auc_pred'].values  # agg raw preds
        weighted_preds[split] = weight * pdf['auc_pred'].values  # agg weighted preds

    # DataFrame containing raw preds
    raw_df = pd.DataFrame(raw_preds)
    raw_df['ens_pred'] = raw_df.mean(axis=1)  # mean

    # DataFrame containing weighted preds
    weighted_df = pd.DataFrame(weighted_preds)
    weighted_df['ens_pred'] = weighted_df.sum(axis=1)  # weighted sum

    # Combine with other meta cols and save to file
    meta_cols = ['model', 'src', 'trg', canc_col_name, drug_col_name, 'auc_true']
    meta_df = pdf[meta_cols]
    # 
    raw_df = pd.concat([meta_df, raw_df], axis=1)
    outpath = outdir / f'{source}_{target}_{model}_mean_preds.csv'
    raw_df.to_csv(outpath, index=False)
    # 
    weighted_df = pd.concat([meta_df, weighted_df], axis=1)
    outpath = outdir / f'{source}_{target}_{model}_weighted_preds.csv'
    weighted_df.to_csv(outpath, index=False)

    return raw_df, weighted_df


# # breakpoint()
# model = 'graphdrp'
# source = 'GDSCv1'
# target = 'CCLE'
# wdf, rdf = prediction_averaging(model=model, source=source, target=target,
#                                 outdir=averaged_preds_dir)
# rs = compute_metrics(rdf['auc_true'], rdf['ens_pred'], metric_type='regression')
# ws = compute_metrics(wdf['auc_true'], wdf['ens_pred'], metric_type='regression')


def compute_scores_from_averaged_predictions(
    outdir: Union[Path, str],
    models: Optional[List]=None,
    sources: Optional[List]=None,
    targets: Optional[List]=None):
    """
    The prediction-averaging approach, averages the predictions across
    splits before computing the performance metrics.

    Load raw predictions dfs on test data, and weights dataframes and
    assign weights to each test set prediction. Then save th prediction files.
    """
    # breakpoint()

    os.makedirs(outdir, exist_ok=True)

    # Glob pred files
    preds_dirname = 'test_preds'
    preds_dir = filepath / preds_dirname
    pred_files = sorted(preds_dir.glob('*'))

    # Extract unique sources and model names
    # File pattenr: <SOURCE>_split_<#>_<MODEL>.csv
    if models is None:
        models = sorted(set([f.stem.split('_')[4] for f in pred_files]))
    if sources is None:
        sources = sorted(set([f.stem.split('_')[0] for f in pred_files]))
    if targets is None:
        targets = sorted(set([f.stem.split('_')[1] for f in pred_files]))

    print('Compute scores from averaged predictions.')
    print('models:', models)
    print('sources:', sources)
    print('targets:', targets)

    # breakpoint()
    # rr = []  # list of dicts containing [model, source, target, split]

    for model in models:  # model
        for src in sources:  # source
            for i, trg in enumerate(targets):  # target
                if src== trg:
                    continue
                wdf, rdf = prediction_averaging(
                    model=model, source=src, target=trg, outdir=outdir)

                # breakpoint()
                rs = compute_metrics(
                    rdf['auc_true'], rdf['ens_pred'], metric_type='regression')
                jj[i] = rs

                ws = compute_metrics(
                    wdf['auc_true'], wdf['ens_pred'], metric_type='regression')

                # Convert dict to df, and aggregate dfs
                breakpoint()
                df = pd.DataFrame(jj)
                df = df.stack().reset_index()
                df.columns = ['met', 'split', 'value']
                df['src'] = src
                df['trg'] = trg
                df['model'] = model
                if df.empty is False:
                    dfs.append(df)

    return None

# models = ['graphdrp']
# sources = None
# # sources = ['GDSCv1']
# targets = None
# compute_scores_from_averaged_predictions(
#     models=models, sources=sources, targets=targets,
#     outdir=averaged_preds_dir)


def compute_scores_from_averaged_splits(
    outdir: Union[Path, str],
    models: Optional[List]=None,
    sources: Optional[List]=None,
    targets: Optional[List]=None):
    """
    The split-averaging approach, computes scores for each split and then
    averages the scores across splits.
    """
    # breakpoint()

    os.makedirs(outdir, exist_ok=True)

    # Glob pred files
    preds_dirname = 'test_preds'
    preds_dir = filepath / preds_dirname
    pred_files = sorted(preds_dir.glob('*'))

    # Extract unique sources and model names
    # File pattenr: <SOURCE>_split_<#>_<MODEL>.csv
    if models is None:
        models = sorted(set([f.stem.split('_')[4] for f in pred_files]))
    if sources is None:
        sources = sorted(set([f.stem.split('_')[0] for f in pred_files]))
    if targets is None:
        targets = sorted(set([f.stem.split('_')[1] for f in pred_files]))

    logging.info('\nCompute scores from averaged splits.')
    logging.info(f'models: {models}')
    logging.info(f'sources: {sources}')
    logging.info(f'targets: {targets}')

    dfs = []
    jj = {}

    for model in models:  # model
        model_dfs = []
        for src in sources:  # source
            for trg in targets:  # target

                files = sorted(preds_dir.glob(f'{src}*{trg}*{model}*'))

                for fname in files:  # filename for all [source, target, model] splits
                    logging.info(f'Loading {fname}')
                    split = int(fname.stem.split('split_')[1].split('_')[0])
                    pdf = pd.read_csv(fname)  # preds dataframe

                    y_true = pdf[f'{y_col_name}_true'].values
                    y_pred = pdf[f'{y_col_name}_pred'].values
                    sc = compute_metrics(y_true, y_pred, metric_type='regression')
                    jj[split] = sc

                # Convert dict to df, and aggregate dfs
                df = pd.DataFrame(jj)
                df = df.stack().reset_index()
                df.columns = ['met', 'split', 'value']
                df['src'] = src
                df['trg'] = trg
                df['model'] = model
                if df.empty is False:
                    model_dfs.append(df)

        model_scores = pd.concat(model_dfs, axis=0).reset_index(drop=True)
        model_scores.to_csv(outdir / f"{model}_scores.csv", index=False)

        dfs.append(model_scores)
        del model_dfs, pdf, jj, df

    # Concat dfs and save
    breakpoint()
    scores = pd.concat(dfs, axis=0)
    scores.to_csv(outdir / "all_model_scores.csv", index=False)
    del dfs

    return scores


breakpoint()
models = ['graphdrp']
sources = None
# sources = ['GDSCv1']
targets = None
compute_scores_from_averaged_splits(
    models=models, sources=sources, targets=targets,
    outdir=averaged_splits_dir)



print("Finished")

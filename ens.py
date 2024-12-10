""" Compute ensemble of model predictions. """

import argparse
import logging
import os
from pathlib import Path
from pprint import pprint
from typing import List, Optional, Union

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

# import seaborn as sns
# import matplotlib.pyplot as plt

from improvelib.metrics import compute_metrics
# from improvelib.workflows.utils.csa.csa_utils import apply_decimal_to_dataframe


filepath = Path(__file__).parent
# filepath = Path(os.path.abspath(''))  # ipynb
print(filepath)

# Arg parser
parser = argparse.ArgumentParser(description='Ensemble.')
parser.add_argument('--outdir',
                    default='.',
                    type=str,
                    help='Output dir.')
args = parser.parse_args()

outdir = Path(args.outdir)
os.makedirs(outdir, exist_ok=True)

weights_dir = outdir / 'weights'
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


def apply_decimal_to_dataframe(df: pd.DataFrame, decimal_places: int=4):
    """
    Applies a specified number of decimal places to all numeric columns in a
    DataFrame, handling potential errors.

    Args:
        df (pd.DataFrame): DataFrame to modify
        decimal_places (int): The desired number of decimal places

    Returns:
        modified DataFrame with the specified decimal format applied where possible
    """

    for col in df.select_dtypes(include='number').columns:
        try:
            # Round values to the specified number of decimal places
            df[col] = df[col].round(decimal_places)
        except Exception as e:
            print(f"Error formatting column '{col}': {e}")

    return df








def compute_scores_from_averaged_splits(
    outdir: Union[Path, str],
    models: Optional[List]=None,
    sources: Optional[List]=None,
    targets: Optional[List]=None,
    filtering: Optional[str]=None
):
    """
    The split-averaging approach, computes scores for each split and then
    averages the scores across splits.
    """
    # breakpoint()

    assert filtering in ['drug_blind', 'cell_blind', 'disjoint', None], f"Invalid 'filtering' ({filtering})"

    os.makedirs(outdir, exist_ok=True)

    # Glob pred files
    preds_dirname = 'test_preds'
    preds_dir = filepath / preds_dirname
    pred_files = sorted(preds_dir.glob('*'))

    def valid_fname(fname):
        if len(fname.stem.split('_')) < 3:
            return False
        return True if fname.stem.split('_')[2] == 'split' else False

    # Extract unique sources and model names
    # File pattenr: <SOURCE>_split_<#>_<MODEL>.csv
    if models is None:
        models = sorted(set([f.stem.split('_')[4] for f in pred_files if valid_fname(f)]))
    if sources is None:
        sources = sorted(set([f.stem.split('_')[0] for f in pred_files if valid_fname(f)]))
    if targets is None:
        targets = sorted(set([f.stem.split('_')[1] for f in pred_files if valid_fname(f)]))

    logging.info('\nCompute scores from averaged splits.')
    logging.info(f'models: {models}')
    logging.info(f'sources: {sources}')
    logging.info(f'targets: {targets}')

    # Y data
    y_data_dir = filepath / 'y_data' 

    dfs = []

    for model in models:  # model
        model_dfs = []
        jj = {}  # aux dict to aggeragte scores for each split

        for src in sources:  # source
            for trg in targets:  # target

                files = sorted(preds_dir.glob(f'{src}*{trg}*{model}*'))

                for fname in files:  # filename for all [source, target, model] splits
                    logging.info(f'Loading {fname}')
                    split = int(fname.stem.split('split_')[1].split('_')[0])
                    pdf = pd.read_csv(fname)  # preds dataframe

                    if filtering is not None:

                        columns_to_load = ['source', canc_col_name, drug_col_name, y_col_name]
                        src_data = pd.read_csv(y_data_dir / f'{model}_{src}_split_{split}_train.csv', usecols=columns_to_load)
                        if src == trg:
                            trg_data = pd.read_csv(y_data_dir / f'{model}_{src}_split_{split}_test.csv', usecols=columns_to_load)
                        else:
                            trg_data = pd.read_csv(y_data_dir / f'{model}_{trg}_all.csv', usecols=columns_to_load)

                        if filtering == 'drug_blind' or filtering == 'disjoint':
                            src_drugs = set(src_data[drug_col_name])
                            trg_drugs = set(trg_data[drug_col_name])
                            unq_trg_drugs = trg_drugs - src_drugs
                            if not pdf.empty:
                                pdf = pdf[pdf[drug_col_name].isin(unq_trg_drugs)].reset_index(drop=True)

                        if filtering == 'cell_blind' or filtering == 'disjoint':
                            src_cells = set(src_data[canc_col_name])
                            trg_cells = set(trg_data[canc_col_name])
                            unq_trg_cells = trg_cells - src_cells
                            if not pdf.empty:
                                pdf = pdf[pdf[canc_col_name].isin(unq_trg_cells)].reset_index(drop=True)

                    # if not pdf.empty:
                    y_true = pdf[f'{y_col_name}_true'].values
                    y_pred = pdf[f'{y_col_name}_pred'].values
                    sc = compute_metrics(y_true, y_pred, metric_type='regression')
                    jj[split] = sc

                # Convert dict to df, and aggregate dfs
                df = pd.DataFrame(jj)
                df = df.stack(dropna=False).reset_index()
                df.columns = ['met', 'split', 'value']
                df['src'] = src
                df['trg'] = trg
                df['model'] = model
                if df.empty is False:
                    model_dfs.append(df)

        # breakpoint()
        model_scores = pd.concat(model_dfs, axis=0).reset_index(drop=True)
        if filtering is not None:
            model_scores.to_csv(outdir / f"{model}_scores_{filtering}.csv", index=False)
        else:
            model_scores.to_csv(outdir / f"{model}_scores.csv", index=False)

        dfs.append(model_scores)
        del model_dfs, pdf, jj, df

    # Concat dfs and save
    scores = pd.concat(dfs, axis=0)
    if filtering is not None:
        scores.to_csv(outdir / f"all_models_scores_{filtering}.csv", index=False)
    else:
        scores.to_csv(outdir / "all_models_scores.csv", index=False)

    return scores


breakpoint()
# models = ['graphdrp']
models = None
sources = None
targets = None
# sources = ['CCLE', 'gCSI']
# targets = ['CCLE', 'gCSI']
# sources = ['CTRPv2', 'GDSCv1']
# targets = ['CCLE', 'CTRPv2', 'GDSCv1']
# sources = [ 'GDSCv2']
# targets = [ 'GDSCv1']
filtering=None
# filtering='cell_blind'
# filtering='drug_blind'
# filtering='disjoint'
scores = compute_scores_from_averaged_splits(
    models=models,
    sources=sources,
    targets=targets,
    outdir=averaged_splits_dir,
    filtering=filtering
)


def compute_csa_tables_from_averaged_splits(
    input_dir: Union[Path, str], 
    outdir: Union[Path, str],
    models: Optional[List]=None,
    sources: Optional[List]=None,
    targets: Optional[List]=None,
    filtering: Optional[str]=None
):
    """
    The split-averaging approach, computes scores for each split and then
    averages the scores across splits.
    """
    # breakpoint()

    os.makedirs(outdir, exist_ok=True)

    assert filtering in ['drug_blind', 'cell_blind', 'disjoint', None], f"Invalid 'filtering' ({filtering})"

    if models is None:
        all_models_scores = pd.read_csv(input_dir / f'all_models_scores.csv')
        models = sorted(all_models_scores['model'].unique().tolist())

    logging.info('\nCompute scores from averaged splits.')
    logging.info(f'models: {models}')
    logging.info(f'sources: {sources}')
    logging.info(f'targets: {targets}')

    for model_name in models:  # model

        # scores = pd.read_csv(input_dir / f'{model_name}_scores.csv')
        if filtering is not None:
            scores = pd.read_csv(input_dir / f"{model_name}_scores_{filtering}.csv")
        else:
            scores = pd.read_csv(input_dir / f'{model_name}_scores.csv')

        # Average across splits (Note! These are not further used)
        sc_mean = scores.groupby(["met", "src", "trg"])["value"].mean().reset_index()
        sc_std = scores.groupby(["met", "src", "trg"])["value"].std().reset_index()

        # Generate csa table
        mean_tb = {}
        std_tb = {}
        for met in scores.met.unique():
            df = scores[scores.met == met]
            # df = df.sort_values(["src", "trg", "met", "split"])
            # df['model'] = model_name  # redundant
            # df.to_csv(outdir / f"{met}_scores.csv", index=True)
            # Mean
            mean = df.groupby(["src", "trg"])["value"].mean()
            mean = mean.unstack()
            mean = apply_decimal_to_dataframe(mean, decimal_places=4)
            print(f"{met} mean:\n{mean}")
            mean_tb[met] = mean
            # Std
            std = df.groupby(["src", "trg"])["value"].std()
            std = std.unstack()
            print(f"{met} std:\n{std}")
            std_tb[met] = std

            if filtering is not None:
                mean.to_csv(outdir / f"{model_name}_{met}_mean_csa_table_{filtering}.csv", index=True)
                std.to_csv(outdir / f"{model_name}_{met}_std_csa_table_{filtering}.csv", index=True)
            else:
                mean.to_csv(outdir / f"{model_name}_{met}_mean_csa_table.csv", index=True)
                std.to_csv(outdir / f"{model_name}_{met}_std_csa_table.csv", index=True)

        # Quick test
        # met="mse"; src="CCLE"; trg="GDSCv1" 
        # print(f"src: {src}; trg: {trg}; met: {met}; mean: {scores[(scores.met==met) & (scores.src==src) & (scores.trg==trg)].value.mean()}")
        # print(f"src: {src}; trg: {trg}; met: {met}; std:  {scores[(scores.met==met) & (scores.src==src) & (scores.trg==trg)].value.std()}")
        # met="mse"; src="CCLE"; trg="GDSCv2" 
        # print(f"src: {src}; trg: {trg}; met: {met}; mean: {scores[(scores.met==met) & (scores.src==src) & (scores.trg==trg)].value.mean()}")
        # print(f"src: {src}; trg: {trg}; met: {met}; std:  {scores[(scores.met==met) & (scores.src==src) & (scores.trg==trg)].value.std()}")

        # Generate densed csa table
        df_on = scores[scores.src == scores.trg].reset_index()
        on_mean = df_on.groupby(["met"])["value"].mean().reset_index().rename(
            columns={"value": "mean"})
        on_std = df_on.groupby(["met"])["value"].std().reset_index().rename(
            columns={"value": "std"})
        on = on_mean.merge(on_std, on="met", how="inner")
        on["summary"] = "within"

        df_off = scores[scores.src != scores.trg]
        off_mean = df_off.groupby(["met"])["value"].mean().reset_index().rename(
            columns={"value": "mean"})
        off_std = df_off.groupby(["met"])["value"].std().reset_index().rename(
            columns={"value": "std"})
        off = off_mean.merge(off_std, on="met", how="inner")
        off["summary"] = "cross"

        print(f"On-diag mean:\n{on_mean}")
        print(f"On-diag std: \n{on_std}")
        print(f"Off-diag mean:\n{off_mean}")
        print(f"Off-diag std: \n{off_std}")

        # Combine dfs
        df = pd.concat([on, off], axis=0).sort_values("met")
        df['model'] = model_name
        # df.to_csv(outdir / f"{model_name}_densed_csa_table.csv", index=False)
        print(f"Densed CSA table:\n{df}")

        if filtering is not None:
            df.to_csv(outdir / f"{model_name}_densed_csa_table_{filtering}.csv", index=False)
        else:
            df.to_csv(outdir / f"{model_name}_densed_csa_table.csv", index=False)

    return None


# if not averaged_splits_dir.exists():
#     models = ['graphdrp']
#     sources = None
#     targets = None
#     compute_csa_tables_from_averaged_splits(
#         models=models, sources=sources, targets=targets,
#         input_dir=averaged_splits_dir, outdir=averaged_splits_dir)

breakpoint()
# models = ['graphdrp']
models = None
sources = None
targets = None
# filtering=None
compute_csa_tables_from_averaged_splits(
    models=models, sources=sources, targets=targets,
    input_dir=averaged_splits_dir, outdir=averaged_splits_dir,
    filtering=filtering
)

breakpoint()









def build_weights_dfs(val_scores: pd.DataFrame,
                      outdir: Union[Path, str]='.'):
    """ Build weights DataFrames. """
    # breakpoint()

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


breakpoint()
if not weights_dir.exists():
    build_weights_dfs(val_scores, outdir=weights_dir)


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
    outpath = outdir / f'{model}_{source}_{target}_mean_preds.csv'
    raw_df.to_csv(outpath, index=False)
    # 
    weighted_df = pd.concat([meta_df, weighted_df], axis=1)
    outpath = outdir / f'{model}_{source}_{target}_weighted_preds.csv'
    weighted_df.to_csv(outpath, index=False)

    return raw_df, weighted_df


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

    logging.info('\nCompute scores from averaged predictions.')
    logging.info(f'models: {models}')
    logging.info(f'sources: {sources}')
    logging.info(f'targets: {targets}')

    rd = []  # aux dict to aggeragte scores for each loop iteration
    wd = []

    for model in models:  # model
        for src in sources:  # source
            for i, trg in enumerate(targets):  # target
                if src== trg:
                    continue
                logging.info(f'{model}; {src}; {trg}')
                wdf, rdf = prediction_averaging(
                    model=model, source=src, target=trg, outdir=outdir)

                # breakpoint()
                r_scores = compute_metrics(
                    rdf['auc_true'], rdf['ens_pred'], metric_type='regression')
                metrics = list(r_scores.keys())
                r_scores['src'] = src
                r_scores['trg'] = trg
                r_scores['model'] = model
                rd.append(r_scores)

                w_scores = compute_metrics(
                    wdf['auc_true'], wdf['ens_pred'], metric_type='regression')
                w_scores['src'] = src
                w_scores['trg'] = trg
                w_scores['model'] = model
                wd.append(w_scores)

    del wdf, rdf, r_scores, w_scores
    extra_cols = ['model', 'src', 'trg']

    rdf = pd.DataFrame(rd)
    rdf = rdf[extra_cols + metrics]
    rdf.to_csv(outdir / f"mean_averaged_pred_scores.csv", index=False)
    rdf_long = rdf.melt(id_vars=['model', 'src', 'trg'], var_name='met')
    rdf_long.to_csv(outdir / f"mean_averaged_pred_scores_long_format.csv", index=False)

    wdf = pd.DataFrame(wd)
    wdf = wdf[extra_cols + metrics]
    wdf.to_csv(outdir / f"weighted_averaged_pred_scores.csv", index=False)
    wdf_long = wdf.melt(id_vars=['model', 'src', 'trg'], var_name='met')
    wdf_long.to_csv(outdir / f"weighted_averaged_pred_scores_long_format.csv", index=False)

    return None


if not averaged_preds_dir.exists():
    models = ['graphdrp']
    sources = None
    # sources = ['CCLE', 'gCSI', 'GDSCv1', 'GDSCv2']
    targets = None
    # targets = ['CCLE', 'gCSI', 'GDSCv2']
    compute_scores_from_averaged_predictions(
        models=models, sources=sources, targets=targets,
        outdir=averaged_preds_dir)


def compute_overlaps(src_data, trg_data, canc_col_name, drug_col_name):
    """ Compute overlaps """
    # Precompute response counts for each unique cell in both src and trg
    src_cell_counts = src_data.groupby(canc_col_name)[y_col_name].count()
    trg_cell_counts = trg_data.groupby(canc_col_name)[y_col_name].count()
    src_drug_counts = src_data.groupby(drug_col_name)[y_col_name].count()
    trg_drug_counts = trg_data.groupby(drug_col_name)[y_col_name].count()

    # Find overlapping cells
    unique_cell_overlap = set(src_cell_counts.index).intersection(trg_cell_counts.index)
    unique_drug_overlap = set(src_drug_counts.index).intersection(trg_drug_counts.index)

    # Compute weighted overlap using the precomputed counts
    sample_cell_overlap = sum(
        src_cell_counts[cell] + trg_cell_counts[cell] for cell in unique_cell_overlap
    )
    sample_drug_overlap = sum(
        src_drug_counts[drug] + trg_drug_counts[drug] for drug in unique_drug_overlap
    )

    overlaps = {
        "unq_drug_un": len(unique_drug_overlap),
        "samp_drug_un": sample_drug_overlap,
        "unq_cell_un": len(unique_cell_overlap),
        "samp_cell_un": sample_cell_overlap
    }
    return overlaps


def compute_overlap_proportion(src_data, trg_data, drug_col_name,
                               canc_col_name, y_col_name):
    """ Compute the proportion of trg samples involving overlapping drugs and
    cells.
    """
    # Find unique drugs and cells in src and trg
    src_drugs = set(src_data[drug_col_name])
    trg_drugs = set(trg_data[drug_col_name])
    src_cells = set(src_data[canc_col_name])
    trg_cells = set(trg_data[canc_col_name])

    # Find overlapping drugs and cells
    overlapping_drugs = trg_drugs.intersection(src_drugs)
    overlapping_cells = trg_cells.intersection(src_cells)

    # Compute the number of trg samples involving overlapping drugs and cells
    trg_drug_overlap_cnt = trg_data[trg_data[drug_col_name].isin(overlapping_drugs)].shape[0]
    trg_cell_overlap_cnt = trg_data[trg_data[canc_col_name].isin(overlapping_cells)].shape[0]

    # Total number of trg samples
    tot_trg_samples = trg_data.shape[0]

    # Compute proportions
    drug_ratio = trg_drug_overlap_cnt / tot_trg_samples
    cell_ratio = trg_cell_overlap_cnt / tot_trg_samples

    return {
        "trg_drug_overlap_ratio": drug_ratio,
        "trg_cell_overlap_ratio": cell_ratio,
        "trg_drug_overlap_cnt": trg_drug_overlap_cnt,
        "trg_cell_overlap_cnt": trg_cell_overlap_cnt,
    }


model_name = 'graphdrp'
perf_data = pd.read_csv(averaged_splits_dir / f'{model_name}_scores.csv')
print(f'Y data: {perf_data.shape}')
pprint(perf_data.nunique())

import time
start = time.time()
columns_to_load = [canc_col_name, drug_col_name, y_col_name]
dfs = []

breakpoint()
for (src, trg, split), group_df in perf_data.groupby(["src", "trg", "split"]):
    print(f'{src}-{trg}; split {split}')
    group_df = group_df.reset_index(drop=True)
    src_data = pd.read_csv(Path('y_data') / f'{model_name}_{src}_split_{split}_train.csv', usecols=columns_to_load)

    if src == trg:
        trg_data = pd.read_csv(Path('y_data') / f'{model_name}_{src}_split_{split}_test.csv', usecols=columns_to_load)
    else:
        trg_data = pd.read_csv(Path('y_data') / f'{model_name}_{trg}_all.csv')

    # overlaps = compute_overlaps(src_data, trg_data, canc_col_name, drug_col_name)
    overlaps = compute_overlap_proportion(src_data, trg_data, drug_col_name,
                                          canc_col_name, y_col_name)
    for k, v in overlaps.items():
        group_df[k] = v

    group_df['src_samples'] = len(src_data)
    group_df['trg_samples'] = len(trg_data)
    dfs.append(group_df)

df = pd.concat(dfs, axis=0)

df.to_csv(averaged_splits_dir / f'{model_name}_scores_with_cell_drug_overlaps.csv', index=False)
end = time.time()
print(f'Runtime: {(end - start) / 60} mins')

res_df = pd.read_csv(averaged_splits_dir / f'{model_name}_scores_with_cell_drug_overlaps.csv')

met = 'r2'
df = res_df[(res_df['met'] == met) &
            (res_df['model'] == model_name)].reset_index(drop=True)

for col in ["trg_drug_overlap_ratio", "trg_cell_overlap_ratio"]:
    if col in df.columns:
        corr, _ = pearsonr(df[col], df['value'])
        print(f"Correlation between {col} and metric_value: {corr}")

breakpoint()

print("Finished")

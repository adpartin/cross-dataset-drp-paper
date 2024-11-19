""" Compute ensemble of model predictions. """

import argparse
import logging
import os
from pathlib import Path

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
                    default='./ens',
                    type=str,
                    help='Output dir.')
args = parser.parse_args()
# filename = Path(args.models_dir)
outdir = Path(args.outdir)
# preds_dirname = Path(args.preds_dirname)
# scores_filename = Path(args.scores_filename)
os.makedirs(outdir, exist_ok=True)

# Cell and drug col names
canc_col_name = "improve_sample_id"
drug_col_name = "improve_chem_id"



# --------------------------------------------
# Obtain model weights for ensemble predictions -- weight based on val performance
# --------------------------------------------

def build_val_scores_df(out_filename: str='val_scores_and_weights.csv'):
    """ Calc performance scores on val data and combine results for multiple
    models. This would be used as weights for ensemble learning.
    """
    breakpoint()

    # Glob pred files
    preds_dirname = 'val_preds'
    preds_dir = filepath / preds_dirname
    pred_files = sorted(preds_dir.glob('*'))

    # Extract unique sources and model names
    # File pattenr: <SOURCE>_split_<#>_<MODEL>.csv
    sources = sorted(set([f.stem.split('_')[0] for f in pred_files]))
    models = sorted(set([f.stem.split('_')[-1] for f in pred_files]))

    rr = []  # list of dicts containing [model, source, split]
    for model in models:  # model

        for source in sources:  # source
            files = sorted(preds_dir.glob(f'{source}_split_*_{model}*'))

            for fname in files:  # filename for all [source, model] combos
                split = fname.stem.split('split_')[1].split('_')[0]
                print(f'{model}; {source}; {split}')
                df = pd.read_csv(fname)  # load raw preds
                sc = compute_metrics(
                    y_true=df['auc_true'], y_pred=df['auc_pred'],
                    metric_type='regression')
                metrics = list(sc.keys())
                sc['model'] = model
                sc['source'] = source
                sc['split'] = split
                rr.append(sc)  # append dict

    df = pd.DataFrame(rr)
    df['set'] = 'val'
    other_cols = [c for c in df.columns if c not in metrics]
    df = df[other_cols + metrics]
    df = df.sort_values(['model', 'source', 'split'], ascending=True)
    df.to_csv(outdir / out_filename, index=False)
    return df

# breakpoint()
out_filename = 'val_scores_agg.csv'
val_scores_fpath = outdir / out_filename
if val_scores_fpath.exists():
    val_scores = pd.read_csv(val_scores_fpath)
else:
    val_scores = build_val_scores_df(out_filename)

# breakpoint()
metrics = val_scores.columns[4:].tolist()
models = sorted(val_scores['model'].unique())
for model in models:
    df = val_scores[val_scores['model'] == model]
    for source in sorted(val_scores['source'].unique()):
        jj = df[df['source'] == source].copy()
        for met in metrics:
            jj[f'{met}_weight'] = jj[met] / jj[met].sum()
            # print(jj.sum(axis=0))
            jj.to_csv(outdir / f'val_scores_and_weights_{model}_{source}.csv', index=False)
del df, jj

# breakpoint()
# df = val_scores.groupby(['model', 'source', 'set']).agg(
#     r2_mean=('r2', 'mean'),
#     scc_mean=('scc', 'mean'),
#     pcc_mean=('pcc', 'mean'),
#     mse_mean=('mse', 'mean'),
#     rmse_mean=('rmse', 'mean')
# ).reset_index()


# ---------------------------
# Raw test model predictions
# ---------------------------
# Glob pred files
preds_dirname = 'test_preds'
preds_dir = filepath / preds_dirname
pred_files = sorted(preds_dir.glob('*'))

# Extract unique sources and model names
# File pattenr: <SOURCE>_split_<#>_<MODEL>.csv
breakpoint()
sources = sorted(set([f.stem.split('_')[0] for f in pred_files]))
targets = sorted(set([f.stem.split('_')[1] for f in pred_files]))
models = sorted(set([f.stem.split('_')[-1] for f in pred_files]))

# Cols to retain from raw pred files
cols = [canc_col_name, drug_col_name, 'auc_true', 'auc_pred', 'model']

rr = []  # list of dicts containing [model, source, split]
for model in models:  # model

    for source in sources:  # source
        # Load weights info
        wdf = pd.read_csv(outdir / f'val_scores_and_weights_{model}_{source}.csv')

        # files = sorted(preds_dir.glob(f'{source}_*_{model}*'))
        for target in targets:  # target
            breakpoint()
            files = sorted(preds_dir.glob(f'{source}*{target}*{model}*'))

            for fname in files:  # filename for all [source, model] combos
                split = fname.stem.split('split_')[1].split('_')[0]
                pdf = pd.read_csv(fname)
                if not all(True if c in df.columns else False for c in cols):
                    print(fr"Don't include {f}")
                    continue
                print(f'Include {fname}')
                assert pdf['model'].unique()[0] == model, 'model name in a file name is not consistent with the data'
                pdf = pdf[cols]

                # for met in metrics:  # met
                met = 'r2'  # weighting metric
                weight = wdf[(wdf.split == 0)][f'{met}_weight'].values
                pdf[f'auc_pred_wgt_{met}'] = weight * pdf['auc_pred']


# Specific CSA experiment [source, target, split]
src = 'CCLE'
trg = 'CCLE'
split = 0

# Glob pred files
test_pred_files = sorted(preds_dir.glob(f'{src}_{trg}_split_{split}_*'))
# model_names = sorted(set([f.stem.split('_')[-1] for f in pred_files]))

# Aggregate raw predictions from all models
# breakpoint()
dfs = []
model_names = []
for i, f in enumerate(test_pred_files):
    df = pd.read_csv(f)
    if not all(True if c in df.columns else False for c in cols):
        print(fr"Don't include {f}")
        continue
    print(f'Include {f}')
    model_names.append(df['model'].unique()[0])
    df = df[cols]
    dfs.append(df)
    # print(df.shape)
    # print(df.iloc[:2,:])
    # breakpoint()

df = pd.concat(dfs, axis=0).reset_index(drop=True)
del i, f, dfs

# Pivot table (create column for each model)
# breakpoint()
sorted_model_names = sorted(model_names)
pred = df.pivot(index=[canc_col_name, drug_col_name, 'auc_true'],
                columns='model', values='auc_pred').reset_index()
cols_sorted = [c for c in pred.columns if c not in sorted_model_names] + sorted_model_names
pred = pred[cols_sorted]


# --------------------------------------------
# Obtain model weights for ensemble predictions
# --------------------------------------------
# Load model performance scores (all models and all scores)
scores_df = pd.read_csv(filepath / scores_filename)
scores_df['model'] = scores_df['model'].map(lambda x: x.lower()) # lowercase model names

# Calc weight for each model for a given [src, trg, met] combo
# breakpoint()
jj = scores_df.groupby(['src', 'trg', 'met', 'model']).agg(
    mean_score=('value', 'mean')).reset_index() # reduce to mean score over splits
jj = jj[jj['model'].isin(sorted_model_names)].reset_index(drop=True)
weight_met = 'r2'  # metric used to determine model weight
ww = jj[(jj.src == src) & (jj.trg == trg) & (jj.met == weight_met)].reset_index(drop=True)

# Normalize scores to be in the range [0, 1]
# TODO. In this normalization, the worst model will have weight of 0!
shifted_scores = ww.mean_score - ww.mean_score.min() # shift scores
ww['weight'] = shifted_scores / shifted_scores.sum() # scale between 0 and 1
ww_dict = {model: weight for model, weight in zip(ww.model, ww.weight)}

# breakpoint()
# Align weights with the predictions DataFrame
ww = ww.sort_values('model', ascending=True) # this is required
weights = ww.set_index('model')['weight']

# --------------------------------------------
# Perform weighted ensemble prediction
# --------------------------------------------
weighted_preds = pred[sorted_model_names] * weights.values  # broadcast weights across columns
pred['ens'] = weighted_preds.sum(axis=1)  # Sum weighted contributions for each row

breakpoint()
ens_scores = compute_metrics(y_true=pred.auc_true, y_pred=pred.ens,
                             metric_type='regression')








breakpoint()
res = None
for model in sorted_model_names:
    if res is None:
        res = pred[model] * ww_dict[model]
    else:
        res += pred[model] * ww_dict[model]
pred['auc_weighted'] = res
print(pred[sorted_model_names + ['auc_weighted']])

df = ww[['model', 'weight']]
df = df.set_index('model').T
df = df.reset_index(drop=True)
# df = df[]

breakpoint()
pr = pred[sorted_model_names]
rr = pp * df[sorted_model_names]
print(pp)
# ww_df = 

for m in model_names:
    pred[m] * ww_dict[m]

# Example predictions DataFrame
pred = pd.DataFrame({
    'deepcdr': [0.8, 0.6, 0.7],
    'graphdrp': [0.7, 0.65, 0.6],
    'igtd': [0.1, 0.2, 0.15],
    'lgbm': [0.9, 0.8, 0.85],
    'tcnns': [0.5, 0.4, 0.45]
})

# Model weights DataFrame from previous step
model_weights = pd.DataFrame({
    'model': ['deepcdr', 'graphdrp', 'igtd', 'lgbm', 'tcnns'],
    'weight': [0.255165, 0.251167, 0.000000, 0.259125, 0.234542]
})

# Align weights with the predictions DataFrame
weights = model_weights.set_index('model')['weight']

# Perform weighted ensemble prediction
weighted_predictions = pred * weights.values  # Broadcast weights across columns
final_predictions = weighted_predictions.sum(axis=1)  # Sum weighted contributions for each row

print("Weighted Predictions:\n", weighted_predictions)
print("\nFinal Ensemble Predictions:\n", final_predictions)

# pp = 

# cols = []
# for col in df.columns:
#     if isinstance(col, tuple):
#         cols.append('_'.join(col).strip('_'))
#     else:
#         cols.append(col)

tt_piv = tt.pivot(index=['drug_name'], columns=['model'], values=['pred_mean']).reset_index()
model_name_cols = [i[1] for i in tt_piv.columns.values[1:]]
tt_piv.columns = ['drug_name'] + model_name_cols
tt_piv['Avg'] = tt_piv[model_name_cols].mean(axis=1)
# tt_piv = tt_piv.sort_values('Avg').reset_index()
tt_piv = tt_piv.sort_values('drug_name').reset_index(drop=True)



# IMPROVE models
source = "CTRPv2"
improve_models_list = ["DeepTTC", "GraphDRP", "HIDRA", "IGTD", "PaccMann_MCA"]
dfs = []
for model in improve_models_list:
    agg_df = pd.read_csv(filepath / "outdir" / f"agg_preds_{model}_{source}_{target}.tsv", sep="\t")
    agg_df["model"] = model
    dfs.append(agg_df)
df_improve = pd.concat(dfs, axis=0)
print(df_improve.shape)

# # UNO
# source = "all"
# model = "UNO"
# df_uno = pd.read_csv(filepath / "outdir" / f"agg_preds_{model}_{source}_{target}.tsv", sep="\t")
# df_uno["model"] = model
# print(df_uno.shape)

# df = pd.concat([df_improve, df_uno], axis=0)
# print(df.shape)


df = pd.read_csv(datadir / filename, sep=',')

pdo_t = "655913~031-T"
pdo_r = "937885~149-R"

tt = df[df[canc_col_name].isin([pdo_t])].reset_index(drop=True)
rr = df[df[canc_col_name].isin([pdo_r])].reset_index(drop=True)

# group_by_cols = [canc_col_name, drug_col_name]
# ff = tt.groupby(group_by_cols).agg(
#     pred_mean=(y_col_name, 'mean'),
#     pred_std=(y_col_name, 'std'))
# ff = ff.reset_index().sort_values([drug_col_name, canc_col_name, 'pred_mean']).reset_index(drop=True)

tt_piv = tt.pivot(index=['drug_name'], columns=['model'], values=['pred_mean']).reset_index()
model_name_cols = [i[1] for i in tt_piv.columns.values[1:]]
tt_piv.columns = ['drug_name'] + model_name_cols
tt_piv['Avg'] = tt_piv[model_name_cols].mean(axis=1)
# tt_piv = tt_piv.sort_values('Avg').reset_index()
tt_piv = tt_piv.sort_values('drug_name').reset_index(drop=True)


df = tt_piv.drop(columns=['Avg'])
df.to_csv(outdir / "table_drug_by_model_auc.tsv", sep="\t", index=False)
ranked_df = df.set_index("drug_name")
ranked_df = ranked_df.rank().astype(int)  # compute ranked from raw auc values
ranked_df.to_csv(outdir / "table_drug_by_model_ranked.tsv", sep="\t", index=True)

# rank_corr = ranked_df.corr(method='spearman', axis=0)

v = df.loc[0, model_name_cols].values
corr, pvalue = spearmanr(v, v)

df.set_index('drug_name', inplace=True)
tran_df = df.transpose()


def calc_ensemble(df: pd.DataFrame):
    df_piv = df.pivot(index=['drug_name'], columns=['model'], values=['pred_mean']).reset_index()
    model_name_cols = [i[1] for i in df_piv.columns.values[1:]]
    df_piv.columns = ['drug_name'] + model_name_cols
    # df_piv = df_piv.set_index('drug_name')
    # df_piv['Avg'] = df_piv.mean(axis=1)
    tt_piv['Avg'] = tt_piv[model_name_cols].mean(axis=1)
    df_piv = df_piv.sort_values('Avg').reset_index(drop=True)
    return df_div


tt_ens = calc_ensemble(tt)
print(tt_ens)

rr_ens = calc_ensemble(rr)
print(rr_ens)


# Aggregate predictions
# For regression or probability predictions
ensemble_pred = np.mean([pred1, pred2, pred3], axis=0)

# For classification with majority voting (assuming predictions are class labels)
ensemble_pred_labels = np.argmax(np.bincount([np.argmax(pred1, axis=1), np.argmax(pred2, axis=1), np.argmax(pred3, axis=1)], axis=0))

print(ensemble_pred)
print(ensemble_pred_labels)



print("Finished")



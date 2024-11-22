""" This script does several things:
- Collects and renames raw predictions from CSA runs of different models and
  saves them into the same direcotory.
- Aggregates runtime info from CSA runs of different models.
- Aggregates scores info from CSA runs of different models.
"""

import argparse
import json
import logging
import os
import warnings
from pathlib import Path
from pprint import pprint
from typing import List

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

filepath = Path(__file__).parent
# filepath = Path(os.path.abspath(''))
print(filepath)

# Arg parser
parser = argparse.ArgumentParser(description='Main post-processing.')
parser.add_argument('--outdir',
                    # default='./agg_results',
                    default='.',
                    type=str,
                    help='Output dir.')
args = parser.parse_args()
outdir = Path(args.outdir)
os.makedirs(outdir, exist_ok=True)
# canc_col_name = "improve_sample_id"
# drug_col_name = "improve_chem_id"

# main_models_path = filepath / 'models'  # dir containing the collection of models
main_models_path = filepath / '../alex/models'  # dir containing the collection of models
models_paths_list  = sorted(main_models_path.glob('*'))  # list of paths to the models

# Cell and drug col names
canc_col_name = 'improve_sample_id'
drug_col_name = 'improve_chem_id'
y_col_name = 'auc'

# Set up logging
logging.basicConfig(
    filename='postprocess.log', # Log file name
    level=logging.INFO, # Log level
    format='%(asctime)s - %(levelname)s - %(message)s' # Log format
)

# Console logging handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logging.getLogger().addHandler(console_handler)


# -----------------------------------
# Aggregate and save raw infer preds
# -----------------------------------
def collect_and_save_raw_preds(models_paths_list: List,
                               outdir: Path,
                               subset_type: str,
                               stage: str) -> None:
    """ Collect and save inference predictions for all models. """
    # breakpoint()

    assert subset_type in ['val', 'test'], f"Invalid 'subset_type' ({subset_type})"
    assert stage in ['train', 'infer'], f"Invalid 'stage' ({subset_type})"

    cols = [canc_col_name, drug_col_name, y_col_name, f'{y_col_name}_true', f'{y_col_name}_pred']
    extra_cols = ['model', 'src', 'trg', 'set', 'split']

    logging.info('\nSave raw model predictions into new files.')
    preds_fname = f'{subset_type}_y_data_predicted.csv'
    out_preds_dir = outdir / f'{subset_type}_preds'
    os.makedirs(out_preds_dir, exist_ok=True)

    for model_dir in models_paths_list:
        model_name = model_dir.name.lower()
        logging.info(model_name)
        stage_path = model_dir / 'improve_output' / f'{stage}'
        exps = sorted(stage_path.glob('*'))

        for i, exp_path in enumerate(exps):
            src = str(exp_path.name).split("-")[0]
            # trg = str(exp_path.name).split("-")[1]
            trg = str(exp_path.name).split("-")[1] if '-' in str(exp_path.name) else None
            splits = sorted(exp_path.glob('*'))

            for i, split_path in enumerate(splits):
                sp = split_path.name.split('split_')[1]
                try:
                    df = pd.read_csv(split_path / preds_fname, sep=',')
                    # org_cols = df.columns.tolist()
                    if not all([True if c in df.columns else False for c in cols]):
                        # print(f'Model {model_name} missing some of the required columns: {cols}')
                        continue
                    df = df[cols]
                    df['model'] = model_name  # extra col
                    df['src'] = src  # extra col
                    df['trg'] = trg  # extra col
                    df['set']= 'val' if trg is None else 'test'
                    df['split'] = sp  # extra col
                    if trg is None:
                        # Val set Predictions
                        fname = f'{src}_split_{sp}_{model_name}.csv'
                    else:
                        # Test set Predictions
                        fname = f'{src}_{trg}_split_{sp}_{model_name}.csv'
                    # df = df[extra_cols + org_cols]
                    df = df[extra_cols + cols]
                    df.to_csv(out_preds_dir / fname, index=False)
                except FileNotFoundError:
                    warnings.warn(f'File not found! {split_path / preds_fname}',
                                  UserWarning)
    return None

# ----------------------------------
# Aggregate runtimes from all models
# ----------------------------------
def agg_runtimes(models_paths_list: List, outdir: Path) -> None:
    """ Runtimes for all models and stages. """
    # breakpoint()

    logging.info('\nAggregate and save runtimes.')
    res_fname = 'runtimes.csv'
    out_fname = 'all_models_' + res_fname

    agg_df_list = []
    missing_files = []

    for model_dir in models_paths_list:
        model_name = model_dir.name
        pp_res_path = model_dir / f'postproc.csa.{model_name}.improve_output'
        logging.info(pp_res_path)
        try:
            rr = pd.read_csv(pp_res_path / res_fname, sep=',')
            agg_df_list.append(rr)
        except FileNotFoundError:
            warnings.warn(f'File not found! {pp_res_path}', UserWarning)
            missing_files.append(pp_res_path)

    with open(outdir / 'missing_runtime_files.csv', 'w') as f:
        for i in missing_files:
            f.write(f'{i}\n')

    df = pd.concat(agg_df_list, axis=0)
    df.to_csv(outdir / out_fname, index=False)
    # pprint(df.shape)
    # pprint(df.nunique())
    # pprint(df[:3])
    return None


# --------------------------------
# Aggregate scores from all models
# --------------------------------
def agg_scores(models_paths_list: List, outdir: Path) -> None:
    """ Note! This is replaced with compute_scores_from_averaged_splits() """
    # breakpoint()

    logging.info('\nAggregate and save performance scores.')
    res_fname = 'all_scores.csv'
    # out_fname = 'all_models_' + res_fname
    out_fname = 'all_models_' + 'test_scores.csv'

    agg_df_list = []
    missing_files = []

    for model_dir in models_paths_list:
        model_name = model_dir.name
        pp_res_path = model_dir / f'postproc.csa.{model_name}.improve_output'
        logging.info(pp_res_path)
        try:
            rr = pd.read_csv(pp_res_path / res_fname, sep=',')
            agg_df_list.append(rr)
        except FileNotFoundError:
            warnings.warn(f'File not found! {pp_res_path}', UserWarning)
            missing_files.append(pp_res_path)

    with open(outdir / 'missing_scores_files.csv', 'w') as f:
        for i in missing_files:
            f.write(f'{i}\n')

    df = pd.concat(agg_df_list, axis=0)
    df.to_csv(outdir / out_fname, index=False)
    # pprint(df.shape)
    # pprint(df.nunique())
    # pprint(df[:3])
    return None



# ------------------------------------------------
# Runtime plots
# ------------------------------------------------
def plot_runtimes(models_paths_list: List, outdir: Path) -> None:
    """ ... """
    # breakpoint()

    plot_runtime_outdir = outdir / 'plot_runtimes'
    os.makedirs(plot_runtime_outdir, exist_ok=True)

    # Creating separate plots for each stage, grouped by src and model.
    # Note that this not useful for infer where trg is the important factor.

    # Load the data
    df = pd.read_csv(outdir / 'all_models_runtimes.csv')

    # Group by src, stage, and model, and calc the mean and standard deviation
    # of tot_mins
    stage_model_src_stats = df.groupby(['src', 'stage', 'model'])['tot_mins'].agg(
        ['mean', 'std', 'count']).reset_index()

    # Calc the standard error of the mean (sem)
    stage_model_src_stats['sem'] = stage_model_src_stats['std'] / \
        stage_model_src_stats['count'] ** 0.5

    # Define a color palette
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    # Create separate plots for each stage
    stages = stage_model_src_stats['stage'].unique()

    for stage in stages:
        plt.figure(figsize=(14, 7))
        stage_data = stage_model_src_stats[stage_model_src_stats['stage'] == stage]
        
        bar_plot = sns.barplot(x='src', y='mean', hue='model',
            data=stage_data, palette=colors, errorbar=None)
        
        # Add error bars for each bar
        for index, bar in enumerate(bar_plot.patches):
            height = bar.get_height()
            sem = stage_data['sem'].iloc[index]
            
            plt.errorbar(x=bar.get_x() + bar.get_width() / 2, 
                         y=height, 
                         yerr=sem, 
                         fmt='none', 
                         c='black', 
                         capsize=5, 
                         elinewidth=1)

        plt.title(f'Distribution of Total Minutes for Stage {stage.upper()} \
                  with Error Bars')
        plt.ylabel('Average Total Minutes')
        plt.xlabel('Source dataset')
        plt.xticks(rotation=45)
        plt.legend(title='Model')
        plt.tight_layout()
        plt.grid(True)
        plt.show()
        fname = f'runtime_{stage}.png'
        plt.savefig(filepath / plot_runtime_outdir / fname, dpi=200,
                    bbox_inchesstr='tight')

    return None


breakpoint()
agg_runtimes(models_paths_list, outdir)
agg_scores(models_paths_list, outdir)
plot_runtimes(models_paths_list, outdir)
collect_and_save_raw_preds(models_paths_list, outdir, subset_type='val', stage='train')
collect_and_save_raw_preds(models_paths_list, outdir, subset_type='test', stage='infer')


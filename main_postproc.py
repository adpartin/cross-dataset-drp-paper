import json
import os
import warnings
from pathlib import Path

from pprint import pprint
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# filepath = Path(__file__).parent
filepath = Path(os.path.abspath(''))
print(filepath)

# canc_col_name = "improve_sample_id"
# drug_col_name = "improve_chem_id"

main_models_path = filepath / 'models'  # dir containing the collection of models
models_paths_list  = sorted(main_models_path.glob('*'))  # list of paths to the models


# ----------------------------------
# Aggregate runtimes from all models
# ----------------------------------
res_fname = 'runtimes.csv'
sep = ','
out_fname = 'all_models_' + res_fname

agg_df_list = []
missing_files = []

for model_dir in models_paths_list:
    model_name = model_dir.name
    pp_res_path = model_dir / f"postproc.csa.{model_name}.improve_output"
    print(pp_res_path)
    try:
        rr = pd.read_csv(pp_res_path / res_fname, sep=sep)
        agg_df_list.append(rr)
    except FileNotFoundError:
        warnings.warn(f"File not found! {pp_res_path}", UserWarning)
        missing_files.append(pp_res_path)

df = pd.concat(agg_df_list, axis=0)
df.to_csv(filepath / out_fname, index=False)
pprint(df.shape)
# pprint(df.nunique())
pprint(df[:3])


# --------------------------------
# Aggregate scores from all models
# --------------------------------
res_fname = 'all_scores.csv'
sep = ','
out_fname = 'all_models_' + res_fname

agg_df_list = []
missing_files = []

for model_dir in models_paths_list:
    model_name = model_dir.name
    pp_res_path = model_dir / f"postproc.csa.{model_name}.improve_output"
    print(pp_res_path)
    try:
        rr = pd.read_csv(pp_res_path / res_fname, sep=sep)
        agg_df_list.append(rr)
    except FileNotFoundError:
        warnings.warn(f"File not found! {pp_res_path}", UserWarning)
        missing_files.append(pp_res_path)

df = pd.concat(agg_df_list, axis=0)
df.to_csv(filepath / out_fname, index=False)
pprint(df.shape)
# pprint(df.nunique())
pprint(df[:3])



### ------------------------------------------------
# Runtime plots
### ------------------------------------------------
main_runtime_outdir = 'plot_runtimes'
os.makedirs(main_runtime_outdir, exist_ok=True)

# Creating separate plots for each stage, grouped by src and model.
# Note that this not useful for infer where trg is the important factor.

# Load the data
df = pd.read_csv('all_models_runtimes.csv')

# Group by src, stage, and model, and calc the mean and standard deviation of tot_mins
stage_model_src_stats = df.groupby(['src', 'stage', 'model'])['tot_mins'].agg(['mean', 'std', 'count']).reset_index()

# Calc the standard error of the mean (sem)
stage_model_src_stats['sem'] = stage_model_src_stats['std'] / stage_model_src_stats['count'] ** 0.5

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

    plt.title(f'Distribution of Total Minutes for Stage {stage.upper()} with Error Bars')
    plt.ylabel('Average Total Minutes')
    plt.xlabel('Source dataset')
    plt.xticks(rotation=45)
    plt.legend(title='Model')
    plt.tight_layout()
    plt.grid(True)
    plt.show()
    fname = f'runtime_{stage}.png'
    plt.savefig(filepath / main_runtime_outdir / fname, dpi=200, bbox_inchesstr='tight')

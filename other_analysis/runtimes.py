"""
Runtime analysis utilities for drug response prediction models.
This script provides functions for:
1. Aggregating runtime information across models
2. Visualizing runtime patterns
3. Comparing computational efficiency across models and datasets
"""

import argparse
import logging
import os
import warnings
from pathlib import Path
from typing import List, Optional

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def setup_logging(log_file: str = 'runtime_analysis.log') -> None:
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

def agg_runtimes(models_paths_list: List[Path], outdir: Path) -> pd.DataFrame:
    """Aggregate runtimes for all models and stages.
    
    Args:
        models_paths_list: List of paths to model directories
        outdir: Output directory for saving results
    
    Returns:
        DataFrame containing aggregated runtime information
    """
    logging.info('Aggregating and saving runtimes.')
    res_fname = 'runtimes.csv'
    out_fname = 'all_models_' + res_fname

    agg_df_list = []
    missing_files = []

    for model_dir in models_paths_list:
        model_name = model_dir.name
        pp_res_path = model_dir / f'postproc.csa.{model_name}.improve_output'
        logging.info(f'Processing {pp_res_path}')
        try:
            rr = pd.read_csv(pp_res_path / res_fname)
            agg_df_list.append(rr)
        except FileNotFoundError:
            logging.warning(f'File not found! {pp_res_path}')
            missing_files.append(pp_res_path)

    # Save missing files info
    with open(outdir / 'missing_runtime_files.csv', 'w') as f:
        for path in missing_files:
            f.write(f'{path}\n')

    # Aggregate and save results
    if agg_df_list:
        df = pd.concat(agg_df_list, axis=0)
        df.to_csv(outdir / out_fname, index=False)
        return df
    return pd.DataFrame()

def plot_runtimes(runtime_df: pd.DataFrame, outdir: Path) -> None:
    """Generate runtime visualization plots.
    
    Args:
        runtime_df: DataFrame containing runtime information
        outdir: Output directory for saving plots
    """
    plot_runtime_outdir = outdir / 'plot_runtimes'
    os.makedirs(plot_runtime_outdir, exist_ok=True)

    # Compute statistics
    stage_model_src_stats = runtime_df.groupby(
        ['src', 'stage', 'model']
    )['tot_mins'].agg(['mean', 'std', 'count']).reset_index()
    
    stage_model_src_stats['sem'] = (
        stage_model_src_stats['std'] / 
        stage_model_src_stats['count'] ** 0.5
    )

    # Plot settings
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    # Create plots for each stage
    for stage in stage_model_src_stats['stage'].unique():
        plt.figure(figsize=(14, 7))
        stage_data = stage_model_src_stats[
            stage_model_src_stats['stage'] == stage
        ]
        
        # Create bar plot
        bar_plot = sns.barplot(
            x='src', y='mean', hue='model',
            data=stage_data, palette=colors, errorbar=None
        )
        
        # Add error bars
        for index, bar in enumerate(bar_plot.patches):
            height = bar.get_height()
            sem = stage_data['sem'].iloc[index]
            plt.errorbar(
                x=bar.get_x() + bar.get_width() / 2,
                y=height,
                yerr=sem,
                fmt='none',
                c='black',
                capsize=5,
                elinewidth=1
            )

        # Customize plot
        plt.title(
            f'Distribution of Total Minutes for Stage {stage.upper()}'
        )
        plt.ylabel('Average Total Minutes')
        plt.xlabel('Source Dataset')
        plt.xticks(rotation=45)
        plt.legend(title='Model')
        plt.tight_layout()
        plt.grid(True)
        
        # Save plot
        fname = f'runtime_{stage}.png'
        plt.savefig(
            plot_runtime_outdir / fname,
            dpi=200,
            bbox_inches='tight'
        )
        plt.close()

def main():
    parser = argparse.ArgumentParser(description='Runtime analysis.')
    parser.add_argument('--outdir', default='.', type=str, help='Output directory')
    parser.add_argument('--runs-dir', default='../run/v1.1', type=str, 
                       help='Directory containing model runs')
    args = parser.parse_args()
    
    outdir = Path(args.outdir)
    setup_logging()
    
    # Get model paths
    main_models_path = Path(args.runs_dir)
    models_paths_list = sorted(p for p in main_models_path.glob('*') 
                             if (p/'improve_output').exists())
    
    # Run analysis
    runtime_df = agg_runtimes(models_paths_list, outdir)
    if not runtime_df.empty:
        plot_runtimes(runtime_df, outdir)

if __name__ == "__main__":
    main()

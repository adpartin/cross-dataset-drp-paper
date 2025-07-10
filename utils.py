import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap, Normalize

# filepath = Path(__file__).parent
# filepath = Path(os.path.abspath(''))

canc_col_name = 'improve_sample_id'
drug_col_name = 'improve_chem_id'

model_name_mapping = {
    "deepcdr": "DeepCDR",
    "deepttc": "DeepTTC",
    "graphdrp": "GraphDRP",
    "hidra": "HiDRA",
    "lgbm": "LGBM",
    "tcnns": "tCNNS",
    "uno": "UNO",
}

metrics_name_mapping = {
    "r2": "R²",
    "mae": "MAE",
    "rmse": "RMSE",
    "stgr": "STGR",
    "stgi": "STGI",
}


def boxplot_violinplot_within_study(
    df: pd.DataFrame, 
    metric_name: str, 
    models_to_include: list = [],
    outdir: Path = Path("."),
    file_format: str = "eps",
    dpi: int = 600,
    palette: str = "muted", # "Set3", "species"
    title: str = None, 
    ylabel: str = None, 
    xlabel: str = None, 
    xlabel_rotation: int = 45,
    ymin: float = None,  # minimum y-axis limit
    ymax: float = None,  # maximum y-axis limit
    title_fontsize: int = 12,
    xtick_fontsize: int = 11,
    ytick_fontsize: int = 11,
    xlabel_fontsize: int = 11,
    ylabel_fontsize: int = 11,
    datasets_order: list = [],
    show: bool = True
):
    """ 
    Generate and save boxplot and violitplot. Each plot shows the performance
    of models for each dataset. This scenario is for the case where the source
    and target datasets are the same (src=trg), i.e. within-study performance
    analysis.
    
    Args:
        df (pd.DataFrame): the data to plot
        metric_name (str): the name of the metric to plot
        models_to_include (list): the models to include in the plot
        outdir (Path): the directory to save the plot
        file_format (str): output file figure format
        dpi (int): figure resolution in dots per inch
        palette (str): the palette to use for the plot
        title (str): the title of the plot
        ylabel (str): the y-axis label of the plot
        xlabel (str): the x-axis label of the plot
        xlabel_rotation (int): the rotation of the x-axis labels
        ymin (float): the minimum y-axis limit
        ymax (float): the maximum y-axis limit
        title_fontsize (int): the font size of the title
        xtick_fontsize (int): the font size of the x-axis ticks
        ytick_fontsize (int): the font size of the y-axis ticks
        xlabel_fontsize (int): the font size of the x-axis label
        ylabel_fontsize (int): the font size of the y-axis label
        show (bool): whether to show the plot

    It is expected the data include the following columns:
    - src: source dataset
    - trg: target dataset
    - model: model name
    - value: metric value

    Returns:
        None
    """
    # Plot settings
    title = f"{metrics_name_mapping[metric_name]} Prediction Performance (source=target)" if title is None else title
    ylabel = f"{metrics_name_mapping[metric_name]} Score" if ylabel is None else ylabel
    xlabel = "Dataset" if xlabel is None else xlabel
    xlabel_rotation = 45 if xlabel_rotation is None else xlabel_rotation

    # Plot: Boxplot for Each Source Dataset (src=trg)
    if len(models_to_include) == 1:
        plt.figure(figsize=(8, 5))
        sns.boxplot(data=df, x="src", y="value", palette=palette, legend=False, order=datasets_order)
    else:
        plt.figure(figsize=(13, 6))
        sns.boxplot(data=df, x="src", y="value", hue="model", palette=palette, legend=True, order=datasets_order)
        plt.legend(title='Model')
    plt.title(title, fontsize=title_fontsize)
    plt.ylabel(ylabel, fontsize=ylabel_fontsize)
    plt.xlabel(xlabel, fontsize=xlabel_fontsize)
    plt.yticks(fontsize=ytick_fontsize)
    plt.xticks(rotation=xlabel_rotation, fontsize=xtick_fontsize)
    plt.grid(True)

    # Set y-axis limits if provided
    if ymin is not None or ymax is not None:
        plt.ylim(ymin if ymin is not None else plt.ylim()[0], ymax if ymax is not None else plt.ylim()[1])

    plt.tight_layout()
    plt.savefig(outdir / f"boxplot_{metric_name}_within_study_multiple_models.{file_format}", format=file_format, dpi=dpi)
    if show:
        plt.show()

    # Plot: Violin Plot for Each Source Dataset (src=trg)
    if len(models_to_include) == 1:
        plt.figure(figsize=(8, 5))
        sns.violinplot(data=df, x="src", y="value", palette=palette, inner="quartile", order=datasets_order)
    else:
        plt.figure(figsize=(13, 6))
        sns.violinplot(data=df, x="src", y="value", hue="model", palette=palette, inner="quartile", dodge=True, order=datasets_order)
        plt.legend(title="Model", loc="lower left")#, bbox_to_anchor=(1, 0.5))
    plt.title(title, fontsize=title_fontsize)
    plt.ylabel(ylabel, fontsize=ylabel_fontsize)
    plt.xlabel(xlabel, fontsize=xlabel_fontsize)
    plt.yticks(fontsize=ytick_fontsize)
    plt.xticks(rotation=xlabel_rotation, fontsize=xtick_fontsize)
    plt.grid(True)

    # Set y-axis limits if provided
    if ymin is not None or ymax is not None:
        plt.ylim(ymin if ymin is not None else plt.ylim()[0], ymax if ymax is not None else plt.ylim()[1])

    plt.tight_layout()
    plt.savefig(outdir / f"violinplot_{metric_name}_within_study_multiple_models.{file_format}", format=file_format, dpi=dpi)
    if show:
        plt.show()

    return None


def boxplot_violinplot_cross_study(
    df: pd.DataFrame,
    source_dataset: str,
    metric_name: str, 
    models_to_include: list = [],
    outdir: Path = Path("."),
    file_format: str = "eps",
    dpi: int = 600,
    palette: str = "muted", # "Set3", "species"
    title: str = None, 
    ylabel: str = None, 
    xlabel: str = None, 
    xlabel_rotation: int = 45,
    ymin: float = None,
    ymax: float = None,
    title_fontsize: int = 12,
    xtick_fontsize: int = 11,
    ytick_fontsize: int = 11,
    xlabel_fontsize: int = 11,
    ylabel_fontsize: int = 11,
    datasets_order: list = [],
    show: bool = True
):
    """
    Generate and save boxplot and violitplot. Each plot shows the performance
    of models for each target dataset. This scenario is for the case where the
    source and target datasets are different (src!=trg), i.e. cross-study
    performance analysis.

    Args:
        df (pd.DataFrame): the data to plot
        source_dataset (str): the source dataset
        metric_name (str): the name of the metric to plot
        models_to_include (list): the models to include in the plot
        outdir (Path): the directory to save the plot
        file_format (str): output file figure format
        dpi (int): figure resolution in dots per inch
        palette (str): the palette to use for the plot
        title (str): the title of the plot
        ylabel (str): the y-axis label of the plot
        xlabel (str): the x-axis label of the plot
        xlabel_rotation (int): the rotation of the x-axis labels
        ymin (float): the minimum y-axis limit
        ymax (float): the maximum y-axis limit
        title_fontsize (int): the font size of the title
        xtick_fontsize (int): the font size of the x-axis ticks
        ytick_fontsize (int): the font size of the y-axis ticks
        xlabel_fontsize (int): the font size of the x-axis label
        ylabel_fontsize (int): the font size of the y-axis label
        datasets_order (list): the order of the datasets in the plot
        show (bool): whether to show the plot

    It is expected the data include the following columns:
    - src: source dataset
    - trg: target dataset
    - model: model name
    - value: metric value

    Returns:
        None
    """
    # Plot settings
    title = f"{metrics_name_mapping[metric_name]} Prediction Generalization from {source_dataset} to Target Datasets (source \u2260 target)" if title is None else title
    ylabel = f"{metrics_name_mapping[metric_name]} Score"
    xlabel = "Target Dataset"
    xlabel_rotation = 45

    # Plot: Boxplot for Each Target Dataset (src != trg)
    # sns.boxplot(data=filtered_data, x="trg", y="value", hue="model", palette=palette)
    if len(models_to_include) == 1:
        plt.figure(figsize=(8, 5))
        sns.boxplot(data=df, x="trg", y="value", palette=palette, legend=False, order=datasets_order)
    else:
        plt.figure(figsize=(13, 6))
        sns.boxplot(data=df, x="trg", y="value", hue="model", palette=palette, legend=True, order=datasets_order)
        plt.legend(title='Model')
    plt.title(title, fontsize=title_fontsize)
    plt.ylabel(ylabel, fontsize=ylabel_fontsize)
    plt.xlabel(xlabel, fontsize=xlabel_fontsize)
    plt.yticks(fontsize=ytick_fontsize)
    plt.xticks(rotation=xlabel_rotation, fontsize=xtick_fontsize)
    plt.grid(True)
    
    # Set y-axis limits if provided
    if ymin is not None or ymax is not None:
        plt.ylim(ymin if ymin is not None else plt.ylim()[0], ymax if ymax is not None else plt.ylim()[1])

    plt.tight_layout()

    # Save the boxplot
    plt.savefig(outdir / f"boxplot_{metric_name}_cross_study_from_{source_dataset}_to_targets.{file_format}", format=file_format, dpi=dpi)
    if show:
        plt.show()
    else:
        plt.close()

    # Plot: Violin Plot for Each Target Dataset (src != trg)
    if len(models_to_include) == 1:
        plt.figure(figsize=(8, 5))
        sns.violinplot(data=df, x="trg", y="value", palette=palette, inner="quartile", order=datasets_order)
    else:
        plt.figure(figsize=(13, 6))
        sns.violinplot(data=df, x="trg", y="value", hue="model", palette=palette, inner="quartile", dodge=True, order=datasets_order)
        plt.legend(title='Model', loc="upper right")
    plt.title(title, fontsize=title_fontsize)
    plt.ylabel(ylabel, fontsize=ylabel_fontsize)
    plt.xlabel(xlabel, fontsize=xlabel_fontsize)
    plt.yticks(fontsize=ytick_fontsize)
    plt.xticks(rotation=xlabel_rotation, fontsize=xtick_fontsize)
    plt.grid(True)

    # Set y-axis limits if provided
    if ymin is not None or ymax is not None:
        plt.ylim(ymin if ymin is not None else plt.ylim()[0], ymax if ymax is not None else plt.ylim()[1])

    plt.tight_layout()

    # Save the violin plot
    plt.savefig(outdir / f"violinplot_{metric_name}_cross_study_from_{source_dataset}_to_targets.{file_format}", format=file_format, dpi=dpi)
    if show:
        plt.show()
    else:
        plt.close()

    return None


def csa_heatmap(
    model_name: str,
    metric_name: str,
    csa_metric_name: str,
    scores_csa_data: pd.DataFrame,
    std_csa_data: Optional[pd.DataFrame] = None,
    vmin: float = -0.5,
    vmax: float = 1,
    outdir: Path = Path("."),
    file_format: str = "eps",
    dpi: int = 600,
    palette: str = "Blues",
    decimal_digits: int = 3,
    show: bool = True,
    title_fontsize: int = 12,
    xtick_fontsize: int = 11,
    ytick_fontsize: int = 11,
    xlabel_fontsize: int = 11,
    ylabel_fontsize: int = 11,
    cbar_fontsize: int = 10,
    annot_size: int = 12,
):
    """
    This function plots the G and Gn matrices as heatmaps.
    For G, we plot mean and std in parentheses.
    For Gn, we plot the mean of normalized scores.

    Args:
        model_name (str): the name of the model
        metric_name (str): the name of the metric
        csa_metric_name (str): the csa metric name (G, Ga, Gn, Gna)
        scores_csa_data (pd.DataFrame): the CSA performance scores (mean across splits)
        std_csa_data (pd.DataFrame): the CSA standard deviations (std across splits)
        vmin (float): the minimum value of the colorbar
        vmax (float): the maximum value of the colorbar
        outdir (Path): the directory to save the plot
        file_format (str): output file figure format
        dpi (int): figure resolution in dots per inch
        palette (str): the palette to use for the plot
        decimal_digits (int): the number of decimal digits to show in annotations
        show (bool): whether to show the plot
        title_fontsize (int): the font size of the title
        xtick_fontsize (int): the font size of the x-axis ticks
        ytick_fontsize (int): the font size of the y-axis ticks
        xlabel_fontsize (int): the font size of the x-axis label
        ylabel_fontsize (int): the font size of the y-axis label
        cbar_fontsize (int): the font size of the colorbar label
        annot_size (int): the font size of the annotations

    Returns:
        None
    """
    # Define colormap
    cmap = sns.color_palette(palette, as_cmap=True)

    # Use Normalize to ensure linear scaling
    threshold = vmin  # Values <= -0.5 will be deep blue
    norm = Normalize(vmin=threshold, vmax=vmax)

    # Combine scores and stds for annotations with specified decimal digits
    if std_csa_data is not None:
        combined_annotations = (
            scores_csa_data.round(decimal_digits).astype(str) + 
            "\n(" + std_csa_data.round(decimal_digits).astype(str) + ")"
        )
        filename = f"{metric_name}_{model_name}_{csa_metric_name}_heatmap_with_stds.{file_format}"
    else:
        combined_annotations = scores_csa_data.round(decimal_digits).astype(str)
        filename = f"{metric_name}_{model_name}_{csa_metric_name}_heatmap.{file_format}"
    title = f"${csa_metric_name}$ matrix for {model_name_mapping[model_name]}"

    # Plot the heatmap
    plt.figure(figsize=(7, 5))
    sns.heatmap(
        scores_csa_data,
        annot=combined_annotations.values,
        fmt="",
        cmap=cmap,
        norm=norm,
        cbar_kws={},
        annot_kws={'size': annot_size},
    )

    # Customize colorbar ticks
    colorbar = plt.gca().collections[0].colorbar
    colorbar.set_ticks([threshold, 0, 0.5, vmax])
    colorbar.set_ticklabels([f"≤ {threshold}", "0", "0.5", f"≤ {vmax}"])
    colorbar.ax.tick_params(labelsize=cbar_fontsize)  # Set font size for colorbar ticks
    # colorbar.set_label(f'{metrics_name_mapping[metric_name]} Score', fontsize=cbar_fontsize)  # Set font size for label
    colorbar.set_label(f'{metric_name} score', fontsize=cbar_fontsize)  # Set font size for label

    # Finalize plot
    plt.title(title, fontsize=title_fontsize)
    plt.xticks(fontsize=xtick_fontsize)
    plt.yticks(fontsize=ytick_fontsize, rotation=0)
    plt.xlabel("Target Dataset", fontsize=xlabel_fontsize)
    plt.ylabel("Source Dataset", fontsize=ylabel_fontsize)
    plt.tight_layout()
    plt.savefig(outdir / filename, format=file_format, dpi=dpi)
    if show:
        plt.show()
    else:
        plt.close()

    return None



# ----------------------------------------------------------------------------
# Custom Metrics
# ----------------------------------------------------------------------------

# Normalized Generalization Matrix (Gn)
"""
Gn is the normalized version of the basic Generalization matrix (G) in the CSA framework.

The Gn quantifies the relative performance of a model trained on a source dataset and
applied to a target dataset. It provides a pairwise comparison by normalizing each 
source-target performance score by the source-source performance.

Gn evaluates generalization quality per source-target pair, informing on dataset 
alignments and model transferability.

Key Characteristics:
    - Pairwise Analysis/Metric: Each score G[i, j] is divided by the within-study performance G[i, i],
        providing insights into source-target performance for each dataset pair.
    - No Aggregation: Retains all normalized values, making it suitable for detailed analysis of
        cross-study generalization.
    - Generalization Insight: Gn highlights the balance between cross-study and 
        within-study performance for each pair.
    - Interpretation: 
        - Gn ≈ 1: Generalization to the target is similar to within-study performance.
        - Gn < 1: Generalization to the target is worse than within-study performance.
        - Gn > 1: Performance on the target exceeds within-study performance, possibly 
            due to target simplicity or alignment.
    - Edge Cases
        - Zero within-study performance: If scores[src, src] == 0, the result is undefined; 
            a default value of 0 is assigned.

Formula:
    Gn[i, j] = Performance(src → trg) / Performance(src → src)
    Where:
        - Performance(src → trg): Cross-study performance for source i on target j.
        - Performance(src → src): Within-study performance for source i.
"""

def compute_Gn_vectorized(scores: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the normalized version of G (Gn) using a vectorized approach.
    This version uses diagonal_values[:, None] for row-wise broadcasting.

    Args:
        scores (pd.DataFrame): A square DataFrame where rows represent source datasets 
            and columns represent target datasets. Each cell contains the performance 
            score for the source-target pair.

    Returns:
        pd.DataFrame: A DataFrame of Gn values.
    """
    diagonal_values = scores.to_numpy().diagonal()  # Extract within-study performances
    normalized_scores = scores.to_numpy() / diagonal_values[:, None]  # Row-wise division
    Gn = pd.DataFrame(normalized_scores, index=scores.index, columns=scores.columns)
    return Gn


def compute_Gn_bruteforce(scores: pd.DataFrame) -> dict:
    """
    Compute the Normalized Generalization Matrix (Gn) using a brute-force approach.

    Args:
        scores (pd.DataFrame): A square DataFrame where rows represent source datasets 
            and columns represent target datasets. Each cell contains the performance 
            score for the source-target pair.

    Returns:
        dict: A nested dictionary where keys are source dataset names, and values are 
              dictionaries with target dataset names as keys and Gn values as values.
    """
    Gn = {}  # Initialize an empty dictionary
    diagonal_values = scores.to_numpy().diagonal()  # Extract within-study performances (diagonal values)
    
    for i, src in enumerate(scores.index):  # Iterate over source datasets
        Gn[src] = {}  # Create a nested dictionary for the source dataset
        within_study = diagonal_values[i]  # Retrieve the within-study performance for the source

        for j, trg in enumerate(scores.columns):  # Iterate over target datasets
            if within_study != 0:  # Avoid division by zero
                Gn[src][trg] = scores.iloc[i, j] / within_study  # Normalize the score for src → trg
            else:
                Gn[src][trg] = 0  # Assign Gn to 0 if within-study performance is zero

    return Gn


# Aggregated Generalization Matrix (Ga and Gna)
"""
The Ga and Gna evaluate the average cross-study generalization of a source dataset by 
aggregating pairwise performance scores across all target datasets (excluding the diagonal).

Ga summarizes the raw performance scores, while Gna normalizes each source-target performance 
score by the within-study performance.

Key Characteristics:
    - Normalization (for Gna): Each score G[i, j] is divided by the within-study performance 
        G[i, i] for the source dataset i.
    - Exclusion of Diagonal: Within-study values (G[i, i]) are excluded from the summation.
    - Generalization: Ga and Gna highlight how well a source dataset generalizes to 
        other target datasets on average.
    - Aggregation: Outputs a single value for each source dataset, making it ideal for
        comparative analysis.
    - Interpretation: Higher Ga or Gna indicates better generalization from the source 
        dataset to other datasets.
        - Ga or Gna > 0: positive average generalization from the source across other datasets.
        - Ga or Gna < 0: poor average generalization performance from the source across other datasets.
    - Caveats
        - Aggregation can mask pairwise nuances: Individual source-target relationships may get obscured.
        - Sensitive to dataset-specific biases: Strong performance on a subset of target datasets can dominate the average.

Formula:
    Ga[i] = (1 / (n - 1)) * Σ[j != i] G[i, j]
    Gna[i] = (1 / (n - 1)) * Σ[j != i] (G[i, j] / G[i, i])
    Where:
        - G[i, j]: Performance of source i on target j
        - G[i, i]: Within-study performance for source i (for Gna)
        - n: Number of datasets (size of the scores matrix)
"""

# def compute_Gna_vectorized(scores: pd.DataFrame, normalize: bool = True) -> pd.Series:
#     """
#     Compute the Aggregated Normalized Generalization Matrix (Gna) using a vectorized approach.
#     This version uses diagonal_values[:, None] for row-wise broadcasting.
#     If normalize is False, the regular mean is computed excluding the source-source score.

#     Args:
#         scores (pd.DataFrame): A square DataFrame where rows represent source datasets 
#             and columns represent target datasets. Each cell contains the performance 
#             score for the source-target pair.
#         normalize (bool): Whether to normalize the scores (for Gna) or compute mean.

#     Returns:
#         pd.Series: A Series of Gna values, one for each source dataset.
#     """
#     diagonal_values = scores.to_numpy().diagonal()  # Extract within-study performances
#     if normalize:
#         normalized_scores = scores.to_numpy() / diagonal_values[:, None]  # Row-wise division
#     else:
#         normalized_scores = scores.to_numpy()

#     # Set diagonal to NaN to exclude from averaging
#     np.fill_diagonal(normalized_scores, np.nan)

#     # Compute mean across columns (ignoring NaN) and return as Series
#     Gna = np.nanmean(normalized_scores, axis=1)
#     Gna = {dataset: value for dataset, value in zip(scores.index, Gna)}
#     return Gna

def compute_aggregated_G_vectorized(scores: pd.DataFrame, normalize: bool = True) -> pd.Series:
    """
    Compute the Aggregated Generalization Matrix (Ga or Gna) using a vectorized approach.
    This version uses diagonal_values[:, None] for row-wise broadcasting.

    Args:
        scores (pd.DataFrame): A square DataFrame where rows represent source datasets 
            and columns represent target datasets. Each cell contains the performance 
            score for the source-target pair.
        normalize (bool): Whether to normalize the scores (for Gna) or compute mean (for Ga).

    Returns:
        pd.Series: A Series of Ga or Gna values, one for each source dataset.
    """
    scores = scores.copy()
    diagonal_values = scores.to_numpy().diagonal()  # Extract within-study performances
    if normalize:
        normalized_scores = scores.to_numpy() / diagonal_values[:, None]  # Row-wise division
    else:
        normalized_scores = scores.to_numpy()

    # Set diagonal to NaN to exclude from averaging
    np.fill_diagonal(normalized_scores, np.nan)

    # Compute mean across columns (ignoring NaN) and return as Series
    aggregated_scores = np.nanmean(normalized_scores, axis=1)
    aggregated_scores = {dataset: value for dataset, value in zip(scores.index, aggregated_scores)}
    return aggregated_scores

# def compute_Gna_bruteforce(scores: pd.DataFrame) -> pd.Series:
#     """
#     Compute the aggregated normalized version of G (Gna) using a brute-force approach
#     (previoulsy called Source-to-Target Generalization Index (STGI)).

#     Args:
#         scores (pd.DataFrame): A square DataFrame where rows represent source datasets 
#             and columns represent target datasets. Each cell contains the performance 
#             score for the source-target pair.

#     Returns:
#         dict: A dictionary where keys are source dataset names, and values are the 
#               computed STGI for each source.
#     """
#     Gna = {}  # Initialize an empty dictionary
#     diagonal_values = scores.to_numpy().diagonal()  # Extract within-study performances (diagonal values)
    
#     for i, src in enumerate(scores.index):  # Iterate over source datasets
#         cross_study = []  # List to collect normalized cross-study scores
#         within_study = diagonal_values[i]  # Retrieve the within-study performance for the source

#         for j, trg in enumerate(scores.columns):  # Iterate over target datasets
#             if i != j:  # Exclude within-study values
#                 if within_study != 0:  # Avoid division by zero
#                     cross_study.append(scores.iloc[i, j] / within_study)  # Normalize the score
#                 else:
#                     cross_study.append(0)  # Assign 0 if within-study performance is zero

#         # Compute the mean of cross-study scores for the current source dataset (src)
#         Gna[src] = sum(cross_study) / len(cross_study) if len(cross_study) > 0 else 0  # Avoid division by zero

#     return Gna

def compute_aggregated_G_bruteforce(scores: pd.DataFrame, normalize: bool = True) -> dict:
    """
    Compute the Aggregated Generalization Matrix (Ga or Gna) using a brute-force approach.

    Args:
        scores (pd.DataFrame): A square DataFrame where rows represent source datasets 
            and columns represent target datasets. Each cell contains the performance 
            score for the source-target pair.
        normalize (bool): Whether to normalize the scores (for Gna) or compute mean (for Ga).

    Returns:
        dict: A dictionary where keys are source dataset names, and values are the 
              computed Ga or Gna for each source.
    """
    scores = scores.copy()
    aggregated_scores = {}  # Initialize an empty dictionary
    diagonal_values = scores.to_numpy().diagonal()  # Extract within-study performances (diagonal values)
    
    for i, src in enumerate(scores.index):  # Iterate over source datasets
        cross_study = []  # List to collect cross-study scores
        within_study = diagonal_values[i]  # Retrieve the within-study performance for the source

        for j, trg in enumerate(scores.columns):  # Iterate over target datasets
            if i != j:  # Exclude within-study values
                if normalize and within_study != 0:  # Normalize if required and avoid division by zero
                    cross_study.append(scores.iloc[i, j] / within_study)  # Normalize the score
                else:
                    cross_study.append(scores.iloc[i, j])  # Use raw score

        # Compute the mean of cross-study scores for the current source dataset (src)
        aggregated_scores[src] = sum(cross_study) / len(cross_study) if len(cross_study) > 0 else 0  # Avoid division by zero

    return aggregated_scores


def aggregated_G_heatmap(
    metric_name: str,
    csa_metric_name: str,
    scores_aggregated_data: pd.DataFrame,
    vmin: float = -0.5,
    vmax: float = 1,
    outdir: Path = Path("."),
    file_format: str = "eps",
    dpi: int = 600,
    palette: str = "plasma",
    decimal_digits: int = 3,
    title_fontsize: int = 12,
    xtick_fontsize: int = 11,
    ytick_fontsize: int = 11,
    xlabel_fontsize: int = 11,
    ylabel_fontsize: int = 11,
    cbar_fontsize: int = 10,
    annot_size: int = 12,
    show: bool = True
):
    """
    Plot the CSA performance scores with standard deviations as a heatmap.

    Args:
        metric_name (str): the name of the metric
        csa_metric_name (str): the name of the CSA metric (Ga or Gna)
        scores_aggregated_data (pd.DataFrame): the aggregated performance scores (mean across splits)
        vmin (float): the minimum value of the colorbar
        vmax (float): the maximum value of the colorbar
        outdir (Path): the directory to save the plot
        file_format (str): output file figure format
        dpi (int): figure resolution in dots per inch
        palette (str): the palette to use for the plot
        decimal_digits (int): the number of decimal digits to show in annotations
        title_fontsize (int): the font size of the title
        xtick_fontsize (int): the font size of the x-axis ticks
        ytick_fontsize (int): the font size of the y-axis ticks
        xlabel_fontsize (int): the font size of the x-axis label
        ylabel_fontsize (int): the font size of the y-axis label
        cbar_fontsize (int): the font size of the colorbar label
        annot_size (int): the font size of the annotations
        show (bool): whether to show the plot

    Returns:
        None
    """
    # Define colormap
    # cmap = sns.color_palette(palette, as_cmap=True)
    if palette in plt.colormaps():  # Check if the palette is a valid Matplotlib colormap
        cmap = plt.get_cmap(palette)
    else:
        cmap = sns.color_palette(palette, as_cmap=True)  # Use Seaborn for custom palettes

    # Use Normalize to ensure linear scaling
    threshold = vmin  # Values <= -0.5 will be deep blue
    norm = Normalize(vmin=threshold, vmax=vmax)

    # Name map for models
    scores_aggregated_data.index = [model_name_mapping[model] for model in scores_aggregated_data.index]

    # Combine scores and stds for annotations with specified decimal digits
    combined_annotations = scores_aggregated_data.round(decimal_digits).astype(str)
    # title = f"Source-to-Target Generalization Index (STGI)"
    title = f"${csa_metric_name}$ (all models combined)"
    filename = f"{metric_name}_{csa_metric_name}_heatmap.{file_format}"

    # Plot the heatmap
    plt.figure(figsize=(7, 5))
    sns.heatmap(
        scores_aggregated_data, 
        annot=combined_annotations.values,
        fmt="", 
        cmap=cmap, 
        norm=norm, 
        # cbar_kws={'label': f'{metrics_name_mapping[metric_name]} Score'},
        cbar_kws={},
        annot_kws={'size': annot_size},
    )

    # Customize colorbar ticks
    colorbar = plt.gca().collections[0].colorbar
    colorbar.set_ticks([threshold, 0, 0.5, vmax])
    colorbar.set_ticklabels([f"≤ {threshold}", "0", "0.5", f"≤ {vmax}"])
    colorbar.ax.tick_params(labelsize=cbar_fontsize)  # Set font size for colorbar ticks
    # colorbar.set_label(f'{metrics_name_mapping[metric_name]} Score', fontsize=cbar_fontsize)  # Set font size for label
    colorbar.set_label(f'{metric_name} score', fontsize=cbar_fontsize)  # Set font size for label

    # Finalize plot
    plt.title(title, fontsize=title_fontsize)
    plt.xticks(fontsize=xtick_fontsize)
    plt.yticks(fontsize=ytick_fontsize)
    plt.xlabel("Source Dataset", fontsize=xlabel_fontsize)
    plt.ylabel("Model", fontsize=ylabel_fontsize)
    plt.tight_layout()
    plt.savefig(outdir / filename, format=file_format, dpi=dpi)
    if show:
        plt.show()
    else:
        plt.close()

    return None
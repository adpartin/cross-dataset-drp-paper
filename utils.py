import json
import os
import warnings
from pathlib import Path
from pprint import pprint
from typing import Optional

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap, Normalize

# filepath = Path(__file__).parent
# filepath = Path(os.path.abspath(''))

canc_col_name = "improve_sample_id"
drug_col_name = "improve_chem_id"

model_name_mapping = {
    "deepcdr": "DeepCDR",
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
}


def boxplot_violinplot_within_study(
    df: pd.DataFrame, 
    metric_name: str, 
    models_to_include: list = [],
    outdir: Path = Path("."), 
    palette: str = "muted", # "Set3", "species"
    title: str = None, 
    ylabel: str = None, 
    xlabel: str = None, 
    xlabel_rotation: int = 45,
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
        palette (str): the palette to use for the plot
        title (str): the title of the plot
        ylabel (str): the y-axis label of the plot
        xlabel (str): the x-axis label of the plot
        xlabel_rotation (int): the rotation of the x-axis labels
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
    title = f"{metrics_name_mapping[metric_name]} Prediction Performance (src=trg)" if title is None else title
    ylabel = f"{metrics_name_mapping[metric_name]} Score" if ylabel is None else ylabel
    xlabel = "Source Dataset" if xlabel is None else xlabel
    xlabel_rotation = 45 if xlabel_rotation is None else xlabel_rotation

    # Plot: Boxplot for Each Source Dataset (src=trg)
    if len(models_to_include) == 1:
        plt.figure(figsize=(8, 5))
        sns.boxplot(data=df, x="src", y="value", palette=palette, legend=False)
    else:
        plt.figure(figsize=(13, 6))
        sns.boxplot(data=df, x="src", y="value", hue="model", palette=palette, legend=True)
        plt.legend(title='Model')
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.xticks(rotation=xlabel_rotation)
    plt.tight_layout()
    plt.savefig(outdir / f"boxplot_{metric_name}_within_study_multiple_models.png")  # Save boxplot
    if show:
        plt.show()

    # Plot: Violin Plot for Each Source Dataset (src=trg)
    if len(models_to_include) == 1:
        plt.figure(figsize=(8, 5))
        sns.violinplot(data=df, x="src", y="value", palette=palette, inner="quartile")
    else:
        plt.figure(figsize=(13, 6))
        sns.violinplot(data=df, x="src", y="value", hue="model", palette=palette, inner="quartile", dodge=True)
        plt.legend(title="Model", loc="upper center")#, bbox_to_anchor=(1, 0.5))
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.xticks(rotation=xlabel_rotation)
    plt.tight_layout()
    plt.savefig(outdir / f"violinplot_{metric_name}_within_study_multiple_models.png")  # Save violinplot
    if show:
        plt.show()

    return None


def boxplot_violinplot_cross_study(
    df: pd.DataFrame,
    source_dataset: str,
    metric_name: str, 
    models_to_include: list = [],
    outdir: Path = Path("."), 
    palette: str = "muted", # "Set3", "species"
    title: str = None, 
    ylabel: str = None, 
    xlabel: str = None, 
    xlabel_rotation: int = 45,
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
        palette (str): the palette to use for the plot
        title (str): the title of the plot
        ylabel (str): the y-axis label of the plot
        xlabel (str): the x-axis label of the plot
        xlabel_rotation (int): the rotation of the x-axis labels
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
    title = f"{metrics_name_mapping[metric_name]} Prediction Generalization from {source_dataset} to Target Datasets (src!=trg)"
    ylabel = f"{metrics_name_mapping[metric_name]} Score"
    xlabel = "Target Dataset"
    xlabel_rotation = 45

    # Plot: Boxplot for Each Target Dataset (src != trg)
    # sns.boxplot(data=filtered_data, x="trg", y="value", hue="model", palette=palette)
    if len(models_to_include) == 1:
        plt.figure(figsize=(8, 5))
        sns.boxplot(data=df, x="trg", y="value", palette=palette, legend=False)
    else:
        plt.figure(figsize=(13, 6))
        sns.boxplot(data=df, x="trg", y="value", hue="model", palette=palette, legend=True)
        plt.legend(title='Model')
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.xticks(rotation=xlabel_rotation)
    plt.tight_layout()

    # Save the boxplot
    plt.savefig(outdir / f"boxplot_{metric_name}_cross_study_from_{source_dataset}_to_targets.png")  # Save boxplot
    if show:
        plt.show()
    else:
        plt.close()

    # Plot: Violin Plot for Each Target Dataset (src != trg)
    if len(models_to_include) == 1:
        plt.figure(figsize=(8, 5))
        sns.violinplot(data=df, x="trg", y="value", palette=palette, inner="quartile")
    else:
        plt.figure(figsize=(13, 6))
        sns.violinplot(data=df, x="trg", y="value", hue="model", palette=palette, inner="quartile", dodge=True)
        plt.legend(title='Model', loc="upper center")
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.xticks(rotation=xlabel_rotation)
    plt.tight_layout()

    # Save the violin plot
    plt.savefig(outdir / f"violinplot_{metric_name}_cross_study_from_{source_dataset}_to_targets.png")  # Save violinplot
    if show:
        plt.show()
    else:
        plt.close()

    return None


def csa_heatmap(
    model_name: str,
    metric_name: str,
    scores_csa_data: pd.DataFrame,
    std_csa_data: Optional[pd.DataFrame] = None,
    vmin: float = -0.5,
    vmax: float = 1,
    outdir: Path = Path("."),
    palette: str = "Blues",
    decimal_digits: int = 3,
    show: bool = True
):
    """
    Plot the CSA performance scores with standard deviations as a heatmap.

    Args:
        model_name (str): the name of the model
        metric_name (str): the name of the metric
        scores_csa_data (pd.DataFrame): the CSA performance scores (mean across splits)
        std_csa_data (pd.DataFrame): the CSA standard deviations (std across splits)
        vmin (float): the minimum value of the colorbar
        vmax (float): the maximum value of the colorbar
        outdir (Path): the directory to save the plot
        palette (str): the palette to use for the plot
        decimal_digits (int): the number of decimal digits to show in annotations
        show (bool): whether to show the plot

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
        title = f"{model_name_mapping[model_name]}; {metrics_name_mapping[metric_name]} CSA Performance Scores with Standard Deviations"
        filename = f"{metric_name}_{model_name}_csa_heatmap_with_stds.png"
    else:
        combined_annotations = scores_csa_data.round(decimal_digits).astype(str)
        title = f"{model_name_mapping[model_name]}; {metrics_name_mapping[metric_name]} CSA Performance Scores"
        filename = f"{metric_name}_{model_name}_csa_heatmap.png"

    # Plot the heatmap
    plt.figure(figsize=(7, 5))
    sns.heatmap(
        scores_csa_data, 
        annot=combined_annotations.values,
        fmt="", 
        cmap=cmap, 
        norm=norm, 
        cbar_kws={'label': f'{metrics_name_mapping[metric_name]} Score'}
    )

    # Customize colorbar ticks
    colorbar = plt.gca().collections[0].colorbar
    colorbar.set_ticks([threshold, 0, 0.5, vmax])
    colorbar.set_ticklabels([f"≤ {threshold}", "0", "0.5", f"≤ {vmax}"])

    # Finalize plot
    plt.title(title)
    plt.xlabel("Target Dataset")
    plt.ylabel("Source Dataset")
    plt.tight_layout()
    plt.savefig(outdir / filename)
    if show:
        plt.show()
    else:
        plt.close()

    return None



# ----------------------------------------------------------------------------
# Custom Metrics
# ----------------------------------------------------------------------------

# Source-Target Generalization Ratio (STG-R)
"""
The STGR quantifies the relative performance of a model trained on a source dataset and
applied to a target dataset. It provides a pairwise comparison by normalizing each 
source-target performance score by the source-source performance.

STGR evaluates generalization quality per source-target pair, informing on dataset 
alignments and model transferability.

Key Characteristics:
    - Pairwise Analysis/Metric: Each score S[i, j] is divided by the within-study performance S[i, i],
        providing insights into source-target performance for each dataset pair.
    - No Aggregation: Retains all normalized values, making it suitable for detailed analysis of
        cross-study generalization.
    - Generalization Insight: STGR highlights the balance between cross-study and 
        within-study performance for each pair.
    - Interpretation: 
        - STGR ≈ 1: Generalization to the target is similar to within-study performance.
        - STGR < 1: Generalization to the target is worse than within-study performance.
        - STGR > 1: Performance on the target exceeds within-study performance, possibly 
            due to target simplicity or alignment.
    - Edge Cases
        - Zero within-study performance: If scores[src, src] == 0, the result is undefined; 
            a default value of 0 is assigned.

Formula:
    STGR[i, j] = Performance(src → trg) / Performance(src → src)
    Where:
        - Performance(src → trg): Cross-study performance for source i on target j.
        - Performance(src → src): Within-study performance for source i.
"""

def compute_stgr_vectorized(scores):
    """
    Compute the Source-Target Generalization Ratio (STGR) using a vectorized approach.
    This version uses diagonal_values[:, None] for row-wise broadcasting.

    Args:
        scores (pd.DataFrame): A square DataFrame where rows represent source datasets 
            and columns represent target datasets. Each cell contains the performance 
            score for the source-target pair.

    Returns:
        pd.DataFrame: A DataFrame of STGR values.
    """
    diagonal_values = scores.to_numpy().diagonal()  # Extract within-study performances
    normalized_scores = scores.to_numpy() / diagonal_values[:, None]  # Row-wise division
    stgr = pd.DataFrame(normalized_scores, index=scores.index, columns=scores.columns)
    return stgr


def compute_stgr_bruteforce(scores):
    """
    Compute the Source-Target Generalization Ratio (STGR) using a brute-force approach.

    Args:
        scores (pd.DataFrame): A square DataFrame where rows represent source datasets 
            and columns represent target datasets. Each cell contains the performance 
            score for the source-target pair.

    Returns:
        dict: A nested dictionary where keys are source dataset names, and values are 
              dictionaries with target dataset names as keys and STGR values as values.
    """
    stgr = {}  # Initialize an empty dictionary
    diagonal_values = scores.to_numpy().diagonal()  # Extract within-study performances (diagonal values)
    
    for i, src in enumerate(scores.index):  # Iterate over source datasets
        stgr[src] = {}  # Create a nested dictionary for the source dataset
        within_study = diagonal_values[i]  # Retrieve the within-study performance for the source

        for j, trg in enumerate(scores.columns):  # Iterate over target datasets
            if within_study != 0:  # Avoid division by zero
                stgr[src][trg] = scores.iloc[i, j] / within_study  # Normalize the score for src → trg
            else:
                stgr[src][trg] = 0  # Assign STGR to 0 if within-study performance is zero

    return stgr


# Source-to-Target Generalization Index (STG-I)
"""
The STGI evaluates the average cross-study generalization of a source dataset by 
aggregating pairwise normalized performance scores across all target datasets 
(excluding the diagonal).

STGI summarizes how well a single source dataset generalizes across many targets, 
helping to understand the intrinsic value of a dataset for training.

Key Characteristics:
    - Normalization: Each score S[i, j] is divided by the within-study performance 
        S[i, i] for the source dataset i.
    - Exclusion of Diagonal: Within-study values (S[i, i]) are excluded from the summation.
    - Generalization: STGI highlights how well a source dataset generalizes to 
        other target datasets on average {as compared to within-study performance}.
    - Aggregation: Outputs a single value for each source dataset, making it ideal for
        comparative analysis.
    - Interpretation: Higher STGI indicates better generalization from the source 
        dataset to other datasets.
        - STGI > 0: Indicates average positive generalization across other datasets.
        - STGI < 0: Suggests poor generalization performance to other datasets.
        - STGI ≈ 0: Indicates no net generalization (e.g., neutral average scores).
    - Caveats
        - Aggregation can mask pairwise nuances: Individual source-target relationships may get obscured.
        - Sensitive to dataset-specific biases: Strong performance on a subset of target datasets can dominate the average.

Formula:
    STGI[i] = (1 / (n - 1)) * Σ[j != i] (S[i, j] / S[i, i])
    Where:
        - S[i, j]: Performance of source i on target j
        - S[i, i]: Within-study performance for source i
        - n: Number of datasets (size of the scores matrix)
"""

def compute_stgi_vectorized(scores):
    """
    Compute the Source-to-Target Generalization Index (STGI) using a vectorized approach.
    This version uses diagonal_values[:, None] for row-wise broadcasting.

    Args:
        scores (pd.DataFrame): A square DataFrame where rows represent source datasets 
            and columns represent target datasets. Each cell contains the performance 
            score for the source-target pair.

    Returns:
        pd.Series: A Series of STGI values, one for each source dataset.
    """
    diagonal_values = scores.to_numpy().diagonal()  # Extract within-study performances
    normalized_scores = scores.to_numpy() / diagonal_values[:, None]  # Row-wise division

    # Set diagonal to NaN to exclude from averaging
    np.fill_diagonal(normalized_scores, np.nan)

    # Compute mean across columns (ignoring NaN) and return as Series
    stgi = np.nanmean(normalized_scores, axis=1)
    # stgi = pd.Series(stgi, index=scores.index)
    stgi = {dataset: value for dataset, value in zip(scores.index, stgi)}
    return stgi


def compute_stgi_bruteforce(scores):
    """
    Compute the Source-to-Target Generalization Index (STGI) for each source dataset.

    Args:
        scores (pd.DataFrame): A square DataFrame where rows represent source datasets 
            and columns represent target datasets. Each cell contains the performance 
            score for the source-target pair.

    Returns:
        dict: A dictionary where keys are source dataset names, and values are the 
              computed STGI for each source.
    """
    stgi = {}  # Initialize an empty dictionary
    diagonal_values = scores.to_numpy().diagonal()  # Extract within-study performances (diagonal values)
    
    for i, src in enumerate(scores.index):  # Iterate over source datasets
        cross_study = []  # List to collect normalized cross-study scores
        within_study = diagonal_values[i]  # Retrieve the within-study performance for the source

        for j, trg in enumerate(scores.columns):  # Iterate over target datasets
            if i != j:  # Exclude within-study values
                if within_study != 0:  # Avoid division by zero
                    cross_study.append(scores.iloc[i, j] / within_study)  # Normalize the score
                else:
                    cross_study.append(0)  # Assign 0 if within-study performance is zero

        # Compute the mean of cross-study scores for the current source dataset (src)
        stgi[src] = sum(cross_study) / len(cross_study) if len(cross_study) > 0 else 0  # Avoid division by zero

    return stgi
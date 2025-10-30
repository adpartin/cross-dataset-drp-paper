"""
Step 3: Compute Ga, Gn, Gna from G matrices and export CSVs
============================================================

This step consumes outputs from Step 1 and Step 2 to compute:
- Gn: normalized G matrices (row-wise normalized by within-study performance)
- Ga: aggregated G across targets per source (mean excluding diagonal)
- Gna: aggregated normalized G (mean of row-wise normalized scores excluding diagonal)

It also exports aggregated Ga/Gna tables across models for convenient plotting.

Inputs:
- ./outputs/G_matrices/: per-model per-metric mean/std CSA tables from Step 2
- ./outputs/scores/all_models_scores.csv: optional, for within-study summaries (not required)

Outputs:
- ./outputs/Ga_Gn_Gna_matrices/
  - <model>_<metric>_Gn_csa_table.csv
  - <model>_<metric>_Ga.csv (row vector over sources)
  - <model>_<metric>_Gna.csv (row vector over sources)
  - <metric>_Ga_all_models.csv (models x sources)
  - <metric>_Gna_all_models.csv (models x sources)
  - within_study/<metric>_mean_within_study_all_models.csv
  - within_study/<metric>_std_within_study_all_models.csv

Usage:
    python s3_compute_Ga_Gn_Gna.py [--scores_dir ./outputs/scores] [--G_dir ./outputs/G_matrices] [--outdir ./outputs]
"""

import argparse
import logging
import os
from pathlib import Path
from typing import Dict, List, Set, Tuple

import pandas as pd

from postprocess_utils import setup_logging
import utils


def discover_models_and_metrics(G_dir: Path) -> Dict[str, Set[str]]:
    """Scan G_dir for files like <MODEL>_<METRIC>_G_mean.csv and return {metric: {models...}}."""
    metrics_to_models: Dict[str, Set[str]] = {}
    for csv_path in G_dir.glob("*_G_mean.csv"):
        name = csv_path.stem  # <model>_<metric>_G_mean
        parts = name.split("_")
        if len(parts) < 3:
            continue
        model, metric = parts[0], parts[1]
        metrics_to_models.setdefault(metric, set()).add(model)
    return metrics_to_models


def load_G_mean_std(G_dir: Path, model: str, metric: str) -> pd.DataFrame:
    """
    Load the mean and std CSA tables for a given model and metric.

    Args:
        G_dir (Path): The directory containing the CSA tables.
        model (str): The model name.
        metric (str): The metric name.

    Returns:
        pd.DataFrame: The mean CSA table.
    """
    # Use new names; legacy no longer expected
    mean_path = G_dir / f"{model}_{metric}_G_mean.csv"
    std_path = G_dir / f"{model}_{metric}_G_std.csv"
    if not mean_path.exists():
        raise FileNotFoundError(f"Missing mean CSA table: {mean_path}")
    if not std_path.exists():
        logging.warning(f"Missing std CSA table: {std_path} (continuing without std)")
    G_mean = pd.read_csv(mean_path)
    if "src" in G_mean.columns:
        G_mean = G_mean.set_index("src")
    return G_mean


# Migration no longer needed; Step 2 writes new names directly.


def compute_and_save_per_model_outputs(
    G_dir: Path, 
    out_dir: Path, 
    metric: str, 
    models: List[str],
    # normalize: bool = True,
    datasets_order: List[str] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Compute Gn, Ga, Gna per model; save CSVs; return aggregated Ga/Gna across models as a DataFrame pair concatenated later."""
    ga_rows: Dict[str, pd.Series] = {}
    gna_rows: Dict[str, pd.Series] = {}

    for model in models:
        logging.info(f"Computing Ga/Gn/Gna for model={model}, metric={metric}")
        G_mean = load_G_mean_std(G_dir, model, metric)

        # Ensure index/columns are in canonical order
        G_mean.index.name = None
        G_mean.columns.name = None

        # Enforce preferred datasets order if provided
        if datasets_order:
            ordered = [d for d in datasets_order if d in G_mean.index]
            if ordered:
                G_mean = G_mean.reindex(index=ordered, columns=ordered)

        # Gn matrix
        Gn_df = utils.compute_Gn_vectorized(G_mean)
        Gn_out = out_dir / f"{model}_{metric}_Gn_mean.csv"
        Gn_df.to_csv(Gn_out)

        # Ga (raw aggregated, no normalization)
        ga_dict = utils.compute_aggregated_G_vectorized(G_mean, normalize=False)
        ga_series = pd.Series(ga_dict)
        if datasets_order:
            ga_series = ga_series.reindex([d for d in datasets_order if d in ga_series.index])
        ga_rows[model] = ga_series
        (out_dir / f"{model}_{metric}_Ga.csv").write_text(ga_series.to_csv(header=False))

        # Gna (aggregated normalized)
        gna_dict = utils.compute_aggregated_G_vectorized(G_mean, normalize=True)
        gna_series = pd.Series(gna_dict)
        if datasets_order:
            gna_series = gna_series.reindex([d for d in datasets_order if d in gna_series.index])
        gna_rows[model] = gna_series
        (out_dir / f"{model}_{metric}_Gna.csv").write_text(gna_series.to_csv(header=False))

    # Assemble aggregated tables across models (rows=models, cols=sources)
    ga_df = pd.DataFrame.from_dict(ga_rows, orient="index").sort_index()
    gna_df = pd.DataFrame.from_dict(gna_rows, orient="index").sort_index()
    if datasets_order:
        ordered_cols = [d for d in datasets_order if d in ga_df.columns]
        if ordered_cols:
            ga_df = ga_df[ordered_cols]
        ordered_cols = [d for d in datasets_order if d in gna_df.columns]
        if ordered_cols:
            gna_df = gna_df[ordered_cols]
    return ga_df, gna_df


def main():
    parser = argparse.ArgumentParser(
        description="Step 3: Compute Ga, Gn, Gna from G matrices."
    )
    parser.add_argument(
        "--scores_dir",
        type=str,
        default="./outputs/scores", 
        help="Scores directory from Step 1."
    )
    parser.add_argument(
        "--G_dir", 
        type=str, 
        default="./outputs/G_matrices", 
        help="G matrices directory from Step 2."
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="./outputs",
        help="Output root directory."
    )
    parser.add_argument(
        "--out_subdir",
        type=str,
        default="Ga_Gn_Gna_matrices",
        help="Subdirectory under outdir for Stage 3 outputs (default: Ga_Gn_Gna_matrices)."
    )
    parser.add_argument(
        "--metrics",
        type=str,
        default="r2",
        help="Comma-separated metrics to process (default: r2)."
    )
    parser.add_argument(
        "--models",
        type=str,
        default="",
        help="Comma-separated models to process (default: discover)."
    )
    parser.add_argument(
        "--datasets_order",
        type=str,
        default="CCLE,CTRPv2,GDSCv1,GDSCv2,gCSI",
        help="Comma-separated dataset order for CSA outputs (default: alphabetical)."
    )
    args = parser.parse_args()

    out_root = Path(args.outdir)
    G_dir = Path(args.G_dir)
    scores_dir = Path(args.scores_dir)

    # Prepare output/log dirs
    csa_out = out_root / args.out_subdir
    csa_out.mkdir(parents=True, exist_ok=True)
    logs_dir = Path("./logs")
    logs_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(log_file=str(logs_dir / "s3_compute_Ga_Gn_Gna.log"))

    logging.info("=" * 50)
    logging.info("Step 3: Compute Ga, Gn, Gna from G matrices")
    logging.info("=" * 50)
    logging.info(f"G_dir: {G_dir}")
    logging.info(f"scores_dir: {scores_dir}")
    logging.info(f"Output: {csa_out}")

    if not G_dir.exists():
        raise FileNotFoundError(f"G_dir not found: {G_dir}")

    # Discover available metrics/models from G_dir unless provided
    discovered = discover_models_and_metrics(G_dir)
    if args.metrics.strip():
        metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]
    else:
        metrics = sorted(discovered.keys())

    provided_models: List[str] = []
    if args.models.strip():
        provided_models = [m.strip() for m in args.models.split(",") if m.strip()]

    # Determine dataset order (CSA uses provided order; within-study uses fixed size order)
    datasets_order_csa = [d.strip() for d in args.datasets_order.split(",") if d.strip()]
    datasets_order_within = ["gCSI", "CCLE", "GDSCv2", "GDSCv1", "CTRPv2"]

    for metric in metrics:
        models = provided_models or sorted(discovered.get(metric, []))
        if not models:
            logging.warning(f"No models discovered for metric={metric}; skipping.")
            continue

        ga_df, gna_df = compute_and_save_per_model_outputs(
            G_dir, csa_out, metric, models, datasets_order=datasets_order_csa
        )

        # Save aggregated tables across models
        ga_path = csa_out / f"{metric}_Ga_all_models.csv"
        gna_path = csa_out / f"{metric}_Gna_all_models.csv"
        ga_df.to_csv(ga_path)
        gna_df.to_csv(gna_path)
        logging.info(f"Saved aggregated Ga -> {ga_path}")
        logging.info(f"Saved aggregated Gna -> {gna_path}")

        # Compute within-study summaries from scores_dir
        scores_path = scores_dir / "all_models_scores.csv"
        if not scores_path.exists():
            logging.warning(f"within-study summaries requested but scores not found: {scores_path}")
        else:
            df = pd.read_csv(scores_path)
            df = df[(df["met"] == metric) & (df["src"] == df["trg"])].copy()
            if df.empty:
                logging.warning(f"No within-study records for metric={metric}")
            else:
                grouped = df.groupby(["model", "src"]).agg(
                    mean_splits=("value", "mean"),
                    std_splits=("value", "std")
                ).reset_index()
                # Mean pivot (models x src) using within-study order (size-based)
                df_mean = grouped.pivot(index="model", columns="src", values="mean_splits").reindex(
                    columns=[d for d in datasets_order_within if d in grouped["src"].unique()]
                )
                # Std pivot (models x src)
                df_std = grouped.pivot(index="model", columns="src", values="std_splits").reindex(
                    columns=[d for d in datasets_order_within if d in grouped["src"].unique()]
                )

                # Add means across datasets and across models
                df_mean.loc["Mean across datasets"] = df_mean.mean(axis=0)
                df_mean["Mean across models"] = df_mean.mean(axis=1)
                df_mean.loc["Mean across datasets", "Mean across models"] = pd.NA

                df_std.loc["Mean across datasets"] = df_std.mean(axis=0)
                df_std["Mean across models"] = df_std.mean(axis=1)
                df_std.loc["Mean across datasets", "Mean across models"] = pd.NA

                # Round to 2 decimal places as requested
                df_mean = df_mean.round(2)
                df_std = df_std.round(2)

                within_study_dir = csa_out / "within_study"
                within_study_dir.mkdir(parents=True, exist_ok=True)
                mean_path = within_study_dir / f"{metric}_mean_within_study_all_models.csv"
                std_path = within_study_dir / f"{metric}_std_within_study_all_models.csv"
                df_mean.to_csv(mean_path)
                df_std.to_csv(std_path)
                logging.info(f"Saved within-study means -> {mean_path}")
                logging.info(f"Saved within-study stds -> {std_path}")

    print(f"\n✅ Finished {Path(__file__).name}!")


if __name__ == "__main__":
    main()

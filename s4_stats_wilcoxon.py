"""
Step 4: Wilcoxon statistical tests over model pairs per (src, trg)
==================================================================

Reads scores from outputs/scores/all_models_scores.csv and computes pairwise
Wilcoxon tests (two-sided) for a selected metric (default: r2), across all
source-target pairs, saving both the full results and per-(src,trg) subsets.

Outputs (under --outdir outputs/s4_stats by default):
- reviewer3_comment1/wilcoxon_tests_r2_all_combos.csv
- reviewer3_comment1/all_wilcoxon_tests/wilcoxon_tests_r2_<src>_<trg>_combos.csv
"""

import argparse
from pathlib import Path
import logging
import pandas as pd
import numpy as np
from itertools import combinations
from scipy.stats import wilcoxon

from postprocess_utils import setup_logging
from utils import model_name_mapping


def compute_wilcoxon(scores: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Compute Wilcoxon tests over model pairs per (src, trg)"""
    df = scores.copy()
    r2_df = df[df['met'] == metric][['src', 'trg', 'model', 'split', 'value']].copy()
    r2_df['model'] = r2_df['model'].map(model_name_mapping)

    models = r2_df['model'].unique()
    src_datasets = r2_df['src'].unique()
    trg_datasets = r2_df['trg'].unique()

    model_pairwise_pairs = list(combinations(models, 2))
    alpha = 0.05 / len(model_pairwise_pairs) if len(model_pairwise_pairs) > 0 else 0.05

    results = []
    for src in src_datasets:
        for trg in trg_datasets:
            pair_df = r2_df[(r2_df['src'] == src) & (r2_df['trg'] == trg)]
            if pair_df.empty:
                continue
            for model1, model2 in model_pairwise_pairs:
                scores1 = pair_df[pair_df['model'] == model1]['value'].values
                scores2 = pair_df[pair_df['model'] == model2]['value'].values
                if len(scores1) == 10 and len(scores2) == 10:
                    try:
                        stat, p = wilcoxon(scores1, scores2, alternative='two-sided')
                        mean_diff = np.mean(scores1 - scores2)
                        results.append({
                            'src': src,
                            'trg': trg,
                            'model1': model1,
                            'model2': model2,
                            'median_model1': np.median(scores1),
                            'median_model2': np.median(scores2),
                            'mean_r2_diff': mean_diff,
                            'p_value': p,
                            'significant': p < alpha
                        })
                    except Exception as e:
                        logging.warning(f"Wilcoxon failed for {model1} vs {model2} on {src}->{trg}: {e}")
                else:
                    logging.info(f"Skipping {model1} vs {model2} on {src}->{trg}: {len(scores1)} vs {len(scores2)} splits")
    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser(description="Step 4: Wilcoxon statistical tests over model pairs.")
    parser.add_argument(
        "--scores_dir",
        type=str,
        default="./outputs/s1_scores",
        help="Directory with all_models_scores.csv"
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="./outputs/s4_stats",
        help="Output directory root for stats (default: ./outputs/s4_stats)"
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="r2",
        help="Metric to test (default: r2)"
    )
    args = parser.parse_args()

    scores_dir = Path(args.scores_dir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    logs_dir = Path("./logs")
    logs_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(log_file=str(logs_dir / "s4_stats_wilcoxon.log"))

    scores_path = scores_dir / "all_models_scores.csv"
    if not scores_path.exists():
        raise FileNotFoundError(f"Scores not found: {scores_path}")

    scores = pd.read_csv(scores_path)
    logging.info(f"Computing Wilcoxon tests for metric: {args.metric}")
    res_df = compute_wilcoxon(scores, args.metric)

    # Save full results
    rdir = outdir / "reviewer3_comment1"
    rdir.mkdir(parents=True, exist_ok=True)
    all_path = rdir / f"wilcoxon_tests_{args.metric}_all_combos.csv"
    res_df.to_csv(all_path, index=False)
    logging.info(f"Saved: {all_path}")

    # Save per (src,trg)
    per_dir = rdir / "all_wilcoxon_tests"
    per_dir.mkdir(parents=True, exist_ok=True)
    for (src, trg), group in res_df.groupby(["src", "trg"]):
        group.to_csv(per_dir / f"wilcoxon_tests_{args.metric}_{src}_{trg}_combos.csv", index=False)

    print(f"\n✅ Finished {Path(__file__).name}!")


if __name__ == "__main__":
    main()

# overlap_vs_performance.py
# -------------------------------------------------------------
# Purpose:
#   Compute simple, interpretable dataset similarity metrics based on
#   DRUG overlap, and compare them to cross-dataset performance (G).
#
#   - Builds per-dataset DRUG sets from one representative prediction CSV per TARGET.
#   - Produces:
#       * drug_overlap_count (|Di ∩ Dj|)
#       * drug_overlap_jaccard (|Di ∩ Dj| / |Di ∪ Dj|)  [symmetric similarity]
#       * drug_directional_coverage (|Ds ∩ Dt| / |Dt|)  [rows=Source, cols=Target]
#   - Compares directional coverage to G(s,t) off-diagonals via Spearman correlation,
#     overall and within same-assay subset.
#   - Saves heatmaps and CSVs.
#
# Notes:
#   * Focuses on DRUG overlap only (can extend later to CELL or SAMPLE overlap).
#   * Assumes prediction CSVs in preds_dir named as:
#       <SRC>_<TRG>_split_<SPLITID>_<MODEL>.csv
#   * Assumes each file with TARGET=T contains predictions for ALL cell–drug pairs in T,
#     except files where SRC==TRG (those typically contain only test splits and are not used).
#   * Requires the column 'improve_chem_id' in the chosen representative CSVs.
#
# Usage example:
#   python overlap_vs_performance.py \
#       --preds_dir /path/to/test_preds \
#       --g_csv /path/to/G_matrix.csv \
#       --outdir /path/to/out
#
# -------------------------------------------------------------

from __future__ import annotations

import argparse
import re
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

filepath = Path(os.path.abspath(''))

results_outdir = filepath / 'results_for_paper_revision_2'
os.makedirs(results_outdir, exist_ok=True)

# ----------------------------
# CONFIG (edit as needed)
# ----------------------------

# Preferred fixed dataset display order (subset will be used based on availability)
DATASET_ORDER = ['CCLE', 'CTRPv2', 'GDSCv1', 'GDSCv2', 'gCSI']

# Known default representative files: <SRC>_<TRG>_split_0_<MODEL>.csv
# (Change MODEL if needed; function will fallback if not found)
DEFAULT_REP_FILES = {
    'CTRPv2': 'CCLE_CTRPv2_split_0_deepcdr.csv',
    'gCSI':   'CCLE_gCSI_split_0_deepcdr.csv',
    'GDSCv1': 'CCLE_GDSCv1_split_0_deepcdr.csv',
    'GDSCv2': 'CCLE_GDSCv2_split_0_deepcdr.csv',
    'CCLE':   'CTRPv2_CCLE_split_0_deepcdr.csv',  # ensure SRC != TRG to cover full CCLE
}

# Required column for drug overlap
REQUIRED_DRUG_COL = 'improve_chem_id'

# Assay mapping for optional stratified correlation (edit to match your manuscript)
ASSAY_MAP = {
    'CCLE': 'CTG',      # CellTiter-Glo
    'CTRPv2': 'CTG',
    'gCSI': 'CTG',
    'GDSCv1': 'Syto60',
    'GDSCv2': 'CTG',
}


# ----------------------------
# Filename parsing helpers
# ----------------------------

FILENAME_RE = re.compile(
    r'^(?P<src>[^_]+)_(?P<trg>[^_]+)_split_(?P<split>\d+)_(?P<model>[^.]+)\.csv$'
)


def parse_filename(path: Path) -> Optional[dict]:
    """Parse a prediction CSV filename into components or return None if pattern does not match."""
    m = FILENAME_RE.match(path.name)
    return m.groupdict() if m else None


def choose_representative_file(preds_dir: Path, target: str, default_map: Dict[str, str]) -> Path:
    """
    Choose one CSV whose TARGET == target and ideally SRC != TRG (to ensure full coverage).
    Preference order:
      (1) default_map[target] if it exists
      (2) any '*_<target>_split_0_*' with SRC != TRG
      (3) any '*_<target>_split_*_*' with SRC != TRG
      (4) any '*_<target>_split_*_*' (fallback, may be SRC==TRG if nothing else)
    Deterministic ties broken lexicographically.
    """
    # (1) Preferred hard-coded representative
    preferred = preds_dir / default_map.get(target, "")
    if preferred.name and preferred.exists():
        return preferred

    # (2)-(4): search patterns
    def pick(pattern: str, prefer_src_neq_trg: bool) -> Optional[Path]:
        matches = sorted(preds_dir.glob(pattern))
        if not matches:
            return None
        if prefer_src_neq_trg:
            for p in matches:
                info = parse_filename(p)
                if info and info['src'] != info['trg']:
                    return p
            return None
        return matches[0]

    # Prefer split==0 and SRC != TRG
    p = pick(f"*_{target}_split_0_*.csv", prefer_src_neq_trg=True)
    if p: return p
    # Any split and SRC != TRG
    p = pick(f"*_{target}_split_*_*.csv", prefer_src_neq_trg=True)
    if p: return p
    # Any split (fallback)
    p = pick(f"*_{target}_split_*_*.csv", prefer_src_neq_trg=False)
    if p: return p

    raise FileNotFoundError(f"No prediction CSV found for target '{target}' in {preds_dir}")


# ----------------------------
# Loading DRUG sets
# ----------------------------

def load_target_drugs(csv_path: Path, required_col: str = REQUIRED_DRUG_COL) -> Set[str]:
    """
    Load the set of unique drug IDs from a representative TARGET CSV.
    Fails with a clear message if the required column is missing.
    """
    # breakpoint()
    # Validate header first for a friendly error
    header = pd.read_csv(csv_path, nrows=0).columns.tolist()
    if required_col not in header:
        raise ValueError(
            f"File {csv_path} does not contain required column '{required_col}'. "
            f"Columns found: {header}"
        )
    # Read only the required column
    series = pd.read_csv(csv_path, usecols=[required_col])[required_col].astype(str)
    return set(series.unique())


def build_drug_sets(
    preds_dir: Path,
    dataset_order: List[str] = DATASET_ORDER,
    default_map: Dict[str, str] = DEFAULT_REP_FILES
    ) -> Dict[str, Set[str]]:
    """
    Discover which TARGET datasets exist in preds_dir and build a dict of
    {dataset: set(drugs)}. Only datasets discovered as TARGETs are included.
    """
    # Discover TARGET datasets from filenames
    targets_found = set()
    for p in preds_dir.glob("*.csv"):
        info = parse_filename(p)
        if info:
            targets_found.add(info['trg'])
    if not targets_found:
        raise RuntimeError(f"No valid prediction files found in: {preds_dir}")

    # Respect the preferred order for the subset that exists
    targets = [d for d in dataset_order if d in targets_found]

    drug_sets: Dict[str, Set[str]] = {}
    for trg in targets:
        rep = choose_representative_file(preds_dir, trg, default_map)
        ds = load_target_drugs(rep, required_col=REQUIRED_DRUG_COL)
        drug_sets[trg] = ds
        print(f"[{trg}] using {rep.name} | unique drugs: {len(ds)}")

    return drug_sets


# ----------------------------
# Overlap matrix computations
# ----------------------------

def compute_drug_overlap_matrices(
    drug_sets: Dict[str, Set[str]],
    dataset_order: Optional[List[str]] = None
    ) -> Dict[str, pd.DataFrame]:
    """
    Given per-dataset drug sets, compute:
      - drug_overlap_count (|Di ∩ Dj|) [symmetric]
      - drug_overlap_jaccard (|Di ∩ Dj| / |Di ∪ Dj|) [symmetric]
      - drug_directional_coverage (|Ds ∩ Dt| / |Dt|) [rows=Source, cols=Target]
    Returns dict of DataFrames with aligned index/columns.
    """
    # breakpoint()
    if dataset_order is None:
        idx = list(drug_sets.keys())
    else:
        idx = [d for d in dataset_order if d in drug_sets]

    n = len(idx)
    count = pd.DataFrame(0, index=idx, columns=idx, dtype=int)
    jacc = pd.DataFrame(0.0, index=idx, columns=idx, dtype=float)
    covg = pd.DataFrame(0.0, index=idx, columns=idx, dtype=float)  # directional S (rows) -> T (cols)

    for i, si in enumerate(idx):
        Di = drug_sets[si] # unique drugs in dataset si
        for j, sj in enumerate(idx):
            Dj = drug_sets[sj] # unique drugs in dataset sj
            inter = len(Di & Dj) # intersection
            union = len(Di | Dj) if (Di or Dj) else 0 # union (handle empty sets)
            count.loc[si, sj] = inter
            jacc.loc[si, sj] = (inter / union) if union > 0 else 0.0
            covg.loc[si, sj] = (inter / len(Dj)) if len(Dj) > 0 else 0.0  # S->T

    return {
        "drug_overlap_count": count,
        "drug_overlap_jaccard": jacc,
        "drug_directional_coverage": covg,
    }


# ----------------------------
# Performance matrix utilities
# ----------------------------

def load_performance_matrix(g_csv: Path) -> pd.DataFrame:
    """
    Load a single G matrix from CSV (rows=sources, cols=targets).
    The function will attempt to coerce to float.
    """
    # breakpoint()
    G = pd.read_csv(g_csv, index_col=0)
    # Try to coerce values to float in case they are strings
    G = G.apply(pd.to_numeric, errors='coerce')
    return G


def flatten_offdiag(A: pd.DataFrame, B: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[Tuple[str, str]]]:
    """
    Align A and B on rows/cols, return off-diagonal entries as flattened arrays
    and the corresponding (source, target) labels.
    """
    # breakpoint()
    A2, B2 = A.align(B, join='inner', axis=0)
    A2, B2 = A2.align(B2, join='inner', axis=1)
    sources = list(A2.index)
    targets = list(A2.columns)

    xs, ys, labels = [], [], []
    for s in sources:
        for t in targets:
            if s == t:
                continue  # off-diagonal only
            a = A2.loc[s, t]
            b = B2.loc[s, t]
            if pd.notna(a) and pd.notna(b):
                xs.append(float(a))
                ys.append(float(b))
                labels.append((s, t))
    return np.array(xs), np.array(ys), labels


def filter_same_assay_pairs(labels: List[Tuple[str, str]], assay_map: Dict[str, str]) -> List[bool]:
    """
    Return a boolean mask indicating (s,t) pairs where assay_map[s] == assay_map[t].
    Unknown datasets default to False (not same assay).
    """
    # breakpoint()
    mask = []
    for s, t in labels:
        same = (assay_map.get(s) is not None) and (assay_map.get(s) == assay_map.get(t))
        mask.append(bool(same))
    return mask


# ----------------------------
# Plotting (matplotlib only)
# ----------------------------

def plot_heatmap(
    df: pd.DataFrame,
    title: str,
    out_png: Path,
    vmin=None,
    vmax=None,
    cmap='viridis',
    fmt='{:.2f}'):
    """Simple annotated heatmap for small matrices."""
    fig, ax = plt.subplots(figsize=(7, 5.5))
    im = ax.imshow(df.values, aspect='auto', vmin=vmin, vmax=vmax, cmap=cmap)
    ax.set_xticks(range(len(df.columns))); ax.set_xticklabels(list(df.columns), rotation=45, ha='right')
    ax.set_yticks(range(len(df.index))); ax.set_yticklabels(list(df.index))
    ax.set_xlabel("Target")
    ax.set_ylabel("Source")
    ax.set_title(title)

    # annotate each cell
    for i in range(len(df.index)):
        for j in range(len(df.columns)):
            val = df.iat[i, j]
            try:
                text = fmt.format(val)
            except Exception:
                text = str(val)
            ax.text(j, i, text, ha='center', va='center', fontsize=8, color='white' if (im.norm(val) > 0.5) else 'black')

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(title, rotation=270, labelpad=12)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()
    return True


def plot_scatter(
    x: np.ndarray,
    y: np.ndarray,
    labels: List[Tuple[str, str]],
    assay_map: Dict[str, str],
    title: str,
    out_png: Path,
    show_assay_split: bool = False
    ):
    """Scatter x vs y for off-diagonal (s,t) pairs, colored by assay concordance."""    
    fig, ax = plt.subplots(figsize=(6.5, 5.2))

    if show_assay_split:
        # Split by assay and show both
        mask_same = np.array(filter_same_assay_pairs(labels, assay_map))
        mask_diff = ~mask_same
        ax.scatter(x[mask_same], y[mask_same], alpha=0.8, label='Same assay', s=40)
        ax.scatter(x[mask_diff], y[mask_diff], alpha=0.8, label='Different assay', s=40)
        ax.legend(loc='lower right', frameon=True)
    else:
        # Show all points in one color
        ax.scatter(x, y, alpha=0.8, s=40, label='All pairs')

    ax.set_xlabel("Directional drug coverage (S→T) = |Ds ∩ Dt| / |Dt|")
    ax.set_ylabel("Cross-dataset performance G(S,T)")
    ax.set_title(title)

    # Compute correlations
    rho_all, p_all = spearmanr(x, y, nan_policy='omit')
    txt = f"Spearman ρ (all pairs) = {rho_all:.2f} (p={p_all:.3g})"
    
    if show_assay_split:
        # Add same-assay correlation if requested
        mask_same = np.array(filter_same_assay_pairs(labels, assay_map))
        if np.any(mask_same):
            rho_same, p_same = spearmanr(x[mask_same], y[mask_same], nan_policy='omit')
            txt += f"\nSpearman ρ (same assay) = {rho_same:.2f} (p={p_same:.3g})"

    ax.text(0.02, 0.98, txt, transform=ax.transAxes, ha='left', va='top', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7, edgecolor='none'))

    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()
    return True


# ----------------------------
# Main CLI
# ----------------------------

def main():
    parser = argparse.ArgumentParser(description="Drug overlap vs performance (G)")
    # parser.add_argument('--preds_dir_name', type=str, default='preds_dir',
    #     help='Directory with prediction CSVs (<SRC>_<TRG>_split_<ID>_<MODEL>.csv)')
    # parser.add_argument('--g_csv', type=str,
    #     help='CSV file for the G matrix (rows=sources, cols=targets)')
    parser.add_argument(
        '--outdir_name', type=str,
        default='reviewer2_comment1',
        help='Output directory for CSVs and plots')
    args = parser.parse_args()

    # breakpoint()
    # Directory with prediction CSVs (<SRC>_<TRG>_split_<ID>_<MODEL>.csv)
    # preds_dir = filepath / Path(args.preds_dir_name)
    preds_dir = filepath / 'test_preds'

    # Directory with G performance matrices
    # g_csv = Path(args.g_csv)
    # splits_averaged = filepath / 'splits_averaged'
    g_dir = filepath / 'results_for_paper'
    model_name = 'lgbm'
    g_csv = g_dir / f'{model_name}_r2_G_mean.csv'

    outdir = results_outdir / Path(args.outdir_name)
    outdir.mkdir(parents=True, exist_ok=True)

    # 1) Build DRUG sets per dataset (from representative TARGET files)
    drug_sets = build_drug_sets(preds_dir, dataset_order=DATASET_ORDER, default_map=DEFAULT_REP_FILES)

    # 2) Compute overlap matrices
    mats = compute_drug_overlap_matrices(drug_sets, dataset_order=DATASET_ORDER)
    drug_count = mats["drug_overlap_count"]
    drug_jacc = mats["drug_overlap_jaccard"]
    drug_covg = mats["drug_directional_coverage"]

    # Save matrices
    drug_count.to_csv(outdir / "drug_overlap_count.csv")
    drug_jacc.to_csv(outdir / "drug_overlap_jaccard.csv")
    drug_covg.to_csv(outdir / "drug_directional_coverage.csv")

    # 3) Load performance matrix G and align
    G = load_performance_matrix(g_csv)

    # Ensure all matrices share the same rows/cols subset
    for name, df in [("count", drug_count), ("jaccard", drug_jacc), ("coverage", drug_covg)]:
        mats[name] = df

    # 4) Plots: coverage heatmap and G heatmap (side-by-side inspection)
    # Align coverage to G for clean plotting
    count_aligned, G_aligned = drug_count.align(G, join='inner', axis=0)
    count_aligned, G_aligned = count_aligned.align(G_aligned, join='inner', axis=1)

    jacc_aligned, _ = drug_jacc.align(G_aligned, join='inner', axis=0)
    jacc_aligned, _ = jacc_aligned.align(G_aligned, join='inner', axis=1)

    covg_aligned, _ = drug_covg.align(G_aligned, join='inner', axis=0)
    covg_aligned, _ = covg_aligned.align(G_aligned, join='inner', axis=1)

    # Count heatmap
    plot_heatmap(
        count_aligned, "Drug overlap count |Di ∩ Dj|",
        outdir / "heatmap_drug_overlap_count.png",
        cmap='YlOrRd',  # YlOrRd good for counts
        fmt='{:.0f}'    # integers for counts
    )

    # Jaccard heatmap
    plot_heatmap(
        jacc_aligned, "Drug overlap Jaccard |Di ∩ Dj| / |Di ∪ Dj|",
        outdir / "heatmap_drug_overlap_jaccard.png",
        cmap='YlOrBr',  # Different colormap from count/coverage
        vmin=0, vmax=1  # Jaccard is always [0,1]
    )

    # Directional coverage heatmap
    plot_heatmap(
        covg_aligned, "Directional drug coverage (S→T)",
        outdir / "heatmap_directional_coverage.png",
        cmap='Oranges', # 'Reds', 'YlOrBr',
        vmin=0, vmax=1  # Coverage is always [0,1]
        )

    # G performance heatmap
    plot_heatmap(
        G_aligned, "Performance G(S,T)",
        outdir / "heatmap_G.png",
        cmap='Blues' # 'Blues', 'YlGnBu',
        )

    # 5) Scatter + Spearman correlation (off-diagonals)
    x, y, labels = flatten_offdiag(covg_aligned, G_aligned)

    # Plot with all pairs (no assay split)
    plot_scatter(
        x, y, labels, ASSAY_MAP,
        "Coverage vs Performance (all pairs)",
        outdir / "scatter_coverage_vs_G_all.png",
        show_assay_split=False
    )

    # Plot with assay split
    plot_scatter(
        x, y, labels, ASSAY_MAP,
        "Coverage vs Performance (by assay)",
        outdir / "scatter_coverage_vs_G_by_assay.png",
        show_assay_split=True
    )

    # 6) Print quick summary to console
    rho_all, p_all = spearmanr(x, y, nan_policy='omit')
    same_mask = np.array(filter_same_assay_pairs(labels, ASSAY_MAP))
    if np.any(same_mask):
        rho_same, p_same = spearmanr(x[same_mask], y[same_mask], nan_policy='omit')
    else:
        rho_same, p_same = np.nan, np.nan

    print("Saved matrices and plots to:", outdir)
    print(f"Off-diagonal Spearman ρ (all pairs): {rho_all:.3f} (p={p_all:.3g})")
    if not np.isnan(rho_same):
        print(f"Off-diagonal Spearman ρ (same-assay pairs): {rho_same:.3f} (p={p_same:.3g})")

    return True


if __name__ == "__main__":
    # Allow running as a script directly in this environment
    # but also let users import functions for testing/extension.
    main()

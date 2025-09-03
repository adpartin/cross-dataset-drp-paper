# stage7_overlap.py
# -------------------------------------------------------------
# Purpose:
#   Compute simple, interpretable dataset similarity metrics based on
#   DRUG and CELL overlap, and compare directional coverage to cross-dataset
#   performance (G).
#
#   Per entity (drug, cell):
#       * overlap_count (|Ei ∩ Ej|)
#       * overlap_jaccard (|Ei ∩ Ej| / |Ei ∪ Ej|)  [symmetric similarity]
#       * directional_coverage (|Es ∩ Et| / |Et|)  [rows=Source, cols=Target]
#   Compare directional coverage to G(s,t) off-diagonals via Spearman correlation,
#   overall and within same-assay subset. Saves heatmaps and CSVs.
#
# Assumptions:
#   * Prediction CSVs in preds_dir named:
#       <SRC>_<TRG>_split_<SPLITID>_<MODEL>.csv
#   * Each file with TARGET=T contains predictions for ALL cell–drug pairs in T,
#     except files where SRC==TRG (those typically contain only test splits; avoided).
#   * Required columns:
#       - 'improve_chem_id'  for drug overlap
#       - 'improve_sample_id' for cell overlap
#
# Usage (example):
#   python stage7_overlap.py --outdir_name reviewer2_comment1
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

DATASET_ORDER = ['CCLE', 'CTRPv2', 'GDSCv1', 'GDSCv2', 'gCSI']

DEFAULT_REP_FILES = {
    'CTRPv2': 'CCLE_CTRPv2_split_0_deepcdr.csv',
    'gCSI':   'CCLE_gCSI_split_0_deepcdr.csv',
    'GDSCv1': 'CCLE_GDSCv1_split_0_deepcdr.csv',
    'GDSCv2': 'CCLE_GDSCv2_split_0_deepcdr.csv',
    'CCLE':   'CTRPv2_CCLE_split_0_deepcdr.csv',  # ensure SRC != TRG to cover full CCLE
}

# Required columns
REQUIRED_DRUG_COL = 'improve_chem_id'
REQUIRED_CELL_COL = 'improve_sample_id'

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
    """
    preferred = preds_dir / default_map.get(target, "")
    if preferred.name and preferred.exists():
        return preferred

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

    p = pick(f"*_{target}_split_0_*.csv", prefer_src_neq_trg=True)
    if p: return p
    p = pick(f"*_{target}_split_*_*.csv", prefer_src_neq_trg=True)
    if p: return p
    p = pick(f"*_{target}_split_*_*.csv", prefer_src_neq_trg=False)
    if p: return p

    raise FileNotFoundError(f"No prediction CSV found for target '{target}' in {preds_dir}")

# ----------------------------
# Loading entity sets (DRUG/CELL)
# ----------------------------

def load_unique_ids(csv_path: Path, required_col: str) -> Set[str]:
    """Load unique values from a single column; fail fast if column missing."""
    header = pd.read_csv(csv_path, nrows=0).columns.tolist()
    if required_col not in header:
        raise ValueError(
            f"File {csv_path} missing required column '{required_col}'. "
            f"Columns found: {header}"
        )
    series = pd.read_csv(csv_path, usecols=[required_col])[required_col].astype(str)
    return set(series.unique())

def build_entity_sets(
    preds_dir: Path,
    required_col: str,
    dataset_order: List[str] = DATASET_ORDER,
    default_map: Dict[str, str] = DEFAULT_REP_FILES,
    log_label: str = "entity"
) -> Dict[str, Set[str]]:
    """
    Discover TARGET datasets in preds_dir and build {dataset: set(ids)} for the given column.
    Uses one representative TARGET CSV per dataset (SRC != TRG preferred).
    """
    targets_found = set()
    for p in preds_dir.glob("*.csv"):
        info = parse_filename(p)
        if info:
            targets_found.add(info['trg'])
    if not targets_found:
        raise RuntimeError(f"No valid prediction files found in: {preds_dir}")

    targets = [d for d in dataset_order if d in targets_found]

    entity_sets: Dict[str, Set[str]] = {}
    for trg in targets:
        rep = choose_representative_file(preds_dir, trg, default_map)
        ids = load_unique_ids(rep, required_col=required_col)
        entity_sets[trg] = ids
        print(f"[{trg}] using {rep.name} | unique {log_label}s: {len(ids)}")

    return entity_sets

# ----------------------------
# Overlap matrix computations
# ----------------------------

def compute_overlap_matrices(
    entity_sets: Dict[str, Set[str]],
    dataset_order: Optional[List[str]] = None
) -> Dict[str, pd.DataFrame]:
    """
    Given per-dataset sets, compute:
      - overlap_count (|Ei ∩ Ej|) [symmetric]
      - overlap_jaccard (|Ei ∩ Ej| / |Ei ∪ Ej|) [symmetric]
      - directional_coverage (|Es ∩ Et| / |Et|) [rows=Source, cols=Target]
    """
    idx = [d for d in (dataset_order or list(entity_sets.keys())) if d in entity_sets]

    count = pd.DataFrame(0, index=idx, columns=idx, dtype=int)
    jacc  = pd.DataFrame(0.0, index=idx, columns=idx, dtype=float)
    covg  = pd.DataFrame(0.0, index=idx, columns=idx, dtype=float)

    for si in idx:
        Ei = entity_sets[si]
        for sj in idx:
            Ej = entity_sets[sj]
            inter = len(Ei & Ej)
            union = len(Ei | Ej) if (Ei or Ej) else 0
            count.loc[si, sj] = inter
            jacc.loc[si, sj]  = (inter / union) if union > 0 else 0.0
            covg.loc[si, sj]  = (inter / len(Ej)) if len(Ej) > 0 else 0.0  # S->T

    return {"overlap_count": count, "overlap_jaccard": jacc, "directional_coverage": covg}

# ----------------------------
# Performance matrix utilities
# ----------------------------

def load_performance_matrix(g_csv: Path) -> pd.DataFrame:
    G = pd.read_csv(g_csv, index_col=0)
    return G.apply(pd.to_numeric, errors='coerce')

def flatten_offdiag(A: pd.DataFrame, B: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[Tuple[str, str]]]:
    A2, B2 = A.align(B, join='inner', axis=0)
    A2, B2 = A2.align(B2, join='inner', axis=1)
    xs, ys, labels = [], [], []
    for s in A2.index:
        for t in A2.columns:
            if s == t:
                continue
            a = A2.loc[s, t]; b = B2.loc[s, t]
            if pd.notna(a) and pd.notna(b):
                xs.append(float(a)); ys.append(float(b)); labels.append((s, t))
    return np.array(xs), np.array(ys), labels

def filter_same_assay_pairs(labels: List[Tuple[str, str]], assay_map: Dict[str, str]) -> List[bool]:
    mask = []
    for s, t in labels:
        same = (assay_map.get(s) is not None) and (assay_map.get(s) == assay_map.get(t))
        mask.append(bool(same))
    return mask

# ----------------------------
# Plotting (matplotlib only)
# ----------------------------

# def plot_heatmap(df: pd.DataFrame, title: str, out_png: Path, vmin=None, vmax=None, cmap='viridis', fmt='{:.2f}'):
#     fig, ax = plt.subplots(figsize=(7, 5.5))
#     im = ax.imshow(df.values, aspect='auto', vmin=vmin, vmax=vmax, cmap=cmap)
#     ax.set_xticks(range(len(df.columns))); ax.set_xticklabels(list(df.columns), rotation=45, ha='right')
#     ax.set_yticks(range(len(df.index))); ax.set_yticklabels(list(df.index))
#     ax.set_xlabel("Target"); ax.set_ylabel("Source"); ax.set_title(title)
#     for i in range(len(df.index)):
#         for j in range(len(df.columns)):
#             val = df.iat[i, j]
#             try: text = fmt.format(val)
#             except Exception: text = str(val)
#             ax.text(j, i, text, ha='center', va='center', fontsize=8,
#                     color='white' if (im.norm(val) > 0.5) else 'black')
#     cbar = plt.colorbar(im, ax=ax); cbar.set_label(title, rotation=270, labelpad=12)
#     plt.tight_layout(); plt.savefig(out_png, dpi=300); plt.close(); return True
def plot_heatmap(
    df: pd.DataFrame,
    title: str,
    out_png: Path,
    vmin=None,
    vmax=None,
    cmap='viridis',
    fmt='{:.2f}'
):
    """Simple annotated heatmap for small matrices (rows=sources, cols=targets)."""
    fig, ax = plt.subplots(figsize=(7, 5.5))
    im = ax.imshow(df.values, aspect='auto', vmin=vmin, vmax=vmax, cmap=cmap)

    ax.set_xticks(range(len(df.columns)))
    ax.set_xticklabels(list(df.columns), rotation=45, ha='right')
    ax.set_yticks(range(len(df.index)))
    ax.set_yticklabels(list(df.index))

    ax.set_xlabel("Target")
    ax.set_ylabel("Source")
    ax.set_title(title)

    # Annotate each cell with a value string; choose text color by normalized intensity
    for i in range(len(df.index)):
        for j in range(len(df.columns)):
            val = df.iat[i, j]
            try:
                text = fmt.format(val)
            except Exception:
                text = str(val)
            ax.text(
                j, i, text,
                ha='center', va='center', fontsize=8,
                color='white' if (im.norm(val) > 0.5) else 'black'
            )

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(title, rotation=270, labelpad=12)

    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()
    return True


# def plot_scatter(x, y, labels, assay_map, title, out_png, xlabel):
#     fig, ax = plt.subplots(figsize=(6.5, 5.2))
#     mask_same = np.array(filter_same_assay_pairs(labels, assay_map))
#     mask_diff = ~mask_same
#     ax.scatter(x[mask_same], y[mask_same], alpha=0.8, label='Same assay', s=40)
#     ax.scatter(x[mask_diff], y[mask_diff], alpha=0.8, label='Different assay', s=40)
#     ax.set_xlabel(xlabel); ax.set_ylabel("Cross-dataset performance G(S,T)"); ax.set_title(title)
#     rho_all, p_all = spearmanr(x, y, nan_policy='omit')
#     rho_same, p_same = (np.nan, np.nan)
#     if np.any(mask_same):
#         rho_same, p_same = spearmanr(x[mask_same], y[mask_same], nan_policy='omit')
#     txt = f"Spearman ρ (all) = {rho_all:.2f} (p={p_all:.3g})"
#     if not np.isnan(rho_same): txt += f"\nSpearman ρ (same assay) = {rho_same:.2f} (p={p_same:.3g})"
#     ax.text(0.02, 0.98, txt, transform=ax.transAxes, ha='left', va='top', fontsize=9,
#             bbox=dict(boxstyle='round', facecolor='white', alpha=0.7, edgecolor='none'))
#     ax.legend(loc='lower right', frameon=True); ax.grid(True, alpha=0.3)
#     plt.tight_layout(); plt.savefig(out_png, dpi=300); plt.close(); return True
def plot_scatter(
    x: np.ndarray,
    y: np.ndarray,
    labels: List[Tuple[str, str]],
    assay_map: Dict[str, str],
    title: str,
    out_png: Path,
    xlabel: str,
    show_assay_split: bool = False,
    # Optional colors to keep visuals consistent with heatmaps:
    # - use color_all for single-color mode,
    # - use color_same / color_diff for by-assay mode.
    color_all: Optional[str] = None,
    color_same: Optional[str] = None,
    color_diff: Optional[str] = None
):
    """Scatter x vs y for off-diagonal (S,T) pairs, with optional assay-based coloring."""
    fig, ax = plt.subplots(figsize=(6.5, 5.2))

    if show_assay_split:
        mask_same = np.array(filter_same_assay_pairs(labels, assay_map))
        mask_diff = ~mask_same

        ax.scatter(
            x[mask_same], y[mask_same],
            alpha=0.85, s=46, label='Same assay',
            color=(color_same or 'tab:blue')
        )
        ax.scatter(
            x[mask_diff], y[mask_diff],
            alpha=0.75, s=46, label='Different assay',
            color=(color_diff or '0.55')  # neutral gray by default
        )
        ax.legend(loc='lower right', frameon=True)
    else:
        ax.scatter(
            x, y,
            alpha=0.85, s=46, label='All pairs',
            color=(color_all or 'tab:blue')
        )

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Cross-dataset performance G(S,T)")
    ax.set_title(title)

    # Correlations (Spearman)
    rho_all, p_all = spearmanr(x, y, nan_policy='omit')
    txt = f"Spearman ρ (all) = {rho_all:.2f} (p={p_all:.3g})"

    if show_assay_split:
        mask_same = np.array(filter_same_assay_pairs(labels, assay_map))
        if np.any(mask_same):
            rho_same, p_same = spearmanr(x[mask_same], y[mask_same], nan_policy='omit')
            txt += f"\nSpearman ρ (same assay) = {rho_same:.2f} (p={p_same:.3g})"

    ax.text(
        0.02, 0.98, txt,
        transform=ax.transAxes, ha='left', va='top', fontsize=9,
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.7, edgecolor='none')
    )

    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()
    return True


# ----------------------------
# Main CLI
# ----------------------------

def main():
    parser = argparse.ArgumentParser(description="Drug & cell overlap vs performance (G)")
    parser.add_argument('--outdir_name', type=str, default='reviewer2_comment1',
                        help='Output directory for CSVs and plots')
    args = parser.parse_args()

    preds_dir = filepath / 'test_preds'
    g_dir = filepath / 'results_for_paper'
    model_name = 'lgbm'
    g_csv = g_dir / f'{model_name}_r2_G_mean.csv'

    outdir = results_outdir / Path(args.outdir_name)
    outdir.mkdir(parents=True, exist_ok=True)

    # === DRUGS ===
    drug_sets = build_entity_sets(preds_dir, REQUIRED_DRUG_COL, DATASET_ORDER, DEFAULT_REP_FILES, log_label="drug")
    drug_mats = compute_overlap_matrices(drug_sets, DATASET_ORDER)
    drug_count = drug_mats["overlap_count"]
    drug_jacc  = drug_mats["overlap_jaccard"]
    drug_covg  = drug_mats["directional_coverage"]
    # Save
    drug_count.to_csv(outdir / "drug_overlap_count.csv")
    drug_jacc.to_csv(outdir / "drug_overlap_jaccard.csv")
    drug_covg.to_csv(outdir / "drug_directional_coverage.csv")

    # === CELLS ===
    cell_sets = build_entity_sets(preds_dir, REQUIRED_CELL_COL, DATASET_ORDER, DEFAULT_REP_FILES, log_label="cell")
    cell_mats = compute_overlap_matrices(cell_sets, DATASET_ORDER)
    cell_count = cell_mats["overlap_count"]
    cell_jacc  = cell_mats["overlap_jaccard"]
    cell_covg  = cell_mats["directional_coverage"]
    # Save
    cell_count.to_csv(outdir / "cell_overlap_count.csv")
    cell_jacc.to_csv(outdir / "cell_overlap_jaccard.csv")
    cell_covg.to_csv(outdir / "cell_directional_coverage.csv")

    # === Load G and align ===
    G = load_performance_matrix(g_csv)

    # Align for plotting
    drug_covg_aligned, G_aligned = drug_covg.align(G, join='inner', axis=0)
    drug_covg_aligned, G_aligned = drug_covg_aligned.align(G_aligned, join='inner', axis=1)

    cell_covg_aligned, _ = cell_covg.align(G_aligned, join='inner', axis=0)
    cell_covg_aligned, _ = cell_covg_aligned.align(G_aligned, join='inner', axis=1)

    # Optional: counts/jaccard heatmaps (drug & cell)
    # (kept since they’re cheap and occasionally informative)
    dcount_aligned, _ = drug_count.align(G_aligned, join='inner', axis=0)
    dcount_aligned, _ = dcount_aligned.align(G_aligned, join='inner', axis=1)
    djacc_aligned, _  = drug_jacc.align(G_aligned, join='inner', axis=0)
    djacc_aligned, _  = djacc_aligned.align(G_aligned, join='inner', axis=1)

    ccount_aligned, _ = cell_count.align(G_aligned, join='inner', axis=0)
    ccount_aligned, _ = ccount_aligned.align(G_aligned, join='inner', axis=1)
    cjacc_aligned, _  = cell_jacc.align(G_aligned, join='inner', axis=0)
    cjacc_aligned, _  = cjacc_aligned.align(G_aligned, join='inner', axis=1)

    # === Heatmaps ===
    plot_heatmap(dcount_aligned, "Drug overlap count |Di ∩ Dj|",
                 outdir / "heatmap_drug_overlap_count.png",
                 cmap='YlOrRd', fmt='{:.0f}')
    plot_heatmap(djacc_aligned, "Drug overlap Jaccard |Di ∩ Dj| / |Di ∪ Dj|",
                 outdir / "heatmap_drug_overlap_jaccard.png",
                 cmap='YlOrBr', vmin=0, vmax=1)
    plot_heatmap(drug_covg_aligned, "Directional drug coverage (S→T)",
                 outdir / "heatmap_directional_drug_coverage.png",
                 cmap='Oranges', vmin=0, vmax=1)

    plot_heatmap(ccount_aligned, "Cell overlap count |Ci ∩ Cj|",
                 outdir / "heatmap_cell_overlap_count.png",
                 cmap='PuBuGn', fmt='{:.0f}')
    plot_heatmap(cjacc_aligned, "Cell overlap Jaccard |Ci ∩ Cj| / |Ci ∪ Cj|",
                 outdir / "heatmap_cell_overlap_jaccard.png",
                 cmap='Greens', vmin=0, vmax=1)
    plot_heatmap(cell_covg_aligned, "Directional cell coverage (S→T)",
                 outdir / "heatmap_directional_cell_coverage.png",
                 cmap='Purples', vmin=0, vmax=1)

    # === Scatter + Spearman (off-diagonals) ===
    # Drug
    x_d, y_d, labels_d = flatten_offdiag(drug_covg_aligned, G_aligned)
    # plot_scatter(
    #     x_d, y_d, labels_d, ASSAY_MAP,
    #     "Drug coverage vs Performance",
    #     outdir / "scatter_drug_coverage_vs_G_all.png",
    #     xlabel="Directional drug coverage (S→T) = |Ds ∩ Dt| / |Dt|",
    #     show_assay_split=False
    # )
    # plot_scatter(
    #     x_d, y_d, labels_d, ASSAY_MAP,
    #     "Drug coverage vs Performance",
    #     outdir / "scatter_drug_coverage_vs_G_by_assay.png",
    #     xlabel="Directional drug coverage (S→T) = |Ds ∩ Dt| / |Dt|",
    #     show_assay_split=True
    # )
    # single-color
    plot_scatter(
        x_d, y_d, labels_d, ASSAY_MAP,
        "Drug coverage vs Performance",
        outdir / "scatter_drug_coverage_vs_G_all.png",
        xlabel="Directional drug coverage (S→T) = |Ds ∩ Dt| / |Dt|",
        show_assay_split=False,
        color_all='tab:orange'
    )
    # assay-split
    plot_scatter(
        x_d, y_d, labels_d, ASSAY_MAP,
        "Drug coverage vs Performance (by assay)",
        outdir / "scatter_drug_coverage_vs_G_by_assay.png",
        xlabel="Directional drug coverage (S→T) = |Ds ∩ Dt| / |Dt|",
        show_assay_split=True,
        color_same='tab:orange',
        color_diff='0.55'   # medium gray
    )
    rho_d_all, p_d_all = spearmanr(x_d, y_d, nan_policy='omit')
    same_mask_d = np.array(filter_same_assay_pairs(labels_d, ASSAY_MAP))
    if np.any(same_mask_d):
        rho_d_same, p_d_same = spearmanr(x_d[same_mask_d], y_d[same_mask_d], nan_policy='omit')
    else:
        rho_d_same, p_d_same = np.nan, np.nan

    # Cell
    x_c, y_c, labels_c = flatten_offdiag(cell_covg_aligned, G_aligned)
    # plot_scatter(
    #     x_c, y_c, labels_c, ASSAY_MAP,
    #     "Cell coverage vs Performance", outdir / "scatter_cell_coverage_vs_G_all.png",
    #     xlabel="Directional cell coverage (S→T) = |Cs ∩ Ct| / |Ct|",
    #     show_assay_split=False
    # )
    # plot_scatter(
    #     x_c, y_c, labels_c, ASSAY_MAP,
    #     "Cell coverage vs Performance",
    #     outdir / "scatter_cell_coverage_vs_G_by_assay.png",
    #     xlabel="Directional cell coverage (S→T) = |Cs ∩ Ct| / |Ct|",
    #     show_assay_split=True
    # )
    # single-color
    plot_scatter(
        x_c, y_c, labels_c, ASSAY_MAP,
        "Cell coverage vs Performance",
        outdir / "scatter_cell_coverage_vs_G_all.png",
        xlabel="Directional cell coverage (S→T) = |Cs ∩ Ct| / |Ct|",
        show_assay_split=False,
        color_all='tab:purple'
    )
    # assay-split
    plot_scatter(
        x_c, y_c, labels_c, ASSAY_MAP,
        "Cell coverage vs Performance (by assay)",
        outdir / "scatter_cell_coverage_vs_G_by_assay.png",
        xlabel="Directional cell coverage (S→T) = |Cs ∩ Ct| / |Ct|",
        show_assay_split=True,
        color_same='tab:purple',
        color_diff='0.55'
    )
    rho_c_all, p_c_all = spearmanr(x_c, y_c, nan_policy='omit')
    same_mask_c = np.array(filter_same_assay_pairs(labels_c, ASSAY_MAP))
    if np.any(same_mask_c):
        rho_c_same, p_c_same = spearmanr(x_c[same_mask_c], y_c[same_mask_c], nan_policy='omit')
    else:
        rho_c_same, p_c_same = np.nan, np.nan

    # === Console summary ===
    print("Saved matrices and plots to:", outdir)
    print(f"[DRUG] Off-diagonal Spearman ρ (all): {rho_d_all:.3f} (p={p_d_all:.3g})")
    if not np.isnan(rho_d_same):
        print(f"[DRUG] Off-diagonal Spearman ρ (same-assay): {rho_d_same:.3f} (p={p_d_same:.3g})")
    print(f"[CELL] Off-diagonal Spearman ρ (all):  {rho_c_all:.3f} (p={p_c_all:.3g})")
    if not np.isnan(rho_c_same):
        print(f"[CELL] Off-diagonal Spearman ρ (same-assay): {rho_c_same:.3f} (p={p_c_same:.3g})")

    return True

if __name__ == "__main__":
    main()

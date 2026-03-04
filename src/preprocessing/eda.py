"""
eda.py — Exploratory Data Analysis for CICIDS2017
==================================================
Usage:
    python src/preprocessing/eda.py --input data/raw/<filename>.csv

Outputs:
    - Console summary statistics
    - outputs/figures/class_distribution.png
    - outputs/figures/correlation_heatmap.png
    - outputs/figures/missing_values.png
    - outputs/reports/eda_summary.csv
"""

import argparse
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────
FIGURES_DIR = "outputs/figures"
REPORTS_DIR = "outputs/reports"
LABEL_COL   = " Label"          # CICIDS2017 label column (note leading space)
SAMPLE_SIZE = 100_000           # Subsample for memory-safe EDA on 8GB RAM

os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)


# ── Helpers ───────────────────────────────────────────────────────────────────
def load_data(filepath: str, sample: int = None) -> pd.DataFrame:
    """Load CSV with optional random sampling for memory efficiency."""
    print(f"[INFO] Loading: {filepath}")
    df = pd.read_csv(filepath, encoding="utf-8", low_memory=False)
    df.columns = df.columns.str.strip()           # Strip whitespace from col names
    print(f"[INFO] Raw shape: {df.shape}")

    if sample and len(df) > sample:
        df = df.sample(n=sample, random_state=42)
        print(f"[INFO] Sampled down to {sample} rows for EDA")

    return df


def summarise_labels(df: pd.DataFrame) -> None:
    """Print and plot class distribution."""
    print("\n── Label Distribution ──────────────────────────────────────")
    counts = df["Label"].value_counts()
    pct    = df["Label"].value_counts(normalize=True) * 100
    summary = pd.DataFrame({"Count": counts, "Percentage (%)": pct.round(2)})
    print(summary.to_string())

    # Save to report
    summary.to_csv(f"{REPORTS_DIR}/class_distribution.csv")

    # Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    counts.plot(kind="bar", color="steelblue", edgecolor="black", ax=ax)
    ax.set_title("Class Distribution — CICIDS2017", fontsize=14, fontweight="bold")
    ax.set_xlabel("Traffic Label")
    ax.set_ylabel("Sample Count")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/class_distribution.png", dpi=150)
    plt.close()
    print(f"[SAVED] {FIGURES_DIR}/class_distribution.png")


def summarise_missing(df: pd.DataFrame) -> None:
    """Detect NaN and Inf values — common issues in CICIDS2017."""
    print("\n── Missing & Infinite Values ───────────────────────────────")

    # Replace inf with NaN for unified handling
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)

    if missing.empty:
        print("[OK] No missing or infinite values detected.")
    else:
        print(f"[WARN] {len(missing)} columns contain NaN/Inf:")
        print(missing.to_string())

        # Plot
        fig, ax = plt.subplots(figsize=(10, 4))
        missing.plot(kind="bar", color="tomato", edgecolor="black", ax=ax)
        ax.set_title("Missing Values per Feature", fontsize=13, fontweight="bold")
        ax.set_ylabel("Missing Count")
        plt.tight_layout()
        plt.savefig(f"{FIGURES_DIR}/missing_values.png", dpi=150)
        plt.close()
        print(f"[SAVED] {FIGURES_DIR}/missing_values.png")

    missing.to_csv(f"{REPORTS_DIR}/missing_values.csv", header=["Missing Count"])


def summarise_dtypes(df: pd.DataFrame) -> None:
    """Print dtype summary and flag non-numeric columns."""
    print("\n── Data Types ──────────────────────────────────────────────")
    non_numeric = df.select_dtypes(exclude=[np.number]).columns.tolist()
    numeric     = df.select_dtypes(include=[np.number]).columns.tolist()
    print(f"  Numeric features  : {len(numeric)}")
    print(f"  Non-numeric cols  : {non_numeric}")


def plot_correlation(df: pd.DataFrame, top_n: int = 20) -> None:
    """
    Plot correlation heatmap of top N most-variant numeric features.
    Using top_n avoids an unreadable 80-feature heatmap.
    """
    print(f"\n── Correlation Heatmap (top {top_n} features) ──────────────")

    numeric_df = df.select_dtypes(include=[np.number])

    # Select top_n features by variance (most informative for heatmap)
    top_features = numeric_df.var().nlargest(top_n).index
    corr = numeric_df[top_features].corr()

    fig, ax = plt.subplots(figsize=(14, 10))
    sns.heatmap(
        corr,
        annot=False,
        cmap="coolwarm",
        center=0,
        linewidths=0.3,
        ax=ax
    )
    ax.set_title(f"Feature Correlation Heatmap (Top {top_n} by Variance)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/correlation_heatmap.png", dpi=150)
    plt.close()
    print(f"[SAVED] {FIGURES_DIR}/correlation_heatmap.png")


def basic_stats(df: pd.DataFrame) -> None:
    """Save descriptive statistics to CSV."""
    print("\n── Descriptive Statistics ──────────────────────────────────")
    stats = df.describe(include=[np.number]).T
    stats.to_csv(f"{REPORTS_DIR}/eda_summary.csv")
    print(f"[SAVED] {REPORTS_DIR}/eda_summary.csv")
    print(stats[["mean", "std", "min", "max"]].head(10).to_string())


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="EDA for CICIDS2017")
    parser.add_argument("--input",  type=str, required=True, help="Path to CSV file")
    parser.add_argument("--sample", type=int, default=SAMPLE_SIZE,
                        help=f"Max rows to load (default: {SAMPLE_SIZE})")
    args = parser.parse_args()

    df = load_data(args.input, sample=args.sample)

    # Standardise label column name
    if "Label" not in df.columns and " Label" in df.columns:
        df.rename(columns={" Label": "Label"}, inplace=True)

    summarise_dtypes(df)
    summarise_missing(df)
    summarise_labels(df)
    basic_stats(df)
    plot_correlation(df)

    print("\n[DONE] EDA complete. Check outputs/figures/ and outputs/reports/")


if __name__ == "__main__":
    main()

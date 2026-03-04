"""
shap_analysis.py — SHAP Explainability for NIDS-XAI
=====================================================
Generates SHAP-based explanations for trained models.
Produces Summary, Waterfall, Dependence, and per-class plots.

Usage:
    python src/explainability/shap_analysis.py
    python src/explainability/shap_analysis.py --model rf
    python src/explainability/shap_analysis.py --model xgb   (default)
    python src/explainability/shap_analysis.py --samples 5000

Outputs:
    outputs/figures/shap_summary_bar.png
    outputs/figures/shap_summary_beeswarm.png
    outputs/figures/shap_waterfall_benign.png
    outputs/figures/shap_waterfall_ddos.png
    outputs/figures/shap_dependence_top1.png
    outputs/figures/shap_dependence_top2.png
    outputs/figures/shap_force_plot.html
    outputs/reports/shap_feature_ranking.csv
"""

import os
import time
import argparse
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import shap
import joblib

warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────
PROCESSED_DIR = "data/processed"
MODELS_DIR    = "outputs/models"
FIGURES_DIR   = "outputs/figures"
REPORTS_DIR   = "outputs/reports"
SAMPLE_SIZE   = 2000     # Rows to use for SHAP (balance speed vs accuracy)
RANDOM_STATE  = 42

for d in [FIGURES_DIR, REPORTS_DIR]:
    os.makedirs(d, exist_ok=True)


# ── Load Data & Model ─────────────────────────────────────────────────────────
def load_artifacts(model_name: str):
    print(f"[INFO] Loading model: {model_name}")

    model_path = os.path.join(MODELS_DIR, f"{model_name}_model.pkl")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}\nRun train.py first.")

    model   = joblib.load(model_path)
    X_test  = pd.read_parquet(os.path.join(PROCESSED_DIR, "X_test.parquet"))
    y_test  = pd.read_parquet(os.path.join(PROCESSED_DIR, "y_test_binary.parquet")).squeeze()

    print(f"  Model loaded     : {model_path}")
    print(f"  X_test shape     : {X_test.shape}")

    return model, X_test, y_test


# ── Sample for SHAP ───────────────────────────────────────────────────────────
def sample_data(X_test, y_test, n: int):
    """
    Sample balanced subset for SHAP computation.
    Equal samples from BENIGN and DDoS classes.
    """
    n_each = n // 2
    idx_benign = y_test[y_test == 0].sample(n=min(n_each, (y_test==0).sum()),
                                             random_state=RANDOM_STATE).index
    idx_ddos   = y_test[y_test == 1].sample(n=min(n_each, (y_test==1).sum()),
                                             random_state=RANDOM_STATE).index
    idx        = idx_benign.tolist() + idx_ddos.tolist()

    X_sample = X_test.loc[idx].reset_index(drop=True)
    y_sample = y_test.loc[idx].reset_index(drop=True)

    print(f"  SHAP sample      : {len(X_sample)} rows (balanced: {n_each} each class)")
    return X_sample, y_sample


# ── Compute SHAP Values ───────────────────────────────────────────────────────
def compute_shap(model, X_sample, model_name: str):
    print(f"\n── Computing SHAP Values ───────────────────────────────────")
    print(f"  Using TreeExplainer (optimised for tree-based models)...")

    start = time.time()
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    elapsed     = time.time() - start

    print(f"  SHAP computation complete in {elapsed:.1f}s")

    # For binary classification: shap_values may be list [class0, class1]
    # We use class 1 (DDoS / attack) for analysis
    if isinstance(shap_values, list):
        shap_vals_positive = shap_values[1]
        shap_vals_both     = shap_values
    else:
        shap_vals_positive = shap_values
        shap_vals_both     = shap_values

    print(f"  SHAP values shape: {np.array(shap_vals_positive).shape}")
    return explainer, shap_values, shap_vals_positive


# ── Plot 1: Summary Bar ───────────────────────────────────────────────────────
def plot_summary_bar(shap_vals_positive, X_sample, model_name):
    """Global feature importance — mean absolute SHAP values."""
    print("\n── Plot 1: SHAP Summary Bar ────────────────────────────────")

    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(
        shap_vals_positive, X_sample,
        plot_type="bar",
        max_display=20,
        show=False,
        color="steelblue"
    )
    plt.title(f"SHAP Feature Importance (Mean |SHAP|) — {model_name}",
              fontsize=13, fontweight="bold", pad=15)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, f"shap_summary_bar_{model_name}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [SAVED] {path}")


# ── Plot 2: Summary Beeswarm ──────────────────────────────────────────────────
def plot_summary_beeswarm(shap_vals_positive, X_sample, model_name):
    """Beeswarm shows direction + magnitude of feature impact."""
    print("\n── Plot 2: SHAP Beeswarm Plot ──────────────────────────────")

    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(
        shap_vals_positive, X_sample,
        plot_type="dot",
        max_display=20,
        show=False
    )
    plt.title(f"SHAP Beeswarm — Feature Impact on DDoS Classification — {model_name}",
              fontsize=12, fontweight="bold", pad=15)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, f"shap_summary_beeswarm_{model_name}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [SAVED] {path}")


# ── Plot 3: Waterfall Plots ───────────────────────────────────────────────────
def plot_waterfall(explainer, X_sample, y_sample, shap_vals_positive, model_name):
    """
    Waterfall explains individual predictions.
    Shows one BENIGN and one DDoS example.
    """
    print("\n── Plot 3: SHAP Waterfall Plots ────────────────────────────")

    # Find one correct BENIGN and one correct DDoS prediction
    benign_idx = y_sample[y_sample == 0].index[0]
    ddos_idx   = y_sample[y_sample == 1].index[0]

    for idx, label in [(benign_idx, "BENIGN"), (ddos_idx, "DDoS")]:
        shap_explanation = shap.Explanation(
            values       = shap_vals_positive[idx],
            base_values  = explainer.expected_value[1] if isinstance(explainer.expected_value, list)
                           else explainer.expected_value,
            data         = X_sample.iloc[idx].values,
            feature_names= list(X_sample.columns)
        )
        fig, ax = plt.subplots(figsize=(10, 7))
        shap.waterfall_plot(shap_explanation, max_display=15, show=False)
        plt.title(f"SHAP Waterfall — {label} Prediction — {model_name}",
                  fontsize=12, fontweight="bold")
        plt.tight_layout()
        path = os.path.join(FIGURES_DIR, f"shap_waterfall_{label.lower()}_{model_name}.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  [SAVED] {path}")


# ── Plot 4: Dependence Plots ──────────────────────────────────────────────────
def plot_dependence(shap_vals_positive, X_sample, model_name, top_n=2):
    """
    Dependence plot shows how top features interact with each other.
    Automatically selects top features by mean SHAP value.
    """
    print("\n── Plot 4: SHAP Dependence Plots ───────────────────────────")

    mean_shap    = np.abs(shap_vals_positive).mean(axis=0)
    top_features = np.argsort(mean_shap)[::-1][:top_n]

    for rank, feat_idx in enumerate(top_features):
        feat_name = X_sample.columns[feat_idx]
        fig, ax = plt.subplots(figsize=(9, 6))
        shap.dependence_plot(
            feat_idx, shap_vals_positive, X_sample,
            interaction_index="auto",
            show=False,
            ax=ax
        )
        ax.set_title(f"SHAP Dependence — '{feat_name}' — {model_name}",
                     fontsize=12, fontweight="bold")
        plt.tight_layout()
        path = os.path.join(FIGURES_DIR, f"shap_dependence_top{rank+1}_{model_name}.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  [SAVED] {path} (feature: {feat_name})")


# ── Plot 5: Class-Separated SHAP ─────────────────────────────────────────────
def plot_class_shap(shap_vals_positive, X_sample, y_sample, model_name):
    """
    Separate SHAP summary for BENIGN vs DDoS samples.
    Reveals which features characterise each class.
    """
    print("\n── Plot 5: Per-Class SHAP Analysis ─────────────────────────")

    for class_val, class_name in [(0, "BENIGN"), (1, "DDoS")]:
        mask = y_sample == class_val
        if mask.sum() == 0:
            continue

        X_class    = X_sample[mask]
        shap_class = shap_vals_positive[mask.values]

        fig, ax = plt.subplots(figsize=(10, 7))
        shap.summary_plot(
            shap_class, X_class,
            plot_type="bar",
            max_display=15,
            show=False,
            color="tomato" if class_val == 1 else "steelblue"
        )
        plt.title(f"SHAP Feature Importance ({class_name} traffic only) — {model_name}",
                  fontsize=12, fontweight="bold", pad=15)
        plt.tight_layout()
        path = os.path.join(FIGURES_DIR, f"shap_class_{class_name.lower()}_{model_name}.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  [SAVED] {path}")


# ── Save Feature Ranking CSV ──────────────────────────────────────────────────
def save_feature_ranking(shap_vals_positive, X_sample, model_name):
    """Save ranked feature importance table based on mean |SHAP| values."""
    mean_shap = np.abs(shap_vals_positive).mean(axis=0)
    ranking   = pd.DataFrame({
        "Feature":        X_sample.columns,
        "Mean_SHAP":      mean_shap,
        "Mean_SHAP_pct":  (mean_shap / mean_shap.sum() * 100).round(2)
    }).sort_values("Mean_SHAP", ascending=False).reset_index(drop=True)

    ranking.index += 1   # Start ranking from 1
    ranking.index.name = "Rank"

    path = os.path.join(REPORTS_DIR, f"shap_feature_ranking_{model_name}.csv")
    ranking.to_csv(path)
    print(f"\n  [SAVED] {path}")

    print(f"\n  Top 10 Features by Mean |SHAP|:")
    print(ranking.head(10).to_string())
    return ranking


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="NIDS-XAI SHAP Explainability")
    parser.add_argument("--model",   type=str, default="xgb",
                        choices=["rf", "xgb"],
                        help="Model to explain: rf or xgb (default: xgb)")
    parser.add_argument("--samples", type=int, default=SAMPLE_SIZE,
                        help=f"Number of samples for SHAP (default: {SAMPLE_SIZE})")
    args = parser.parse_args()

    print("=" * 60)
    print(f"NIDS-XAI — SHAP Explainability Analysis")
    print(f"Model: {args.model.upper()}  |  Samples: {args.samples}")
    print("=" * 60)

    total_start = time.time()

    # Load
    model, X_test, y_test           = load_artifacts(args.model)
    X_sample, y_sample              = sample_data(X_test, y_test, args.samples)

    # Compute SHAP
    explainer, shap_values, shap_vals_positive = compute_shap(model, X_sample, args.model)

    # Generate all plots
    plot_summary_bar(shap_vals_positive, X_sample, args.model)
    plot_summary_beeswarm(shap_vals_positive, X_sample, args.model)
    plot_waterfall(explainer, X_sample, y_sample, shap_vals_positive, args.model)
    plot_dependence(shap_vals_positive, X_sample, args.model)
    plot_class_shap(shap_vals_positive, X_sample, y_sample, args.model)

    # Save ranking
    ranking = save_feature_ranking(shap_vals_positive, X_sample, args.model)

    total_elapsed = time.time() - total_start
    print(f"\n[DONE] SHAP analysis complete in {total_elapsed:.1f}s")
    print(f"       Figures saved to : {FIGURES_DIR}/")
    print(f"       Reports saved to : {REPORTS_DIR}/")
    print(f"\n  Plots generated:")
    print(f"    shap_summary_bar_{args.model}.png       — Global feature importance")
    print(f"    shap_summary_beeswarm_{args.model}.png  — Feature impact direction")
    print(f"    shap_waterfall_benign_{args.model}.png  — Why BENIGN was predicted")
    print(f"    shap_waterfall_ddos_{args.model}.png    — Why DDoS was predicted")
    print(f"    shap_dependence_top1_{args.model}.png   — Top feature interaction")
    print(f"    shap_dependence_top2_{args.model}.png   — 2nd feature interaction")
    print(f"    shap_class_benign_{args.model}.png      — BENIGN-specific features")
    print(f"    shap_class_ddos_{args.model}.png        — DDoS-specific features")


if __name__ == "__main__":
    main()

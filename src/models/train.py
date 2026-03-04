"""
train.py — Model Training & Evaluation for NIDS-XAI
=====================================================
Trains Random Forest and/or XGBoost on preprocessed CICIDS2017 data.
Evaluates using F1-score, Precision-Recall AUC, and Confusion Matrix.
Saves trained models and evaluation plots.

Usage:
    python src/models/train.py --model rf           # Train Random Forest
    python src/models/train.py --model xgb          # Train XGBoost
    python src/models/train.py --model both         # Train both (default)
    python src/models/train.py --model xgb --tune   # XGBoost + hyperparameter tuning

Outputs:
    outputs/models/rf_model.pkl
    outputs/models/xgb_model.pkl
    outputs/figures/confusion_matrix_rf.png
    outputs/figures/confusion_matrix_xgb.png
    outputs/figures/precision_recall_rf.png
    outputs/figures/precision_recall_xgb.png
    outputs/figures/feature_importance_rf.png
    outputs/figures/feature_importance_xgb.png
    outputs/reports/classification_report_rf.txt
    outputs/reports/classification_report_xgb.txt
    outputs/reports/model_comparison.csv
"""

import os
import time
import argparse
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix,
    precision_recall_curve, average_precision_score,
    f1_score, roc_auc_score
)
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────
PROCESSED_DIR = "data/processed"
MODELS_DIR    = "outputs/models"
FIGURES_DIR   = "outputs/figures"
REPORTS_DIR   = "outputs/reports"
RANDOM_STATE  = 42

for d in [MODELS_DIR, FIGURES_DIR, REPORTS_DIR]:
    os.makedirs(d, exist_ok=True)

# ── Load Data ─────────────────────────────────────────────────────────────────
def load_data():
    print("[INFO] Loading preprocessed data from parquet...")
    X_train = pd.read_parquet(os.path.join(PROCESSED_DIR, "X_train.parquet"))
    X_test  = pd.read_parquet(os.path.join(PROCESSED_DIR, "X_test.parquet"))
    y_train = pd.read_parquet(os.path.join(PROCESSED_DIR, "y_train_binary.parquet")).squeeze()
    y_test  = pd.read_parquet(os.path.join(PROCESSED_DIR, "y_test_binary.parquet")).squeeze()

    print(f"  X_train : {X_train.shape}")
    print(f"  X_test  : {X_test.shape}")
    print(f"  y_train : {y_train.value_counts().to_dict()}")
    print(f"  y_test  : {y_test.value_counts().to_dict()}")
    return X_train, X_test, y_train, y_test


# ── Plot: Confusion Matrix ────────────────────────────────────────────────────
def plot_confusion_matrix(y_test, y_pred, model_name):
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["BENIGN", "DDoS"],
        yticklabels=["BENIGN", "DDoS"],
        ax=ax, linewidths=0.5
    )
    ax.set_title(f"Confusion Matrix — {model_name}", fontsize=13, fontweight="bold")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, f"confusion_matrix_{model_name.lower().replace(' ', '_')}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  [SAVED] {path}")


# ── Plot: Precision-Recall Curve ──────────────────────────────────────────────
def plot_precision_recall(y_test, y_proba, model_name):
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    ap = average_precision_score(y_test, y_proba)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(recall, precision, color="steelblue", lw=2,
            label=f"PR Curve (AP = {ap:.4f})")
    ax.fill_between(recall, precision, alpha=0.1, color="steelblue")
    ax.axhline(y=y_test.mean(), color="gray", linestyle="--",
               label=f"Baseline (prevalence = {y_test.mean():.2f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision-Recall Curve — {model_name}", fontsize=13, fontweight="bold")
    ax.legend(loc="lower left")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, f"precision_recall_{model_name.lower().replace(' ', '_')}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  [SAVED] {path}")
    return ap


# ── Plot: Feature Importance ──────────────────────────────────────────────────
def plot_feature_importance(model, feature_names, model_name, top_n=20):
    importances = model.feature_importances_
    indices     = np.argsort(importances)[::-1][:top_n]
    top_features = [feature_names[i] for i in indices]
    top_values   = importances[indices]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(range(top_n), top_values[::-1], color="steelblue", edgecolor="black", linewidth=0.5)
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(top_features[::-1], fontsize=9)
    ax.set_xlabel("Feature Importance Score")
    ax.set_title(f"Top {top_n} Feature Importances — {model_name}", fontsize=13, fontweight="bold")

    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, top_values[::-1])):
        ax.text(val + 0.0005, bar.get_y() + bar.get_height()/2,
                f"{val:.4f}", va="center", fontsize=8)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, f"feature_importance_{model_name.lower().replace(' ', '_')}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  [SAVED] {path}")


# ── Evaluate Model ────────────────────────────────────────────────────────────
def evaluate_model(model, X_test, y_test, model_name):
    print(f"\n── Evaluation: {model_name} ────────────────────────────────")

    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    f1      = f1_score(y_test, y_pred, average="macro")
    f1_bin  = f1_score(y_test, y_pred, average="binary")
    roc_auc = roc_auc_score(y_test, y_proba)
    ap      = average_precision_score(y_test, y_proba)

    print(f"  F1-Score (Macro)   : {f1:.4f}")
    print(f"  F1-Score (Binary)  : {f1_bin:.4f}")
    print(f"  ROC-AUC            : {roc_auc:.4f}")
    print(f"  PR-AUC (Avg Prec)  : {ap:.4f}")
    print(f"\n  Classification Report:")
    report = classification_report(y_test, y_pred, target_names=["BENIGN", "DDoS"])
    print(report)

    # Save classification report
    report_path = os.path.join(REPORTS_DIR, f"classification_report_{model_name.lower().replace(' ', '_')}.txt")
    with open(report_path, "w") as f:
        f.write(f"Model: {model_name}\n")
        f.write("=" * 50 + "\n")
        f.write(report)
        f.write(f"\nF1-Score (Macro)  : {f1:.4f}\n")
        f.write(f"ROC-AUC           : {roc_auc:.4f}\n")
        f.write(f"PR-AUC            : {ap:.4f}\n")
    print(f"  [SAVED] {report_path}")

    # Generate plots
    plot_confusion_matrix(y_test, y_pred, model_name)
    plot_precision_recall(y_test, y_proba, model_name)
    plot_feature_importance(model, list(X_test.columns), model_name)

    return {"Model": model_name, "F1_Macro": f1, "F1_Binary": f1_bin, "ROC_AUC": roc_auc, "PR_AUC": ap}


# ── Train: Random Forest ──────────────────────────────────────────────────────
def train_random_forest(X_train, y_train):
    print("\n" + "=" * 60)
    print("Training: Random Forest")
    print("=" * 60)

    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features="sqrt",
        n_jobs=-1,                  # Use all CPU cores
        random_state=RANDOM_STATE,
        class_weight="balanced"     # Handles any residual imbalance
    )

    print("  Fitting Random Forest (n_estimators=200, max_depth=20)...")
    start = time.time()
    rf.fit(X_train, y_train)
    elapsed = time.time() - start
    print(f"  Training complete in {elapsed:.1f}s")

    # Save model
    model_path = os.path.join(MODELS_DIR, "rf_model.pkl")
    joblib.dump(rf, model_path)
    print(f"  [SAVED] {model_path}")

    return rf


# ── Train: XGBoost ────────────────────────────────────────────────────────────
def train_xgboost(X_train, y_train):
    print("\n" + "=" * 60)
    print("Training: XGBoost")
    print("=" * 60)

    # Calculate scale_pos_weight for any class imbalance
    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    scale_pos_weight = neg / pos
    print(f"  scale_pos_weight : {scale_pos_weight:.3f} (neg/pos ratio)")

    xgb = XGBClassifier(
        n_estimators=300,
        max_depth=8,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        use_label_encoder=False,
        eval_metric="logloss",
        n_jobs=-1,
        random_state=RANDOM_STATE,
        tree_method="hist"          # Faster CPU training
    )

    print("  Fitting XGBoost (n_estimators=300, max_depth=8)...")
    start = time.time()
    xgb.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train)],
        verbose=False
    )
    elapsed = time.time() - start
    print(f"  Training complete in {elapsed:.1f}s")

    # Save model
    model_path = os.path.join(MODELS_DIR, "xgb_model.pkl")
    joblib.dump(xgb, model_path)
    print(f"  [SAVED] {model_path}")

    return xgb


# ── Hyperparameter Tuning: XGBoost ────────────────────────────────────────────
def tune_xgboost(X_train, y_train):
    print("\n" + "=" * 60)
    print("Hyperparameter Tuning: XGBoost (RandomizedSearchCV)")
    print("=" * 60)

    param_dist = {
        "n_estimators":      [100, 200, 300, 400],
        "max_depth":         [4, 6, 8, 10],
        "learning_rate":     [0.01, 0.05, 0.1, 0.2],
        "subsample":         [0.6, 0.7, 0.8, 0.9],
        "colsample_bytree":  [0.6, 0.7, 0.8, 0.9],
        "min_child_weight":  [1, 3, 5],
        "gamma":             [0, 0.1, 0.2, 0.3],
    }

    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()

    base_xgb = XGBClassifier(
        scale_pos_weight=neg/pos,
        use_label_encoder=False,
        eval_metric="logloss",
        n_jobs=-1,
        random_state=RANDOM_STATE,
        tree_method="hist"
    )

    search = RandomizedSearchCV(
        base_xgb,
        param_distributions=param_dist,
        n_iter=20,              # 20 random combinations
        scoring="f1_macro",
        cv=3,                   # 3-fold cross-validation
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=2
    )

    print("  Running RandomizedSearchCV (n_iter=20, cv=3)...")
    print("  Estimated time: 20-45 minutes on i5-12450H...")
    start = time.time()
    search.fit(X_train, y_train)
    elapsed = time.time() - start

    print(f"\n  Tuning complete in {elapsed/60:.1f} minutes")
    print(f"  Best F1 (macro) : {search.best_score_:.4f}")
    print(f"  Best params     : {search.best_params_}")

    # Save tuned model
    model_path = os.path.join(MODELS_DIR, "xgb_tuned_model.pkl")
    joblib.dump(search.best_estimator_, model_path)
    print(f"  [SAVED] {model_path}")

    # Save best params
    params_path = os.path.join(REPORTS_DIR, "xgb_best_params.txt")
    with open(params_path, "w") as f:
        f.write(f"Best F1 (macro): {search.best_score_:.4f}\n\n")
        f.write("Best Parameters:\n")
        for k, v in search.best_params_.items():
            f.write(f"  {k}: {v}\n")
    print(f"  [SAVED] {params_path}")

    return search.best_estimator_


# ── Save Comparison Table ─────────────────────────────────────────────────────
def save_comparison(results: list):
    df = pd.DataFrame(results)
    df = df.sort_values("F1_Macro", ascending=False)
    path = os.path.join(REPORTS_DIR, "model_comparison.csv")
    df.to_csv(path, index=False)

    print("\n" + "=" * 60)
    print("Model Comparison Summary")
    print("=" * 60)
    print(df.to_string(index=False))
    print(f"\n[SAVED] {path}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="NIDS-XAI Model Training")
    parser.add_argument("--model", type=str, default="both",
                        choices=["rf", "xgb", "both"],
                        help="Model to train: rf, xgb, or both (default: both)")
    parser.add_argument("--tune", action="store_true",
                        help="Run hyperparameter tuning on XGBoost")
    args = parser.parse_args()

    print("=" * 60)
    print("NIDS-XAI — Model Training Pipeline")
    print("=" * 60)

    X_train, X_test, y_train, y_test = load_data()
    results = []

    # ── Random Forest ──
    if args.model in ["rf", "both"]:
        rf = train_random_forest(X_train, y_train)
        rf_results = evaluate_model(rf, X_test, y_test, "Random_Forest")
        results.append(rf_results)

    # ── XGBoost ──
    if args.model in ["xgb", "both"]:
        if args.tune:
            xgb = tune_xgboost(X_train, y_train)
            xgb_results = evaluate_model(xgb, X_test, y_test, "XGBoost_Tuned")
        else:
            xgb = train_xgboost(X_train, y_train)
            xgb_results = evaluate_model(xgb, X_test, y_test, "XGBoost")
        results.append(xgb_results)

    # ── Comparison Table ──
    if len(results) > 1:
        save_comparison(results)
    else:
        print(f"\n[DONE] {results[0]['Model']} — F1 Macro: {results[0]['F1_Macro']:.4f}")

    print("\n[DONE] All models trained and evaluated.")
    print(f"       Models saved to  : {MODELS_DIR}/")
    print(f"       Figures saved to : {FIGURES_DIR}/")
    print(f"       Reports saved to : {REPORTS_DIR}/")


if __name__ == "__main__":
    main()

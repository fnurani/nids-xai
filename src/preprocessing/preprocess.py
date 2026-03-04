"""
preprocess.py — Data Preprocessing Pipeline for CICIDS2017
===========================================================
Handles:
    - Infinite value replacement
    - NaN dropping
    - Constant/duplicate column removal
    - Feature scaling (StandardScaler)
    - Stratified train/test split
    - Optional SMOTE for class imbalance
    - Save as .parquet for fast I/O

Usage:
    python src/preprocessing/preprocess.py
    python src/preprocessing/preprocess.py --smote          # Enable SMOTE
    python src/preprocessing/preprocess.py --all-days       # Load all 8 CSV files

Outputs:
    data/processed/X_train.parquet
    data/processed/X_test.parquet
    data/processed/y_train.parquet
    data/processed/y_test.parquet
    outputs/reports/preprocessing_report.txt
"""

import os
import time
import argparse
import warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import joblib

warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────
RAW_DIR       = "data/raw"
PROCESSED_DIR = "data/processed"
REPORTS_DIR   = "outputs/reports"
MODELS_DIR    = "outputs/models"

LABEL_COL     = "Label"
TEST_SIZE     = 0.2
RANDOM_STATE  = 42

# Target CSV file (single-day mode)
TARGET_FILE   = "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"

# All 8 CICIDS2017 CSV files (multi-day mode)
ALL_FILES = [
    "Monday-WorkingHours.pcap_ISCX.csv",
    "Tuesday-WorkingHours.pcap_ISCX.csv",
    "Wednesday-workingHours.pcap_ISCX.csv",
    "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
    "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
    "Friday-WorkingHours-Morning.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
]

for d in [PROCESSED_DIR, REPORTS_DIR, MODELS_DIR]:
    os.makedirs(d, exist_ok=True)


# ── Step 1: Load Data ─────────────────────────────────────────────────────────
def load_data(all_days: bool = False) -> pd.DataFrame:
    """Load one or all CICIDS2017 CSV files."""
    if all_days:
        frames = []
        for fname in ALL_FILES:
            fpath = os.path.join(RAW_DIR, fname)
            if os.path.exists(fpath):
                print(f"  [LOAD] {fname}")
                df = pd.read_csv(fpath, encoding="utf-8", low_memory=False)
                df.columns = df.columns.str.strip()
                frames.append(df)
            else:
                print(f"  [SKIP] Not found: {fname}")
        df = pd.concat(frames, ignore_index=True)
        print(f"[INFO] Combined shape: {df.shape}")
    else:
        fpath = os.path.join(RAW_DIR, TARGET_FILE)
        print(f"[INFO] Loading: {fpath}")
        df = pd.read_csv(fpath, encoding="utf-8", low_memory=False)
        df.columns = df.columns.str.strip()
        print(f"[INFO] Raw shape: {df.shape}")

    return df


# ── Step 2: Clean Data ────────────────────────────────────────────────────────
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Remove inf, NaN, duplicates, and constant columns."""
    print("\n── Step 2: Cleaning ────────────────────────────────────────")
    initial_rows = len(df)

    # Replace inf/-inf with NaN
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    inf_replaced = df.isnull().sum().sum()
    print(f"  Inf values replaced with NaN : {inf_replaced}")

    # Drop NaN rows
    df.dropna(inplace=True)
    nan_dropped = initial_rows - len(df)
    print(f"  Rows dropped (NaN)           : {nan_dropped}")

    # Drop duplicate rows
    df.drop_duplicates(inplace=True)
    dup_dropped = (initial_rows - nan_dropped) - len(df)
    print(f"  Duplicate rows dropped       : {dup_dropped}")

    # Drop constant columns (zero variance — no predictive value)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    constant_cols = [c for c in numeric_cols if df[c].nunique() <= 1]
    df.drop(columns=constant_cols, inplace=True)
    print(f"  Constant columns removed     : {len(constant_cols)} {constant_cols}")

    print(f"  Final shape after cleaning   : {df.shape}")
    return df


# ── Step 3: Encode Labels ─────────────────────────────────────────────────────
def encode_labels(df: pd.DataFrame) -> tuple[pd.DataFrame, LabelEncoder, dict]:
    """
    Encode string labels to integers.
    For binary classification: BENIGN=0, all attacks=1
    For multiclass: each attack type gets unique integer
    """
    print("\n── Step 3: Label Encoding ──────────────────────────────────")

    # Binary encoding: BENIGN vs. ALL attacks
    df["Label_Binary"] = df[LABEL_COL].apply(lambda x: 0 if x.strip() == "BENIGN" else 1)

    # Multiclass encoding
    le = LabelEncoder()
    df["Label_Multi"] = le.fit_transform(df[LABEL_COL].str.strip())

    class_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    print(f"  Classes        : {list(le.classes_)}")
    print(f"  Class mapping  : {class_mapping}")

    # Save encoder for inference
    joblib.dump(le, os.path.join(MODELS_DIR, "label_encoder.pkl"))
    print(f"  [SAVED] label_encoder.pkl")

    return df, le, class_mapping


# ── Step 4: Feature Engineering ───────────────────────────────────────────────
def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Separate features from labels."""
    print("\n── Step 4: Feature Preparation ────────────────────────────")

    drop_cols = [LABEL_COL, "Label_Binary", "Label_Multi"]
    feature_cols = [c for c in df.columns if c not in drop_cols]

    X = df[feature_cols].copy()
    y_binary = df["Label_Binary"].copy()
    y_multi  = df["Label_Multi"].copy()

    print(f"  Feature count  : {X.shape[1]}")
    print(f"  Sample count   : {X.shape[0]}")
    print(f"  Binary labels  : {y_binary.value_counts().to_dict()}")

    return X, y_binary, y_multi


# ── Step 5: Train/Test Split ──────────────────────────────────────────────────
def split_data(X, y_binary, y_multi):
    """Stratified split to preserve class distribution."""
    print("\n── Step 5: Train/Test Split ────────────────────────────────")

    X_train, X_test, y_train_bin, y_test_bin, y_train_multi, y_test_multi = train_test_split(
        X, y_binary, y_multi,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_binary
    )

    print(f"  Training set   : {X_train.shape[0]} rows ({100*(1-TEST_SIZE):.0f}%)")
    print(f"  Test set       : {X_test.shape[0]} rows ({100*TEST_SIZE:.0f}%)")
    print(f"  Train labels   : {y_train_bin.value_counts().to_dict()}")
    print(f"  Test labels    : {y_test_bin.value_counts().to_dict()}")

    return X_train, X_test, y_train_bin, y_test_bin, y_train_multi, y_test_multi


# ── Step 6: Scaling ───────────────────────────────────────────────────────────
def scale_features(X_train, X_test):
    """StandardScaler fitted on training set only to prevent data leakage."""
    print("\n── Step 6: Feature Scaling (StandardScaler) ────────────────")

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index
    )

    # Save scaler for inference
    joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.pkl"))
    print(f"  Scaler fitted on training set only (no data leakage)")
    print(f"  [SAVED] scaler.pkl")

    return X_train_scaled, X_test_scaled, scaler


# ── Step 7: SMOTE (Optional) ──────────────────────────────────────────────────
def apply_smote(X_train, y_train):
    """
    Apply SMOTE to training set only.
    NOTE: For the Friday DDoS file, distribution is already balanced (56/44).
    SMOTE is more critical when training on the full multi-day dataset.
    """
    print("\n── Step 7: SMOTE Oversampling ──────────────────────────────")
    print(f"  Before SMOTE   : {y_train.value_counts().to_dict()}")

    smote = SMOTE(random_state=RANDOM_STATE, k_neighbors=5)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

    X_resampled = pd.DataFrame(X_resampled, columns=X_train.columns)
    y_resampled = pd.Series(y_resampled, name=y_train.name)

    print(f"  After SMOTE    : {y_resampled.value_counts().to_dict()}")
    print(f"  New shape      : {X_resampled.shape}")

    return X_resampled, y_resampled


# ── Step 8: Save Processed Data ───────────────────────────────────────────────
def save_data(X_train, X_test, y_train_bin, y_test_bin, y_train_multi, y_test_multi):
    """Save all splits as .parquet for fast loading in training scripts."""
    print("\n── Step 8: Saving Processed Data ───────────────────────────")

    X_train.to_parquet(os.path.join(PROCESSED_DIR, "X_train.parquet"), index=False)
    X_test.to_parquet(os.path.join(PROCESSED_DIR,  "X_test.parquet"),  index=False)

    y_train_bin.to_frame().to_parquet(os.path.join(PROCESSED_DIR,    "y_train_binary.parquet"), index=False)
    y_test_bin.to_frame().to_parquet(os.path.join(PROCESSED_DIR,     "y_test_binary.parquet"),  index=False)

    y_train_multi.to_frame().to_parquet(os.path.join(PROCESSED_DIR,  "y_train_multi.parquet"),  index=False)
    y_test_multi.to_frame().to_parquet(os.path.join(PROCESSED_DIR,   "y_test_multi.parquet"),   index=False)

    files = ["X_train.parquet", "X_test.parquet",
             "y_train_binary.parquet", "y_test_binary.parquet",
             "y_train_multi.parquet", "y_test_multi.parquet"]

    for f in files:
        size_kb = os.path.getsize(os.path.join(PROCESSED_DIR, f)) / 1024
        print(f"  [SAVED] {f:<30} ({size_kb:.1f} KB)")


# ── Step 9: Preprocessing Report ─────────────────────────────────────────────
def save_report(df_original_shape, df_clean_shape, class_mapping, smote_applied):
    """Save a text summary of preprocessing steps and outcomes."""
    report_path = os.path.join(REPORTS_DIR, "preprocessing_report.txt")

    with open(report_path, "w") as f:
        f.write("=" * 60 + "\n")
        f.write("NIDS-XAI — Preprocessing Report\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Original shape       : {df_original_shape}\n")
        f.write(f"After cleaning       : {df_clean_shape}\n")
        f.write(f"Rows removed         : {df_original_shape[0] - df_clean_shape[0]}\n")
        f.write(f"Features used        : {df_clean_shape[1] - 3}\n")  # minus 3 label cols
        f.write(f"Test size            : {TEST_SIZE * 100:.0f}%\n")
        f.write(f"SMOTE applied        : {smote_applied}\n")
        f.write(f"Scaler               : StandardScaler\n")
        f.write(f"Split strategy       : Stratified\n")
        f.write(f"Random state         : {RANDOM_STATE}\n\n")
        f.write("Class Mapping (Multiclass):\n")
        for cls, idx in class_mapping.items():
            f.write(f"  {idx} -> {cls}\n")

    print(f"\n  [SAVED] {report_path}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="CICIDS2017 Preprocessing Pipeline")
    parser.add_argument("--smote",    action="store_true", help="Apply SMOTE to training set")
    parser.add_argument("--all-days", action="store_true", help="Load all 8 CICIDS2017 CSV files")
    args = parser.parse_args()

    print("=" * 60)
    print("NIDS-XAI — Preprocessing Pipeline")
    print("=" * 60)
    start = time.time()

    # Pipeline
    df                                          = load_data(all_days=args.all_days)
    original_shape                              = df.shape
    df                                          = clean_data(df)
    clean_shape                                 = df.shape
    df, le, class_mapping                       = encode_labels(df)
    X, y_binary, y_multi                        = prepare_features(df)
    X_train, X_test, y_train_bin, y_test_bin, \
    y_train_multi, y_test_multi                 = split_data(X, y_binary, y_multi)
    X_train, X_test, scaler                     = scale_features(X_train, X_test)

    if args.smote:
        X_train, y_train_bin = apply_smote(X_train, y_train_bin)
    else:
        print("\n── Step 7: SMOTE Skipped (use --smote flag to enable) ──")

    save_data(X_train, X_test, y_train_bin, y_test_bin, y_train_multi, y_test_multi)
    save_report(original_shape, clean_shape, class_mapping, args.smote)

    elapsed = time.time() - start
    print(f"\n[DONE] Preprocessing complete in {elapsed:.1f}s")
    print(f"       Processed files saved to: {PROCESSED_DIR}/")
    print(f"       Scaler + encoder saved to: {MODELS_DIR}/")


if __name__ == "__main__":
    main()

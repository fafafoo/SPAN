# SPAN: Subgraph Progressive Attention Networks for microRNA-disease association prediction
# Copyright (c) 2025 SPAN Authors
# Licensed under the Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0).
# You may use, modify, and distribute this software for non-commercial purposes only.
# For commercial use, please contact the authors to obtain a commercial license.
# See LICENSE for details or visit https://creativecommons.org/licenses/by-nc/4.0/
"""
Step 1: Generate global prediction score matrix with calibration.

Input:  retest_aggregated.csv
        mirna_name.csv
        disease_name.csv
Output: prediction_matrix.csv          
        prediction_data_long.csv       
        prediction_matrix_minmax.csv
        prediction_matrix_percentile.csv
        prediction_matrix_platt.csv
        prediction_matrix_per_disease.csv
        prediction_matrix_labels.csv
        confidence_matrix.csv

Normalization methods (scientifically justified):
  1. Min-Max normalization on candidate pairs:
     x_norm = (x - min_positive) / (max_positive - min_positive)
     Applied only to candidate pairs (true_label=0) with score > 0.
     Known associations remain 1.0.

  2. Percentile rank normalization:
     x_rank = percentile_rank(x) among all candidate pair scores.
     Distribution-free, robust to outliers and skewness.

  3. Platt Scaling calibration:
     Fits a logistic regression (a*x + b) to map raw scores to calibrated
     probabilities using known associations as positive samples and
     zero-score candidates as negative samples. This makes scores
     interpretable as probabilities, enabling threshold-based decisions
     for biomarker discovery.

  4. Per-disease normalization:
     Independent Min-Max normalization within each disease column.
     Ensures cross-disease comparability by mapping each disease's
     candidate scores to [0, 1] based on its own score distribution.

  5. Confidence matrix (SNR-based):
     For each (miRNA, disease) pair, confidence = mean / (std + epsilon),
     where mean and std are computed across retest rounds.
     High SNR indicates stable, reliable predictions.
"""

import csv
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

BASE_DIR = Path(__file__).parent

RETEST_CSV = BASE_DIR / "retest_aggregated.csv"
MIRNA_NAME_CSV = BASE_DIR / "mirna_name.csv"
DISEASE_NAME_CSV = BASE_DIR / "disease_name.csv"
OUTPUT_CSV = BASE_DIR / "prediction_matrix.csv"
OUTPUT_LONG_CSV = BASE_DIR / "prediction_data_long.csv"


def load_names(filepath):
    names = []
    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            if row:
                names.append(row[0].strip())
    return names


def minmax_normalize(score_matrix, label_matrix):
    normalized = score_matrix.copy()
    candidate_mask = (label_matrix == 0) & (score_matrix > 0)
    positive_scores = score_matrix[candidate_mask]
    if len(positive_scores) == 0:
        return normalized
    min_val = positive_scores.min()
    max_val = positive_scores.max()
    if max_val == min_val:
        normalized[candidate_mask] = 0.5
    else:
        normalized[candidate_mask] = (positive_scores - min_val) / (max_val - min_val)
    known_mask = label_matrix == 1
    normalized[known_mask] = 1.0
    return normalized


def percentile_normalize(score_matrix, label_matrix):
    normalized = score_matrix.copy()
    candidate_mask = (label_matrix == 0) & (score_matrix > 0)
    positive_scores = score_matrix[candidate_mask]
    if len(positive_scores) == 0:
        return normalized
    sorted_scores = np.sort(positive_scores)
    ranks = np.searchsorted(sorted_scores, positive_scores, side="right") / len(sorted_scores)
    normalized[candidate_mask] = ranks
    known_mask = label_matrix == 1
    normalized[known_mask] = 1.0
    return normalized


def platt_calibrate(score_matrix, label_matrix):
    calibrated = score_matrix.copy()

    known_mask = label_matrix == 1
    candidate_mask = (label_matrix == 0) & (score_matrix > 0)
    zero_mask = (label_matrix == 0) & (score_matrix == 0)

    pos_scores = score_matrix[known_mask]
    neg_scores = score_matrix[zero_mask]

    if len(pos_scores) == 0 or len(neg_scores) == 0:
        print("  WARNING: Cannot perform Platt calibration (insufficient pos/neg samples)")
        return calibrated

    X_pos = pos_scores.reshape(-1, 1)
    X_neg = neg_scores.reshape(-1, 1)
    X_train = np.vstack([X_pos, X_neg])
    y_train = np.concatenate([np.ones(len(pos_scores)), np.zeros(len(neg_scores))])

    lr = LogisticRegression(C=1e10, solver="lbfgs", max_iter=1000)
    lr.fit(X_train, y_train)

    all_scores = score_matrix[candidate_mask | known_mask].reshape(-1, 1)
    probs = lr.predict_proba(all_scores)[:, 1]

    idx = 0
    for i in range(score_matrix.shape[0]):
        for j in range(score_matrix.shape[1]):
            if candidate_mask[i, j] or known_mask[i, j]:
                calibrated[i, j] = probs[idx]
                idx += 1

    calibrated[zero_mask] = 0.0

    print(f"  Platt Scaling: coef={lr.coef_[0,0]:.4f}, intercept={lr.intercept_[0]:.4f}")
    return calibrated


def per_disease_normalize(score_matrix, label_matrix):
    normalized = score_matrix.copy()
    n_diseases = score_matrix.shape[1]

    for j in range(n_diseases):
        col_scores = score_matrix[:, j]
        col_labels = label_matrix[:, j]
        candidate_mask = (col_labels == 0) & (col_scores > 0)
        positive_scores = col_scores[candidate_mask]

        if len(positive_scores) == 0:
            continue

        min_val = positive_scores.min()
        max_val = positive_scores.max()
        if max_val == min_val:
            normalized[candidate_mask, j] = 0.5
        else:
            normalized[candidate_mask, j] = (positive_scores - min_val) / (max_val - min_val)

        known_mask = col_labels == 1
        normalized[known_mask, j] = 1.0

    return normalized


def compute_confidence_matrix(retest_csv, mirna_names, disease_names):
    mirna_to_idx = {name: i for i, name in enumerate(mirna_names)}
    disease_to_idx = {name: i for i, name in enumerate(disease_names)}

    score_accum = {}
    score_sq_accum = {}
    count_accum = {}

    with open(retest_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            mirna_name = row["mirna_name"]
            disease_name = row["disease_name"]
            predicted_mean = float(row["predicted_mean"])

            mi = mirna_to_idx.get(mirna_name)
            di = disease_to_idx.get(disease_name)
            if mi is None or di is None:
                continue

            key = (mi, di)
            score_accum[key] = score_accum.get(key, 0.0) + predicted_mean
            score_sq_accum[key] = score_sq_accum.get(key, 0.0) + predicted_mean ** 2
            count_accum[key] = count_accum.get(key, 0) + 1

    n_mirna = len(mirna_names)
    n_disease = len(disease_names)
    confidence = np.zeros((n_mirna, n_disease), dtype=np.float64)
    mean_matrix = np.zeros((n_mirna, n_disease), dtype=np.float64)
    std_matrix = np.zeros((n_mirna, n_disease), dtype=np.float64)

    epsilon = 1e-8
    for (mi, di), cnt in count_accum.items():
        mean_val = score_accum[(mi, di)] / cnt
        var_val = score_sq_accum[(mi, di)] / cnt - mean_val ** 2
        std_val = np.sqrt(max(var_val, 0.0))
        mean_matrix[mi, di] = mean_val
        std_matrix[mi, di] = std_val
        confidence[mi, di] = mean_val / (std_val + epsilon)

    return confidence, mean_matrix, std_matrix


def main():
    mirna_names = load_names(MIRNA_NAME_CSV)
    disease_names = load_names(DISEASE_NAME_CSV)
    print(f"miRNAs: {len(mirna_names)}, Diseases: {len(disease_names)}")

    mirna_to_idx = {name: i for i, name in enumerate(mirna_names)}
    disease_to_idx = {name: i for i, name in enumerate(disease_names)}

    score_matrix = np.zeros((len(mirna_names), len(disease_names)), dtype=np.float64)
    label_matrix = np.zeros((len(mirna_names), len(disease_names)), dtype=np.int8)

    with open(RETEST_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        count = 0
        for row in reader:
            mirna_name = row["mirna_name"]
            disease_name = row["disease_name"]
            predicted_mean = float(row["predicted_mean"])
            true_label = int(row["true_label"])

            mi = mirna_to_idx.get(mirna_name)
            di = disease_to_idx.get(disease_name)

            if mi is not None and di is not None:
                score_matrix[mi, di] = predicted_mean
                label_matrix[mi, di] = true_label
                count += 1

    print(f"Filled {count} entries in prediction matrix")

    minmax_matrix = minmax_normalize(score_matrix, label_matrix)
    print("Min-Max normalization applied")

    percentile_matrix = percentile_normalize(score_matrix, label_matrix)
    print("Percentile rank normalization applied")

    print("Applying Platt Scaling calibration...")
    platt_matrix = platt_calibrate(score_matrix, label_matrix)
    print("Platt Scaling calibration applied")

    print("Applying Per-disease normalization...")
    per_disease_matrix = per_disease_normalize(score_matrix, label_matrix)
    print("Per-disease normalization applied")

    print("Computing confidence matrix (SNR-based)...")
    confidence_matrix, _, _ = compute_confidence_matrix(RETEST_CSV, mirna_names, disease_names)
    print("Confidence matrix computed")

    score_df = pd.DataFrame(score_matrix, index=mirna_names, columns=disease_names)
    minmax_df = pd.DataFrame(minmax_matrix, index=mirna_names, columns=disease_names)
    percentile_df = pd.DataFrame(percentile_matrix, index=mirna_names, columns=disease_names)
    platt_df = pd.DataFrame(platt_matrix, index=mirna_names, columns=disease_names)
    per_disease_df = pd.DataFrame(per_disease_matrix, index=mirna_names, columns=disease_names)
    label_df = pd.DataFrame(label_matrix, index=mirna_names, columns=disease_names)
    confidence_df = pd.DataFrame(confidence_matrix, index=mirna_names, columns=disease_names)

    score_df.to_csv(OUTPUT_CSV)
    print(f"Saved raw prediction matrix CSV to {OUTPUT_CSV}")

    minmax_df.to_csv(BASE_DIR / "prediction_matrix_minmax.csv")
    print("Saved Min-Max normalized matrix CSV")
    percentile_df.to_csv(BASE_DIR / "prediction_matrix_percentile.csv")
    print("Saved Percentile rank matrix CSV")
    platt_df.to_csv(BASE_DIR / "prediction_matrix_platt.csv")
    print("Saved Platt calibrated matrix CSV")
    per_disease_df.to_csv(BASE_DIR / "prediction_matrix_per_disease.csv")
    print("Saved Per-disease normalized matrix CSV")
    label_df.to_csv(BASE_DIR / "prediction_matrix_labels.csv")
    print("Saved labels matrix CSV")
    confidence_df.to_csv(BASE_DIR / "confidence_matrix.csv")
    print("Saved confidence matrix CSV")

    long_rows = []
    for i, mname in enumerate(mirna_names):
        for j, dname in enumerate(disease_names):
            if score_matrix[i, j] > 0 or label_matrix[i, j] == 1:
                long_rows.append({
                    "mirna_id": i,
                    "disease_id": j,
                    "mirna_name": mname,
                    "disease_name": dname,
                    "predicted_score": score_matrix[i, j],
                    "platt_score": platt_matrix[i, j],
                    "per_disease_score": per_disease_matrix[i, j],
                    "confidence_snr": confidence_matrix[i, j],
                    "true_label": int(label_matrix[i, j]),
                })
    long_df = pd.DataFrame(long_rows)
    long_df.to_csv(OUTPUT_LONG_CSV, index=False)
    print(f"Saved long-format prediction data to {OUTPUT_LONG_CSV} ({len(long_rows)} entries)")

    candidate_mask = (label_matrix == 0) & (score_matrix > 0)
    positive_scores = score_matrix[candidate_mask]
    print(f"\nMatrix shape: {score_matrix.shape}")
    print(f"  Known associations: {label_matrix.sum()}")
    print(f"  Candidate pairs (score>0): {candidate_mask.sum()}")
    print(f"\nRaw scores (candidate pairs with score>0):")
    print(f"  Range: [{positive_scores.min():.6f}, {positive_scores.max():.6f}]")
    print(f"  Mean: {positive_scores.mean():.6f}, Median: {np.median(positive_scores):.6f}")
    print(f"\nPlatt calibrated scores (candidate pairs):")
    platt_positive = platt_matrix[candidate_mask]
    print(f"  Range: [{platt_positive.min():.6f}, {platt_positive.max():.6f}]")
    print(f"  Mean: {platt_positive.mean():.6f}, Median: {np.median(platt_positive):.6f}")
    print(f"\nPer-disease normalized scores (candidate pairs):")
    pd_positive = per_disease_matrix[candidate_mask]
    print(f"  Range: [{pd_positive.min():.6f}, {pd_positive.max():.6f}]")
    print(f"  Mean: {pd_positive.mean():.6f}, Median: {np.median(pd_positive):.6f}")
    print(f"\nConfidence SNR (candidate pairs with score>0):")
    conf_positive = confidence_matrix[candidate_mask]
    print(f"  Range: [{conf_positive.min():.2f}, {conf_positive.max():.2f}]")
    print(f"  Mean: {conf_positive.mean():.2f}, Median: {np.median(conf_positive):.2f}")


if __name__ == "__main__":
    main()

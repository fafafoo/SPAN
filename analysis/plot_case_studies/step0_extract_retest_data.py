# SPAN: Subgraph Progressive Attention Networks for microRNA-disease association prediction
# Copyright (c) 2025 SPAN Authors
# Licensed under the Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0).
# You may use, modify, and distribute this software for non-commercial purposes only.
# For commercial use, please contact the authors to obtain a commercial license.
# See LICENSE for details or visit https://creativecommons.org/licenses/by-nc/4.0/
"""
Step 0: Extract and aggregate retest predictions from 50 log files.

Input:  SPAN/logs/retest-round_{0-9}-fold_{0-4}.log  (50 files)
Output: SPAN/analysis/plot_case_studies/retest_aggregated.csv
        SPAN/analysis/plot_case_studies/retest_round_predictions.csv

Aggregation strategy:
  - Each round has 5 folds covering all miRNA-disease pairs (disjoint test sets).
  - Combining 5 folds in a round yields one prediction per pair.
  - Across 10 rounds, each pair has 10 predictions.
  - Final prediction = mean across 10 rounds; also compute std.

retest_round_predictions.csv (round-level, one row per candidate pair per
round) is produced for step2_compute_topk_validation.py, which needs per-round
predictions to compute top-k metrics across rounds (mean/std). It matches the
original retest_round_predictions_0.csv layout (fields: round, disease_name,
mirna_name, predicted) and is restricted to the 7 TARGET_DISEASES with
candidate pairs only (true_label == 0), consistent with case_studies step3.
"""

import os
import re
import csv
import collections
from pathlib import Path

# ── Configuration ──────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
LOG_DIR = BASE_DIR / ".." / ".." / "logs"
OUTPUT_CSV = BASE_DIR / "retest_aggregated.csv"
ROUND_OUTPUT_CSV = BASE_DIR / "retest_round_predictions.csv"

N_ROUNDS = 10
N_FOLDS = 5

# Only these target diseases are kept (matches case_studies step3 / plot step2).
TARGET_DISEASES = {
    "Breast Neoplasms",
    "Ovarian Neoplasms",
    "Prostatic Neoplasms",
    "Lung Neoplasms",
    "Stomach Neoplasms",
    "Melanoma",
    "Leukemia",
}

# ── Parse a single log file ───────────────────────────────────────────
# Format: "miRNA_id, disease_id: miRNA_name, disease_name, predicted_value, true_label"
# Note: disease_name may contain commas (e.g., "Adenocarcinoma, Sebaceous")
PATTERN = re.compile(
    r"^(\d+),\s*(\d+):\s*(.+)$"
)


def parse_log_file(filepath):
    """Parse one retest log file, return list of dicts."""
    records = []
    header_found = False
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not header_found:
                if line.startswith("miRNA_id, disease_id:"):
                    header_found = True
                continue
            if not line:
                continue
            m = PATTERN.match(line)
            if not m:
                continue
            mirna_id = int(m.group(1))
            disease_id = int(m.group(2))
            rest = m.group(3)
            # Split from the right: predicted_value and true_label are always last
            parts = rest.rsplit(", ", 2)
            if len(parts) != 3:
                continue
            name_and_disease = parts[0]
            predicted_value = float(parts[1])
            true_label = int(parts[2])
            # Split name_and_disease on first comma (miRNA names never contain commas)
            name_parts = name_and_disease.split(", ", 1)
            if len(name_parts) != 2:
                continue
            mirna_name = name_parts[0]
            disease_name = name_parts[1]

            records.append({
                "mirna_id": mirna_id,
                "disease_id": disease_id,
                "mirna_name": mirna_name,
                "disease_name": disease_name,
                "predicted": predicted_value,
                "true_label": true_label,
            })
    return records


def main():
    # Step 1: Parse all 50 log files, grouped by round
    # round_data[round_id][(mirna_id, disease_id)] = predicted_value
    # Also collect true_label and names from first occurrence
    round_data = [{} for _ in range(N_ROUNDS)]
    pair_info = {}  # (mirna_id, disease_id) -> {mirna_name, disease_name, true_label}

    for r in range(N_ROUNDS):
        for f in range(N_FOLDS):
            log_path = LOG_DIR / f"retest-round_{r}-fold_{f}.log"
            if not log_path.exists():
                print(f"WARNING: {log_path} not found, skipping.")
                continue
            records = parse_log_file(log_path)
            for rec in records:
                key = (rec["mirna_id"], rec["disease_id"])
                round_data[r][key] = rec["predicted"]
                if key not in pair_info:
                    pair_info[key] = {
                        "mirna_name": rec["mirna_name"],
                        "disease_name": rec["disease_name"],
                        "true_label": rec["true_label"],
                    }

    # Step 2: Aggregate across rounds
    # For each pair, compute mean and std of predicted values across rounds
    all_pairs = sorted(pair_info.keys())
    print(f"Total unique pairs: {len(all_pairs)}")

    results = []
    for key in all_pairs:
        mirna_id, disease_id = key
        info = pair_info[key]
        preds = []
        for r in range(N_ROUNDS):
            if key in round_data[r]:
                preds.append(round_data[r][key])
        if not preds:
            continue
        pred_mean = sum(preds) / len(preds)
        pred_std = (sum((p - pred_mean) ** 2 for p in preds) / len(preds)) ** 0.5 if len(preds) > 1 else 0.0
        results.append({
            "mirna_id": mirna_id,
            "disease_id": disease_id,
            "mirna_name": info["mirna_name"],
            "disease_name": info["disease_name"],
            "predicted_mean": round(pred_mean, 6),
            "predicted_std": round(pred_std, 6),
            "true_label": info["true_label"],
        })

    # Step 3: Write output CSV
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "mirna_id", "disease_id", "mirna_name", "disease_name",
            "predicted_mean", "predicted_std", "true_label"
        ])
        writer.writeheader()
        writer.writerows(results)

    print(f"Written {len(results)} records to {OUTPUT_CSV}")

    # Step 4: Write round-level candidate predictions (consumed by step2_compute_topk_validation.py)
    # Match the original retest_round_predictions_0.csv layout:
    #   - Only the 7 TARGET_DISEASES are kept (same as case_studies step3 / plot step2).
    #   - Only candidate pairs (true_label == 0) are included.
    #   - Fields must match what step2 reads: round, disease_name, mirna_name, predicted
    round_rows = []
    for r in range(N_ROUNDS):
        for key, pred in round_data[r].items():
            info = pair_info[key]
            if info["disease_name"] not in TARGET_DISEASES:
                continue
            if info["true_label"] != 0:
                continue
            round_rows.append({
                "round": r,
                "disease_name": info["disease_name"],
                "mirna_name": info["mirna_name"],
                "predicted": round(pred, 6),
            })
    with open(ROUND_OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "round", "disease_name", "mirna_name", "predicted"
        ])
        writer.writeheader()
        writer.writerows(round_rows)

    print(f"Written {len(round_rows)} round-level candidate records to {ROUND_OUTPUT_CSV}")


if __name__ == "__main__":
    main()

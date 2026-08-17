# SPAN: Subgraph Progressive Attention Networks for microRNA-disease association prediction
# Copyright (c) 2025 SPAN Authors
# Licensed under the Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0).
# You may use, modify, and distribute this software for non-commercial purposes only.
# For commercial use, please contact the authors to obtain a commercial license.
# See LICENSE for details or visit https://creativecommons.org/licenses/by-nc/4.0/
"""
Step 3: Extract Top-100 candidate miRNA per disease type for Panel B.

Input:  prediction_data_long.csv    
Output: panel_b_top100.csv          

KEY: Only consider candidate pairs (true_label=0),
i.e., miRNA-disease pairs NOT in the known association matrix.
Sort by predicted_score descending, take Top-100.
"""

import csv
from pathlib import Path

BASE_DIR = Path(__file__).parent

INPUT_CSV = BASE_DIR / "prediction_data_long.csv"
OUTPUT_CSV = BASE_DIR / "panel_b_top100.csv"

TARGET_DISEASES = [
    "Breast Neoplasms",
    "Ovarian Neoplasms",
    "Prostatic Neoplasms",
    "Lung Neoplasms",
    "Stomach Neoplasms",
    "Melanoma",
    "Leukemia",
]
TOP_K = 100


def main():
    disease_data = {}

    total_records = 0
    known_records = 0
    candidate_records = 0

    with open(INPUT_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            dn = row["disease_name"]
            if dn not in TARGET_DISEASES:
                continue
            total_records += 1
            true_label = int(row["true_label"])

            if true_label == 1:
                known_records += 1
                continue

            candidate_records += 1
            disease_data.setdefault(dn, []).append({
                "mirna_id": int(row["mirna_id"]),
                "disease_id": int(row["disease_id"]),
                "mirna_name": row["mirna_name"],
                "disease_name": dn,
                "predicted_score": float(row["predicted_score"]),
                "true_label": true_label,
            })

    print(f"Total pairs for target diseases: {total_records}")
    print(f"  Known (true_label=1): {known_records} -> EXCLUDED")
    print(f"  Candidates (true_label=0): {candidate_records} -> used for ranking")

    results = []
    for dn in TARGET_DISEASES:
        records = disease_data.get(dn, [])
        records.sort(key=lambda x: x["predicted_score"], reverse=True)
        print(f"\n{dn}: {len(records)} candidates, taking Top-{min(TOP_K, len(records))}")
        for rank, rec in enumerate(records[:TOP_K], start=1):
            rec["rank"] = rank
            results.append(rec)

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "disease", "mirna_id", "disease_id", "mirna_name",
            "predicted_score", "true_label", "rank"
        ])
        writer.writeheader()
        for r in results:
            writer.writerow({
                "disease": r["disease_name"],
                "mirna_id": r["mirna_id"],
                "disease_id": r["disease_id"],
                "mirna_name": r["mirna_name"],
                "predicted_score": r["predicted_score"],
                "true_label": r["true_label"],
                "rank": r["rank"],
            })

    for dn in TARGET_DISEASES:
        count = sum(1 for r in results if r["disease_name"] == dn)
        print(f"  {dn}: {count} candidate miRNAs in Top-100")

    print(f"\nWritten {len(results)} records to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()

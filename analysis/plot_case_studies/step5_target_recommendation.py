# SPAN: Subgraph Progressive Attention Networks for microRNA-disease association prediction
# Copyright (c) 2025 SPAN Authors
# Licensed under the Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0).
# You may use, modify, and distribute this software for non-commercial purposes only.
# For commercial use, please contact the authors to obtain a commercial license.
# See LICENSE for details or visit https://creativecommons.org/licenses/by-nc/4.0/
"""
Step 5: Generate Target Recommendation Report.

This step produces a structured report for each of the 7 target diseases,
providing actionable guidance for biomarker discovery and therapeutic target
identification.

Input:
  - panel_b_edges.csv           (from step4)
  - prediction_data_long.csv    (from step1)
Output:
  - target_recommendations.csv  
  - target_report_summary.txt   

Report structure per disease:
  1. High-confidence validated targets (both DBs, high score, high SNR)
  2. Single-DB validated targets with high prediction confidence
  3. Novel predictions ranked by calibrated probability (Platt score)
  4. Cross-disease shared miRNA targets
"""

import csv
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).parent

PANEL_B_CSV = BASE_DIR / "panel_b_edges.csv"
PREDICTION_CSV = BASE_DIR / "prediction_data_long.csv"
OUTPUT_CSV = BASE_DIR / "target_recommendations.csv"
OUTPUT_SUMMARY = BASE_DIR / "target_report_summary.txt"

TARGET_DISEASES = [
    "Breast Neoplasms",
    "Ovarian Neoplasms",
    "Prostatic Neoplasms",
    "Lung Neoplasms",
    "Stomach Neoplasms",
    "Melanoma",
    "Leukemia",
]

DISEASE_SHORT = {
    "Breast Neoplasms": "Breast Cancer",
    "Ovarian Neoplasms": "Ovarian Cancer",
    "Prostatic Neoplasms": "Prostate Cancer",
    "Lung Neoplasms": "Lung Cancer",
    "Stomach Neoplasms": "Stomach Cancer",
    "Melanoma": "Melanoma",
    "Leukemia": "Leukemia",
}

PLATT_THRESHOLD_HIGH = 0.8
PLATT_THRESHOLD_MEDIUM = 0.5
SNR_THRESHOLD = 10.0


def load_panel_b_data():
    records = []
    with open(PANEL_B_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Count evidence IDs from pmids fields
            hmdd_pmids_str = row.get("hmdd_pmids", "")
            dbdemc_pmids_str = row.get("dbdemc_pmids", "")
            hmdd_count = len([p for p in hmdd_pmids_str.split(";") if p.strip()]) if hmdd_pmids_str else 0
            dbdemc_count = len([p for p in dbdemc_pmids_str.split(";") if p.strip()]) if dbdemc_pmids_str else 0

            records.append({
                "disease": row["disease"],
                "mirna_name": row["mirna_name"],
                "predicted_score": float(row["predicted_score"]),
                "true_label": int(row["true_label"]),
                "rank": int(row["rank"]),
                "in_dbdemc": row["in_dbdemc"] == "True",
                "in_hmddv4": row["in_hmddv4"] == "True",
                "edge_type": row["edge_type"],
                "dbdemc_count": dbdemc_count,
                "hmdd_count": hmdd_count,
            })
    return records


def load_prediction_data():
    if not PREDICTION_CSV.exists():
        return None
    return pd.read_csv(PREDICTION_CSV)


def classify_recommendation(rec, platt_score=None, snr=None):
    if rec["edge_type"] == "both":
        if platt_score is not None and platt_score >= PLATT_THRESHOLD_HIGH:
            return "Tier-1: High-confidence validated"
        return "Tier-2: Validated (both DBs)"
    elif rec["edge_type"] == "single":
        if platt_score is not None and platt_score >= PLATT_THRESHOLD_MEDIUM:
            return "Tier-3: Validated (single DB, high confidence)"
        return "Tier-4: Validated (single DB)"
    else:
        if platt_score is not None and platt_score >= PLATT_THRESHOLD_MEDIUM:
            return "Tier-5: Novel (high confidence)"
        if snr is not None and snr >= SNR_THRESHOLD:
            return "Tier-6: Novel (stable prediction)"
        return "Tier-7: Novel (exploratory)"


def main():
    panel_b = load_panel_b_data()
    print(f"Loaded {len(panel_b)} records from panel_b_edges.csv")

    pred_df = load_prediction_data()
    platt_lookup = {}
    snr_lookup = {}
    if pred_df is not None:
        for _, row in pred_df.iterrows():
            key = (row["mirna_name"], row["disease_name"])
            if "platt_score" in pred_df.columns:
                platt_lookup[key] = row["platt_score"]
            if "confidence_snr" in pred_df.columns:
                snr_lookup[key] = row["confidence_snr"]
        print(f"Loaded Platt/SNR data for {len(platt_lookup)} pairs")

    recommendations = []
    for rec in panel_b:
        if rec["disease"] not in TARGET_DISEASES:
            continue
        key = (rec["mirna_name"], rec["disease"])
        platt_score = platt_lookup.get(key)
        snr = snr_lookup.get(key)
        tier = classify_recommendation(rec, platt_score, snr)
        recommendations.append({
            "disease": rec["disease"],
            "disease_short": DISEASE_SHORT[rec["disease"]],
            "mirna_name": rec["mirna_name"],
            "rank": rec["rank"],
            "predicted_score": rec["predicted_score"],
            "platt_score": platt_score if platt_score is not None else "",
            "confidence_snr": snr if snr is not None else "",
            "edge_type": rec["edge_type"],
            "in_dbdemc": rec["in_dbdemc"],
            "in_hmddv4": rec["in_hmddv4"],
            "dbdemc_count": rec["dbdemc_count"],
            "hmdd_count": rec["hmdd_count"],
            "recommendation_tier": tier,
        })

    rec_df = pd.DataFrame(recommendations)
    rec_df = rec_df.sort_values(["disease", "rank"])
    rec_df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved target recommendations to {OUTPUT_CSV}")

    mirna_diseases = defaultdict(set)
    for rec in panel_b:
        if rec["disease"] in TARGET_DISEASES and rec["rank"] <= 50:
            mirna_diseases[rec["mirna_name"]].add(rec["disease"])

    shared_mirnas = {m: ds for m, ds in mirna_diseases.items() if len(ds) >= 2}

    lines = []
    lines.append("=" * 80)
    lines.append("TARGET RECOMMENDATION REPORT — SPAN Case Studies")
    lines.append("=" * 80)
    lines.append("")

    for dn in TARGET_DISEASES:
        short = DISEASE_SHORT[dn]
        disease_recs = [r for r in recommendations if r["disease"] == dn]
        if not disease_recs:
            lines.append(f"\n{'─' * 60}")
            lines.append(f"  {short} ({dn})")
            lines.append(f"  No Top-100 candidates found.")
            continue

        lines.append(f"\n{'─' * 60}")
        lines.append(f"  {short} ({dn})")
        lines.append(f"  Total Top-100 candidates: {len(disease_recs)}")

        both = [r for r in disease_recs if r["edge_type"] == "both"]
        single = [r for r in disease_recs if r["edge_type"] == "single"]
        novel = [r for r in disease_recs if r["edge_type"] == "novel"]

        lines.append(f"  Validated (both DBs):  {len(both)} ({len(both)/len(disease_recs):.1%})")
        lines.append(f"  Validated (single DB): {len(single)} ({len(single)/len(disease_recs):.1%})")
        lines.append(f"  Novel predictions:     {len(novel)} ({len(novel)/len(disease_recs):.1%})")
        lines.append("")

        tier_counts = defaultdict(int)
        for r in disease_recs:
            tier_counts[r["recommendation_tier"]] += 1
        lines.append("  Recommendation tier breakdown:")
        for tier in sorted(tier_counts.keys()):
            lines.append(f"    {tier}: {tier_counts[tier]}")
        lines.append("")

        if novel:
            lines.append("  Top-5 Novel predictions (ranked by prediction score):")
            novel_sorted = sorted(novel, key=lambda x: x["predicted_score"], reverse=True)[:5]
            for r in novel_sorted:
                platt_str = f", Platt={r['platt_score']:.4f}" if r["platt_score"] != "" else ""
                snr_str = f", SNR={r['confidence_snr']:.1f}" if r["confidence_snr"] != "" else ""
                lines.append(f"    #{r['rank']:3d} {r['mirna_name']:20s} "
                             f"score={r['predicted_score']:.5f}{platt_str}{snr_str}")
            lines.append("")

        top5 = sorted(disease_recs, key=lambda x: x["predicted_score"], reverse=True)[:5]
        lines.append("  Top-5 highest-scoring candidates:")
        for r in top5:
            platt_str = f", Platt={r['platt_score']:.4f}" if r["platt_score"] != "" else ""
            lines.append(f"    #{r['rank']:3d} {r['mirna_name']:20s} "
                         f"score={r['predicted_score']:.5f}{platt_str} [{r['edge_type']}]")
        lines.append("")

    lines.append("\n" + "=" * 80)
    lines.append("CROSS-DISEASE SHARED miRNA TARGETS (Top-50)")
    lines.append("=" * 80)
    lines.append("")

    if shared_mirnas:
        sorted_shared = sorted(shared_mirnas.items(), key=lambda x: len(x[1]), reverse=True)
        for mirna, diseases in sorted_shared:
            disease_names = ", ".join(DISEASE_SHORT[d] for d in sorted(diseases))
            lines.append(f"  {mirna:20s} -> {disease_names} ({len(diseases)} diseases)")
    else:
        lines.append("  No shared miRNA targets found across diseases.")

    lines.append("")
    lines.append("=" * 80)
    lines.append("END OF REPORT")
    lines.append("=" * 80)

    with open(OUTPUT_SUMMARY, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Saved report summary to {OUTPUT_SUMMARY}")


if __name__ == "__main__":
    main()

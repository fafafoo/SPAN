# SPAN: Subgraph Progressive Attention Networks for microRNA-disease association prediction
# Copyright (c) 2025 SPAN Authors
# Licensed under the Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0).
# You may use, modify, and distribute this software for non-commercial purposes only.
# For commercial use, please contact the authors to obtain a commercial license.
# See LICENSE for details or visit https://creativecommons.org/licenses/by-nc/4.0/
"""
Step 2: Compute Top-k validation rate for Panel A.

Input:  retest_round_predictions.csv  (pre-extracted from retest logs)
        dataset_HMDD v4.txt
        miRExpAll_dbDEMC 3.0.txt
Output: panel_a_data.csv

For each cancer type, EXCLUDE known associations (true_label=1),
sort remaining candidate miRNAs by predicted score (descending),
then compute the validation rate (fraction validated by external databases)
at each k. Repeat per round to get mean and std across 10 rounds.

External validation sources:
  - dbDEMC 3.0:  miRExpAll_dbDEMC 3.0.txt (differential expression data)
  - HMDD v4.0:   dataset_HMDD v4.txt (local copy)

Note: retest_round_predictions.csv contains pre-extracted candidate miRNA
  predictions (true_label=0 only) for the 7 target diseases across 10 rounds,
  aggregated from 5-fold cross-validation log files.
"""

import csv
import math
from pathlib import Path
from collections import defaultdict

BASE_DIR = Path(__file__).parent

RETEST_PREDICTIONS_CSV = BASE_DIR / "retest_round_predictions.csv"
OUTPUT_CSV = BASE_DIR / "panel_a_data.csv"

MIREXPALL_PATH = BASE_DIR / "miRExpAll_dbDEMC 3.0.txt"
HMDD_PATH = BASE_DIR / "dataset_HMDD v4.txt"

TARGET_DISEASES = {
    "Breast Neoplasms": "Breast Cancer",
    "Ovarian Neoplasms": "Ovarian Cancer",
    "Prostatic Neoplasms": "Prostate Cancer",
    "Lung Neoplasms": "Lung Cancer",
    "Stomach Neoplasms": "Stomach Cancer",
    "Melanoma": "Melanoma",
    "Leukemia": "Leukemia",
}

K_VALUES = [25, 50, 75]

N_ROUNDS = 10

DISEASE_MAP_HMDD = {
    "Breast Neoplasms": "Breast Neoplasms",
    "Triple Negative Breast Neoplasms": "Breast Neoplasms",
    "Ovarian Neoplasms": "Ovarian Neoplasms",
    "Carcinoma, Ovarian Epithelial": "Ovarian Neoplasms",
    "High Grade Ovarian Neoplasms": "Ovarian Neoplasms",
    "Prostatic Neoplasms": "Prostatic Neoplasms",
    "Lung Neoplasms": "Lung Neoplasms",
    "Carcinoma, Non-Small-Cell Lung": "Lung Neoplasms",
    "Small Cell Lung Carcinoma": "Lung Neoplasms",
    "Pancreatic Neoplasms": "Pancreatic Neoplasms",
    "Carcinoma, Pancreatic Ductal": "Pancreatic Neoplasms",
    "Liver Neoplasms": "Liver Neoplasms",
    "Hepatocellular Carcinoma": "Liver Neoplasms",
    "Liver Cell Carcinoma": "Liver Neoplasms",
    "Carcinoma, Hepatocellular": "Liver Neoplasms",
    "Stomach Neoplasms": "Stomach Neoplasms",
    "Gastric Neoplasms": "Stomach Neoplasms",
    "Stomach Carcinoma": "Stomach Neoplasms",
    "Gastric Carcinoma": "Stomach Neoplasms",
    "Melanoma": "Melanoma",
    "Leukemia": "Leukemia",
    "Leukemia, Myeloid, Acute": "Leukemia",
    "Leukemia, Lymphoid, Acute": "Leukemia",
    "Leukemia, Myeloid, Chronic": "Leukemia",
    "Leukemia, Lymphocytic, Chronic": "Leukemia",
}

DBDEMC_CANCER_MAP = {
    "breast cancer": "Breast Neoplasms",
    "breast carcinoma": "Breast Neoplasms",
    "ovarian cancer": "Ovarian Neoplasms",
    "ovarian carcinoma": "Ovarian Neoplasms",
    "prostate cancer": "Prostatic Neoplasms",
    "prostate carcinoma": "Prostatic Neoplasms",
    "prostatic cancer": "Prostatic Neoplasms",
    "prostatic carcinoma": "Prostatic Neoplasms",
    "prostate adenocarcinoma": "Prostatic Neoplasms",
    "castration-resistant prostate cancer": "Prostatic Neoplasms",
    "lung cancer": "Lung Neoplasms",
    "lung carcinoma": "Lung Neoplasms",
    "non-small cell lung cancer": "Lung Neoplasms",
    "nsclc": "Lung Neoplasms",
    "small cell lung cancer": "Lung Neoplasms",
    "sclc": "Lung Neoplasms",
    "pancreatic cancer": "Pancreatic Neoplasms",
    "pancreatic carcinoma": "Pancreatic Neoplasms",
    "pancreatic ductal adenocarcinoma": "Pancreatic Neoplasms",
    "pdac": "Pancreatic Neoplasms",
    "liver cancer": "Liver Neoplasms",
    "liver carcinoma": "Liver Neoplasms",
    "hepatocellular carcinoma": "Liver Neoplasms",
    "hcc": "Liver Neoplasms",
    "hepatic carcinoma": "Liver Neoplasms",
    "stomach cancer": "Stomach Neoplasms",
    "gastric cancer": "Stomach Neoplasms",
    "gastric carcinoma": "Stomach Neoplasms",
    "melanoma": "Melanoma",
    "malignant melanoma": "Melanoma",
    "cutaneous melanoma": "Melanoma",
    "leukemia": "Leukemia",
    "acute myeloid leukemia": "Leukemia",
    "aml": "Leukemia",
    "acute lymphoblastic leukemia": "Leukemia",
    "all": "Leukemia",
    "chronic myeloid leukemia": "Leukemia",
    "cml": "Leukemia",
    "chronic lymphocytic leukemia": "Leukemia",
    "cll": "Leukemia",
}

DBDEMC_SUBTYPE_MAP = {
    "er positive": "Breast Neoplasms",
    "er negative": "Breast Neoplasms",
    "her2 positive": "Breast Neoplasms",
    "triple negative": "Breast Neoplasms",
    "ovarian carcinoma": "Ovarian Neoplasms",
    "prostate adenocarcinoma": "Prostatic Neoplasms",
    "castration-resistant prostate cancer": "Prostatic Neoplasms",
    "lung adenocarcinoma": "Lung Neoplasms",
    "lung squamous cell carcinoma": "Lung Neoplasms",
    "pancreatic ductal adenocarcinoma": "Pancreatic Neoplasms",
    "hepatocellular carcinoma": "Liver Neoplasms",
    "gastric adenocarcinoma": "Stomach Neoplasms",
}


def normalize_mirna(name):
    name = name.strip()
    if name.startswith("has-mir-") or name.startswith("has-miR-"):
        name = "hsa-miR-" + name[8:]
    elif name.startswith("hsa-mir-"):
        name = "hsa-miR-" + name[8:]
    elif name.startswith("miR-") or name.startswith("mir-"):
        name = "hsa-miR-" + name[4:]
    elif name.startswith("hsa-let-"):
        pass
    elif name.startswith("let-"):
        name = "hsa-let-" + name[4:]
    return name


def normalize_mirna_for_lookup(name):
    name = normalize_mirna(name)
    variants = {name}
    if name.endswith("*"):
        base = name[:-1]
        variants.add(base)
        variants.add(base + "-3p")
        name = base
    for suffix in ["-5p", "-3p"]:
        if name.endswith(suffix):
            variants.add(name[:-len(suffix)])
    parts = name.split("-")
    if len(parts) >= 4 and parts[-1].isdigit():
        variants.add("-".join(parts[:-1]))
    if name.startswith("hsa-miR-"):
        stripped = name
        for suffix in ["-5p", "-3p"]:
            if stripped.endswith(suffix):
                stripped = stripped[:-len(suffix)]
        parts2 = stripped.split("-")
        is_precursor = True
        if len(parts2) >= 4 and parts2[-1].isdigit():
            is_precursor = False
        if is_precursor:
            variants.add(name + "-5p")
            variants.add(name + "-3p")
            for suffix in ["-5p", "-3p"]:
                if name.endswith(suffix):
                    base_no_arm = name[:-len(suffix)]
                    variants.add(base_no_arm + "-5p")
                    variants.add(base_no_arm + "-3p")
                    break
    return variants


def load_hmdd_associations():
    associations = set()
    with open(HMDD_PATH, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader)
        for row in reader:
            if len(row) < 4:
                continue
            mirna = row[2].strip()
            disease = row[3].strip()
            if disease in DISEASE_MAP_HMDD:
                our_disease = DISEASE_MAP_HMDD[disease]
                for variant in normalize_mirna_for_lookup(mirna):
                    associations.add((variant, our_disease))
    print(f"  HMDD v4.0: {len(associations)} (mirna, disease) lookup entries")
    return associations


def load_mirexpall_associations():
    associations = set()
    skipped_nonhuman = 0
    skipped_nonsig = 0
    cancer_counts = defaultdict(int)

    with open(MIREXPALL_PATH, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader)
        for row in reader:
            if len(row) < 16:
                continue
            mirna_id = row[0].strip()
            mirbase_id = row[1].strip()
            cancer_type = row[4].strip()
            cancer_subtype = row[5].strip()
            adj_pvalue_str = row[12].strip()
            species = row[14].strip()

            if species != "Homo sapiens":
                skipped_nonhuman += 1
                continue

            try:
                adj_pvalue = float(adj_pvalue_str)
            except ValueError:
                skipped_nonsig += 1
                continue
            if adj_pvalue >= 0.05:
                skipped_nonsig += 1
                continue

            our_disease = None
            cancer_type_lower = cancer_type.lower()
            for dbdemc_name, our_name in DBDEMC_CANCER_MAP.items():
                if dbdemc_name.lower() == cancer_type_lower:
                    our_disease = our_name
                    break

            if our_disease is None and cancer_subtype:
                subtype_lower = cancer_subtype.lower()
                for subtype_name, our_name in DBDEMC_SUBTYPE_MAP.items():
                    if subtype_name.lower() == subtype_lower:
                        our_disease = our_name
                        break

            if our_disease is None:
                continue

            for variant in normalize_mirna_for_lookup(mirna_id):
                associations.add((variant, our_disease))
            if mirbase_id and mirbase_id != "NA" and (mirbase_id.startswith("hsa-") or mirbase_id.startswith("has-")):
                for variant in normalize_mirna_for_lookup(mirbase_id):
                    associations.add((variant, our_disease))
            cancer_counts[our_disease] += 1

    print(f"  miRExpAll dbDEMC 3.0: {len(associations)} (mirna, disease) lookup entries")
    print(f"  Skipped {skipped_nonhuman} non-human, {skipped_nonsig} non-significant records")
    for disease, count in sorted(cancer_counts.items()):
        print(f"    {disease}: {count} raw records")
    return associations


def is_externally_validated(mirna_name, disease_name, hmdd_set, dbdemc_set):
    for variant in normalize_mirna_for_lookup(mirna_name):
        if (variant, disease_name) in hmdd_set or (variant, disease_name) in dbdemc_set:
            return True
    return False


def main():
    print("Loading external validation databases...")
    hmdd_assoc = load_hmdd_associations()
    dbdemc_assoc = load_mirexpall_associations()

    # Load pre-extracted retest round predictions (candidate pairs only)
    round_candidates = [{} for _ in range(N_ROUNDS)]

    with open(RETEST_PREDICTIONS_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            r = int(row["round"])
            dn = row["disease_name"]
            mn = row["mirna_name"]
            pred = float(row["predicted"])
            round_candidates[r][(dn, mn)] = pred

    print(f"Loaded retest predictions: {sum(len(rc) for rc in round_candidates)} records across {N_ROUNDS} rounds")

    rates = {dn: {k: [] for k in K_VALUES} for dn in TARGET_DISEASES}

    for dn in TARGET_DISEASES:
        for r in range(N_ROUNDS):
            mirna_preds = []
            for (d, mname), pred in round_candidates[r].items():
                if d == dn:
                    mirna_preds.append((mname, pred))

            mirna_preds.sort(key=lambda x: x[1], reverse=True)

            for k in K_VALUES:
                if len(mirna_preds) < k:
                    continue
                top_k = mirna_preds[:k]
                validated = sum(
                    1 for mname, _ in top_k
                    if is_externally_validated(mname, dn, hmdd_assoc, dbdemc_assoc)
                )
                rate = validated / k
                rates[dn][k].append(rate)

    results = []
    for dn in TARGET_DISEASES:
        label = TARGET_DISEASES[dn]
        for k in K_VALUES:
            vals = rates[dn][k]
            if not vals:
                continue
            mean = sum(vals) / len(vals)
            std = (sum((v - mean) ** 2 for v in vals) / len(vals)) ** 0.5 if len(vals) > 1 else 0.0
            results.append({
                "disease": label,
                "disease_raw": dn,
                "k": k,
                "rate_mean": round(mean, 6),
                "rate_std": round(std, 6),
                "n_rounds": len(vals),
            })

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "disease", "disease_raw", "k", "rate_mean", "rate_std", "n_rounds"
        ])
        writer.writeheader()
        writer.writerows(results)

    print(f"\nWritten {len(results)} records to {OUTPUT_CSV}")

    for dn in TARGET_DISEASES:
        label = TARGET_DISEASES[dn]
        for k in K_VALUES:
            vals = rates[dn][k]
            if vals:
                mean = sum(vals) / len(vals)
                print(f"  {label} k={k}: rate={mean:.4f} (n={len(vals)} rounds)")


if __name__ == "__main__":
    main()

# SPAN: Subgraph Progressive Attention Networks for microRNA-disease association prediction
# Copyright (c) 2025 SPAN Authors
# Licensed under the Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0).
# You may use, modify, and distribute this software for non-commercial purposes only.
# For commercial use, please contact the authors to obtain a commercial license.
# See LICENSE for details or visit https://creativecommons.org/licenses/by-nc/4.0/
"""
Step 4: Map external validation status from dbDEMC 3.0 and HMDD v4.0.

Input:
  - panel_b_top100.csv          (from step3)
  - miRExpAll_dbDEMC 3.0.txt
  - dataset_HMDD v4.txt
  - mirna_name.csv

Output:
  - panel_b_edges.csv           

Edge type classification:
  - "both"       : Validated in both dbDEMC and HMDD v4.0 -> deep blue solid line
  - "single"     : Validated in exactly one database     -> sky blue dashed line
  - "novel"      : Not validated in either database      -> red dotted line

KEY NOTES:
  1. Use miRExpAll_dbDEMC 3.0.txt (contains differential expression data
     with miRNA names, cancer types, logFC, and adjusted p-values).
     This provides far more coverage than the curated subset.
  2. Fuzzy miRNA name matching: hsa-mir-XXX -> hsa-miR-XXX, strip -5p/-3p suffixes,
     and generate mature miRNA variants for precursor names.
  3. Detailed matching statistics for debugging.
"""

import csv
from pathlib import Path
from collections import defaultdict

BASE_DIR = Path(__file__).parent

TOP100_CSV = BASE_DIR / "panel_b_top100.csv"
MIREXPALL_PATH = BASE_DIR / "miRExpAll_dbDEMC 3.0.txt"
HMDD_PATH = BASE_DIR / "dataset_HMDD v4.txt"
MIRNA_NAME_PATH = BASE_DIR / "mirna_name.csv"
OUTPUT_CSV = BASE_DIR / "panel_b_edges.csv"

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
    associations = defaultdict(set)
    match_stats = defaultdict(int)

    with open(HMDD_PATH, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader)
        for row in reader:
            if len(row) < 4:
                continue
            code, pmid, mirna, disease = row[0], row[1], row[2], row[3]
            if disease in DISEASE_MAP_HMDD:
                our_disease = DISEASE_MAP_HMDD[disease]
                for variant in normalize_mirna_for_lookup(mirna):
                    associations[(variant, our_disease)].add(pmid)
                match_stats[our_disease] += 1

    print(f"  HMDD v4.0: {len(associations)} unique (miRNA, disease) lookup entries")
    for disease, count in sorted(match_stats.items()):
        print(f"    {disease}: {count} raw records")
    return associations


def load_mirexpall_associations():
    associations = defaultdict(set)
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
            source_data_id = row[3].strip()
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

            evidence_id = source_data_id

            for variant in normalize_mirna_for_lookup(mirna_id):
                associations[(variant, our_disease)].add(evidence_id)
            if mirbase_id and mirbase_id != "NA" and (mirbase_id.startswith("hsa-") or mirbase_id.startswith("has-")):
                for variant in normalize_mirna_for_lookup(mirbase_id):
                    associations[(variant, our_disease)].add(evidence_id)
            cancer_counts[our_disease] += 1

    print(f"  miRExpAll dbDEMC 3.0: {len(associations)} unique (miRNA, disease) lookup entries")
    print(f"  Skipped {skipped_nonhuman} non-human, {skipped_nonsig} non-significant records")
    for disease, count in sorted(cancer_counts.items()):
        print(f"    {disease}: {count} raw records")
    return associations


def check_validation(mirna_name, disease_name, hmdd_assoc, dbdemc_assoc):
    in_hmdd = False
    in_dbdemc = False
    hmdd_pmids = set()
    dbdemc_pmids = set()

    for variant in normalize_mirna_for_lookup(mirna_name):
        key = (variant, disease_name)
        if key in hmdd_assoc:
            in_hmdd = True
            hmdd_pmids.update(hmdd_assoc[key])
        if key in dbdemc_assoc:
            in_dbdemc = True
            dbdemc_pmids.update(dbdemc_assoc[key])

    return (in_hmdd, in_dbdemc,
            ";".join(sorted(hmdd_pmids)),
            ";".join(sorted(dbdemc_pmids)))


def main():
    top100_records = []
    with open(TOP100_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            top100_records.append(row)

    print(f"Loaded {len(top100_records)} Top-100 records")

    print("Loading HMDD v4.0...")
    hmdd_assoc = load_hmdd_associations()
    print(f"  HMDD v4.0: {len(hmdd_assoc)} unique (miRNA, disease) associations")

    print("\nLoading miRExpAll dbDEMC 3.0...")
    dbdemc_assoc = load_mirexpall_associations()
    print(f"  miRExpAll dbDEMC 3.0: {len(dbdemc_assoc)} unique (miRNA, disease) associations")

    results = []
    match_count_hmdd = 0
    match_count_dbdemc = 0

    for rec in top100_records:
        mirna_name = rec["mirna_name"]
        disease_name = rec["disease"]
        predicted_score = float(rec["predicted_score"])
        true_label = int(rec["true_label"])
        mirna_id = int(rec["mirna_id"])
        disease_id = int(rec["disease_id"])
        rank = int(rec["rank"])

        in_hmdd, in_dbdemc, hmdd_pmids, dbdemc_pmids = check_validation(
            mirna_name, disease_name, hmdd_assoc, dbdemc_assoc
        )

        if in_hmdd:
            match_count_hmdd += 1
        if in_dbdemc:
            match_count_dbdemc += 1

        if in_dbdemc and in_hmdd:
            edge_type = "both"
        elif in_dbdemc or in_hmdd:
            edge_type = "single"
        else:
            edge_type = "novel"

        results.append({
            "disease": disease_name,
            "mirna_id": mirna_id,
            "disease_id": disease_id,
            "mirna_name": mirna_name,
            "predicted_score": predicted_score,
            "true_label": true_label,
            "rank": rank,
            "in_dbdemc": in_dbdemc,
            "in_hmddv4": in_hmdd,
            "edge_type": edge_type,
            "hmdd_pmids": hmdd_pmids,
            "dbdemc_pmids": dbdemc_pmids,
        })

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "disease", "mirna_id", "disease_id", "mirna_name",
            "predicted_score", "true_label", "rank",
            "in_dbdemc", "in_hmddv4", "edge_type", "hmdd_pmids", "dbdemc_pmids"
        ])
        writer.writeheader()
        writer.writerows(results)

    print(f"\n=== Matching Summary ===")
    print(f"Total records: {len(results)}")
    print(f"Matched in HMDD v4.0: {match_count_hmdd}")
    print(f"Matched in dbDEMC 3.0 (miRExpAll): {match_count_dbdemc}")

    ALL_DISEASES = [
        "Breast Neoplasms", "Ovarian Neoplasms", "Prostatic Neoplasms",
        "Lung Neoplasms", "Stomach Neoplasms",
        "Melanoma", "Leukemia",
    ]
    for dn in ALL_DISEASES:
        subset = [r for r in results if r["disease"] == dn]
        both = sum(1 for r in subset if r["edge_type"] == "both")
        single = sum(1 for r in subset if r["edge_type"] == "single")
        novel = sum(1 for r in subset if r["edge_type"] == "novel")
        print(f"\n{dn}: Both={both}, Single={single}, Novel={novel}")

    print(f"\nWritten {len(results)} records to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()

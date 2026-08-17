# SPAN: Subgraph Progressive Attention Networks for microRNA-disease association prediction
# Copyright (c) 2025 SPAN Authors
# Licensed under the Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0).
# You may use, modify, and distribute this software for non-commercial purposes only.
# For commercial use, please contact the authors to obtain a commercial license.
# See LICENSE for details or visit https://creativecommons.org/licenses/by-nc/4.0/
import os
import re
import glob
from collections import defaultdict
import pandas as pd

# ========== Path setup (relative, no modification of existing programs) ==========
script_dir = os.path.dirname(os.path.abspath(__file__))          # .../plot_performance_comparison
project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
logs_dir = os.path.join(project_root, 'logs')
log_pattern = os.path.join(logs_dir, "retest-round_*-fold_*.log")

compared_csv = os.path.join(script_dir, 'compared_systems.csv')
span_out_csv = os.path.join(script_dir, 'span_10runs.csv')
means_out_csv = os.path.join(script_dir, 'system_means.csv')

metric_cols = ['AUC', 'AUPR', 'ACC', 'F1', 'Precision', 'Recall']

# ========== Parse SPAN logs ==========
log_files = sorted(glob.glob(log_pattern))

all_values = []          # list of [6 floats] for all folds
round_values = defaultdict(list)   # round_num -> list of [6 floats]

for log_file in log_files:
    filename = os.path.basename(log_file)
    match = re.match(r"retest-round_(\d+)-fold_(\d+)\.log", filename)
    if not match:
        print(f"Warning: {filename} does not match expected pattern, skipping.")
        continue

    round_num = int(match.group(1))
    fold_num = int(match.group(2))

    with open(log_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    if len(lines) < 34:
        print(f"Warning: {filename} has fewer than 34 lines, skipping.")
        continue

    data_line = lines[33].strip()
    values = re.split(r'\s+', data_line)

    if len(values) != 6:
        print(f"Warning: {filename} line 34 has unexpected format: {data_line}")
        continue

    float_values = [float(v) for v in values]
    all_values.append(float_values)
    round_values[round_num].append(float_values)

if not all_values:
    raise SystemExit("No valid SPAN log files found. Check that logs/ contains retest-round_*-fold_*.log.")

# ========== Build span_10runs.csv (Y scheme: per-round 5-fold average, 10 rows) ==========
# Keep only the first 10 rounds (Run 0-9).
available_rounds = sorted(round_values.keys())[:10]
span_rows = []
for run_idx, round_num in enumerate(available_rounds):
    cols = round_values[round_num]
    if not cols:
        raise SystemExit(f"Missing folds for round {round_num}.")
    avg = [sum(c) / len(c) for c in zip(*cols)]
    # Round to 4 decimals to match result_span.txt per-round rows (average-then-round).
    avg = [round(v, 4) for v in avg]
    span_rows.append({'Run': run_idx,
                      'AUC': avg[0], 'AUPR': avg[1], 'ACC': avg[2],
                      'F1': avg[3], 'Precision': avg[4], 'Recall': avg[5]})

span_df = pd.DataFrame(span_rows, columns=['Run'] + metric_cols)
span_df.to_csv(span_out_csv, index=False, float_format='%.4f')

# ========== Build system_means.csv (SPAN total mean + baseline rows from compared_systems.csv) ==========
span_total_raw = list(span_df[metric_cols].mean())
# span_total_raw = [sum(c) / len(c) for c in zip(*all_values)]
span_total = [round(v, 4) for v in span_total_raw]

# Read baseline means from compared_systems.csv (skip its SPAN placeholder row of zeros)
compared_df = pd.read_csv(compared_csv)
baseline_df = compared_df[compared_df['System'] != 'SPAN'].copy()

rows = [{'System': 'SPAN',
         'AUC': span_total[0], 'AUPR': span_total[1], 'ACC': span_total[2],
         'F1': span_total[3], 'Precision': span_total[4], 'Recall': span_total[5]}]
for _, r in baseline_df.iterrows():
    rows.append({'System': r['System'],
                 'AUC': r['AUC'], 'AUPR': r['AUPR'], 'ACC': r['ACC'],
                 'F1': r['F1'], 'Precision': r['Precision'], 'Recall': r['Recall']})

means_df = pd.DataFrame(rows, columns=['System'] + metric_cols)
means_df.to_csv(means_out_csv, index=False, float_format='%.4f')

print(f"Generated {span_out_csv} with {len(span_df)} runs.")
print(f"Generated {means_out_csv} with {len(means_df)} systems.")

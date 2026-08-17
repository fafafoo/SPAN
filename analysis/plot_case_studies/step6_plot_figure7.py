# SPAN: Subgraph Progressive Attention Networks for microRNA-disease association prediction
# Copyright (c) 2025 SPAN Authors
# Licensed under the Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0).
# You may use, modify, and distribute this software for non-commercial purposes only.
# For commercial use, please contact the authors to obtain a commercial license.
# See LICENSE for details or visit https://creativecommons.org/licenses/by-nc/4.0/
"""
Step 6: Merge Panel A and Panel B into Figure 7.

Instead of loading pre-rendered PNG images (which causes quality loss from
double rasterization), this script directly renders both panels as subplots
in a single matplotlib figure.

Style: Nature Communications — clean, professional, high-impact journal standard.

Layout: Vertical (Panel A on top, Panel B on bottom)

Panel B uses deduplicated concentric ring layout:
  - Inner ring (r=2.0): 9 disease nodes, evenly distributed.
  - Middle ring (r=5.0): shared miRNAs (linked to >=3 diseases).
  - Inner-outer ring (r=7.0): pair miRNAs (linked to 2 diseases).
  - Outer sectors (r=8.0/9.5/11.0): single-disease miRNAs, grouped by disease
    into +/-32 degree sectors, distributed across 3 radial layers by score.
  - Edges: solid deep blue=both, dashed sky blue=single, solid red=novel.
  - miRNA nodes: large, light-colored circles with names overlapping.
  - For miRNAs with novel edges masked by higher validation, a red border ring is added.
  - Disease labels placed OUTSIDE the inner ring.

Input:
  - panel_a_data.csv       (from step2)
  - panel_b_edges.csv      (from step4)
Output:
  - figure7.png            (raster, 600 DPI)
  - figure7.pdf            (vector, for submission)
"""

import csv
import math
import numpy as np
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch
import matplotlib.gridspec as gridspec

BASE_DIR = Path(__file__).parent

PANEL_A_CSV = BASE_DIR / "panel_a_data.csv"
PANEL_B_CSV = BASE_DIR / "panel_b_edges.csv"
OUTPUT_PNG = BASE_DIR / "figure7.png"
OUTPUT_PDF = BASE_DIR / "figure7.pdf"

PANEL_B_K = 50

COLORS_A = {
    "Breast Cancer": "#C0392B",
    "Ovarian Cancer": "#2471A3",
    "Prostate Cancer": "#D55E00",
    "Lung Cancer": "#8E44AD",
    "Stomach Cancer": "#E91E63",
    "Melanoma": "#607D8B",
    "Leukemia": "#795548",
}
MARKERS_A = {
    "Breast Cancer": "o",
    "Ovarian Cancer": "s",
    "Prostate Cancer": "D",
    "Lung Cancer": "^",
    "Stomach Cancer": "*",
    "Melanoma": "<",
    "Leukemia": ">",
}

DISEASE_ORDER_A = [
    "Ovarian Cancer", "Breast Cancer", "Prostate Cancer",
    "Lung Cancer", "Stomach Cancer",
    "Melanoma", "Leukemia",
]

DISEASE_CONFIG = {
    "Melanoma": {
        "label": "Melanoma", "color": "#607D8B",
        "marker": "<", "angle": 193,
    },
    "Leukemia": {
        "label": "Leukemia", "color": "#795548",
        "marker": ">", "angle": 141,
    },
    "Lung Neoplasms": {
        "label": "Lung\nCancer", "color": "#8E44AD",
        "marker": "^", "angle": 296,
    },
    "Ovarian Neoplasms": {
        "label": "Ovarian\nCancer", "color": "#2471A3",
        "marker": "s", "angle": 39,
    },
    "Breast Neoplasms": {
        "label": "Breast\nCancer", "color": "#C0392B",
        "marker": "o", "angle": 90,
    },
    "Prostatic Neoplasms": {
        "label": "Prostate\nCancer", "color": "#D55E00",
        "marker": "D", "angle": 347,
    },
    "Stomach Neoplasms": {
        "label": "Stomach\nCancer", "color": "#E91E63",
        "marker": "*", "angle": 244,
    },
}

EDGE_STYLES = {
    "both":   {"color": "#1A5276", "linestyle": "-",  "linewidth": 0.9, "alpha": 0.55},
    "single": {"color": "#56B4E9", "linestyle": "--", "linewidth": 0.6, "alpha": 0.40},
    "novel":  {"color": "#CD0101", "linestyle": ":",  "linewidth": 1.2, "alpha": 0.70},
}

NODE_FILL = {
    "both":   "#1A5276",
    "single": "#56B4E9",
    "novel":  "#FFFFFF",
}

NODE_EDGE = {
    "both":   "#1A5276",
    "single": "#56B4E9",
    "novel":  "#CD0101",
}

R_DISEASE = 22.0
R_SHARED = 70.858125
R_PAIR = 98.11125
R_SINGLE_RINGS = [114.463125, 136.265625, 158.068125]
ARC_HALF_SINGLE = math.radians(24)

# ===== 手动逐区域微调（Panel B 重叠修复）=====
# 在 figure7.png 中找到重叠区域，记下涉及的 miRNA 名字，填入下表后重新绘图。
# 尺度参考：1 个数据单位 ≈ 12.5 px；一个标签宽约 12~14 单位，
#           因此把两个重叠标签分开通常需要各自移动 ~8~15 单位。
#   TRANSLATE: 把单个节点（含圆点/连线/标签）平移 (dx, dy)，dx正值右移、负值左移，dy正值上移，负值下移。
#   ROTATE   : 把一组节点绕圆心旋转 deg 度（保持在各自环上，适合"同方式整体挪动"）
MANUAL_NODE_TRANSLATE = {
    "hsa-mir-543": (0, -4), 
    "hsa-mir-665": (2, -6), 
    "hsa-mir-95": (24, 4), 
    "hsa-mir-639": (-8, -6), 
    "hsa-mir-371a": (20, 36), 
    "hsa-mir-423": (28, 36), 
    "hsa-mir-770": (8, 0), 
    "hsa-mir-196a-2": (26, 16), 
    "hsa-mir-198": (-24, 32), 
    "hsa-mir-24-1": (-20, 24), 
    "hsa-mir-26b": (22, -12), 

    "hsa-mir-595": (-6, 6), 
    "hsa-mir-520f": (0, 6), 
    "hsa-mir-622": (0, 6), 
    "hsa-mir-490": (2, 6), 

    "hsa-mir-1468": (0, 4), 

    "hsa-mir-92a-2": (-8, -8), 

    "hsa-mir-1827": (-8, 12), 
    "hsa-mir-429": (-4, 8), 
    "hsa-mir-193b": (-2, -2), 
    "hsa-mir-199b": (-2, -10), 
    "hsa-mir-181d": (8, -8), 
    "hsa-mir-181b-2": (2, 4), 
    "hsa-mir-202": (-6, 0), 
    "hsa-mir-491": (2, -4), 
    "hsa-mir-33b": (4, 4), 
    "hsa-mir-125a": (-40, -4), 
    "hsa-mir-181a-2": (-2, 2),
    "hsa-mir-10b": (2.5, 2.5),

    "hsa-mir-99b": (12, 12), 
    "hsa-mir-18b": (12, 12), 
    "hsa-mir-7-3": (20, 12), 
    "hsa-mir-7-2": (22, 12),
    "hsa-mir-422a": (22, 12),

    "hsa-mir-1271": (20, 0), 
    "hsa-mir-7-1": (20, 0), 
    "hsa-mir-9-3": (20, 0),
    "hsa-mir-216b": (10, 0),

    "hsa-mir-758": (-30, 2), 
    "hsa-mir-192": (-30, 2), 
    "hsa-mir-138-2": (-29, -2), 
    "hsa-mir-335": (-12, -9), 
    "hsa-mir-29c": (-2, 0), 
    "hsa-mir-133b": (4, 0),
    "hsa-mir-30c-2": (16, 16),
    "hsa-mir-139": (0, -2),
    "hsa-mir-660": (-2, 2),
    "hsa-mir-187": (16, 16), 
    "hsa-mir-133a-1": (-24, -24), 
    "hsa-mir-425": (-48, -40), 
    "hsa-mir-454": (-24, -40), 

    "hsa-mir-516b-2": (-2, -8),
    "hsa-mir-10a": (16, -12),

    "hsa-mir-331": (-10, -2),

    "hsa-mir-485": (-16, 4), 
}
MANUAL_NODE_ROTATE = {
    # "cluster1": (["miR-A", "miR-B", "miR-C"], 4.0),  # (名字列表, 度数)
}


def mean_angle(angles):
    s = np.mean(np.sin(angles))
    c = np.mean(np.cos(angles))
    return math.atan2(s, c)


def compute_layout(filtered):
    disease_positions = {}
    disease_angle_map = {}
    for dn, cfg in DISEASE_CONFIG.items():
        angle = math.radians(cfg["angle"])
        x = R_DISEASE * math.cos(angle)
        y = R_DISEASE * math.sin(angle)
        disease_positions[dn] = (x, y)
        disease_angle_map[dn] = angle

    mirna_data = defaultdict(
        lambda: {"diseases": set(), "edges": [], "max_score": -1.0}
    )
    for rec in filtered:
        mn = rec["mirna_name"]
        mirna_data[mn]["diseases"].add(rec["disease"])
        mirna_data[mn]["edges"].append((rec["disease"], rec["edge_type"], rec["predicted_score"]))
        if rec["predicted_score"] > mirna_data[mn]["max_score"]:
            mirna_data[mn]["max_score"] = rec["predicted_score"]

    for data in mirna_data.values():
        types = [t for _, t, _ in data["edges"]]
        if "both" in types:
            data["best_type"] = "both"
        elif "single" in types:
            data["best_type"] = "single"
        else:
            data["best_type"] = "novel"
        data["n_linked"] = len(data["diseases"])
        data["has_novel_edge"] = "novel" in types

    mirna_positions = {}
    mirna_info = {}

    shared_items = [(mn, d) for mn, d in mirna_data.items() if d["n_linked"] >= 3]
    shared_items.sort(key=lambda x: mean_angle([disease_angle_map[d_name] for d_name in x[1]["diseases"]]))
    shared_base_angles = []
    for mn, data in shared_items:
        angles = [disease_angle_map[d_name] for d_name in data["diseases"]]
        shared_base_angles.append(mean_angle(angles))
    min_angular_gap = math.radians(10)
    for idx in range(1, len(shared_base_angles)):
        prev = shared_base_angles[idx - 1]
        curr = shared_base_angles[idx]
        diff = curr - prev
        while diff < -math.pi:
            diff += 2 * math.pi
        while diff > math.pi:
            diff -= 2 * math.pi
        if abs(diff) < min_angular_gap:
            if diff >= 0:
                shared_base_angles[idx] = prev + min_angular_gap
            else:
                shared_base_angles[idx] = prev - min_angular_gap
    for idx, (mn, data) in enumerate(shared_items):
        angle = shared_base_angles[idx]
        x = R_SHARED * math.cos(angle)
        y = R_SHARED * math.sin(angle)
        mirna_positions[mn] = (x, y)
        mirna_info[mn] = data

    pair_items = [(mn, d) for mn, d in mirna_data.items() if d["n_linked"] == 2]
    pair_items.sort(key=lambda x: mean_angle([disease_angle_map[d_name] for d_name in x[1]["diseases"]]))
    pair_base_angles = []
    for mn, data in pair_items:
        angles = [disease_angle_map[d_name] for d_name in data["diseases"]]
        pair_base_angles.append(mean_angle(angles))
    for idx in range(1, len(pair_base_angles)):
        prev = pair_base_angles[idx - 1]
        curr = pair_base_angles[idx]
        diff = curr - prev
        while diff < -math.pi:
            diff += 2 * math.pi
        while diff > math.pi:
            diff -= 2 * math.pi
        if abs(diff) < min_angular_gap:
            if diff >= 0:
                pair_base_angles[idx] = prev + min_angular_gap
            else:
                pair_base_angles[idx] = prev - min_angular_gap
    for idx, (mn, data) in enumerate(pair_items):
        angle = pair_base_angles[idx]
        x = R_PAIR * math.cos(angle)
        y = R_PAIR * math.sin(angle)
        mirna_positions[mn] = (x, y)
        mirna_info[mn] = data

    single_by_disease = defaultdict(list)
    for mn, data in mirna_data.items():
        if data["n_linked"] == 1:
            d_name = list(data["diseases"])[0]
            single_by_disease[d_name].append((mn, data["max_score"], data["best_type"]))

    min_single_angular_gap = math.radians(5)
    for d_name, items in single_by_disease.items():
        items_sorted = sorted(items, key=lambda x: x[1], reverse=True)
        base_angle = disease_angle_map[d_name]
        n = len(items_sorted)

        n_inner = max(int(n * 0.20), 1)
        n_mid = max(int(n * 0.35), 1)
        layers = [
            items_sorted[:n_inner],
            items_sorted[n_inner:n_inner + n_mid],
            items_sorted[n_inner + n_mid:],
        ]

        for layer_idx, layer_items in enumerate(layers):
            r_base = R_SINGLE_RINGS[layer_idx]
            n_layer = len(layer_items)
            layer_angles = []
            if n_layer == 1:
                layer_angles.append(base_angle)
            else:
                # Compute the angular span needed with min gap
                needed_span = (n_layer - 1) * min_single_angular_gap
                available_span = 2 * ARC_HALF_SINGLE
                if needed_span > available_span:
                    # Compress: reduce gap so nodes fit within sector
                    actual_gap = available_span / (n_layer - 1)
                else:
                    actual_gap = min_single_angular_gap
                actual_span = (n_layer - 1) * actual_gap
                start_angle = base_angle - actual_span / 2
                for j in range(n_layer):
                    layer_angles.append(start_angle + actual_gap * j)
            for j, (mn, score, best_type) in enumerate(layer_items):
                angle = layer_angles[j]
                x = r_base * math.cos(angle)
                y = r_base * math.sin(angle)
                mirna_positions[mn] = (x, y)
                mirna_info[mn] = mirna_data[mn]

    return disease_positions, disease_angle_map, mirna_positions, mirna_info


def resolve_label_overlaps(positions, texts, node_positions, char_w, char_h,
                           max_iter=100, max_drift=1.5):
    """Iterative repulsion to resolve label overlaps."""
    n = len(positions)
    if n == 0:
        return []

    pos = [list(p) for p in positions]

    for iteration in range(max_iter):
        step = 0.5 * (1 - iteration / max_iter)
        moved = False
        for i in range(n):
            for j in range(i + 1, n):
                wi = max(len(texts[i]), 3) * char_w
                hi = char_h * 1.2
                wj = max(len(texts[j]), 3) * char_w
                hj = char_h * 1.2

                dx = pos[j][0] - pos[i][0]
                dy = pos[j][1] - pos[i][1]

                min_dx = (wi + wj) / 2 + 0.05
                min_dy = (hi + hj) / 2 + 0.05

                if abs(dx) < min_dx and abs(dy) < min_dy:
                    dist = math.hypot(dx, dy)
                    if dist < 0.01:
                        dx, dy, dist = 1.0, 0.0, 1.0

                    nx, ny = dx / dist, dy / dist
                    pos[i][0] -= nx * step
                    pos[i][1] -= ny * step
                    pos[j][0] += nx * step
                    pos[j][1] += ny * step
                    moved = True

        if not moved:
            break

    # Constrain labels to stay within max_drift of their nodes
    for i in range(n):
        nx, ny = node_positions[i]
        dx = pos[i][0] - nx
        dy = pos[i][1] - ny
        dist = math.hypot(dx, dy)
        if dist > max_drift:
            pos[i][0] = nx + dx * max_drift / dist
            pos[i][1] = ny + dy * max_drift / dist

    return [(p[0], p[1]) for p in pos]


def resolve_node_overlaps(mirna_positions, min_gap=0.4, max_iter=20):
    """Resolve node circular-icon overlaps by pushing overlapping nodes apart.

    After compute_layout(), some miRNA nodes may be too close (especially at
    sector boundaries), causing their scatter circles to overlap and become
    invisible.  This function iteratively pushes such nodes apart along the
    line connecting them.
    """
    names = list(mirna_positions.keys())
    n = len(names)
    if n <= 1:
        return dict(mirna_positions)

    pos = {name: list(mirna_positions[name]) for name in names}

    for _iteration in range(max_iter):
        moved = False
        for i in range(n):
            for j in range(i + 1, n):
                xi, yi = pos[names[i]]
                xj, yj = pos[names[j]]
                dx = xj - xi
                dy = yj - yi
                dist = math.hypot(dx, dy)
                if dist < min_gap:
                    if dist < 0.01:
                        angle = 2 * math.pi * (i + j) / n
                        dx, dy, dist = math.cos(angle), math.sin(angle), 1.0
                    push = (min_gap - dist) / 2 + 0.05
                    nx, ny = dx / dist, dy / dist
                    pos[names[i]][0] -= nx * push
                    pos[names[i]][1] -= ny * push
                    pos[names[j]][0] += nx * push
                    pos[names[j]][1] += ny * push
                    moved = True
        if not moved:
            break

    return {name: tuple(pos[name]) for name in names}


def stagger_inner_mirnas(mirna_positions, mirna_info, offset=0.35):
    """Stagger shared & pair miRNAs radially to reduce visual crowding.

    For miRNAs on the shared (>=3 diseases) and pair (2 diseases) rings,
    sort by angle and alternate: 1st shifts toward center, 2nd shifts away,
    3rd toward center, etc.  This creates a zig-zag pattern that makes
    overlapping nodes distinguishable without changing the angular layout.
    """
    updated = dict(mirna_positions)

    for ring_label in ("shared", "pair"):
        # Collect miRNAs belonging to this ring
        if ring_label == "shared":
            group = [(mn, pos) for mn, pos in mirna_positions.items()
                     if mirna_info[mn]["n_linked"] >= 3]
        else:
            group = [(mn, pos) for mn, pos in mirna_positions.items()
                     if mirna_info[mn]["n_linked"] == 2]

        if len(group) <= 1:
            continue

        # Sort by angle (atan2)
        group.sort(key=lambda item: math.atan2(item[1][1], item[1][0]))

        # Alternate radial offset: even index -> toward center, odd -> away
        for idx, (mn, (x, y)) in enumerate(group):
            dist = math.hypot(x, y)
            if dist < 0.01:
                continue
            direction = -1 if idx % 2 == 0 else 1  # -1 = toward center, +1 = away
            new_dist = dist + direction * offset
            if new_dist < 0.5:  # safety: don't push too close to center
                new_dist = 0.5
            scale = new_dist / dist
            updated[mn] = (x * scale, y * scale)

    return updated


def draw_panel_a(ax):
    """Grouped bar chart: 7 diseases x 3 k-values (25, 50, 75).

    Bar fill color = disease color (matches Panel B).
    Top-k distinguished by alpha: 1.0 / 0.65 / 0.35.
    """
    data = {}
    with open(PANEL_A_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            disease = row["disease"]
            k = int(row["k"])
            mean = float(row["rate_mean"])
            std = float(row["rate_std"])
            data.setdefault(disease, []).append((k, mean, std))

    k_values = [25, 50, 75]
    k_alphas = [1.0, 0.65, 0.35]
    k_labels = ["Top-25", "Top-50", "Top-75"]
    n_diseases = len(DISEASE_ORDER_A)
    n_ks = len(k_values)

    bar_width = 0.08
    group_gap = 0.15
    group_width = n_ks * bar_width + group_gap

    x_centers = np.arange(n_diseases) * group_width

    for ki, k_val in enumerate(k_values):
        offsets = (ki - (n_ks - 1) / 2) * bar_width
        xs = x_centers + offsets
        means = []
        colors = []
        for di, disease in enumerate(DISEASE_ORDER_A):
            if disease not in data:
                means.append(0)
                colors.append("#CCCCCC")
                continue
            row_map = {r[0]: r[1] for r in data[disease]}
            means.append(row_map.get(k_val, 0))
            colors.append(COLORS_A[disease])

        ax.bar(xs, means, width=bar_width,
               color=colors, alpha=k_alphas[ki],
               edgecolor="white", linewidth=0.4,
               zorder=3)

    ax.set_xticks(x_centers)
    ax.set_xticklabels([d.replace(" ", "\n") for d in DISEASE_ORDER_A], fontsize=7)
    # ax.set_ylabel("External validation rate", fontsize=7)
    ax.set_ylim(0.85, 1.02)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0, decimals=0))
    ax.yaxis.set_major_locator(mticker.MultipleLocator(0.05))
    ax.yaxis.set_minor_locator(mticker.MultipleLocator(0.01))
    ax.tick_params(axis="y", labelsize=7)

    # 99% reference line
    ax.axhline(y=0.99, color="#888888", linestyle="--", linewidth=0.8, zorder=0)
    ax.text(n_diseases * group_width - group_gap / 2, 0.99, " 99%",
            fontsize=7, color="#888888", va="bottom", ha="right", zorder=0)

    ax.grid(True, axis="y", which="major", color="#E0E0E0", linewidth=0.5, linestyle="-")
    ax.grid(True, axis="y", which="minor", color="#F0F0F0", linewidth=0.3, linestyle="-")

    # Legend: 3 gray patches with different alpha for Top-k
    legend_handles = []
    for ki, k_val in enumerate(k_values):
        patch = plt.Rectangle((0, 0), 1, 1, facecolor="#808080",
                              alpha=k_alphas[ki], edgecolor="white", linewidth=0.4)
        legend_handles.append((patch, k_labels[ki]))
    ax.legend([h[0] for h in legend_handles], [h[1] for h in legend_handles],
              fontsize=7, loc="lower left", framealpha=0.95,
              borderpad=0.4, handlelength=1.2, handletextpad=0.4,
              labelspacing=0.3, title="Top-k", title_fontsize=7)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.0)
    ax.spines["bottom"].set_linewidth(1.0)
    ax.tick_params(width=1.0, length=4)

    ax.set_title("External validation rate by cancer type", fontsize=7, pad=4)

    ax.text(-0.08, 1.14, "a", transform=ax.transAxes,
            fontsize=8, fontweight="bold", va="top", ha="right")


def draw_panel_b(ax, k=PANEL_B_K):
    records = []
    with open(PANEL_B_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rank = int(row["rank"])
            if rank <= k:
                records.append({
                    "disease": row["disease"],
                    "mirna_name": row["mirna_name"],
                    "predicted_score": float(row["predicted_score"]),
                    "rank": rank,
                    "edge_type": row["edge_type"],
                })

    disease_positions, disease_angle_map, mirna_positions, mirna_info = compute_layout(records)

    # Resolve node overlaps — push overlapping circular icons apart
    mirna_positions = resolve_node_overlaps(mirna_positions, min_gap=4.3605, max_iter=20)

    # Stagger inner-ring miRNAs (shared & pair) radially to reduce crowding
    mirna_positions = stagger_inner_mirnas(mirna_positions, mirna_info, offset=3.8475)

    # --- 手动逐区域微调：放在最后 ---
    for mn, (dx, dy) in MANUAL_NODE_TRANSLATE.items():
        if mn in mirna_positions:
            mx, my = mirna_positions[mn]
            mirna_positions[mn] = (mx + dx, my + dy)
        else:
            print(f"[WARN] MANUAL_NODE_TRANSLATE: {mn!r} not in layout, skipped")

    for _g, (_names, _deg) in MANUAL_NODE_ROTATE.items():
        _rad = math.radians(_deg)
        _ca, _sa = math.cos(_rad), math.sin(_rad)
        for mn in _names:
            if mn in mirna_positions:
                mx, my = mirna_positions[mn]
                mirna_positions[mn] = (mx * _ca - my * _sa, mx * _sa + my * _ca)
            else:
                print(f"[WARN] MANUAL_NODE_ROTATE: {mn!r} not in layout, skipped")

    # All miRNA nodes use a fixed size (no longer scaled by predicted_score)
    MIRNA_NODE_SIZE = 88

    drawn_edges = set()
    for rec in records:
        dn, mn = rec["disease"], rec["mirna_name"]
        key = (dn, mn)
        if key in drawn_edges:
            continue
        drawn_edges.add(key)

        dx, dy = disease_positions[dn]
        mx, my = mirna_positions[mn]
        style = EDGE_STYLES[rec["edge_type"]]

        arrow = FancyArrowPatch(
            (dx, dy), (mx, my),
            arrowstyle="-",
            connectionstyle="arc3,rad=0.12",
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=style["linewidth"],
            alpha=style["alpha"],
            zorder=1,
        )
        ax.add_patch(arrow)

    for mn, (mx, my) in mirna_positions.items():
        info = mirna_info[mn]
        size = MIRNA_NODE_SIZE
        etype = info["best_type"]

        facecolor = NODE_FILL[etype]
        edgecolor = NODE_EDGE[etype]
        alpha = 0.75
        linewidth = 1.0

        # Novel miRNA: distinguish by thicker red edge only (size kept equal)
        if etype == "novel":
            linewidth = 2.0

        ax.scatter(mx, my, s=size, c=facecolor, edgecolors=edgecolor,
                   linewidths=linewidth, alpha=alpha, zorder=2)

    # --- Red border rings for miRNAs with masked novel edges (zorder=2) ---
    for mn, (mx, my) in mirna_positions.items():
        info = mirna_info[mn]
        etype = info["best_type"]
        if info.get("has_novel_edge") and etype != "novel":
            base_size = MIRNA_NODE_SIZE
            ring_size = base_size * 1.5
            ax.scatter(mx, my, s=ring_size, c="none", edgecolors="#CD0101",
                       linewidths=2.0, alpha=0.75, zorder=2)

    # --- miRNA name labels on nodes (zorder=5) ---
    # Collect label positions, resolve overlaps, then draw
    label_items = []
    for mn, (mx, my) in mirna_positions.items():
        info = mirna_info[mn]
        dist = math.hypot(mx, my)
        if dist > 0:
            offset = 3.8475
            lx = mx + offset * mx / dist
            ly = my + offset * my / dist
        else:
            lx, ly = mx, my
        label_items.append((mn, lx, ly, mx, my, info["best_type"]))

    label_positions = [(it[1], it[2]) for it in label_items]
    label_texts = [it[0] for it in label_items]
    label_nodes = [(it[3], it[4]) for it in label_items]
    adjusted = resolve_label_overlaps(label_positions, label_texts, label_nodes,
                                      char_w=0.011, char_h=0.019, max_drift=8.0,
                                      max_iter=100)

    for i, (mn, _, _, _, _, etype) in enumerate(label_items):
        lx, ly = adjusted[i]
        text_color = "#A30303" if etype == "novel" else "#020732" 
        ax.text(lx, ly, mn, fontsize=5, fontweight="normal",
                color=text_color, ha="center", va="center", zorder=5)

    for dn, cfg in DISEASE_CONFIG.items():
        dx, dy = disease_positions[dn]
        ax.scatter(dx, dy, s=134, c=cfg["color"], marker=cfg["marker"],
                   edgecolors="none", linewidths=1.5, zorder=3)

    DISEASE_LEGEND_ORDER = [
        "Ovarian Neoplasms", "Breast Neoplasms", "Prostatic Neoplasms",
        "Lung Neoplasms", "Stomach Neoplasms",
        "Melanoma", "Leukemia",
    ]
    legend_elements = [
        Line2D([0], [0], color=EDGE_STYLES["novel"]["color"],
               linestyle=EDGE_STYLES["novel"]["linestyle"],
               linewidth=1.5, label="Novel"),
        Line2D([0], [0], color=EDGE_STYLES["both"]["color"],
               linestyle=EDGE_STYLES["both"]["linestyle"],
               linewidth=1.5, label="Both DBs"),
        Line2D([0], [0], color=EDGE_STYLES["single"]["color"],
               linestyle=EDGE_STYLES["single"]["linestyle"],
               linewidth=1.5, label="Single DB"),
        Line2D([0], [0], color="white", label=""),
    ]
    for dn in DISEASE_LEGEND_ORDER:
        cfg = DISEASE_CONFIG[dn]
        legend_elements.append(
            Line2D([0], [0], marker=cfg["marker"], color="w",
                   markerfacecolor=cfg["color"],
                   markersize=7, markeredgecolor="none",
                   label=cfg["label"].replace("\n", " "))
        )
    legend_elements.append(Line2D([0], [0], color="white", label=""))
    legend_elements.extend([
        Line2D([0], [0], marker="o", color="w", markerfacecolor=NODE_FILL["novel"],
               markersize=7, markeredgecolor=NODE_EDGE["novel"],
               markeredgewidth=1.0, label="Novel (no validation)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=NODE_FILL["both"],
               markersize=7, markeredgecolor=NODE_EDGE["both"],
               markeredgewidth=1.0, label="Validated (both DBs)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=NODE_FILL["single"],
               markersize=7, markeredgecolor=NODE_EDGE["single"],
               markeredgewidth=1.0, label="Validated (single DB)"),
    ])

    ax.text(-0.08, 1.00, "b", transform=ax.transAxes,
            fontsize=8, fontweight="bold", va="top", ha="right")

    # Dynamically fit all content with a margin for node/ring radius
    pts = list(mirna_positions.values()) + [disease_positions[dn] for dn in disease_positions]
    content_r = max(math.hypot(x, y) for x, y in pts)
    margin = content_r + 15.0   # +15 covers scatter marker + red ring radius
    ax.set_xlim(-margin, margin)
    ax.set_ylim(-margin, margin)
    ax.set_aspect("equal")
    ax.axis("off")

    leg = ax.legend(handles=legend_elements, loc="lower left", fontsize=7,
                    bbox_to_anchor=(-0.10, -0.00),
                    frameon=False, framealpha=0.9, ncol=1,
                    borderpad=0.5, handlelength=1.5, handletextpad=0.5,
                    labelspacing=0.3)
    leg.set_zorder(0)


def main():
    plt.rcParams.update({
        "font.family": "Arial",
        "font.size": 7,
        "axes.linewidth": 1.0,
        "xtick.major.width": 1.0,
        "ytick.major.width": 1.0,
        "xtick.major.size": 4,
        "ytick.major.size": 4,
        "xtick.minor.size": 2,
        "ytick.minor.size": 2,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": False,
        "ytick.right": False,
        "legend.frameon": True,
        "legend.edgecolor": "#CCCCCC",
        "legend.fancybox": False,
    })

    fig = plt.figure(figsize=(7.2, 12.0))

    # Vertical layout: Panel A (top, full width, short) | Panel B (bottom, full width, tall)
    outer = gridspec.GridSpec(2, 1, height_ratios=[0.10, 0.90], hspace=-0.35)

    # Panel A: top row, full width
    ax_a = fig.add_subplot(outer[0])

    # Panel B: bottom row, full width
    ax_b = fig.add_subplot(outer[1])

    draw_panel_a(ax_a)
    draw_panel_b(ax_b, k=PANEL_B_K)

    fig.tight_layout(w_pad=0.5)

    fig.savefig(OUTPUT_PNG, dpi=600, bbox_inches="tight")
    fig.savefig(OUTPUT_PDF, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved Figure 7 to {OUTPUT_PNG}")
    print(f"Saved Figure 7 to {OUTPUT_PDF}")


if __name__ == "__main__":
    main()

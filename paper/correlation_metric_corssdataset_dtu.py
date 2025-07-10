import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap

from collections import defaultdict
from gta.common import *
import re

FEATURES = (
    DUST3R,
    MAST3R,
    MIDAS,
    DINOV2,
    DINO16,
    SAM,
    CLIP,
    RADIO,
    MAE,
    DIFT,
    IUVRGB,
)

METHODS = (
    METHOD_G,
    # METHOD_T,
    # METHOD_A,
)

METRICS = (
    METRIC_PSNR,
    METRIC_SSIM,
    METRIC_LPIPS,
    METRIC_ACC,
    # METRIC_ACC_MED,
    METRIC_COMP,
    # METRIC_COMP_MED,
    METRIC_DIST,
    # METRIC_DIST_MED,
)

SCENES = (*SCENES_DTU,)
morandi_colors = ['#F2D0A9', '#D98E73', '#B24C33']
morandi_cmap = mcolors.LinearSegmentedColormap.from_list("morandi", morandi_colors, N=100)


if __name__ == "__main__":
    df = load_metrics(SCENES, METHODS, FEATURES, METRICS)

    corr_matrices = {}

    grouped = df.groupby(["method_tag", "metric_tag", "feature_tag"])[
        "metric_value"
    ].mean()

    rows = defaultdict(list)
    metric_groups = {}
    for method in METHODS:
        for metric in METRICS:
            for feature in FEATURES:
                # method_name = method.full_name.replace("feat2gs-", "")
                method_name = re.search(r'\\textbf\{(.+?)\}', method.full_name)
                value = grouped[method.tag][metric.tag][feature.tag]
                # if metric.tag == METRIC_LPIPS.tag:
                if metric.order == -1:
                    value = -value  # negate the value of the metric
                # rows[f"{metric.full_name}-{method_name.group(1)}"].append(value)
                rows[f"{metric.full_name}"].append(value)
                metric_groups[metric.full_name] = metric.group



                # rows[f"{metric.tag}-{method_name}"].append(
                #     grouped[method.tag][metric.tag][feature.tag]
                # )

    data = pd.DataFrame(rows)

    corr_matrix = data.corr(method='spearman').values

    features = data.columns.tolist()

    plt.figure(figsize=(6, 6))
    ax = plt.gca()
    ax.set_aspect('equal')
    # ax.set_title(f'Correlation Matrix', fontsize=25, pad=20)

    # group_colors = {
    #     '2d': '#5E7460',
    #     '3d': '#4A6B8A'
    # }

    # group_colors = {
    #     '2d': '#415548',
    #     '3d': '#003248'
    # }

    group_colors = {
        '2d': '#E19645',
        '3d': '#871510'
    }


    sns.set(font_scale=1.1)
    heatmap = sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap=morandi_cmap,
                        xticklabels=features, yticklabels=features,
                        vmin=0.4, vmax=1, center=0.7, ax=ax, 
                        cbar=False,
                        linewidths=0,
                        linecolor='none', 
                        square=True,
                        rasterized=True,
                        # cbar_kws={'label': ''},
                        annot_kws={"size": 18})

    ax = plt.gca()
    for tick_label in ax.get_xticklabels():
        metric_name = tick_label.get_text()
        tick_label.set_color(group_colors[metric_groups[metric_name]])
    
    for tick_label in ax.get_yticklabels():
        metric_name = tick_label.get_text()
        tick_label.set_color(group_colors[metric_groups[metric_name]])

    current_pos = 0
    for group in ['2d', '3d']:
        group_metrics = [m for m in features if metric_groups[m] == group]
        if group_metrics:
            group_size = len(group_metrics)
            mid_pos = current_pos + group_size/2
            ax.text(-0.6, mid_pos, f"{group.upper()} Metrics", 
                   ha='right', va='center', fontsize=22, rotation=90,
                   color=group_colors[group], weight='bold')
            ax.text(mid_pos, len(features) + 1., f"{group.upper()} Metrics", 
                   ha='center', va='bottom', fontsize=22,
                   color=group_colors[group], weight='bold')
            current_pos += group_size

    d2_metrics = [m for m in features if metric_groups[m] == '2d']
    d2_size = len(d2_metrics)
    
    if d2_size > 0 and d2_size < len(features):
        rect = plt.Rectangle((d2_size, 0), 0.01, len(features), 
                           fill=True, color='#ffffff', lw=2)
        ax.add_patch(rect)
        
        rect = plt.Rectangle((0, d2_size), len(features), 0.01, 
                           fill=True, color='#ffffff', lw=2)
        ax.add_patch(rect)

    plt.xticks(rotation=0, fontsize=18)
    plt.yticks(rotation=90, fontsize=18)

    plt.tight_layout()

    plt.savefig('figs/DTU.png', dpi=300, bbox_inches='tight')
    plt.savefig('figs/DTU.pdf', format='pdf', dpi=300, bbox_inches='tight')

    plt.close()
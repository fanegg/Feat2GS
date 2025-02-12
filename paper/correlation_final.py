import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.gridspec as gridspec

from collections import defaultdict
from gta.common import *

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
    METHOD_T,
    METHOD_A,
)

METRICS = (
    METRIC_PSNR,
    METRIC_SSIM,
    METRIC_LPIPS,
)

SCENES = (*SCENES_LLFF, *SCENES_DL3DV, *SCENES_CASUAL, *SCENES_MIPNERF360, *SCENES_MVIMGNET, *SCENES_TANDT)

morandi_colors = ['#F2D0A9', '#D98E73', '#B24C33']
morandi_cmap = mcolors.LinearSegmentedColormap.from_list("morandi", morandi_colors, N=100)

def create_correlation_heatmap(corr_matrix, features, metric_tag, metric_full_name):
    plt.figure(figsize=(12, 10))
    ax = plt.gca()
    
    sns.set(font_scale=1.1)
    heatmap = sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap=morandi_cmap,
                          xticklabels=features, yticklabels=features,
                          vmin=0.4, vmax=1, center=0.7, ax=ax, cbar_kws={'label': ''})
    
    plt.xticks(rotation=0)
    plt.yticks(rotation=0)

    plt.title(f'Correlation Matrix - {metric_full_name}', fontsize=16, pad=20)

    plt.tight_layout()
    
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=10)
    cbar.ax.yaxis.set_ticks_position('left')
    cbar.ax.yaxis.set_label_position('left')
    
    plt.savefig(f'figs/correlation_{metric_tag}_crossdataset.png', dpi=300, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    df = load_metrics(SCENES, METHODS, FEATURES, METRICS)

    corr_matrices = {}

    for metric in METRICS:
        grouped = df.groupby(["method_tag", "metric_tag", "feature_tag"])[
            "metric_value"
        ].mean()

        rows = defaultdict(list)

        for method in METHODS:
            for feature in FEATURES:
                method_name = method.tag.replace("feat2gs-", "")
                rows[f"{method_name}"].append(
                    grouped[method.tag][metric.tag][feature.tag]
                )

        data = pd.DataFrame(rows)

        corr_matrices[metric.tag] = data.corr(method='spearman').values

        features = data.columns.tolist()

        # create_correlation_heatmap(corr_matrices[metric.tag], features, metric.tag, metric.full_name)

    # Create combined heatmap
    fig = plt.figure(figsize=(23, 8))
    gs = gridspec.GridSpec(1, 4, width_ratios=[1, 1, 1, 0.08])

    for i, (metric_tag, corr_matrix) in enumerate(corr_matrices.items()):
        ax = fig.add_subplot(gs[i])
        
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap=morandi_cmap,
                    xticklabels=features, yticklabels=features,
                    vmin=0.4, vmax=1, center=0.7, ax=ax, cbar=False,
                    annot_kws={"size": 40})
        
        ax.set_title(f'{METRICS[i].full_name}', fontsize=40, pad=20)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0, fontsize=30)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=30)
        ax.tick_params(axis='both', which='major', length=10, width=2)
        
    cbar_ax = fig.add_subplot(gs[3])
    sm = plt.cm.ScalarMappable(cmap=morandi_cmap, norm=plt.Normalize(vmin=0.4, vmax=1))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.ax.tick_params(labelsize=30)
    cbar.ax.yaxis.set_ticks_position('left')
    cbar.ax.yaxis.set_label_position('left')

    cbar.outline.set_visible(False)

    plt.tight_layout()
    plt.savefig('figs/correlation.png', dpi=300, bbox_inches='tight')
    plt.savefig('figs/correlation.pdf', format='pdf', dpi=300, bbox_inches='tight')
    plt.close()

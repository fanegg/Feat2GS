import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.gridspec as gridspec
import re

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
DATASETS = tuple(dict.fromkeys(scene.dataset for scene in SCENES))



morandi_colors = ['#F2D0A9', '#D98E73', '#B24C33']
morandi_cmap = mcolors.LinearSegmentedColormap.from_list("morandi", morandi_colors)


def topsis(decision_matrix, weights, is_benefit):
    normalized = decision_matrix / np.sqrt(np.sum(decision_matrix**2, axis=0))
    
    weighted = normalized * weights
    
    ideal_best = np.max(weighted, axis=0) * is_benefit + np.min(weighted, axis=0) * (1 - is_benefit)
    ideal_worst = np.min(weighted, axis=0) * is_benefit + np.max(weighted, axis=0) * (1 - is_benefit)
    
    s_best = np.sqrt(np.sum((weighted - ideal_best)**2, axis=1))
    s_worst = np.sqrt(np.sum((weighted - ideal_worst)**2, axis=1))
    
    scores = s_worst / (s_best + s_worst)
    
    ranks = len(scores) - np.argsort(np.argsort(scores))

    return scores, ranks

if __name__ == "__main__":
    df = load_metrics(SCENES, METHODS, FEATURES, METRICS)

    weights = np.array([1/3, 1/3, 1/3])
    is_benefit = np.array([1, 1, 0])

    corr_matrices = {}
    
    for method in METHODS:
        rows = defaultdict(list)
        for dataset in DATASETS:
            decision_matrix = []
            for feature in FEATURES:
                feature_metrics = []
                for metric in METRICS:
                    metric_value = df[(df['dataset_tag'] == dataset.tag) & 
                                      (df['method_tag'] == method.tag) & 
                                      (df['feature_tag'] == feature.tag) & 
                                      (df['metric_tag'] == metric.tag)]['metric_value'].mean()
                    feature_metrics.append(metric_value)
                decision_matrix.append(feature_metrics)
            
            decision_matrix = np.array(decision_matrix)
            scores, ranks = topsis(decision_matrix, weights, is_benefit)
            
            method_name = re.sub(r'\\textbf{(.*?)}', r'\1', method.full_name)
            dataset_name = dataset.full_name.replace("T\\&T", "T&T").replace("MipNeRF 360", "MipNeRF")
            for feature, score, rank in zip(FEATURES, scores, ranks):
                rows[dataset_name].append(score)

        data = pd.DataFrame(rows, index=[feature.full_name for feature in FEATURES])

        features = list(rows.keys())
        # features = [col.split('-')[0] for col in data.columns]
        # features = list(dict.fromkeys(features))

        corr_matrices[method_name] = data.corr(method='spearman').values

    # Create combined heatmap
    fig = plt.figure(figsize=(30, 8))
    
    outer_gs = gridspec.GridSpec(1, 2, width_ratios=[23.5, 0.5], wspace=0.06)  # 控制colorbar和热力图组之间的间距
    
    heatmap_gs = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=outer_gs[0], wspace=0.25)  # 控制热力图之间的间距
    
    cbar_gs = outer_gs[1]

    for i, (key, corr_matrix) in enumerate(corr_matrices.items()):
        ax = fig.add_subplot(heatmap_gs[i])
        
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap=morandi_cmap,
                    xticklabels=features, yticklabels=features,
                    vmin=0.4, vmax=1, center=0.7, ax=ax, cbar=False,
                    annot_kws={"size": 20})
        
        ax.set_title(key, fontsize=30, pad=20)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, fontsize=20, ha='center')
        ax.set_yticklabels(ax.get_yticklabels(), rotation=45, fontsize=20, ha='right')
        ax.tick_params(axis='both', which='major', length=10, width=2)
        
    cbar_ax = fig.add_subplot(cbar_gs)
    sm = plt.cm.ScalarMappable(cmap=morandi_cmap, norm=plt.Normalize(vmin=0.4, vmax=1))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.ax.tick_params(labelsize=20)
    cbar.ax.yaxis.set_ticks_position('left')
    cbar.ax.yaxis.set_label_position('left')

    cbar.outline.set_visible(False)

    plt.tight_layout()
    plt.savefig('figs/correlation_dataset.png', dpi=300, bbox_inches='tight')
    plt.savefig('figs/correlation_dataset.pdf', format='pdf', dpi=300, bbox_inches='tight')
    plt.close()
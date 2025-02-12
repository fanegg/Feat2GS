import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap

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

def plot_correlation_heatmap(corr_matrix, features, title, filename, vmin=-1, vmax=1, center=0):
    plt.figure(figsize=(12, 10))
    ax = plt.gca()

    sns.set(font_scale=1.1)
    heatmap = sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap=morandi_cmap,
                        xticklabels=features, yticklabels=features,
                        vmin=vmin, vmax=vmax, center=center, ax=ax, cbar_kws={'label': ''})

    datasets = list(set(feature.split('-')[1] for feature in features))
    for i, dataset1 in enumerate(datasets):
        for j, dataset2 in enumerate(datasets):
            dataset1_features = [f for f in features if f.endswith(f'-{dataset1}')]
            dataset2_features = [f for f in features if f.endswith(f'-{dataset2}')]
            dataset1_size = len(dataset1_features)
            dataset2_size = len(dataset2_features)
            
            start_x = sum(len([f for f in features if f.endswith(f'-{d}')]) for d in datasets[:j])
            start_y = sum(len([f for f in features if f.endswith(f'-{d}')]) for d in datasets[:i])
            
            rect = plt.Rectangle((start_x, start_y), dataset2_size, dataset1_size, 
                                fill=False, edgecolor='#ffffff', lw=2)
            ax.add_patch(rect)

    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)

    plt.title(title)

    plt.tight_layout()

    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=10)
    cbar.ax.yaxis.set_ticks_position('left')
    cbar.ax.yaxis.set_label_position('left')

    plt.savefig(f'figs/{filename}', dpi=300, bbox_inches='tight')
    plt.close()



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

    rows = defaultdict(list)
    rankings = defaultdict(list)
    
    for method in METHODS:
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
            
            method_name = method.full_name.replace("Feat2gs-", "")
            dataset_name = dataset.full_name.replace("T\\&T", "T&T").replace("MipNeRF 360", "MipNeRF")
            for feature, score, rank in zip(FEATURES, scores, ranks):
                rows[f"{dataset_name}-{method_name}"].append(score)
                rankings[f"{dataset_name}-{method_name}"].append(rank)

    data = pd.DataFrame(rows, index=[feature.full_name for feature in FEATURES])
    ranking_data = pd.DataFrame(rankings, index=[feature.full_name for feature in FEATURES])

    features = list(rows.keys())
    # features = [col.split('-')[0] for col in data.columns]
    # features = list(dict.fromkeys(features))

    combined_corr = data.corr(method='spearman').values
    ranking_corr = ranking_data.corr(method='spearman').values

    plot_correlation_heatmap(combined_corr, features, 
                             "Score Correlation Matrix", 
                             "score_correlation.png",
                             vmin=0.4, vmax=1, center=0.7)

    # plot_correlation_heatmap(ranking_corr, features, 
    #                          "Ranking Correlation Matrix", 
    #                          "ranking_correlation.png",
    #                          vmin=0.4, vmax=1, center=0.7)
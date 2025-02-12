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

if __name__ == "__main__":
    df = load_metrics(SCENES, METHODS, FEATURES, METRICS)

    for metric in METRICS:
        grouped = df.groupby(["dataset_tag", "method_tag", "metric_tag", "feature_tag"])[
            "metric_value"
        ].mean()

        rows = defaultdict(list)

        for dataset in DATASETS:
            for method in METHODS:
                for feature in FEATURES:
                    method_name = method.full_name.replace("Feat2gs-", "")
                    dataset_name = dataset.full_name.replace("T\\&T", "T&T").replace("MipNeRF 360", "MipNeRF")
                    rows[f"{method_name}-{dataset_name}"].append(
                        grouped[dataset.tag][method.tag][metric.tag][feature.tag]
                    )

        data = pd.DataFrame(rows)

        # corr_data = data.corr().values
        corr_data = data.corr(method='spearman').values
        # corr_data = data.corr(method='kendall').values
        

        features = data.columns.tolist()

        links = ['#4A6B8A', '#F2D0A9', '#D98E73', '#B24C33', '#5E7460'] #links

        # morandi_colors = ['#E0D4C3', '#CFC0BD', '#B4A89F', '#98877D', '#7D6B66']
        # morandi_colors = ['#F2D0A9', '#D98E73', '#B24C33']
        # morandi_colors = ['#4A6B8A', '#F2D0A9', '#D98E73', '#B24C33']
        # # morandi_colors = ['#8AA2A9', '#F2D0A9', '#D98E73', '#B24C33']
        # positions = [0, 0.5, 0.75, 1]

        # morandi_colors = ['#4A6B8A', '#F2D0A9', '#F2D0A9', '#D98E73', '#B24C33']
        # positions = [0, 0.25, 0.75, (0.75+1)/2, 1] #[0, 0.3, 0.7, 0.85, 1]
        # n_bins = 100
        # morandi_cmap = LinearSegmentedColormap.from_list("custom_morandi", list(zip(positions, morandi_colors)), N=n_bins)

        morandi_colors = ['#F2D0A9', '#D98E73', '#B24C33']
        morandi_cmap = mcolors.LinearSegmentedColormap.from_list("morandi", morandi_colors)

        plt.figure(figsize=(12, 10))
        ax = plt.gca()

        sns.set(font_scale=1.1)
        heatmap = sns.heatmap(corr_data, annot=True, fmt='.2f', cmap=morandi_cmap,
                            xticklabels=features, yticklabels=features,
                            vmin=0.4, vmax=1, center=0.7, ax=ax, cbar_kws={'label': ''})

        datasets = list(set(feature.split('-')[1] for feature in features))
        current_pos = 0
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

        plt.title(f'Correlation Matrix - {metric.full_name}', fontsize=16, pad=20)

        plt.tight_layout()

        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=10)
        cbar.ax.yaxis.set_ticks_position('left')
        cbar.ax.yaxis.set_label_position('left')

        plt.savefig(f'figs/correlation_{metric.full_name}.png', dpi=300, bbox_inches='tight')

        plt.close()
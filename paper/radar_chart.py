import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
from collections import defaultdict
from gta.common import *

from matplotlib import rcParams
rcParams['font.family'] = 'Liberation Sans'
# rcParams['font.family'] = 'Nimbus Sans'


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
    # IUVRGB,
)

METHODS = (
    METHOD_T,
    METHOD_A,
    METHOD_G,
)

METRICS = (
    METRIC_PSNR,
    METRIC_SSIM,
    METRIC_LPIPS,
)

SCENES = (*SCENES_LLFF, *SCENES_DL3DV, *SCENES_CASUAL, *SCENES_MIPNERF360, *SCENES_MVIMGNET, *SCENES_TANDT)

df = load_metrics(SCENES, METHODS, FEATURES, METRICS)
rows = defaultdict(list)

grouped = df.groupby(["method_tag", "metric_tag", "feature_tag"])[
    "metric_value"
].mean()

for method in METHODS:
    for metric in METRICS:
        for feature in FEATURES:
            method_name = method.tag.replace("feat2gs-", "")
            rows[f"{method_name}-{metric.tag}"].append(
                grouped[method.tag][metric.tag][feature.tag]
            )

data = pd.DataFrame(rows)

categories = data.columns.tolist()

angles = np.linspace((0.5-1/len(categories))*np.pi + 2*np.pi, (0.5-1/len(categories))*np.pi, len(categories), endpoint=False)

fig, ax = plt.subplots(figsize=(8, 8))

extension_factor = 1.075

colors = [
    '#B2CBC2',  # 浅青色
    # '#DA8E4F',  # 深橙色
    '#BBC990',  # 浅橄榄色
    # '#FCB275',  # 亮橙色
    # '#CAC691',  # 浅橄榄色
    # '#A8C990',  # 嫩绿色
    '#6B859E',  # 灰蓝色
    '#B45342',  # 红棕色
    # '#C67752',  # 橙红色
    # '#84291C',  # 红棕色
    # '#B9551F',  # 橙色
    # '#736A93',  # 紫色

    '#DCAC99',  # 浅粉色
    # '#8F6B93',  # 薰衣草紫色
    '#6F936B',  # 绿色
    '#EBA062',  # 浅橙色

    '#FED273',  # 浅黄色
    # '#3D8C84',  # 青蓝色
    # '#8C956C',  # 橄榄绿色
    # '#C39EC6',  # 淡薰衣草紫色
    # '#ABA0C7',  # 淡紫色
    '#9A8EB4',  # 芋泥色
    # '#C18830',  # 土黄色
    # '#374B6C',  # 深蓝色
    # '#817265',  # 咖啡色
    '#706052',  # 深咖啡色
]

def draw_polygon(values, label, color):
    values = np.array(values)
    x = np.cos(angles) * values
    y = np.sin(angles) * values
    x = np.append(x, x[0])
    y = np.append(y, y[0])
    ax.plot(x, y, 'o-', linewidth=2, label=label, color=color)
    ax.fill(x, y, alpha=0.25, color=color)

def normalize_data(data, is_lpips=False):
    min_val = min(data)
    max_val = max(data)
    if is_lpips:
        return [0.1 + 0.9 * (max_val - x) / (max_val - min_val) for x in data]
    else:
        return [0.1 + 0.9 * (x - min_val) / (max_val - min_val) for x in data]

normalized_data = data.copy()
for column in normalized_data.columns:
    if 'LPIPS' in column:
        normalized_data[column] = normalize_data(normalized_data[column], is_lpips=True)
    else:
        normalized_data[column] = normalize_data(normalized_data[column])

for i, (index, row) in enumerate(normalized_data.iterrows()):
    feature_name = FEATURES[i].full_name
    # if feature_name == "DUST3R":
    #     feature_name = "DUSt3R"
    # elif feature_name == "MAST3R":
    #     feature_name = "MASt3R"
    # elif feature_name == "MIDAS":
    #     feature_name = "MiDaS"
    # elif feature_name == "DINOV2":
    #     feature_name = "DINOv2"
    draw_polygon(row.values, feature_name, colors[i])

tolerance = 1e-10
for i, point in enumerate(categories):
    angle = angles[i]
    x = np.cos(angle) * extension_factor
    y = np.sin(angle) * extension_factor
    ha = 'center' if abs(x) < tolerance else 'left' if x > 0 else 'right'
    va = 'center' if abs(y) < tolerance else 'bottom' if y > 0 else 'top'

    if point.startswith('G-'):
        text = r'$\mathbf{G}$' + point[1:]
        ax.text(x, y, text, color='#9A6500', ha=ha, va=va, fontsize=14)
    elif point.startswith('T-'):
        text = r'$\mathbf{T}$' + point[1:]
        ax.text(x, y, text, color='#564668', ha=ha, va=va, fontsize=14)
    elif point.startswith('A-'):
        text = r'$\mathbf{A}$' + point[1:]
        ax.text(x, y, text, color='#444B0B', ha=ha, va=va, fontsize=14)
    else:
        ax.text(x, y, point, ha=ha, va=va, fontsize=14)

lim_max = 1.2
ax.set_ylim(-lim_max, lim_max)
ax.set_xlim(-lim_max, lim_max)

ax.set_axis_off()

ax.set_aspect('equal')

plt.legend(loc='upper right', bbox_to_anchor=(1.05, 1.1), ncol=1, fontsize=12, 
           borderpad=0.3, labelspacing=0.35, handletextpad=0.5)

ax.set_facecolor('#f0f0f0')

ax.grid(True, color='gray', alpha=0.3)

for i in np.arange(0.2, 1.2, 0.2):
    values = [i] * len(categories)
    x = np.cos(angles) * values
    y = np.sin(angles) * values
    x = np.append(x, x[0])
    y = np.append(y, y[0])
    ax.plot(x, y, '--', color='gray', alpha=0.3)

for angle in angles:
    x = np.cos(angle) * extension_factor
    y = np.sin(angle) * extension_factor
    ax.plot([0, x], [0, y], '--', color='gray', alpha=0.3)

rcParams['text.usetex'] = False

x, y = -0.25, 1.3

def color_text(ax, text, color, x_offset):
    t = ax.text(x + x_offset, y, r'$\mathbf{' + text + r'}$', color=color, fontweight='bold', 
                ha='left', va='center', fontsize=16)
    return t

ax.text(x-0.82, y, "= Geometry         = Texture         = Geometry & Texture", 
        ha='left', va='center', fontsize=16, color='black')

color_text(ax, "G", '#9A6500', -0.9)
color_text(ax, "T", '#564668', -0.3)
color_text(ax, "A", '#444B0B', 0.2)

plt.tight_layout()

plt.savefig('figs/radar_chart.png', dpi=300, bbox_inches='tight')
plt.savefig('figs/radar_chart.pdf', format='pdf', dpi=300, bbox_inches='tight')

plt.close(fig)

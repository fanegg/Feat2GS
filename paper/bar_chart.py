import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import os

from collections import defaultdict
from pathlib import Path
from gta.common import *

Analysis = 2
if Analysis == 1:
    symbols = ['+', '+', '=', '≈']
    FEATURES = (
        DINOV2,
        CLIP,
        SAM,
        DN2_C_S,
        RADIO,
    )

    METHODS = (
        METHOD_G,
    )
elif Analysis == 2:
    symbols = ['+', '+', '=', '>']
    FEATURES = (
        RADIO,
        MAE,
        IUVRGB,
        R_MAE_I,
        MAST3R,
    )
    METHODS = (
        METHOD_A,
    )

METRICS = (
    METRIC_PSNR,
    METRIC_SSIM,
    METRIC_LPIPS,
)

SCENES = (
    (*SCENES_CASUAL,),
    (*SCENES_DL3DV,),
    (*SCENES_LLFF,),
    (*SCENES_TANDT,),
    (*SCENES_MVIMGNET,), 
    (*SCENES_MIPNERF360,), 
)



if __name__ == "__main__":
    df = load_metrics(sum(SCENES, ()), METHODS, FEATURES, METRICS)
    grouped = df.groupby(["method_tag", "metric_tag", "feature_tag"])[
        "metric_value"
    ].mean()

    rows = defaultdict(list)

    for method in METHODS:
        for feature in FEATURES:
            for metric in METRICS:
                rows[f"{metric.full_name}"].append(
                    grouped[method.tag][metric.tag][feature.tag]
                )

colors = ['#8AA2A9', '#C98474', '#F2D0A9', '#8D9F87', '#A7A7A7'] 
# colors = ['#F2D0A9', '#D98E73', '#B24C33', '#5E7460', '#4A6B8A']
# colors = ['#4A6B8A', '#B24C33', '#D9C148', '#5E7460', '#A7A7A7']
# colors = ['#A3C1DA', '#E8A0A2', '#F3E5AB', '#B5CDA3', '#D3D3D3']

rank_symbols = {
    "PSNR": r"$\uparrow$",
    "SSIM": r"$\uparrow$",
    "LPIPS": r"$\downarrow$",
}
categories = [metric.full_name for metric in METRICS]
models = [feature.full_name for feature in FEATURES]

group_spacing = 2.5
x = np.arange(len(categories)) * group_spacing
width = 0.3 
spacing = 0.15

fig, ax1 = plt.subplots(figsize=(18, 5))
ax1.set_xlim(-1.3, 6)

ax2 = ax1.twinx()
ax3 = ax1.twinx()

if Analysis == 1:
    ax1.set_ylim(19.3, 20.0)
    ax2.set_ylim(0.63, 0.66)
    ax3.set_ylim(0.300, 0.350)
elif Analysis == 2:
    ax1.set_ylim(14.0, 21.0)
    ax2.set_ylim(0.56, 0.68)
    ax3.set_ylim(0.34, 0.52)

def format_func1(x, p):
    return f"{x:.1f}"

def format_func2(x, p):
    return f"{x:.3f}"

def format_func3(x, p):
    return f"{x:.2f}"

yfontsize=14
ax1.yaxis.set_major_formatter(FuncFormatter(format_func1))
ax2.yaxis.set_major_formatter(FuncFormatter(format_func2))
ax3.yaxis.set_major_formatter(FuncFormatter(format_func3))

ax1.spines['left'].set_position(('outward', 100))
ax1.yaxis.set_label_position('left')
ax1.yaxis.set_ticks_position('left')
ax1.set_ylabel('PSNR', rotation=30, fontsize=yfontsize)

ax2.spines['left'].set_position(('outward', 45))
ax2.yaxis.set_label_position('left')
ax2.yaxis.set_ticks_position('left')
ax2.set_ylabel('SSIM', rotation=30, fontsize=yfontsize)

ax3.spines['left'].set_position(('outward', 0))
ax3.yaxis.set_label_position('left')
ax3.yaxis.set_ticks_position('left')
ax3.set_ylabel('LPIPS', rotation=30, fontsize=yfontsize)

ax1.tick_params(axis='y', labelsize=14)
ax2.tick_params(axis='y', labelsize=14)
ax3.tick_params(axis='y', labelsize=14)

ax1.yaxis.set_label_coords(-0.095, -0.08)
ax2.yaxis.set_label_coords(-0.045, -0.08)
ax3.yaxis.set_label_coords(0, -0.08)

group_width = len(models) * (width + spacing) - spacing
for i, category in enumerate(categories):
    for j, model in enumerate(models):
        x_pos = i * group_spacing + j * (width + spacing) - group_width / 2
        if category == 'PSNR':
            bar = ax1.bar(x_pos, rows[category][j], width, color=colors[j], label=model if i == 0 else "")
            ax = ax1
        elif category == 'SSIM':
            bar = ax2.bar(x_pos, rows[category][j], width, color=colors[j])
            ax = ax2
        else:  # LPIPS
            bar = ax3.bar(x_pos, rows[category][j], width, color=colors[j])
            ax = ax3
        
        if j < len(models) - 1:
            symbol_x = x_pos + (width + spacing) / 2
            symbol_y = (sorted(rows[category])[1] + ax.get_ylim()[0]) / 2
            ax.text(symbol_x, symbol_y, symbols[j], ha='center', va='center', fontsize=24)
            
# group_width = len(models) * (width + spacing) - spacing
# for i, category in enumerate(categories):
#     for j, model in enumerate(models):
#         x_pos = i * group_spacing + j * (width + spacing) - group_width / 2
#         if category == 'PSNR':
#             ax1.bar(x_pos, rows[category][j], width, color=colors[j], label=model if i == 0 else "")
#         elif category == 'SSIM':
#             ax2.bar(x_pos, rows[category][j], width, color=colors[j])
#         else:  # LPIPS
#             ax3.bar(x_pos, rows[category][j], width, color=colors[j])

ax1.set_xticks([i * group_spacing - width/2 for i in range(len(categories))])
ax1.set_xticklabels([f"{category}{rank_symbols[category]}" for category in categories], 
                    rotation=0, ha='center', fontsize=24)

ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)

if Analysis == 1:
    handles = [plt.Rectangle((0,0),1,1, color=colors[i]) for i in range(len(models))]
    legend = ax1.legend(handles, models, loc='upper center', bbox_to_anchor=(0.5, 1.0), 
           ncol=len(models), frameon=False, fontsize=24, 
           handlelength=1, handleheight=1, handletextpad=0.8, borderpad=-0.5)
    for rect in legend.get_patches():
        rect.set_width(30)
        rect.set_height(20)

if Analysis == 2:
    models = ['Best-G:\nRADIO', 'Best-T:\nMAE', 'Best-T:\nIUVRGB', 'New Best-A:\nRADIO+MAE+IUVRGB', 'Original Best-A:\nMASt3R']
    handles = [plt.Rectangle((0,0),1,1, color=colors[i]) for i in range(len(models))]
    legend = ax1.legend(handles, models, loc='upper center', bbox_to_anchor=(0.5, 1.0), 
            ncol=len(models), frameon=False, fontsize=24, 
           handlelength=1, handleheight=1, handletextpad=0.5, borderpad=-1)
    for rect in legend.get_patches():
        rect.set_width(30)
        rect.set_height(20)

    # for handle in legend.get_patches():
    #     handle.set_y(-8)

plt.tight_layout(rect=[0, 0, 1, 0.95])

if not os.path.exists('figs'):
    os.makedirs('figs')

if Analysis == 1:
    fig_name = 'cat_vs_radio'
elif Analysis == 2:
    fig_name = 'cat_vs_mast3r'
plt.savefig(f'figs/{fig_name}.png', dpi=300, bbox_inches='tight')
plt.savefig(f'figs/{fig_name}.pdf', format='pdf', dpi=300, bbox_inches='tight')
plt.close()

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import gridspec

import numpy as np

features_1 = [
    "RADIO", "RADIO+MAST3R", "RADIO+MAST3R+DUST3R", "RADIO+MAST3R+DUST3R+DINO",
    "RADIO+MAST3R+DUST3R+DINO+MAE", "RADIO+MAST3R+DUST3R+DINO+MAE+MIDAS",
    "RADIO+MAST3R+DUST3R+DINO+MAE+MIDAS+CLIP",
    "RADIO+MAST3R+DUST3R+DINO+MAE+MIDAS+CLIP+DINOV2",
    "RADIO+MAST3R+DUST3R+DINO+MAE+MIDAS+CLIP+DINOV2+SAM",
    "RADIO+MAST3R+DUST3R+DINO+MAE+MIDAS+CLIP+DINOV2+SAM+SD",
    "RADIO+MAST3R+DUST3R+DINO+MAE+MIDAS+CLIP+DINOV2+SAM+SD+IUVRGB",
]

psnr_1 = [19.73, 19.66, 19.71, 19.69, 19.73, 19.71, 19.73, 19.74, 19.74, 19.76, 19.80]
ssim_1 = [0.6513, 0.6504, 0.6529, 0.6522, 0.6535, 0.6535, 0.6537, 0.6535, 0.6540, 0.6550, 0.6545]
lpips_1 = [0.3143, 0.3137, 0.3107, 0.3112, 0.3107, 0.3106, 0.3114, 0.3106, 0.3108, 0.3097, 0.3105]

features_2 = [
    "IUVRGB", "IUVRGB+SD", "IUVRGB+SD+SAM", "IUVRGB+SD+SAM+DINOV2",
    "IUVRGB+SD+SAM+DINOV2+CLIP", "IUVRGB+SD+SAM+DINOV2+CLIP+MIDAS",
    "IUVRGB+SD+SAM+DINOV2+CLIP+MIDAS+MAE",
    "IUVRGB+SD+SAM+DINOV2+CLIP+MIDAS+MAE+DINO",
    "IUVRGB+SD+SAM+DINOV2+CLIP+MIDAS+MAE+DINO+DUST3R",
    "IUVRGB+SD+SAM+DINOV2+CLIP+MIDAS+MAE+DINO+DUST3R+MAST3R",
    "IUVRGB+SD+SAM+DINOV2+CLIP+MIDAS+MAE+DINO+DUST3R+MAST3R+RADIO",
]

psnr_2 = [15.01, 19.70, 19.72, 19.75, 19.77, 19.74, 19.77, 19.70, 19.78, 19.74, 19.80]
ssim_2 = [0.5422, 0.6524, 0.6516, 0.6522, 0.6524, 0.6528, 0.6526, 0.6514, 0.6521, 0.6526, 0.6545]
lpips_2 = [0.4845, 0.3147, 0.3142, 0.3136, 0.3122, 0.3140, 0.3136, 0.3136, 0.3135, 0.3132, 0.3105]

# fig, ax1 = plt.subplots(figsize=(20, 8))

# rank_1 = 1
# rank_2 = 1

# # PSNR
# color = 'tab:orange'
# ax1.set_xlabel('Ranking', fontsize=20)
# ax1.set_ylabel(r'PSNR$\uparrow$', color=color, fontsize=20)
# ax1.plot(range(rank_1, len(features_1)+1), psnr_1[rank_1-1:], '-', color=color, linewidth=3)
# ax1.plot(range(rank_2, len(features_2)+1), psnr_2[rank_2-1:], '--', color=color, linewidth=3)
# ax1.tick_params(axis='y', labelcolor=color, labelsize=20)

# # SSIM
# ax2 = ax1.twinx()
# color = 'tab:green'
# ax2.set_ylabel(r'SSIM$\uparrow$', color=color, fontsize=20)
# ax2.plot(range(rank_1, len(features_1)+1), ssim_1[rank_1-1:], '-', color=color, linewidth=3)
# ax2.plot(range(rank_2, len(features_2)+1), ssim_2[rank_2-1:], '--', color=color, linewidth=3)
# ax2.tick_params(axis='y', labelcolor=color, labelsize=20)

# # LPIPS
# ax3 = ax1.twinx()
# ax3.spines['right'].set_position(('axes', 1.1))
# color = 'tab:blue'
# ax3.set_ylabel(r'LPIPS$\downarrow$', color=color, fontsize=20)
# ax3.plot(range(rank_1, len(features_1)+1), lpips_1[rank_1-1:], '-', color=color, linewidth=3)
# ax3.plot(range(rank_2, len(features_2)+1), lpips_2[rank_2-1:], '--', color=color, linewidth=3)
# ax3.tick_params(axis='y', labelcolor=color, labelsize=20)

# ax1.set_xticks(range(1, max(len(features_1), len(features_2))+1))
# ax1.set_xticklabels(range(1, max(len(features_1), len(features_2))+1), fontsize=20)

# legend_elements = [plt.Line2D([0], [0], color='black', linestyle='-', linewidth=3, label='TOP'),
#                    plt.Line2D([0], [0], color='black', linestyle='--', linewidth=3, label='LAST')]
# ax1.legend(handles=legend_elements, loc='upper left', ncol=1, fontsize=20)

# plt.tight_layout()
# plt.savefig('figs/cat_feat_line.png', dpi=300, bbox_inches='tight')
# plt.close()


colors = {
    'RADIO': '#FED273',   # 浅黄色
    'MAST3R': '#BBC990',  # 浅橄榄色
    'DUST3R': '#B2CBC2',  # 浅青色
    'DINO': '#DCAC99',  # 浅粉色
    'MAE': '#9A8EB4',     # 芋泥色
    'MIDAS': '#6B859E',   # 灰蓝色
    'CLIP': '#EBA062',    # 浅橙色
    'DINOV2': '#B45342',  # 红棕色
    'SAM': '#6F936B',     # 绿色
    'SD': '#706052',    # 深咖啡色
    'IUVRGB': '#E9E5E5',  # 浅灰色
}

fig = plt.figure(figsize=(16, 8))
gs = gridspec.GridSpec(2, 1, height_ratios=[1, 3])

ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1])

color = 'black'

x_min = 0.5
x_max = max(len(features_1), len(features_2)) + 0.5


def add_color_blocks(ax, x, y, features, is_solid_line, transform=None):
    block_width = 0.3 
    block_height = 0.0003
    for i, feature in enumerate(features):
        x_pos = x[i]
        y_pos = y[i]
        feature_parts = feature.split('+')
        num_parts = len(feature_parts)
        total_height = block_height * num_parts
        y_start = y_pos - total_height/2 if is_solid_line else y_pos - block_height/2 + (num_parts-1)*block_height/2
        if num_parts in [5, 7, 9]:
            y_start = y_start + block_height/2 if is_solid_line else y_start - block_height/2

        for j, part in enumerate(feature_parts):
            if part in colors:
                rect = plt.Rectangle((x_pos - block_width/2, y_start + (j*block_height if is_solid_line else -j*block_height)), 
                                     block_width, block_height, 
                                     facecolor=colors[part], edgecolor='none',
                                     transform=transform if transform else ax.transData)
                ax.add_patch(rect)

add_color_blocks(ax2, range(1, len(features_1)), lpips_1[:-1], features_1[:-1], False)

add_color_blocks(ax2, range(2, len(features_2)+1), lpips_2[1:], features_2[1:], True)

rect = plt.Rectangle((1-0.21*3/4, lpips_2[0]-0.018*3/4), 0.21*3/2, 0.018*3/2, facecolor=colors['IUVRGB'], edgecolor='none')
ax1.add_patch(rect)

ax1.plot([1, 2], lpips_2[:2], '--', color=color, linewidth=3, markersize=10)
ax1.set_ylim(0.3150, 0.505)
ax1.set_xlim(x_min, x_max)
ax1.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
ax1.tick_params(axis='y', labelsize=20)
ax1.spines['bottom'].set_visible(False)

ax2.plot(range(1, len(features_1)+1), lpips_1, '-', color=color, linewidth=3)
ax2.plot(range(2, len(features_2)+1), lpips_2[1:], '--', color=color, linewidth=3)
ax2.set_ylim(0.308, 0.3150)
ax2.set_xlim(x_min, x_max)
ax2.spines['top'].set_visible(False)


def draw_wave(ax, x, y, width, height, num_waves=3):
    x_wave = np.linspace(x, x + width, 100)
    y_wave = y + height * np.sin(2 * np.pi * num_waves * (x_wave - x) / width)
    ax.plot(x_wave, y_wave, color='k', lw=1, clip_on=False, transform=ax.transAxes)

draw_wave(ax1, -0.005, -0.01, 0.01, 0.005, num_waves=3)
draw_wave(ax1, 0.995, -0.01, 0.01, 0.005, num_waves=3)
draw_wave(ax2, -0.005, 1.01, 0.01, 0.005, num_waves=3)
draw_wave(ax2, 0.995, 1.01, 0.01, 0.005, num_waves=3)

ax2.set_xlabel('Concatenated Feature Number', fontsize=24)
fig.text(-0.01, 0.5, r'LPIPS$\downarrow$', va='center', rotation='vertical', fontsize=24)

ax2.set_xticks(range(1, int(x_max) + 1))
ax2.set_xticklabels(range(1, int(x_max) + 1), fontsize=20)
ax2.tick_params(axis='y', labelsize=20)

legend_elements = [plt.Line2D([0], [0], color=color, linestyle='-', linewidth=3, label='Descending'),
                   plt.Line2D([0], [0], color=color, linestyle='--', linewidth=3, label='Ascending')]
ax1.legend(handles=legend_elements, loc='upper right', ncol=1, fontsize=24)

import matplotlib.patches as mpatches
legend_elements = []
for feature, color in colors.items():
    feature_name = feature
    if feature_name == "DUST3R":
        feature_name = "DUSt3R"
    elif feature_name == "MAST3R":
        feature_name = "MASt3R"
    elif feature_name == "MIDAS":
        feature_name = "MiDaS"
    elif feature_name == "DINOV2":
        feature_name = "DINOv2"
    legend_elements.append(mpatches.Rectangle((0, 0), 1, 1, facecolor=color, edgecolor='none', label=feature_name))

legend = ax2.legend(handles=legend_elements, loc='lower left', fontsize=16, ncol=1, 
                    handlelength=1, handleheight=1, handletextpad=1.2, borderpad=0.5,
                    labelspacing=0.5)

for rect in legend.get_patches():
    rect.set_width(25)
    rect.set_height(15)

plt.tight_layout()
plt.savefig('figs/line_cat_feat.png', dpi=300, bbox_inches='tight')
plt.savefig('figs/line_cat_feat.pdf', format='pdf', dpi=300, bbox_inches='tight')
plt.close()
import os
import torch
import torchvision.transforms as tvf
import torch.nn.functional as F
import numpy as np

from dust3r.utils.device import to_numpy

from dust3r.inference import inference
from dust3r.model import AsymmetricCroCo3DStereo
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
from utils.dust3r_utils import compute_global_alignment

from mast3r.model import AsymmetricMASt3R
from mast3r.cloud_opt.sparse_ga import sparse_global_alignment
from mast3r.cloud_opt.tsdf_optimizer import TSDFPostProcess

from hydra.utils import instantiate
from omegaconf import OmegaConf


class TorchPCA(object):

    def __init__(self, n_components):
        self.n_components = n_components

    def fit(self, X):
        self.mean_ = X.mean(dim=0)
        unbiased = X - self.mean_.unsqueeze(0)
        U, S, V = torch.pca_lowrank(unbiased, q=self.n_components, center=False, niter=50)
        self.components_ = V.T
        self.singular_values_ = S
        return self

    def transform(self, X):
        t0 = X - self.mean_.unsqueeze(0)
        projected = t0 @ self.components_.T
        return projected

def pca(stacked_feat, dim):
    flattened_feats = []
    for feat in stacked_feat:
        H, W, C = feat.shape
        feat = feat.reshape(H * W, C).detach()
        flattened_feats.append(feat)
    x = torch.cat(flattened_feats, dim=0)
    fit_pca = TorchPCA(n_components=dim).fit(x)

    projected_feats = []
    for feat in stacked_feat:
        H, W, C = feat.shape
        feat = feat.reshape(H * W, C).detach()
        x_red = fit_pca.transform(feat)
        projected_feats.append(x_red.reshape(H, W, dim))
    projected_feats = torch.stack(projected_feats)
    return projected_feats


def upsampler(feature, upsampled_height, upsampled_width, max_chunk=None):
    """
    Upsample the feature tensor to the specified height and width.

    Args:
    - feature (torch.Tensor): The input tensor with size [B, H, W, C].
    - upsampled_height (int): The target height after upsampling.
    - upsampled_width (int): The target width after upsampling.

    Returns:
    - upsampled_feature (torch.Tensor): The upsampled tensor with size [B, upsampled_height, upsampled_width, C].
    """
    # Permute the tensor to [B, C, H, W] for interpolation
    feature = feature.permute(0, 3, 1, 2)
    
    # Perform the upsampling
    if max_chunk:
        upsampled_chunks = []

        for i in range(0, len(feature), max_chunk):
            chunk = feature[i:i+max_chunk]
            
            upsampled_chunk = F.interpolate(chunk, size=(upsampled_height, upsampled_width), mode='bilinear', align_corners=False)
            upsampled_chunks.append(upsampled_chunk)
        
        upsampled_feature = torch.cat(upsampled_chunks, dim=0)
    else:
        upsampled_feature = F.interpolate(feature, size=(upsampled_height, upsampled_width), mode='bilinear', align_corners=False)
    
    # Permute back to [B, H, W, C]
    upsampled_feature = upsampled_feature.permute(0, 2, 3, 1)
    
    return upsampled_feature

def visualizer(features, images, save_dir, dim=9, feat_type=None, file_name=None):
    """
    Visualize features and corresponding images, and save the result.

    Args:
        features (torch.Tensor): Feature tensor with shape [B, H, W, C].
        images (list): List of dictionaries containing images with keys 'img'. Each image tensor has shape [1, 3, H, W]
                       and values in the range [-1, 1].
        save_dir (str): Directory to save the resulting visualization.
        feat_type (list): List of feature types.
        file_name (str): Name of the file to save.
    """
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    import torchvision.utils as vutils

    assert features.dim() == 4, "Input tensor must have 4 dimensions (B, H, W, C)"
    
    B, H, W, C = features.size()

    features = features[..., dim-9:]
    # Normalize the 3-dimensional feature to range [0, 1]
    features_min = features.min(dim=0, keepdim=True).values.min(dim=1, keepdim=True).values.min(dim=2, keepdim=True).values
    features_max = features.max(dim=0, keepdim=True).values.max(dim=1, keepdim=True).values.max(dim=2, keepdim=True).values
    features = (features - features_min) / (features_max - features_min)

    ##### Save individual feature maps
    # # Create subdirectory for feature visualizations
    # feat_dir = os.path.join(save_dir, 'feature_maps')
    # if feat_type:
    #     feat_dir = os.path.join(feat_dir, '-'.join(feat_type))
    # os.makedirs(feat_dir, exist_ok=True)

    # for i in range(B):
    #     # Extract and save the feature map (channels 3-6)
    #     feat_map = features[i, :, :, 3:6].permute(2, 0, 1)  # [3, H, W]
    #     save_path = os.path.join(feat_dir, f'{i}_feat.png')
    #     vutils.save_image(feat_map, save_path, normalize=False)

    # return feat_dir

    ##### Save feature maps in a single image
    # Set the size of the plot
    fig, axes = plt.subplots(B, 4, figsize=(W*4*0.01, H*B*0.01))
    
    for i in range(B):
        # Get the original image
        image_tensor = images[i]['img']
        assert image_tensor.dim() == 4 and image_tensor.size(0) == 1 and image_tensor.size(1) == 3, "Image tensor must have shape [1, 3, H, W]"
        image = image_tensor.squeeze(0).permute(1, 2, 0).numpy()  # Convert to (H, W, 3)
        
        # Scale image values from [-1, 1] to [0, 1]
        image = (image + 1) / 2
        
        ax = axes[i, 0] if B > 1 else axes[0]
        ax.imshow(image)
        ax.axis('off')
        
        # Visualize each 3-dimensional feature
        for j in range(3):
            ax = axes[i, j+1] if B > 1 else axes[j+1]
            if j * 3 < min(C, dim):  # Check if the feature channels are available
                feature_to_plot = features[i, :, :, j*3:(j+1)*3].cpu().numpy()
                ax.imshow(feature_to_plot)
            else:  # Plot white image if features are not available
                ax.imshow(torch.ones(H, W, 3).numpy())
            ax.axis('off')
    
    # Reduce margins and spaces between images
    plt.subplots_adjust(wspace=0.005, hspace=0.005, left=0.01, right=0.99, top=0.99, bottom=0.01)
    
    # Save the entire plot
    if file_name is None:
        file_name = f'feat_dim{dim-9}-{dim}'
    if feat_type:
        feat_type_str = '-'.join(feat_type)
        file_name = file_name + f'_{feat_type_str}'
    save_path = os.path.join(save_dir, file_name + '.png')
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()

    return save_path


#### Open it if you visualize feature maps in Feat2GS's teaser
# import matplotlib.colors as mcolors
# from PIL import Image

# morandi_colors = [
#     '#8AA2A9', '#C98474', '#F2D0A9', '#8D9F87', '#A7A7A7', '#D98E73', '#B24C33', '#5E7460', '#4A6B8A', '#B2CBC2', 
#     '#BBC990', '#6B859E', '#B45342', '#4E0000', '#3D0000', '#2C0000', '#1B0000', '#0A0000', '#DCAC99', '#6F936B', 
#     '#EBA062', '#FED273', '#9A8EB4', '#706052', '#E9E5E5', '#C4D8D2', '#F2CBBD', '#F6F9F1', '#C5CABC', '#A3968B', 
#     '#5C6974', '#BE7B6E', '#C67752', '#C18830', '#8C956C', '#CAC691', '#819992', '#4D797F', '#95AEB2', '#B6C4CF', 
#     '#84291C', '#B9551F', '#A96400', '#374B6C', '#C8B493', '#677D5D', '#9882A2', '#2D5F53', '#D2A0AC', '#658D9A', 
#     '#9A7265', '#EFE1D2', '#DDD8D1', '#D2C6BC', '#E3C9BC', '#B8AB9F', '#D8BEA4', '#E0D4C5', '#B8B8B6', '#D0CAC3', 
#     '#9AA8B5', '#BBC9B9', '#E3E8D8', '#ADB3A4', '#C5C9BB', '#A3968B', '#C2A995', '#EDE1D1', '#EDE8E1', '#EDEBE1', 
#     '#CFCFCC', '#AABAC6', '#DCDEE0', '#EAE5E7', '#B7AB9F', '#F7EFE3', '#DED8CF', '#ABCA99', '#C5CD8F', '#959491', 
#     '#FFE481', '#C18E99', '#B07C86', '#9F6A73', '#8E5860', '#DEAD44', '#CD9B31', '#BC891E', '#AB770B', '#9A6500', 
#     '#778144', '#666F31', '#555D1E', '#444B0B', '#333900', '#67587B', '#564668', '#684563', '#573350', '#684550', 
#     '#57333D', '#46212A', '#350F17', '#240004',
# ]

# def rgb_to_hsv(rgb):
#     rgb = rgb.clamp(0, 1)
    
#     cmax, cmax_idx = rgb.max(dim=-1)
#     cmin = rgb.min(dim=-1).values
    
#     diff = cmax - cmin
    
#     h = torch.zeros_like(cmax)
#     h[cmax_idx == 0] = (((rgb[..., 1] - rgb[..., 2]) / diff) % 6)[cmax_idx == 0]
#     h[cmax_idx == 1] = (((rgb[..., 2] - rgb[..., 0]) / diff) + 2)[cmax_idx == 1]
#     h[cmax_idx == 2] = (((rgb[..., 0] - rgb[..., 1]) / diff) + 4)[cmax_idx == 2]
#     h[diff == 0] = 0  # If cmax == cmin
#     h = h / 6
    
#     s = torch.zeros_like(cmax)
#     s[cmax != 0] = (diff / cmax)[cmax != 0]
    
#     v = cmax
    
#     return torch.stack([h, s, v], dim=-1)

# def hsv_to_rgb(hsv):
#     h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    
#     c = v * s
#     x = c * (1 - torch.abs((h * 6) % 2 - 1))
#     m = v - c
    
#     rgb = torch.zeros_like(hsv)
#     mask = (h < 1/6)
#     rgb[mask] = torch.stack([c[mask], x[mask], torch.zeros_like(x[mask])], dim=-1)
#     mask = (1/6 <= h) & (h < 2/6)
#     rgb[mask] = torch.stack([x[mask], c[mask], torch.zeros_like(x[mask])], dim=-1)
#     mask = (2/6 <= h) & (h < 3/6)
#     rgb[mask] = torch.stack([torch.zeros_like(x[mask]), c[mask], x[mask]], dim=-1)
#     mask = (3/6 <= h) & (h < 4/6)
#     rgb[mask] = torch.stack([torch.zeros_like(x[mask]), x[mask], c[mask]], dim=-1)
#     mask = (4/6 <= h) & (h < 5/6)
#     rgb[mask] = torch.stack([x[mask], torch.zeros_like(x[mask]), c[mask]], dim=-1)
#     mask = (5/6 <= h)
#     rgb[mask] = torch.stack([c[mask], torch.zeros_like(x[mask]), x[mask]], dim=-1)
    
#     return rgb + m.unsqueeze(-1)

# def interpolate_colors(colors, n_colors):
#     # Convert colors to RGB tensor
#     rgb_colors = torch.tensor([mcolors.to_rgb(color) for color in colors])
    
#     # Convert RGB to HSV
#     hsv_colors = rgb_to_hsv(rgb_colors)
    
#     # Sort by hue
#     sorted_indices = torch.argsort(hsv_colors[:, 0])
#     sorted_hsv_colors = hsv_colors[sorted_indices]
    
#     # Create interpolation indices
#     indices = torch.linspace(0, len(sorted_hsv_colors) - 1, n_colors)
    
#     # Perform interpolation
#     interpolated_hsv = torch.stack([
#         torch.lerp(sorted_hsv_colors[int(i)], 
#                    sorted_hsv_colors[min(int(i) + 1, len(sorted_hsv_colors) - 1)], 
#                    i - int(i))
#         for i in indices
#     ])
    
#     # Convert interpolated result back to RGB
#     interpolated_rgb = hsv_to_rgb(interpolated_hsv)
    
#     return interpolated_rgb


# def project_to_morandi(features, morandi_colors):
#     features_flat = features.reshape(-1, 3)
#     distances = torch.cdist(features_flat, morandi_colors)
    
#     # Get the indices of the closest colors
#     closest_color_indices = torch.argmin(distances, dim=1)
    
#     # Use the closest Morandi colors directly
#     features_morandi = morandi_colors[closest_color_indices]
    
#     features_morandi = features_morandi.reshape(features.shape)
#     return features_morandi


# def smooth_color_transform(features, morandi_colors, smoothness=0.1):
#     features_flat = features.reshape(-1, 3)
#     distances = torch.cdist(features_flat, morandi_colors)
    
#     # Calculate weights
#     weights = torch.exp(-distances / smoothness)
#     weights = weights / weights.sum(dim=1, keepdim=True)
    
#     # Weighted average
#     features_morandi = torch.matmul(weights, morandi_colors)
    
#     features_morandi = features_morandi.reshape(features.shape)
#     return features_morandi

# def histogram_matching(source, template):
#     """
#     Match the histogram of the source tensor to that of the template tensor.
    
#     :param source: Source tensor with shape [B, H, W, 3]
#     :param template: Template tensor with shape [N, 3], where N is the number of colors
#     :return: Source tensor after histogram matching
#     """
#     def match_cumulative_cdf(source, template):
#         src_values, src_indices, src_counts = torch.unique(source, return_inverse=True, return_counts=True)
#         tmpl_values, tmpl_counts = torch.unique(template, return_counts=True)
        
#         src_quantiles = torch.cumsum(src_counts.float(), 0) / source.numel()
#         tmpl_quantiles = torch.cumsum(tmpl_counts.float(), 0) / template.numel()
        
#         idx = torch.searchsorted(tmpl_quantiles, src_quantiles)
#         idx = torch.clamp(idx, 1, len(tmpl_quantiles)-1)
        
#         slope = (tmpl_values[idx] - tmpl_values[idx-1]) / (tmpl_quantiles[idx] - tmpl_quantiles[idx-1])
#         interp_a_values = torch.lerp(tmpl_values[idx-1], tmpl_values[idx], 
#                                      (src_quantiles - tmpl_quantiles[idx-1]) * slope)
        
#         return interp_a_values[src_indices].reshape(source.shape)
    
#     matched = torch.stack([match_cumulative_cdf(source[..., i], template[:, i]) for i in range(3)], dim=-1)
#     return matched

# def process_features(features):
#     device = features.device
    
#     n_colors = 1024
#     morandi_colors_tensor = interpolate_colors(morandi_colors, n_colors).to(device)
#     # morandi_colors_tensor = torch.tensor([mcolors.to_rgb(color) for color in morandi_colors]).to(device)

#     # features_morandi = project_to_morandi(features, morandi_colors_tensor)
#     # features_morandi = histogram_matching(features, morandi_colors_tensor)
#     features_morandi = smooth_color_transform(features, morandi_colors_tensor, smoothness=0.05)
    
#     return features_morandi.cpu().numpy()

# def visualizer(features, images, save_dir, dim=9, feat_type=None, file_name=None):
#     import matplotlib
#     matplotlib.use('Agg')
    
#     import matplotlib.pyplot as plt
#     import numpy as np
#     import os

#     assert features.dim() == 4, "Input tensor must have 4 dimensions (B, H, W, C)"
    
#     B, H, W, C = features.size()

#     # Ensure features have at least 3 channels for RGB visualization
#     assert C >= 3, "Features must have at least 3 channels for RGB visualization"
#     features = features[..., :3]

#     # Normalize features to [0, 1] range
#     features_min = features.min(dim=0, keepdim=True).values.min(dim=1, keepdim=True).values.min(dim=2, keepdim=True).values
#     features_max = features.max(dim=0, keepdim=True).values.max(dim=1, keepdim=True).values.max(dim=2, keepdim=True).values
#     features = (features - features_min) / (features_max - features_min)

#     features_processed = process_features(features)

#     # Create the directory structure
#     vis_dir = os.path.join(save_dir, 'vis')

#     if feat_type:
#         feat_type_str = '-'.join(feat_type)
#         vis_dir = os.path.join(vis_dir, feat_type_str)
#     os.makedirs(vis_dir, exist_ok=True)

#     # Save individual images for each feature map
#     for i in range(B):
#         if file_name is None:
#             file_name = 'feat_morandi'
#         save_path = os.path.join(vis_dir, f'{file_name}_{i}.png')
        
#         # Convert to uint8 and save directly
#         img = Image.fromarray((features_processed[i] * 255).astype(np.uint8))
#         img.save(save_path)

#     print(f"Feature maps have been saved in the directory: {vis_dir}")
#     return vis_dir

def mv_visualizer(features, images, save_dir, dim=9, feat_type=None, file_name=None):
    """
    Visualize features and corresponding images, and save the result. (For MASt3R decoder or head features)
    """
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    import os

    B, H, W, _ = features.size()
    features = features[..., dim-9:]

    # Normalize the 3-dimensional feature to range [0, 1]
    features_min = features.min(dim=0, keepdim=True).values.min(dim=1, keepdim=True).values.min(dim=2, keepdim=True).values
    features_max = features.max(dim=0, keepdim=True).values.max(dim=1, keepdim=True).values.max(dim=2, keepdim=True).values
    features = (features - features_min) / (features_max - features_min)

    rows = (B + 1) // 2  # Adjust rows for odd B
    fig, axes = plt.subplots(rows, 8, figsize=(W*8*0.01, H*rows*0.01))
    
    for i in range(B//2):
        # Odd row: image and features
        image = (images[2*i]['img'].squeeze(0).permute(1, 2, 0).numpy() + 1) / 2
        axes[i, 0].imshow(image)
        axes[i, 0].axis('off')
        for j in range(3):
            axes[i, j+1].imshow(features[2*i, :, :, j*3:(j+1)*3].cpu().numpy())
            axes[i, j+1].axis('off')
        
        # Even row: image and features
        if 2*i + 1 < B:
            image = (images[2*i + 1]['img'].squeeze(0).permute(1, 2, 0).numpy() + 1) / 2
            axes[i, 4].imshow(image)
            axes[i, 4].axis('off')
            for j in range(3):
                axes[i, j+5].imshow(features[2*i + 1, :, :, j*3:(j+1)*3].cpu().numpy())
                axes[i, j+5].axis('off')

    # Handle last row if B is odd
    if B % 2 != 0:
        image = (images[-1]['img'].squeeze(0).permute(1, 2, 0).numpy() + 1) / 2
        axes[-1, 0].imshow(image)
        axes[-1, 0].axis('off')
        for j in range(3):
            axes[-1, j+1].imshow(features[-1, :, :, j*3:(j+1)*3].cpu().numpy())
            axes[-1, j+1].axis('off')

        # Hide unused columns in last row
        for j in range(4, 8):
            axes[-1, j].axis('off')

    plt.subplots_adjust(wspace=0.005, hspace=0.005, left=0.01, right=0.99, top=0.99, bottom=0.01)
    
    # Save the plot 
    if file_name is None:
        file_name = f'feat_dim{dim-9}-{dim}'
    if feat_type:
        feat_type_str = '-'.join(feat_type)
        file_name = file_name + f'_{feat_type_str}'
    save_path = os.path.join(save_dir, file_name + '.png')
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()

    return save_path


def adjust_norm(image: torch.Tensor) -> torch.Tensor:

    inv_normalize = tvf.Normalize(
        mean=[-1, -1, -1],
        std=[1/0.5, 1/0.5, 1/0.5]
    )

    correct_normalize = tvf.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    image = inv_normalize(image)
    image = correct_normalize(image)

    return image

def adjust_midas_norm(image: torch.Tensor) -> torch.Tensor:

    inv_normalize = tvf.Normalize(
        mean=[-1, -1, -1],
        std=[1/0.5, 1/0.5, 1/0.5]
    )

    correct_normalize = tvf.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )

    image = inv_normalize(image)
    image = correct_normalize(image)

    return image

def adjust_clip_norm(image: torch.Tensor) -> torch.Tensor:

    inv_normalize = tvf.Normalize(
        mean=[-1, -1, -1],
        std=[1/0.5, 1/0.5, 1/0.5]
    )

    correct_normalize = tvf.Normalize(
        mean=[0.48145466, 0.4578275, 0.40821073],
        std=[0.26862954, 0.26130258, 0.27577711]
    )

    image = inv_normalize(image)
    image = correct_normalize(image)

    return image

class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image):
        image2 = torch.clone(image)
        if len(image2.shape) == 4:
            image2 = image2.permute(1, 0, 2, 3)
        for t, m, s in zip(image2, self.mean, self.std):
            t.mul_(s).add_(m)
        return image2.permute(1, 0, 2, 3)


norm = tvf.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
unnorm = UnNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

midas_norm = tvf.Normalize([0.5] * 3, [0.5] * 3)
midas_unnorm = UnNormalize([0.5] * 3, [0.5] * 3)


def generate_iuv(B, H, W):
    i_coords = torch.arange(B).view(B, 1, 1, 1).expand(B, H, W, 1).float() / (B - 1)
    u_coords = torch.linspace(0, 1, W).view(1, 1, W, 1).expand(B, H, W, 1)
    v_coords = torch.linspace(0, 1, H).view(1, H, 1, 1).expand(B, H, W, 1)
    iuv_coords = torch.cat([i_coords, u_coords, v_coords], dim=-1)
    return iuv_coords

class FeatureExtractor:
    """
    Extracts and processes features from images using VFMs for per point(per pixel).
    Supports multiple VFM features, dimensionality reduction, feature upsampling, and visualization.

    Parameters:
        images (list): List of image info.
        method (str): Pointmap Init method, choose in ["dust3r", "mast3r"].
        device (str): 'cuda'.
        feat_type (list): VFM, choose in ["dust3r", "mast3r", "dift", "dino_b16", "dinov2_b14", "radio", "clip_b16", "mae_b16", "midas_l16", "sam_base", "iuvrgb"].
        feat_dim (int): PCA dimensions.
        img_base_path (str): Training view data directory path.
        model_path (str): Model path, './submodules/mast3r/checkpoints/'.
        vis_feat (bool): Visualize and save feature maps.
        vis_key (str): Feature type to visualize(only for mast3r), choose in ["decfeat", "desc"].
        focal_avg (bool): Use averaging focal.
    """
    def __init__(self, images, args, method):
        self.images = images
        self.method = method
        self.device = args.device
        self.feat_type = args.feat_type
        self.feat_dim = args.feat_dim
        self.img_base_path = args.img_base_path
        # self.use_featup = args.use_featup
        self.model_path = args.model_path
        self.vis_feat = args.vis_feat
        self.vis_key = args.vis_key
        self.focal_avg = args.focal_avg

    def get_dust3r_feat(self, **kw):
        model_path = os.path.join(self.model_path, "DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth")
        model = AsymmetricCroCo3DStereo.from_pretrained(model_path).to(self.device)
        output = inference(kw['pairs'], model, self.device, batch_size=1)
        scene = global_aligner(output, device=self.device, mode=GlobalAlignerMode.PointCloudOptimizer)
        if self.vis_key:
            assert self.vis_key == 'decfeat', f"Expected vis_key to be 'decfeat', but got {self.vis_key}"
            self.vis_decfeat(kw['pairs'], output=output)

        # del model, output
        # torch.cuda.empty_cache()

        return scene.stacked_feat

    def get_mast3r_feat(self, **kw):
        model_path = os.path.join(self.model_path, "MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth")
        model = AsymmetricMASt3R.from_pretrained(model_path).to(self.device)
        cache_dir = os.path.join(self.img_base_path, "cache")
        if os.path.exists(cache_dir):
            os.system(f'rm -rf {cache_dir}')
        scene = sparse_global_alignment(kw['train_img_list'], kw['pairs'], cache_dir,
                                        model, lr1=0.07, niter1=500, lr2=0.014, niter2=200, device=self.device,
                                        opt_depth='depth' in 'refine', shared_intrinsics=self.focal_avg,
                                        matching_conf_thr=5.)
        if self.vis_key:
            assert self.vis_key in ['decfeat', 'desc'], f"Expected vis_key to be 'decfeat' or 'desc', but got {self.vis_key}"
            self.vis_decfeat(kw['pairs'], model=model)

        # del model
        # torch.cuda.empty_cache()

        return scene.stacked_feat

    def get_feat(self, feat_type):
        """
        Get features using Probe3D.
        """
        cfg = OmegaConf.load(f"configs/backbone/{feat_type}.yaml")
        model = instantiate(cfg.model, output="dense", return_multilayer=False)
        model = model.to(self.device)
        if 'midas' in feat_type:
            image_norm = adjust_midas_norm(torch.cat([i['img'] for i in self.images])).to(self.device)
        # elif 'clip' in self.feat_type:
        #     image_norm = adjust_clip_norm(torch.cat([i['img'] for i in self.images])).to(self.device)
        else:
            image_norm = adjust_norm(torch.cat([i['img'] for i in self.images])).to(self.device)
        
        with torch.no_grad():
            feats = model(image_norm).permute(0, 2, 3, 1)

        # del model
        # torch.cuda.empty_cache()

        return feats

    # def get_feat(self, feat_type):
    #     """
    #     Get features using FeatUp.
    #     """
    #     original_feat_type = feat_type
    #     use_norm = False if 'maskclip' in feat_type else True
    #     if 'featup' in original_feat_type:
    #         feat_type = feat_type.split('_featup')[0]
    #     # feat_upsampler = torch.hub.load("mhamilton723/FeatUp", feat_type, use_norm=use_norm).to(device)
    #     feat_upsampler = torch.hub.load("/home/chenyue/.cache/torch/hub/mhamilton723_FeatUp_main/", feat_type, use_norm=use_norm, source='local').to(self.device)     ## offline
    #     image_norm = adjust_norm(torch.cat([i['img'] for i in self.images])).to(self.device)
    #     image_norm = F.interpolate(image_norm, size=(224, 224), mode='bilinear', align_corners=False)
    #     if 'featup' in original_feat_type:
    #         feats = feat_upsampler(image_norm).permute(0, 2, 3, 1)
    #     else:
    #         feats = feat_upsampler.model(image_norm).permute(0, 2, 3, 1)
    #     return feats

    def get_iuvrgb(self):
        rgb = torch.cat([i['img'] for i in self.images]).permute(0, 2, 3, 1).to(self.device)
        feats = torch.cat([generate_iuv(*rgb.shape[:-1]).to(self.device), rgb], dim=-1)
        return feats

    def get_iuv(self):
        rgb = torch.cat([i['img'] for i in self.images]).permute(0, 2, 3, 1).to(self.device)
        feats = generate_iuv(*rgb.shape[:-1]).to(self.device)
        return feats

    def preprocess(self, feature, feat_dim, vis_feat=False, is_upsample=True):
        """
        Preprocess features by applying PCA, upsampling, and optionally visualizing.
        """
        if feat_dim:
            feature = pca(feature, feat_dim)
        # else:
        #     feature_min = feature.min(dim=0, keepdim=True).values.min(dim=1, keepdim=True).values
        #     feature_max = feature.max(dim=0, keepdim=True).values.max(dim=1, keepdim=True).values
        #     feature = (feature - feature_min) / (feature_max - feature_min + 1e-6)
        #     feature = feature - feature.mean(dim=[0,1,2], keepdim=True)
        
        torch.cuda.empty_cache()

        if (feature[0].shape[0:-1] != self.images[0]['true_shape'][0]).all() and is_upsample:
            feature = upsampler(feature, *[s for s in self.images[0]['true_shape'][0]])

        print(f"Feature map size >>> height: {feature[0].shape[0]}, width: {feature[0].shape[1]}, channels: {feature[0].shape[2]}")
        if vis_feat:
            save_path = visualizer(feature, self.images, self.img_base_path, feat_type=self.feat_type)
            print(f"The encoder feature visualization has been saved at >>>>> {save_path}")
        
        return feature
    
    def vis_decfeat(self, pairs, **kw):
        """
        Visualize decoder or head(only for mast3r) features.
        """
        if 'output' in kw:
            output = kw['output']
        else:
            output = inference(pairs, kw['model'], self.device, batch_size=1, verbose=False)
        decfeat1 = output['pred1'][self.vis_key].detach()
        decfeat2 = output['pred2'][self.vis_key].detach()
        # decfeat1 = pca(decfeat1, 9)
        # decfeat2 = pca(decfeat2, 9)
        decfeat = torch.stack([decfeat1, decfeat2], dim=1).view(-1, *decfeat1.shape[1:])
        decfeat = torch.cat(torch.chunk(decfeat,2)[::-1], dim=0)
        decfeat = pca(decfeat, 9)
        if (decfeat.shape[1:-1] != self.images[0]['true_shape'][0]).all():
            decfeat = upsampler(decfeat, *[s for s in self.images[0]['true_shape'][0]])
        pair_images = [im for p in pairs[3:] + pairs[:3] for im in p]
        save_path = mv_visualizer(decfeat, pair_images, self.img_base_path, 
                                  feat_type=self.feat_type, file_name=f'{self.vis_key}_pcaall_dim0-9')
        print(f"The decoder feature visualization has been saved at >>>>> {save_path}")

    def forward(self, **kw):
        feat_dim = self.feat_dim
        vis_feat = self.vis_feat and len(self.feat_type) == 1
        is_upsample = len(self.feat_type) == 1

        all_feats = {}
        for feat_type in self.feat_type:
            if feat_type == self.method:
                feats = kw['scene'].stacked_feat
            elif feat_type in ['dust3r', 'mast3r']:
                feats = getattr(self, f"get_{feat_type}_feat")(**kw)
            elif feat_type in ['iuv', 'iuvrgb']:
                feats = getattr(self, f"get_{feat_type}")()
                feat_dim = None
            else:
                feats = self.get_feat(feat_type)
            
            # feats = to_numpy(self.preprocess(feats))
            all_feats[feat_type] = self.preprocess(feats.detach().clone(), feat_dim, vis_feat, is_upsample)

        if len(self.feat_type) > 1:
            all_feats = {k: (v - v.min()) / (v.max() - v.min()) for k, v in all_feats.items()}

            target_size = tuple(s // 16 for s in self.images[0]['true_shape'][0][:2])
            tmp_feats = []
            kickoff = []

            for k, v in all_feats.items():
                if k in ['iuv', 'iuvrgb']:
                    # self.feat_dim -= v.shape[-1]
                    kickoff.append(v)
                else:
                    if v.shape[1:3] != target_size:
                        v = F.interpolate(v.permute(0, 3, 1, 2), size=target_size, 
                                        mode='bilinear', align_corners=False).permute(0, 2, 3, 1)
                    tmp_feats.append(v)

            feats = self.preprocess(torch.cat(tmp_feats, dim=-1), self.feat_dim, self.vis_feat and not kickoff)

            if kickoff:
                feats = torch.cat([feats] + kickoff, dim=-1)
                feats = self.preprocess(feats, self.feat_dim, self.vis_feat, is_upsample=False)

        else:
            feats = all_feats[self.feat_type[0]]

        torch.cuda.empty_cache()
        return to_numpy(feats)

    def __call__(self, **kw):
        return self.forward(**kw)


class InitMethod:
    """
    Initialize pointmap and camera param via DUSt3R or MASt3R.
    """
    def __init__(self, args):
        self.method = args.method
        self.n_views = args.n_views
        self.device = args.device
        self.img_base_path = args.img_base_path
        self.focal_avg = args.focal_avg
        self.tsdf_thresh = args.tsdf_thresh
        self.min_conf_thr = args.min_conf_thr 
        if self.method == 'dust3r':
            self.model_path = os.path.join(args.model_path, "DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth")
        else:
            self.model_path = os.path.join(args.model_path, "MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth")

    def get_dust3r(self):
        return AsymmetricCroCo3DStereo.from_pretrained(self.model_path).to(self.device)

    def get_mast3r(self):
        return AsymmetricMASt3R.from_pretrained(self.model_path).to(self.device)

    def infer_dust3r(self, **kw):
        output = inference(kw['pairs'], kw['model'], self.device, batch_size=1)
        scene = global_aligner(output, device=self.device, mode=GlobalAlignerMode.PointCloudOptimizer)
        loss = compute_global_alignment(scene=scene, init="mst", niter=300, schedule='linear', lr=0.01, 
                                        focal_avg=self.focal_avg, known_focal=kw.get('known_focal', None))
        scene = scene.clean_pointcloud()
        return scene

    def infer_mast3r(self, **kw):
        cache_dir = os.path.join(self.img_base_path, "cache")
        if os.path.exists(cache_dir):
            os.system(f'rm -rf {cache_dir}')

        scene = sparse_global_alignment(kw['train_img_list'], kw['pairs'], cache_dir,
                                        kw['model'], lr1=0.07, niter1=500, lr2=0.014, niter2=200, device=self.device,
                                        opt_depth='depth' in 'refine', shared_intrinsics=self.focal_avg,
                                        matching_conf_thr=5.)
        return scene

    def get_dust3r_info(self, scene):
        imgs = to_numpy(scene.imgs)
        focals = scene.get_focals()
        poses = to_numpy(scene.get_im_poses())
        pts3d = to_numpy(scene.get_pts3d())
        # pts3d = to_numpy(scene.get_planes3d())
        scene.min_conf_thr = float(scene.conf_trf(torch.tensor(1.0)))
        confidence_masks = to_numpy(scene.get_masks())
        intrinsics = to_numpy(scene.get_intrinsics())
        return imgs, focals, poses, intrinsics, pts3d, confidence_masks

    def get_mast3r_info(self, scene):
        imgs = to_numpy(scene.imgs)
        focals = scene.get_focals()[:,None]
        poses = to_numpy(scene.get_im_poses())
        intrinsics = to_numpy(scene.intrinsics)
        tsdf = TSDFPostProcess(scene, TSDF_thresh=self.tsdf_thresh)
        pts3d, _, confs = to_numpy(tsdf.get_dense_pts3d(clean_depth=True))
        pts3d = [arr.reshape((*imgs[0].shape[:2], 3)) for arr in pts3d]
        confidence_masks = np.array(to_numpy([c > self.min_conf_thr for c in confs]))
        return imgs, focals, poses, intrinsics, pts3d, confidence_masks

    def get_dust3r_depth(self, scene):
        return to_numpy(scene.get_depthmaps())

    def get_mast3r_depth(self, scene):
        imgs = to_numpy(scene.imgs)
        tsdf = TSDFPostProcess(scene, TSDF_thresh=self.tsdf_thresh)
        _, depthmaps, _ = to_numpy(tsdf.get_dense_pts3d(clean_depth=True))
        depthmaps = [arr.reshape((*imgs[0].shape[:2], 3)) for arr in depthmaps]
        return depthmaps

    def get_model(self):
        return getattr(self, f"get_{self.method}")()

    def infer(self, **kw):
        return getattr(self, f"infer_{self.method}")(**kw)

    def get_info(self, scene):
        return getattr(self, f"get_{self.method}_info")(scene)

    def get_depth(self, scene):
        return getattr(self, f"get_{self.method}_depth")(scene)







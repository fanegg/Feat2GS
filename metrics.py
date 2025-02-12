#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from pathlib import Path
import os
from PIL import Image
import torch
import torchvision.transforms.functional as tf
from utils.loss_utils import ssim, masked_ssim
from lpipsPyTorch import lpips
import json
from tqdm import tqdm
from utils.image_utils import psnr, masked_psnr
from argparse import ArgumentParser
import numpy as np
import matplotlib.pyplot as plt

def readImages(renders_dir, gt_dir, msk_dir):
    renders = []
    gts = []
    masks = []
    image_names = []
    for fname in os.listdir(renders_dir):
        render = Image.open(renders_dir / fname)
        gt = Image.open(gt_dir / fname)
        mask = Image.open(msk_dir / fname)
        renders.append(tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda())
        gts.append(tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda())
        masks.append(tf.to_tensor(mask).int().unsqueeze(0)[:, 0, :, :].cuda())
        image_names.append(fname)
    return renders, gts, masks, image_names

def evaluate(args):
    
    full_dict = {}
    per_view_dict = {}

    print("")

    for scene_dir in args.model_paths:
        print("Scene:", scene_dir)
        full_dict[scene_dir] = {}
        per_view_dict[scene_dir] = {}

        test_dir = Path(scene_dir) / "test"

        for method in os.listdir(test_dir):
            print("Method:", method)

            full_dict[scene_dir][method] = {}
            per_view_dict[scene_dir][method] = {}

            method_dir = test_dir / method
            out_f = open(method_dir / 'metrics.txt', 'w') 
            gt_dir = method_dir/ "gt"
            renders_dir = method_dir / "renders"
            msk_dir = method_dir/ "masks"
            renders, gts, masks, image_names = readImages(renders_dir, gt_dir, msk_dir)

            ssims = []
            psnrs = []
            lpipss = []

            for idx in tqdm(range(len(renders)), desc="Metric evaluation progress"):

                render_indx, gt_indx, mask_indx = renders[idx], gts[idx], masks[idx][:, None, ...]
                render_indx = render_indx * mask_indx
                gt_indx = gt_indx * mask_indx

                # s=masked_ssim(render_indx, gt_indx, mask_indx)
                # p=masked_psnr(render_indx, gt_indx, mask_indx)

                psnr_map = 20 * torch.log10(1.0 /( 1e-2 + torch.sqrt(((render_indx - gt_indx) ** 2).mean(1))))
                psnr_normalized = (psnr_map / 40).clamp(0, 1)
                psnr_image = (psnr_normalized.squeeze().cpu().numpy() * 255).astype(np.uint8)
                psnr_pil = Image.fromarray(psnr_image)
                psnr_pil.save(method_dir / f"psnr_map_{image_names[idx]}")

                ssim_map = ssim(render_indx, gt_indx, get_ssim_map=True).mean(1)
                ssim_normalized = ((ssim_map + 1)/ 2).clamp(0, 1)
                ssim_image = (ssim_normalized.squeeze().cpu().numpy() * 255).astype(np.uint8)
                ssim_pil = Image.fromarray(ssim_image)
                ssim_pil.save(method_dir / f"ssim_map_{image_names[idx]}")

                # mean square error
                jet_cmap = plt.cm.jet
                mse_map = ((render_indx - gt_indx) ** 2).mean(1)
                mse_map = mse_map.clamp(0, 0.2)*5
                mse_colored = jet_cmap(mse_map.squeeze().cpu().numpy())
                mse_image = (mse_colored[:, :, :3] * 255).astype(np.uint8)
                mse_pil = Image.fromarray(mse_image)
                mse_pil.save(method_dir / f"mse_map_{image_names[idx]}")

                # lpips
                lpips_map = lpips(render_indx, gt_indx, net_type='vgg', return_spatial_map=True)
                lpips_map = lpips_map.clamp(0, 1.0)
                lpips_colored = jet_cmap(lpips_map.squeeze().cpu().numpy())
                lpips_image = (lpips_colored[:, :, :3] * 255).astype(np.uint8)
                lpips_pil = Image.fromarray(lpips_image)
                lpips_pil.save(method_dir / f"lpips_map_{image_names[idx]}")

                s=ssim(render_indx, gt_indx)
                p=psnr(render_indx, gt_indx)
                l=lpips(render_indx, gt_indx, net_type='vgg')

                out_f.write(f"image name{image_names[idx]}, image idx: {idx}, PSNR: {p.item():.2f}, SSIM: {s:.4f}, LPIPS: {l.item():.4f}\n")
                ssims.append(s)
                psnrs.append(p)
                lpipss.append(l)

            print("  SSIM : {:>12.7f}".format(torch.tensor(ssims).mean(), ".5"))
            print("  PSNR : {:>12.7f}".format(torch.tensor(psnrs).mean(), ".5"))
            print("  LPIPS: {:>12.7f}".format(torch.tensor(lpipss).mean(), ".5"))
            print("")

            full_dict[scene_dir][method].update({"SSIM": torch.tensor(ssims).mean().item(),
                                                    "PSNR": torch.tensor(psnrs).mean().item(),
                                                    "LPIPS": torch.tensor(lpipss).mean().item()})
            per_view_dict[scene_dir][method].update({"SSIM": {name: ssim for ssim, name in zip(torch.tensor(ssims).tolist(), image_names)},
                                                        "PSNR": {name: psnr for psnr, name in zip(torch.tensor(psnrs).tolist(), image_names)},
                                                        "LPIPS": {name: lp for lp, name in zip(torch.tensor(lpipss).tolist(), image_names)}})
            
        with open(scene_dir + "/results.json", 'w') as fp:
            json.dump(full_dict[scene_dir], fp, indent=True)
        with open(scene_dir + "/per_view.json", 'w') as fp:
            json.dump(per_view_dict[scene_dir], fp, indent=True)


if __name__ == "__main__":
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument('--model_paths', '-m', required=True, nargs="+", type=str, default=[])
    parser.add_argument('--iteration', type=int, default=1000)    
    parser.add_argument("--n_views", default=None, type=int)
    args = parser.parse_args()
    evaluate(args)

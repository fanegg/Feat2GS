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

import torch

def mse(img1, img2):
    return (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)

def psnr(img1, img2):
    mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

def masked_psnr(img1, img2, mask):
    mse = ((((img1 - img2)) ** 2) * mask).sum() / (3. * mask.sum())
    return 20 * torch.log10(1.0 / torch.sqrt(mse))


def accuracy_torch(gt_points, rec_points, gt_normals=None, rec_normals=None, batch_size=5000):
    n_points = rec_points.shape[0]
    all_distances = []
    all_indices = []
    
    for i in range(0, n_points, batch_size):
        end_idx = min(i + batch_size, n_points)
        batch_points = rec_points[i:end_idx]
        
        distances = torch.cdist(batch_points, gt_points)  # (batch_size, M)
        batch_distances, batch_indices = torch.min(distances, dim=1)  # (batch_size,)
        
        all_distances.append(batch_distances)
        all_indices.append(batch_indices)
    
    distances = torch.cat(all_distances)
    indices = torch.cat(all_indices)
    
    acc = torch.mean(distances)
    acc_median = torch.median(distances)

    if gt_normals is not None and rec_normals is not None:
        normal_dot = torch.sum(gt_normals[indices] * rec_normals, dim=-1)
        normal_dot = torch.abs(normal_dot)
        return acc, acc_median, torch.mean(normal_dot), torch.median(normal_dot)

    return acc, acc_median

def completion_torch(gt_points, rec_points, gt_normals=None, rec_normals=None, batch_size=5000):

    n_points = gt_points.shape[0]
    all_distances = []
    all_indices = []
    
    for i in range(0, n_points, batch_size):
        end_idx = min(i + batch_size, n_points)
        batch_points = gt_points[i:end_idx]
        
        distances = torch.cdist(batch_points, rec_points)  # (batch_size, M)
        batch_distances, batch_indices = torch.min(distances, dim=1)  # (batch_size,)
        
        all_distances.append(batch_distances)
        all_indices.append(batch_indices)
    
    distances = torch.cat(all_distances)
    indices = torch.cat(all_indices)
    
    comp = torch.mean(distances)
    comp_median = torch.median(distances)

    if gt_normals is not None and rec_normals is not None:
        normal_dot = torch.sum(gt_normals * rec_normals[indices], dim=-1)
        normal_dot = torch.abs(normal_dot)
        return comp, comp_median, torch.mean(normal_dot), torch.median(normal_dot)
    
    return comp, comp_median

def accuracy_per_point(gt_points, rec_points, batch_size=5000):
    n_points = rec_points.shape[0]
    all_distances = []
    all_indices = []
    
    for i in range(0, n_points, batch_size):
        end_idx = min(i + batch_size, n_points)
        batch_points = rec_points[i:end_idx]
        
        distances = torch.cdist(batch_points, gt_points)  # (batch_size, M)
        batch_distances, batch_indices = torch.min(distances, dim=1)  # (batch_size,)
        
        all_distances.append(batch_distances)
        all_indices.append(batch_indices)
    
    distances = torch.cat(all_distances)
    return distances

def completion_per_point(gt_points, rec_points, batch_size=5000):

    n_points = gt_points.shape[0]
    all_distances = []
    all_indices = []
    
    for i in range(0, n_points, batch_size):
        end_idx = min(i + batch_size, n_points)
        batch_points = gt_points[i:end_idx]
        
        distances = torch.cdist(batch_points, rec_points)  # (batch_size, M)
        batch_distances, batch_indices = torch.min(distances, dim=1)  # (batch_size,)
        
        all_distances.append(batch_distances)
        all_indices.append(batch_indices)
    
    distances = torch.cat(all_distances)
    return distances
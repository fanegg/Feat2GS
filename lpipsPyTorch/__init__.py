import torch

from .modules.lpips import LPIPS


def lpips(x: torch.Tensor,
          y: torch.Tensor,
          net_type: str = 'alex',
          version: str = '0.1',
          return_spatial_map=False):
    r"""Function that measures
    Learned Perceptual Image Patch Similarity (LPIPS).

    Arguments:
        x, y (torch.Tensor): the input tensors to compare.
        net_type (str): the network type to compare the features: 
                        'alex' | 'squeeze' | 'vgg'. Default: 'alex'.
        version (str): the version of LPIPS. Default: 0.1.
        return_spatial_map (bool): whether to return the spatial map. Default: False.
    """
    device = x.device
    criterion = LPIPS(net_type, version).to(device)
    return criterion(x, y, return_spatial_map)

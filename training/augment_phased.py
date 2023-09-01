# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Augmentation pipeline from the paper
"Training Generative Adversarial Networks with Limited Data".
Matches the original implementation by Karras et al. at
https://github.com/NVlabs/stylegan2-ada/blob/main/training/augment.py"""

import torch
from torch_utils import persistence
from torch_utils import misc
from torch_utils.ops import grid_sample_gradfix
import torch.nn.functional as F
import numpy as np


""" Phased (normal) augmentaion 
NOTE: This code is for 256*256 traning data 
"""

def matrix(*rows, device=None):
    assert all(len(row) == len(rows[0]) for row in rows)
    elems = [x for row in rows for x in row]
    ref = [x for x in elems if isinstance(x, torch.Tensor)]
    if len(ref) == 0:
        return misc.constant(np.asarray(rows), device=device)
    assert device is None or device == ref[0].device
    elems = [x if isinstance(x, torch.Tensor) else misc.constant(x, shape=ref[0].shape, device=ref[0].device) for x in elems]
    return torch.stack(elems, dim=-1).reshape(ref[0].shape + (len(rows), -1))

def rotate2d(theta, **kwargs):
    return matrix(
        [torch.cos(theta), torch.sin(-theta), 0],
        [torch.sin(theta), torch.cos(theta),  0],
        [0,                0,                 1],
        **kwargs)

def rotate2d_inv(theta, **kwargs):
    return rotate2d(-theta, **kwargs)

@persistence.persistent_class
class AugmentPipe(torch.nn.Module):
    def __init__(self, rot=180, zoom_shift=True, color=0.3, flip=True):
        super().__init__()
        self.register_buffer('p', torch.ones([]))       # This in not used
        self.rot = rot
        self.zoom_shift = zoom_shift
        self.color = color 
        self.flip = flip

    def forward(self, images, debug_percentile=None):
        assert isinstance(images, torch.Tensor) and images.ndim == 4
        batch_size, num_channels, height, width = images.shape
        device = images.device

        ###rotate (and center crop, upsampling)
        # NOTE: bilinear, fill=none (this border mode is not same as my cv2 implemention 
        # in tf VQ-VAE-2 code, but the border does not apper because of upsampling and cropping)

        I_3 = torch.eye(3, device=device)
        G_inv = I_3

        theta_single = (torch.rand(1, device=device) * 2 - 1) * self.rot * (np.pi / 180.0)
        theta = theta_single.repeat(batch_size)
        G_inv = G_inv @ rotate2d_inv(-theta) 

        shape = [batch_size, num_channels, height, width]
        grid = torch.nn.functional.affine_grid(theta=G_inv[:,:2,:], size=shape, align_corners=False)
        images = grid_sample_gradfix.grid_sample(images, grid)

        images = images[:, :, 37:(256-38), 37:(256-38)]

        # NOTE: bilinear
        images = F.interpolate(images, size=(256, 256), mode='bilinear', align_corners=True)

        ###flip
        if self.flip:
            flip = (torch.rand(1) > 0.5).item()
        else:
            flip = 0

        if flip:
            images = torch.flip(images, [3])

        ###zoom shift (upsampling and random cropping)
        if self.zoom_shift == True:
            size = torch.round(height * (torch.rand(2, device=device) / 4 + 1.05)).long()#1.05 ~ 1.30 times (269~)
            
            # NOTE: bilinear
            images = F.interpolate(images, size=(size[0], size[1]), mode='bilinear', align_corners=True)

            # NOTE: This range length is smaller than the value of (size[0] - 256).
            shift = torch.round(torch.rand(2, device=device) * 10 - 5).long()
            
            sh = torch.round((size[0] - 256) // 2 + shift[0]).item()
            sw = torch.round((size[1] - 256) // 2 + shift[1]).item()
            
            images = images[:, :, sh:sh+height, sw:sw+height]
    
        ###brightness
        delta = (torch.rand(1, device=device) * 2 - 1) * self.color
        images = images + delta

        ###saturation
        magnitude = (torch.rand(1, 1, 1, 1, device=device) * 2 - 1) * self.color + 1
        x_mean = images.mean(dim=1, keepdim=True)
        images = (images - x_mean) * magnitude + x_mean

        ###contrast
        magnitude = (torch.rand(1, 1, 1, 1, device=device) * 2 - 1) * self.color + 1
        x_mean = images.mean(dim=[1, 2, 3], keepdim=True)
        images = (images - x_mean) * magnitude + x_mean
        # print("After contrast adjustment, requires_grad:", images.requires_grad)

        return images

#----------------------------------------------------------------------------

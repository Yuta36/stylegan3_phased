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

# Phased modified 4
# This rand_cutout function is from "Data-Efficient GANs with DiffAugment" (Pytorch)
# https://github.com/mit-han-lab/data-efficient-gans/blob/master/DiffAugment-stylegan2-pytorch/DiffAugment_pytorch.py
def rand_cutout(x, ratio=0.5):
    cutout_size = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)
    offset_x = torch.randint(0, x.size(2) + (1 - cutout_size[0] % 2), size=[x.size(0), 1, 1], device=x.device)
    offset_y = torch.randint(0, x.size(3) + (1 - cutout_size[1] % 2), size=[x.size(0), 1, 1], device=x.device)
    grid_batch, grid_x, grid_y = torch.meshgrid(
        torch.arange(x.size(0), dtype=torch.long, device=x.device),
        torch.arange(cutout_size[0], dtype=torch.long, device=x.device),
        torch.arange(cutout_size[1], dtype=torch.long, device=x.device),
    )
    grid_x = torch.clamp(grid_x + offset_x - cutout_size[0] // 2, min=0, max=x.size(2) - 1)
    grid_y = torch.clamp(grid_y + offset_y - cutout_size[1] // 2, min=0, max=x.size(3) - 1)
    mask = torch.ones(x.size(0), x.size(2), x.size(3), dtype=x.dtype, device=x.device)
    mask[grid_batch, grid_x, grid_y] = 0
    x = x * mask.unsqueeze(1)
    return x

@persistence.persistent_class
class AugmentPipe(torch.nn.Module):
    def __init__(self, rot=180, zoom_shift=True, color=0.3, flip=True, cutout=0.0):
        super().__init__()
        self.register_buffer('p', torch.ones([]))       # This in not used
        self.register_buffer('rot', torch.tensor(rot).float())
        self.register_buffer('zoom_shift', torch.tensor(zoom_shift).bool())
        self.register_buffer('color', torch.tensor(color).float()) 
        self.register_buffer('flip', torch.tensor(flip).bool())
        self.register_buffer('cutout', torch.tensor(cutout).float()) 

    def forward(self, images, debug_percentile=None):
        assert isinstance(images, torch.Tensor) and images.ndim == 4
        batch_size, num_channels, height, width = images.shape
        device = images.device

        ###rotate (and center crop, upsampling)
        # NOTE: bilinear, fill=none (this border mode is not same as my cv2 implemention 
        # in tf VQ-VAE-2 code, but the border does not apper because of upsampling and cropping)

        # Modified
        if self.rot != 0:
            I_3 = torch.eye(3, device=device)
            G_inv = I_3

            theta_single = (torch.rand(1, device=device) * 2 - 1) * self.rot * (np.pi / 180.0)
            theta = theta_single.repeat(batch_size)
            G_inv = G_inv @ rotate2d_inv(-theta) 

            shape = [batch_size, num_channels, height, width]
            grid = torch.nn.functional.affine_grid(theta=G_inv[:,:2,:], size=shape, align_corners=False)
            images = grid_sample_gradfix.grid_sample(images, grid)

        # Modified, no zoom
        # images = images[:, :, 37:(256-38), 37:(256-38)]

        # # NOTE: bilinear
        # images = F.interpolate(images, size=(256, 256), mode='bilinear', align_corners=True)

        ###flip
        if self.flip:
            flip = (torch.rand(1) > 0.5).item()
        else:
            flip = 0

        if flip:
            images = torch.flip(images, [3])

        ###zoom shift (upsampling and random cropping)
        # Modified2, 3
        if self.zoom_shift == True:
            
            # NOTE: This size is sampled under uniform distribution.
            #       To equalize the probability of getting a size of 256 pixels with other sizes, I subtract 0.5 pixels.
            size = torch.round(height * (torch.rand(2, device=device) / 4 + 1.0) - 0.5).long()#1.0 ~ 1.25 times (256 ~ (320 - 1))
            
            # NOTE: bilinear
            images = F.interpolate(images, size=(size[0], size[1]), mode='bilinear', align_corners=True)

            # NOTE: This range length is smaller than the value of (size[0] - 256).
            #       If odd max_shift value is used, the range should be open interval to avoid a cropping error.
            max_shift = size - 256
            value_to_exclude_endpoints = torch.tensor([0.00001, 0.00001], device=device)
            # Modified3
            shift_range = max_shift - value_to_exclude_endpoints
            shift = torch.round(torch.rand(2, device=device) * shift_range - shift_range / 2).long()
            
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

        # cutout 
        # 0.5 → 0.05 → 0.0(disenable)
        if self.cutout > 0.0:
            images = rand_cutout(images, self.cutout) 
        
        return images

#----------------------------------------------------------------------------

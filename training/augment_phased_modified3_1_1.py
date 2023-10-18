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
modefided3_1 → modified3_1_1: remove clamping
NOTE: This code is for 256*256 traning data.
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
        self.register_buffer('rot', torch.tensor(rot).float())
        self.register_buffer('zoom_shift', torch.tensor(zoom_shift).bool())
        self.register_buffer('color', torch.tensor(color).float()) 
        self.register_buffer('flip', torch.tensor(flip).bool())

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

            # Modified3_1, changed random parameters in a minibatch
            theta = (torch.rand(batch_size, device=device) * 2 - 1) * self.rot * (np.pi / 180.0)
            G_inv = I_3 @ rotate2d_inv(-theta) 

            shape = [batch_size, num_channels, height, width]
            grid = torch.nn.functional.affine_grid(theta=G_inv[:,:2,:], size=shape, align_corners=False)
            images = grid_sample_gradfix.grid_sample(images, grid)

        # Modified, no zoom
        # images = images[:, :, 37:(256-38), 37:(256-38)]
        # # NOTE: bilinear
        # images = F.interpolate(images, size=(256, 256), mode='bilinear', align_corners=True)

        ###flip
        # Modified3_1, changed random parameters in a minibatch
        if self.flip:
            flip = torch.rand(batch_size, device=device) > 0.5
        else:
            flip = torch.zeros(batch_size, device=device, dtype=torch.bool)

        flipped_images = torch.flip(images.clone(), [3])
        images = torch.where(flip[:, None, None, None], flipped_images, images)

        ###zoom shift (upsampling and random cropping)
        # Modified2, 3
        # Modified3_1, changed random parameters in a minibatch
        if self.zoom_shift == True:
            
            # NOTE: This size is sampled under uniform distribution.
            #       To equalize the probability of getting a size of 256 pixels with other sizes, I subtract 0.5 pixels.
            sizes = torch.round(height * (torch.rand(batch_size, 2, device=device) / 4 + 1.0) - 0.5).long()#1.0 ~ 1.25 times (256 ~ (320 - 1))

            # NOTE: This range length is smaller than the value of (size[0] - 256).
            #       If odd max_shift value is used, the range should be open interval to avoid a cropping error.
            max_shifts = sizes - 256
            values_to_exclude_endpoints = torch.tensor([0.00001, 0.00001], device=device).unsqueeze(0).repeat(batch_size, 1)
            # Modified3
            shift_ranges = max_shifts - values_to_exclude_endpoints
            shifts = torch.round(torch.rand(batch_size, 2, device=device) * shift_ranges - shift_ranges / 2).long()

            for i in range(batch_size):
                img = images[i].unsqueeze(0)
                # NOTE: bilinear
                img = F.interpolate(img, size=(sizes[i, 0].item(), sizes[i, 1].item()), mode='bilinear', align_corners=True)
                sh = ((sizes[i, 0] - 256) // 2 + shifts[i, 0]).item()
                sw = ((sizes[i, 1] - 256) // 2 + shifts[i, 1]).item()
            
                img = img[:, :, sh:sh+height, sw:sw+width]

                images[i] = img.squeeze(0)
    
        ###brightness
        # Modified3_1, changed random parameters in a minibatch
        delta = (torch.rand(batch_size, 1, 1, 1, device=device) * 2 - 1) * self.color
        images = images + delta

        ###saturation
        # Modified3_1, changed random parameters in a minibatch
        magnitude = (torch.rand(batch_size, 1, 1, 1, device=device) * 2 - 1) * self.color + 1
        x_mean = images.mean(dim=1, keepdim=True)
        images = (images - x_mean) * magnitude + x_mean

        ###contrast
        # Modified3_1, changed random parameters in a minibatch
        magnitude = (torch.rand(batch_size, 1, 1, 1, device=device) * 2 - 1) * self.color + 1
        x_mean = images.mean(dim=[1, 2, 3], keepdim=True)
        images = (images - x_mean) * magnitude + x_mean
        # print("After contrast adjustment, requires_grad:", images.requires_grad)

        return images

#----------------------------------------------------------------------------

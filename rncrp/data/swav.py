"""
Script to pass ImageNet through pretrained SwAV and save activations to disk.

Modified from https://github.com/facebookresearch/swav/
"""

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import random

import numpy as np
import os

import torchvision.datasets
from PIL import ImageFilter
from pl_bolts.models.self_supervised import SwAV
import torch
import torch.nn.functional
import torch.utils.data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from typing import List

path_to_imagenet = '/om2/user/akhilan'
path_to_write_data = '/om2/user/rylansch/FieteLab-Recursive-Nonstationary-CRP/data/swav_imagenet_2021'
os.makedirs(path_to_write_data, exist_ok=True)
model_str = 'resnet50'
assert model_str in {'resnet50', 'resnet50w2', 'resnet50w4', 'resnet50w5'}

path_to_imagenet_train = os.path.join(path_to_imagenet, 'train')
path_to_imagenet_val = os.path.join(path_to_imagenet, 'val')

# class MultiCropDataset(datasets.ImageFolder):
#
#     # Defaults copied from main_swav.py
#     def __init__(
#             self,
#             data_path,
#             size_crops: List[int] = [224],
#             nmb_crops: List[int] = [2],
#             min_scale_crops: List[float] = [0.14],
#             max_scale_crops: List[float] = [1],
#             size_dataset: int = -1,
#             return_index: bool = False,
#     ):
#         super(MultiCropDataset, self).__init__(data_path)
#         assert len(size_crops) == len(nmb_crops)
#         assert len(min_scale_crops) == len(nmb_crops)
#         assert len(max_scale_crops) == len(nmb_crops)
#         if size_dataset >= 0:
#             self.samples = self.samples[:size_dataset]
#         self.return_index = return_index
#
#         color_transform = [get_color_distortion(), PILRandomGaussianBlur()]
#         mean = [0.485, 0.456, 0.406]
#         std = [0.228, 0.224, 0.225]
#         trans = []
#         for i in range(len(size_crops)):
#             randomresizedcrop = transforms.RandomResizedCrop(
#                 size_crops[i],
#                 scale=(min_scale_crops[i], max_scale_crops[i]),
#             )
#             trans.extend([transforms.Compose([
#                 randomresizedcrop,
#                 transforms.RandomHorizontalFlip(p=0.5),
#                 transforms.Compose(color_transform),
#                 transforms.ToTensor(),
#                 transforms.Normalize(mean=mean, std=std)])
#                          ] * nmb_crops[i])
#         self.trans = trans
#
#     def __getitem__(self, index):
#         path, _ = self.samples[index]
#         image = self.loader(path)
#         multi_crops = list(map(lambda trans: trans(image), self.trans))
#         if self.return_index:
#             return index, multi_crops
#         return multi_crops
#
#
# class PILRandomGaussianBlur(object):
#     """
#     Apply Gaussian Blur to the PIL image. Take the radius and probability of
#     application as the parameter.
#     This transform was used in SimCLR - https://arxiv.org/abs/2002.05709
#     """
#
#     def __init__(self, p=0.5, radius_min=0.1, radius_max=2.):
#         self.prob = p
#         self.radius_min = radius_min
#         self.radius_max = radius_max
#
#     def __call__(self, img):
#         do_it = np.random.rand() <= self.prob
#         if not do_it:
#             return img
#
#         return img.filter(
#             ImageFilter.GaussianBlur(
#                 radius=random.uniform(self.radius_min, self.radius_max)
#             )
#         )
#
#
# def get_color_distortion(s=1.0):
#     # s is the strength of color distortion.
#     color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
#     rnd_color_jitter = transforms.RandomApply([color_jitter], p=0.8)
#     rnd_gray = transforms.RandomGrayscale(p=0.2)
#     color_distort = transforms.Compose([rnd_color_jitter, rnd_gray])
#     return color_distort


# model = torch.hub.load('facebookresearch/swav:main',
#                        model=model_str,
#                        # pretrained=True,
#                        )

# Print available methods
# print(dir(model))

# Print what arguments model requires to run
# print(help(model))

weight_path = 'https://pl-bolts-weights.s3.us-east-2.amazonaws.com/swav/swav_imagenet/swav_imagenet.pth.tar'
swav = SwAV.load_from_checkpoint(weight_path, strict=True)
swav.freeze()
# swav = swav.cuda()

# # Check out number of redundant prototypes
# p = swav.model.prototypes.weight.numpy()
# # Exclude diagonal elements
# tmp = np.sort((p @ p.T)[~np.eye(3000, dtype=bool)])[::-1]
# num_overlapping_prototypes = np.sum(tmp > 0.95)
# frac_overlapping_prototypes = np.mean(tmp > 0.95)

# model = torchvision.models.resnet50()
# ckp = torch.hub.load_state_dict_from_url(
#     'https://dl.fbaipublicfiles.com/deepcluster/swav_800ep_pretrain.pth.tar',
#     map_location=torch.device('cpu'))
#
#
# for k in ckp.keys():
#     if "projection_head" in k or "prototypes" in k:
#         print(k, ckp[k].shape)

# checkpoint = torch.load('.user/swav_800ep_pretrain.pth.tar')
# model.load_state_dict(checkpoint, strict=False)
#
#
# model.load_state_dict(ckp)

print(f'Loaded model: {model_str}')

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize,
])

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize,
])

for split in ['val', 'train']:

    if split == 'train':
        dataset = torchvision.datasets.ImageFolder(
            path_to_imagenet_train,
            transform=train_transform)
    elif split == 'val':
        dataset = torchvision.datasets.ImageFolder(
            path_to_imagenet_val,
            transform=val_transform)
    else:
        raise ValueError

    print('Created dataset.')

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=20,  # SwAV Default arg = 64
        num_workers=2,  # SwAV Default arg = 10
        drop_last=False,
    )
    print('Created dataloader.')

    embeddings, targets = [], []

    for batch_index, (input_tensor, target_tensor) in enumerate(dataloader, start=1):

        print(f'Batch index: {batch_index}')

        # normalize the prototypes
        with torch.no_grad():

            # Note! This is different than swav(input_tensor) because it also passes through
            # the MLP projection head
            embedding, _ = swav.model.forward(input_tensor)
            # embedding = embedding.detach()
            embeddings.append(embedding.detach().numpy())
            targets.append(target_tensor.numpy())

        # 20 embeddings per batch * 50 batches per write = 10k embeddings per write
        if (batch_index % 50) == 0:

            embeddings_array = np.concatenate(embeddings)
            targets_array = np.concatenate(targets)

            # Imagenet Classes: https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a
            np.savez(file=os.path.join(path_to_write_data, split + f'_batch={batch_index}.npz'),
                     embeddings=embeddings_array,
                     targets=targets_array,
                     prototypes=swav.model.prototypes.weight.numpy())

            print('Wrote representations to disk.')

            embeddings, targets = [], []

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
import torch
import torch.nn.functional
import torch.utils.data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from typing import List


path_to_imagenet = '/om2/user/akhilan'
path_to_write_data = 'data/swav_imagenet_2021'
os.makedirs(path_to_write_data, exist_ok=True)
model_str = 'resnet50'
assert model_str in {'resnet50', 'resnet50w2', 'resnet50w4', 'resnet50w5'}

path_to_imagenet_train = os.path.join(path_to_imagenet, 'train')
path_to_imagenet_val = os.path.join(path_to_imagenet, 'val')


class MultiCropDataset(datasets.ImageFolder):

    # Defaults copied from main_swav.py
    def __init__(
        self,
        data_path,
        size_crops: List[int] = [224],
        nmb_crops: List[int] = [2],
        min_scale_crops: List[float] = [0.14],
        max_scale_crops: List[float] = [1],
        size_dataset: int = -1,
        return_index: bool = False,
    ):
        super(MultiCropDataset, self).__init__(data_path)
        assert len(size_crops) == len(nmb_crops)
        assert len(min_scale_crops) == len(nmb_crops)
        assert len(max_scale_crops) == len(nmb_crops)
        if size_dataset >= 0:
            self.samples = self.samples[:size_dataset]
        self.return_index = return_index

        color_transform = [get_color_distortion(), PILRandomGaussianBlur()]
        mean = [0.485, 0.456, 0.406]
        std = [0.228, 0.224, 0.225]
        trans = []
        for i in range(len(size_crops)):
            randomresizedcrop = transforms.RandomResizedCrop(
                size_crops[i],
                scale=(min_scale_crops[i], max_scale_crops[i]),
            )
            trans.extend([transforms.Compose([
                randomresizedcrop,
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.Compose(color_transform),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)])
            ] * nmb_crops[i])
        self.trans = trans

    def __getitem__(self, index):
        path, _ = self.samples[index]
        image = self.loader(path)
        multi_crops = list(map(lambda trans: trans(image), self.trans))
        if self.return_index:
            return index, multi_crops
        return multi_crops


class PILRandomGaussianBlur(object):
    """
    Apply Gaussian Blur to the PIL image. Take the radius and probability of
    application as the parameter.
    This transform was used in SimCLR - https://arxiv.org/abs/2002.05709
    """

    def __init__(self, p=0.5, radius_min=0.1, radius_max=2.):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        do_it = np.random.rand() <= self.prob
        if not do_it:
            return img

        return img.filter(
            ImageFilter.GaussianBlur(
                radius=random.uniform(self.radius_min, self.radius_max)
            )
        )


def get_color_distortion(s=1.0):
    # s is the strength of color distortion.
    color_jitter = transforms.ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.2*s)
    rnd_color_jitter = transforms.RandomApply([color_jitter], p=0.8)
    rnd_gray = transforms.RandomGrayscale(p=0.2)
    color_distort = transforms.Compose([rnd_color_jitter, rnd_gray])
    return color_distort


# model = torch.hub.load('facebookresearch/swav:main',
#                        model=model_str,
#                        # pretrained=True,
#                        )

# Print available methods
# print(dir(model))

# Print what arguments model requires to run
# print(help(model))

from pl_bolts.models.self_supervised import SwAV

weight_path = 'https://pl-bolts-weights.s3.us-east-2.amazonaws.com/swav/swav_imagenet/swav_imagenet.pth.tar'
swav = SwAV.load_from_checkpoint(weight_path, strict=True)

swav.freeze()


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

train_dataset = torchvision.datasets.ImageFolder(
    root=path_to_imagenet_train,
    transform=torchvision.transforms.Compose([
        torchvision.transforms.ToTensor()]))

# train_dataset = torchvision.datasets.ImageNet(
#     root=path_to_imagenet)

# train_dataset = MultiCropDataset(
#     data_path=path_to_imagenet_train)

# sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    # sampler=sampler,
    batch_size=1,  # SwAV Default arg
    num_workers=1,  # SwAV Default arg
    pin_memory=True,
    drop_last=False,
)

# train_loader.sampler.set_epoch(0)

embeddings, targets = [], []

for batch_index, (input_tensor, target_tensor) in enumerate(train_loader):

    print(f'Batch index: {batch_index}')

    # normalize the prototypes
    with torch.no_grad():
        # w = model.module.prototypes.weight.data.clone()
        # w = torch.nn.functional.normalize(w, dim=1, p=2)
        # model.module.prototypes.weight.copy_(w)

        # Note! This is different than swav(input_tensor) because it also passes through
        # the MLP projection head
        embedding, _ = swav.model.forward(input_tensor)
        # embedding = embedding.detach()
        embeddings.append(embedding.detach().numpy())
        targets.append(target_tensor.numpy())

    if batch_index > 10:
        break

print('Finished extracting ImageNet representations.')

embeddings = np.concatenate(embeddings)
targets = np.concatenate(targets)

print('Concatenated representations and converted to NumPy.')

# Imagenet Classes: https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a
np.save(file=path_to_write_data,
        outputs=embeddings,
        targets=targets,
        prototypes=swav.model.prototypes.weight.numpy())

print('Wrote representations to disk.')



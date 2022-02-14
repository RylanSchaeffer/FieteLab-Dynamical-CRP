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
from logging import getLogger

from PIL import ImageFilter
import numpy as np
import torch
import torch.nn.functional
import torch.utils.data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from typing import List

logger = getLogger()


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



model = torch.hub.load('facebookresearch/swav:main', 'resnet50')

path_to_imagenet = '/home/akhilan/om2/train'

train_dataset = MultiCropDataset(
    data_path=path_to_imagenet)

sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    sampler=sampler,
    batch_size=64,  # SwAV Default arg
    num_workers=10,  # SwAV Default arg
    pin_memory=True,
    drop_last=True
)

for it, inputs in enumerate(train_loader):

    # normalize the prototypes
    with torch.no_grad():
        w = model.module.prototypes.weight.data.clone()
        w = torch.nn.functional.normalize(w, dim=1, p=2)
        model.module.prototypes.weight.copy_(w)

    embedding, output = model(inputs)
    embedding = embedding.detach()


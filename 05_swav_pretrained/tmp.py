import numpy as np
import os
from torch.utils.data import DataLoader

from rncrp.data.real_nontabular import SwavImageNet2021Dataset, ChangingWeightedClassesRandomSampler


data_dir = 'data'
split = 'val'
n_samples = 5000
include_images = False
dataset_dir = os.path.join(data_dir, 'swav_imagenet_2021')
swav_imagenet_dataset = SwavImageNet2021Dataset(
    dataset_dir=dataset_dir,
    split=split,
    include_images=include_images,
    n_samples=n_samples)

initial_classes = np.random.choice(
    np.arange(len(swav_imagenet_dataset.labels_to_indices_list)),
    size=5)
sampler = ChangingWeightedClassesRandomSampler(
    dataset=swav_imagenet_dataset,
    initial_classes=initial_classes,
)

default_dataloader_kwargs = {
    'batch_size': 1,
    'shuffle': False,  # do the permutation in the dataset, not the dataloader
    'num_workers': 2,
    'drop_last': False,
}

swav_imagenet_dataloader = DataLoader(
    dataset=swav_imagenet_dataset,
    sampler=sampler,
    **default_dataloader_kwargs
)

for batch_idx, batch_tensors in enumerate(swav_imagenet_dataloader):
    print(10)

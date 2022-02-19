import numpy as np
import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms
from typing import Dict, Union


class SwavImageNet2011Dataset(Dataset):

    def __init__(self,
                 dataset_dir,
                 transforms=None,
                 split: str = 'val',
                 include_images: bool = False,
                 n_samples: int = None):

        assert split in {'train', 'test', 'val'}
        self.dataset_dir = dataset_dir
        self.split = split
        self.dataset_split_dir = os.path.join(dataset_dir, split)

        if transforms is None:
            self.transforms = torchvision.transforms.ToTensor()

        self.transforms = transforms
        self.include_images = include_images
        self.file_paths = np.random.permutation([
            os.path.join(self.dataset_split_dir, filename)
            for filename in os.listdir(self.dataset_split_dir)])
        self.n_samples = n_samples
        if n_samples is not None:
            self.file_paths = self.file_paths[:n_samples]

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx: int):
        with np.load(self.file_paths[idx]) as np_array_fp:
            embedding = np_array_fp['embedding']
            target = np_array_fp['target']
            sample = {'observations': embedding,
                      'target': target}
            if self.include_images:
                sample['image'] = np_array_fp['image']

        if self.transforms:
            sample = self.transforms(sample)

        return sample


def load_dataloader_swav_imagenet_2021(data_dir: str = 'data',
                                       split: str = 'val',
                                       include_images: bool = False,
                                       n_samples: int = None,
                                       dataloader_kwargs: Dict = None):
    dataset_dir = os.path.join(data_dir, 'swav_imagenet_2021')

    dataset = SwavImageNet2011Dataset(
        dataset_dir=dataset_dir,
        split=split,
        include_images=include_images,
        n_samples=n_samples)

    default_dataloader_kwargs = {
        'batch_size': 1,
        'shuffle': False,  # do the permutation in the dataset, not the dataloader
        'num_workers': 2,
        'drop_last': False,
    }

    # Overwrite default Dataloader kwargs if given.
    if dataloader_kwargs is not None:
        default_dataloader_kwargs.update(dataloader_kwargs)

    dataloader = DataLoader(
        dataset=dataset,
        **default_dataloader_kwargs
    )

    return dataloader


def load_dataset_yilun_nav_2d_2022(data_dir: str = 'data',
                                   **kwargs,
                                   ) -> Dict[str, Union[np.ndarray, pd.DataFrame]]:

    dataset_dir = os.path.join(data_dir, 'yilun_nav_2d_2022')
    data_path = os.path.join(dataset_dir, 'data_traj.npz')

    with np.load(data_path) as np_fp:
        landmarks = np_fp['landmarks']  # Shape: (num envs, num landmarks, 2 for xy position)
        points = np_fp['points']  # Shape: (num envs, num vis points, 2 for xy position)
        room_ids = np_fp['room_ids']  # Shape: (num envs, num vis points)
        vis_matrix = np_fp['vis_matrix']  # (num envs, num , num)

    dataset_dict = dict(
        landmarks=landmarks,
        points=points,
        room_ids=room_ids,
        vis_matrix=vis_matrix,
    )

    return dataset_dict

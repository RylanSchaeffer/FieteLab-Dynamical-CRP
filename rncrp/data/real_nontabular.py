import numpy as np
import os
import torch
from torch.utils.data import Dataset, DataLoader


class SwavImageNet2011Dataset(Dataset):

    def __init__(self,
                 dataset_dir,
                 transforms=None,
                 split: str = 'val',
                 include_images: bool = False):

        assert split in {'train', 'test', 'val'}
        self.dataset_dir = dataset_dir
        self.split = split
        self.dataset_split_dir = os.path.join(dataset_dir, split)
        if transforms is not None:
            raise NotImplementedError('Transforms have not yet been implemented.')
        self.transforms = transforms
        self.include_images = include_images
        self.file_paths = sorted([
            os.path.join(dataset_dir, filename)
            for filename in os.listdir(self.dataset_split_dir)])

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx: int):
        with np.load(self.file_paths[idx]) as np_array_fp:
            embedding = np_array_fp['embedding']
            target = np_array_fp['target']
            if self.include_images:
                image = np_array_fp['image']

        if self.include_images:
            return embedding, target, image
        else:
            return embedding, target


def load_dataset_swav_imagenet_2011(data_dir: str = 'data',
                                    batch_size: int = 32,
                                    split: str = 'val',
                                    include_images: bool = False):

    dataset_dir = os.path.join(data_dir, 'swav_imagenet_2021')

    dataset = SwavImageNet2011Dataset(
        dataset_dir=dataset_dir,
        split=split,
        include_images=include_images)

    return dataset

import numpy as np
import os
import torch
from torch.utils.data import Dataset, DataLoader


class SwavImageNet2011Dataset(Dataset):

    def __init__(self,
                 dataset_dir,
                 transforms=None,
                 split: str = 'val'):
        assert split in {'train', 'test', 'val'}
        self.dataset_dir = dataset_dir
        if transforms is not None:
            raise NotImplementedError('Transforms have not yet been implemented.')
        self.transforms = transforms
        self.file_paths = sorted([
            os.path.join(dataset_dir, filename) for filename in os.listdir(dataset_dir)
            if filename.startswith(split)])

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx: int):
        with open(self.file_paths[idx], 'r') as np_array_file:
            embedding = np_array_file['embeddings']
            target = np_array_file['targets']
        return embedding, target


def load_dataset_swav_imagenet_2011(data_dir: str = 'data',
                                    batch_size: int = 32,
                                    split: str = 'val',
                                    include_images: bool = False):

    dataset_dir = os.path.join(data_dir, 'swav_imagenet_2021')

    dataset = SwavImageNet2011Dataset(
        dataset_dir=dataset_dir,
        split='val',
    )

    return dataset

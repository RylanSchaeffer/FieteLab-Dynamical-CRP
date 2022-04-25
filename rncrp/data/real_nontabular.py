import numpy as np
import os
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data.sampler import Sampler
from typing import Iterator, Optional, Sequence, List, TypeVar, Generic, Sized
import torch.nn.functional
from torch.utils.data import Dataset, DataLoader, BatchSampler
import torchvision.transforms
from typing import Dict, List, Tuple, Union


class SwavImageNet2021Dataset(Dataset):

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

        self.labels_to_indices_list = self._create_labels_to_indices_list()

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

    def _create_labels_to_indices_list(self) -> List[np.ndarray]:
        labels = np.zeros(self.__len__(), dtype=np.int)
        indices = np.arange(self.__len__())
        for file_idx, file_path in enumerate(self.file_paths):
            labels[file_idx] = self.__getitem__(idx=file_idx)['target']
        labels_to_indices_list = []
        for unique_label in np.unique(labels):
            labels_to_indices_list.append(indices[labels == unique_label])
        return labels_to_indices_list


class ChangingWeightedClassesRandomSampler(BatchSampler):
    """
    Great forum: https://discuss.pytorch.org/t/load-the-same-number-of-data-per-class/65198/4?
    """
    curr_classes: Tensor
    initial_class_weights: Tensor
    curr_classes: Tensor
    num_samples: int

    def __init__(self,
                 dataset: SwavImageNet2021Dataset,
                 initial_classes: np.ndarray,
                 transition_prob: float = 0.005,
                 num_samples: int = 1,
                 generator=None) -> None:

        self.dataset = dataset
        self.initial_classes = initial_classes
        self.curr_classes = np.copy(self.initial_classes)
        self.possible_classes = list(range(len(self.dataset.labels_to_indices_list)))
        self.transition_prob = transition_prob
        self.num_samples = num_samples
        self.generator = generator
        self.count = 0

    def __iter__(self):
        self.count = 0
        while self.count < len(self.dataset):
            # First, sample the class index.
            class_idx = np.random.choice(self.curr_classes)
            # Next, sample uniformly from within the class.
            sample_idx = np.random.choice(self.dataset.labels_to_indices_list[class_idx])
            yield sample_idx
            self.count += 1
            for class_idx, curr_class in enumerate(self.curr_classes):
                # With low probability, transition to new classes.
                if np.random.random() < self.transition_prob:
                    self.curr_classes[class_idx] = np.random.choice(
                        self.possible_classes[:curr_class] + self.possible_classes[curr_class+1:])

    def __len__(self) -> int:
        return len(self.dataset)


def load_dataloader_swav_imagenet_2021(data_dir: str = 'data',
                                       split: str = 'val',
                                       include_images: bool = False,
                                       n_samples: int = None,
                                       n_starting_classes: int = 5,
                                       transition_prob: float = 0.005,
                                       dataloader_kwargs: Dict = None):

    dataset_dir = os.path.join(data_dir, 'swav_imagenet_2021')

    swav_imagenet_dataset = SwavImageNet2021Dataset(
        dataset_dir=dataset_dir,
        split=split,
        include_images=include_images,
        n_samples=n_samples)

    initial_classes = np.random.choice(
        np.arange(len(swav_imagenet_dataset.labels_to_indices_list)),
        size=n_starting_classes)
    sampler = ChangingWeightedClassesRandomSampler(
        dataset=swav_imagenet_dataset,
        initial_classes=initial_classes,
        transition_prob=transition_prob,
    )

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
        dataset=swav_imagenet_dataset,
        sampler=sampler,
        **default_dataloader_kwargs
    )

    return dataloader


def load_dataset_omniglot(data_dir: str = 'data',
                          num_data: int = None,
                          center_crop: bool = True,
                          avg_pool: bool = False,
                          feature_extractor_method: str = 'pca',
                          shuffle=True):
    """

    """

    assert feature_extractor_method in {'pca', 'cnn', 'vae', 'vae_old', None}

    # https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
    transforms = [torchvision.transforms.ToTensor()]
    if center_crop:
        transforms.append(torchvision.transforms.CenterCrop((80, 80)))
    if avg_pool:
        transforms.append(torchvision.transforms.Lambda(
            lambda x: torch.nn.functional.avg_pool2d(x, kernel_size=9, stride=3)))

    omniglot_dataset = torchvision.datasets.Omniglot(
        root=data_dir,
        download=True,
        transform=torchvision.transforms.Compose(transforms))

    # truncate dataset for now
    if num_data is None:
        num_data = len(omniglot_dataset._flat_character_images)
    omniglot_dataset._flat_character_images = omniglot_dataset._flat_character_images[:num_data]
    dataset_size = len(omniglot_dataset._flat_character_images)

    omniglot_dataloader = torch.utils.data.DataLoader(
        dataset=omniglot_dataset,
        batch_size=1,
        shuffle=False)

    images, labels = [], []
    for image, label in omniglot_dataloader:
        labels.append(label)
        images.append(image[0, 0, :, :])
        # uncomment to deterministically append the first image
        # images.append(omniglot_dataset[0][0][0, :, :])
    images = torch.stack(images).numpy()

    # These might be swapped, but I think height = width for omniglot.
    _, image_height, image_width = images.shape
    labels = np.array(labels)

    if feature_extractor_method == 'pca':
        from sklearn.decomposition import PCA
        reshaped_images = np.reshape(images, newshape=(dataset_size, image_height * image_width))
        pca = PCA(n_components=20)
        pca_latents = pca.fit_transform(reshaped_images)
        image_features = pca.inverse_transform(pca_latents)
        # image_features = np.reshape(pca.inverse_transform(pca_latents),
        #                             newshape=(dataset_size, image_height, image_width))
        feature_extractor = pca
    elif feature_extractor_method == 'cnn':
        # # for some reason, omniglot uses 1 for background and 0 for stroke
        # # whereas MNIST uses 0 for background and 1 for stroke
        # # for consistency, we'll invert omniglot
        # images = 1. - images
        #
        # from utils.omniglot_feature_extraction import cnn_load
        # lenet = cnn_load()
        #
        # from skimage.transform import resize
        # downsized_images = np.stack([resize(image, output_shape=(28, 28))
        #                              for image in images])
        #
        # # import matplotlib.pyplot as plt
        # # plt.imshow(downsized_images[0], cmap='gray')
        # # plt.title('Test Downsized Omniglot')
        # # plt.show()
        #
        # # add channel dimension for CNN
        # reshaped_images = np.expand_dims(downsized_images, axis=1)
        #
        # # make sure dropout is turned off
        # lenet.eval()
        # image_features = lenet(torch.from_numpy(reshaped_images)).detach().numpy()
        #
        # feature_extractor = lenet

        raise NotImplementedError
    elif feature_extractor_method is None:
        image_features = np.reshape(
            images,
            newshape=(dataset_size, image_height * image_width))
        feature_extractor = None
    else:
        raise ValueError(f'Impermissible feature method: {feature_extractor_method}')

    # # visualize images if curious
    # import matplotlib.pyplot as plt
    # for idx in range(10):
    #     plt.imshow(image_features[idx], cmap='gray')
    #     plt.show()

    if shuffle:
        random_indices = np.random.choice(
            np.arange(num_data),
            size=num_data,
            replace=False)
        images = images[random_indices]
        labels = labels[random_indices]
        image_features = image_features[random_indices]

    omniglot_dataset_results = dict(
        images=images,
        labels=labels,
        feature_extractor_method=feature_extractor_method,
        feature_extractor=feature_extractor,
        image_features=image_features,
    )

    return omniglot_dataset_results


def load_dataset_omniglot_vae(data_dir: str = 'data',
                              num_data: int = None,
                              feature_extractor_method: str = 'pca'):
    """
    Load Omniglot VAE embeddings from VAE with single Gaussian latent.
    """

    # The file `omniglot_data.npz` contains Gaussian latent vectors from
    # a VAE with a single Gaussian prior, generated using
    # https://github.com/jmtomczak/vae_vampprior.
    # vae_data = np.load(os.path.join(data_dir, 'omniglot_vae/omniglot_data.npz'))
    vae_data = np.load(os.path.join(data_dir, 'omniglot_vae/omniglot_data_with_language_labels.npz'))

    # transforms = [torchvision.transforms.ToTensor()]
    # omniglot_dataset = torchvision.datasets.Omniglot(
    #     root=data_dir,
    #     download=True,
    #     transform=torchvision.transforms.Compose(transforms))

    # make sure labels are sorted before slicing so we get multiple instances of the same class
    total_num_data = len(vae_data['targets'])
    indices_to_sort_labels = np.random.choice(
        np.arange(total_num_data),
        size=total_num_data,
        replace=False)
    labels = vae_data['targets'][indices_to_sort_labels]
    images = vae_data['images'][indices_to_sort_labels]
    image_features = vae_data['latents'][indices_to_sort_labels]

    if num_data is not None:
        labels = labels[:num_data]
        images = images[:num_data, :, :]
        image_features = image_features[:num_data, :]

    omniglot_dataset_results = dict(
        images=images,
        labels=labels,
        feature_extractor_method=feature_extractor_method,
        image_features=image_features,
    )

    return omniglot_dataset_results


def load_dataset_yilun_nav_2d_2022(data_dir: str = 'data',
                                   narrow_hallways: bool = False,
                                   finite_vision: bool = False,
                                   **kwargs,
                                   ) -> Dict[str, Union[np.ndarray, pd.DataFrame]]:
    dataset_dir = os.path.join(data_dir, 'yilun_nav_2d_2022')
    if narrow_hallways:
        if finite_vision:
            data_path = os.path.join(dataset_dir, 'data_traj_narrow_hallways_limited_vision.npz')
        else:
            data_path = os.path.join(dataset_dir, 'data_traj_narrow_hallways.npz')
    else:
        data_path = os.path.join(dataset_dir, 'data_traj.npz')

    with np.load(data_path) as np_fp:
        # WARNING: for no reason, the number of landmarks is the same as the number of vis points
        # Don't screw up the dimensions!
        landmarks = np_fp['landmarks']  # Shape: (num envs, num landmarks, 2 for xy position)
        points = np_fp['points']  # Shape: (num envs, num vis points, 2 for xy position)
        room_ids = np_fp['room_ids']  # Shape: (num envs, num vis points)
        room_lists = np_fp['room_lists']  # Shape: (num envs, 5, 4)
        vis_matrix = np_fp['vis_matrix']  # Shape: (num envs, num vis points, num landmarks)
        edges = np_fp['edges']  # Shape: (num envs, 5)

    dataset_dict = dict(
        landmarks=landmarks,
        points=points,
        room_ids=room_ids,
        vis_matrix=vis_matrix,
        edges=edges,
        room_lists=room_lists,
    )

    return dataset_dict

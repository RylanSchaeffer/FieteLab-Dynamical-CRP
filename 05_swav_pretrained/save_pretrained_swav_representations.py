"""
Script to pass ImageNet through pretrained SwAV and save activations to disk.

"""


import torch
import torch.nn.functional
import torch.utils.data

from multicropdataset import MultiCropDataset


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



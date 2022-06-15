from torchvision import transforms, datasets
from torch_geometric import datasets
from typing import *
import torch
import os
from torch.utils.data import Dataset
import process_data

# list of all datasets
DATASETS = ["modelnet40","modelnet40_128","modelnet40_256","modelnet40_512",
            "modelnet40_1024","modelnet40_32","modelnet40_16", "shapenet"]


def get_dataset(dataset: str, split: str) -> Dataset:
    """Return the dataset as a PyTorch Dataset object"""
    if dataset == "modelnet40":
        return modelnet40(64,split)
    elif dataset == "modelnet40_128":
        return modelnet40(128,split)
    elif dataset == "modelnet40_256":
        return modelnet40(256,split)
    elif dataset == "modelnet40_512":
        return modelnet40(512,split)
    elif dataset == "modelnet40_1024":
        return modelnet40(1024,split)
    elif dataset == "modelnet40_32":
        return modelnet40(32,split)
    elif dataset == "modelnet40_16":
        return modelnet40(16,split)
    elif dataset == "shapenet":
        return shapenet(64,split)


def get_num_classes(dataset: str):
    """Return the number of classes in the dataset. """
    if "modelnet40" in dataset:
        return 40
    elif "shapenet" in dataset:
        return 50


def modelnet40(num_points: int = 1024, split: str = 'train') -> datasets.modelnet.ModelNet:
    dataset_root = "./dataset_cache/modelnet40fp"
    assert 1 <= num_points <= 1024, "num_points must be between 1 and 1024"
    train = split == 'train'

    pre_transforms = transforms.Compose([
        process_data.SamplePoints(num=4096),
        process_data.FarthestPoints(num_points=1024)
    ])

    # if add_noise is None:
    #     add_noise = split == 'train'

    if train:
        post_transforms = transforms.Compose([
            process_data.ConvertFromGeometric(),
            process_data.NormalizeUnitSphere(),
            process_data.SelectPoints(num_points),
            process_data.RemoveNones()
        ])
    else:
        post_transforms = transforms.Compose([
            process_data.ConvertFromGeometric(),
            process_data.NormalizeUnitSphere(),
            process_data.SelectPoints(num_points),
            process_data.RemoveNones()
        ])
    # if add_noise:
    #     post_transforms.transforms.append(transformers.GaussianNoise())

    return datasets.modelnet.ModelNet(root=dataset_root, name='40', train=train,
                                      pre_transform=pre_transforms, transform=post_transforms)

def shapenet(num_points: int = 1024, split: str = 'train') -> datasets.shapenet.ShapeNet:
    dataset_root = "./dataset_cache/shapenet"
    assert 1 <= num_points <= 1024, "num_points must be between 1 and 1024"
    assert split in ['train', 'test'], "split must either be 'train' or 'test'"
    train = split == 'train'

    pre_transforms = transforms.Compose([
        process_data.FarthestPoints(num_points=1024)
    ])

    if train:
        split = 'trainval'
        post_transforms = transforms.Compose([
            process_data.ConvertFromGeometric(),
            process_data.NormalizeUnitSphere(),
            process_data.SelectPoints(num_points),
            process_data.RemoveNones()
        ])
    else:
        split = 'test'
        post_transforms = transforms.Compose([
            process_data.ConvertFromGeometric(),
            process_data.NormalizeUnitSphere(),
            process_data.SelectPoints(num_points),
            process_data.RemoveNones()
        ])
    return datasets.shapenet.ShapeNet(root=dataset_root, include_normals=False, split=split,
                                      pre_transform=pre_transforms, transform=post_transforms)
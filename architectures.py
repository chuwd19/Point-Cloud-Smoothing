import torch
import torch.backends.cudnn as cudnn
from archs.pointnet import PointNet, PointNetSegmentation
from torch.nn.functional import interpolate
import torch.nn as nn

ARCHITECTURES = ['pointnet','pointnet128','pointnet256','pointnet512',
                'pointnet1024','pointnet32','pointnet16','pointnet_segmentation']

def get_architecture(arch: str, dataset: str) -> torch.nn.Module:
    """ Return a neural network (with random weights)

    :param arch: the architecture - should be in the ARCHITECTURES list above
    :param dataset: the dataset - should be in the datasets.DATASETS list
    :return: a Pytorch module
    """
    if arch == "pointnet":
        model = PointNet(number_points=64,num_classes=40)
        model = model
    elif arch == "pointnet128":
        model = PointNet(number_points=128,num_classes=40)
        model = model
    elif arch == "pointnet256":
        model = PointNet(number_points=256,num_classes=40)
        model = model
    elif arch == "pointnet512":
        model = PointNet(number_points=512,num_classes=40)
        model = model
    elif arch == "pointnet1024":
        model = PointNet(number_points=1024,num_classes=40)
        model = model
    elif arch == "pointnet32":
        model = PointNet(number_points=32,num_classes=40)
        model = model
    elif arch == "pointnet16":
        model = PointNet(number_points=16,num_classes=40)
        model = model
    elif arch == "pointnet_segmentation":
        model = PointNetSegmentation(number_points=64,num_seg_classes=50)
    return model

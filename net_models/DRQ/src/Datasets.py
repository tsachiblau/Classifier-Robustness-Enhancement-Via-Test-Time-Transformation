from .CustomEnums import DataSetName
from robustbench.data import load_cifar10c
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset
import torch

import os

def get_data_set(conf):
    if conf.ood_dataset:
        dataset = conf.ood_dataset
    else:
        dataset = conf.dataset
    batch_size = conf.batch_size

    if dataset == DataSetName.cifar10:
        test = get_cifar10(conf)
    elif dataset == DataSetName.cifar10c:
        x, y = get_cifar10c(conf)
        y = torch.tensor(y, dtype=torch.long)
        test = TensorDataset(x, y)
    else:
        raise  Exception("Dataset not defined")

    loader = get_loader(test, batch_size)

    return loader

def get_cifar10c(conf):
    return load_cifar10c(1000 * len(conf.corruptions), severity=conf.severity, data_dir=conf.data_path, shuffle=False, corruptions=conf.corruptions)

def get_cifar10(conf):
    transform = get_transform()

    test = datasets.CIFAR10(os.path.join(conf.data_path, "CIFAR10-data"), train=False, download=True,
                                         transform=transform)
    return test

def get_transform():
    t = []
    t.append(transforms.ToTensor())
    transform = transforms.Compose(t)
    return transform

def get_mean_std(data_set):
    mean = 0.
    std = 1.
    # Standatarize for CIFAR10
    if data_set == DataSetName.cifar_10:
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
    channels = get_dataset_information(data_set)[1]
    if channels == 3:
        return torch.tensor(mean).view(3, 1, 1).cuda(), torch.tensor(std).view(3, 1, 1).cuda()
    else:
        return torch.tensor(mean), torch.tensor(std)

def get_lower_and_upper_limits():
    lower_limit = 0.
    upper_limit = 1.
    return lower_limit, upper_limit

def get_loader(test, batch_size):
    loader = DataLoader(test, batch_size=batch_size, shuffle=False, pin_memory=True)

    return loader

def get_dataset_information(dataset):
    if dataset == DataSetName.cifar10 or dataset == DataSetName.cifar10c:
        return {"classes":10, 'channels':3, 'shape':(1, 3, 32, 32)}
    elif dataset == DataSetName.cifar100 or dataset == DataSetName.cifar100c:
        return  {"classes":100, 'channels':3, 'shape':(1, 3, 32, 32)}
    elif dataset == DataSetName.svhn:
        return {"classes":10, 'channels':3, 'shape':(1, 3, 32, 32)}
    elif dataset == DataSetName.ImageNet_A:
        return {"classes":200, 'channels':3, 'shape':(1, 3, 224, 224)}
    elif dataset == DataSetName.ImageNet_C:
        return {"classes":1000, 'channels':3, 'shape':(1, 3, 224, 224)}
    elif dataset == DataSetName.imagenet:
        return {"classes":1000, 'channels':3, 'shape':(1, 3, 224, 224)}
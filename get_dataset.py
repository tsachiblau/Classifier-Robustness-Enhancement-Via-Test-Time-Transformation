import os.path

import torchvision
from torchvision import transforms
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt


def get_data_loader(args):
    if args.dataset == 'cifar10':
        if args.aa_dataset == False:
            testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                                   download=True, transform=transforms.ToTensor())
        else:
            testset = custom_dataset(dataset = args.dataset,
                                     data_path=args.aa_dataset_path,
                                     labels_path=args.aa_labels_path,
                                     transform=transforms.ToTensor())
    elif args.dataset == 'cifar100':
        if args.aa_dataset == False:
            testset = torchvision.datasets.CIFAR100(root='./data', train=False,
                                                    download=True, transform=transforms.ToTensor())
        else:
            testset = custom_dataset(dataset = args.dataset,
                                     data_path=args.aa_dataset_path,
                                     labels_path=args.aa_labels_path,
                                     transform=transforms.ToTensor())
    elif args.dataset == 'cifar10-c':
        dataset_files = ['brightness.npy', 'contrast.npy', 'defocus_blur.npy', 'elastic_transform.npy',
                         'fog.npy', 'frost.npy', 'gaussian_blur.npy', 'gaussian_noise.npy', 'glass_blur.npy',
                         'impulse_noise.npy', 'jpeg_compression.npy', 'motion_blur.npy', 'pixelate.npy',
                         'saturate.npy', 'shot_noise.npy', 'snow.npy', 'spatter.npy', 'speckle_noise.npy',
                         'zoom_blur.npy']

        labels = torch.from_numpy(np.load(os.path.join('data', 'CIFAR-10-C', 'labels.npy')))

        tensor_labels = None
        tensor_data = None

        for dataset_idx in range(len(dataset_files)):
            imgs = torch.from_numpy(np.load(os.path.join('data', 'CIFAR-10-C', dataset_files[dataset_idx])))
            imgs = imgs / 255.
            imgs = imgs.permute(0, 3, 1, 2)
            if tensor_labels is not None:
                tensor_labels = torch.concat((tensor_labels, labels), dim=0)
                tensor_data = torch.concat((tensor_data, imgs), dim=0)
            else:
                tensor_labels = labels
                tensor_data = imgs

        testset = torch.utils.data.TensorDataset(tensor_data, tensor_labels)

    elif args.dataset == 'imagenet':
        if args.aa_dataset == False:
            test_trans = torchvision.transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor()]
            )
            testset = torchvision.datasets.ImageNet(root='./data/imagenet', split='val', transform=test_trans)
        else:
            testset = custom_dataset(dataset = args.dataset,
                                     data_path=args.aa_dataset_path,
                                     labels_path=args.aa_labels_path,
                                     transform=transforms.ToTensor())

    else:
        raise Exception('Dont find dataset')

    if args.plot or (args.quick_review and args.dataset == 'imagenet'):
        testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    else:
        testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    return testloader




class custom_dataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, dataset, data_path, labels_path, transform=None):
        self.data_path = data_path
        self.labels_path = labels_path
        self.transform = transform
        self.data_aa = torch.load(self.data_path)
        self.lables = torch.load(self.labels_path)
        if dataset == 'cifar10':
            self.data = torchvision.datasets.CIFAR10(root='./data', train=False,
                                                     download=True, transform=transforms.ToTensor())
        elif dataset == 'cifar100':
            self.data = torchvision.datasets.CIFAR100(root='./data', train=False,
                                                    download=True, transform=transforms.ToTensor())
        elif dataset == 'imagenet':
            test_trans = torchvision.transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor()]
            )
            self.data = torchvision.datasets.ImageNet(root='./data/imagenet', split='val', transform=test_trans)
        else:
            raise Exception('this dataset name does not exists')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample_aa = self.data_aa[idx]
        clear_sample = self.data[idx][0]
        y = self.lables[idx]
        '''
        plt.figure()
        plt.imshow(sample_aa.permute(1, 2, 0).detach().cpu())
        plt.show()
        
        plt.figure()
        plt.imshow(clear_sample.permute(1, 2, 0).detach().cpu())
        plt.show()

        '''

        if self.transform and isinstance(sample_aa, torch.Tensor) == False:
            sample_aa = self.transform(sample_aa)

        if self.transform and isinstance(clear_sample, torch.Tensor) == False:
            clear_sample = self.transform(clear_sample)

        return sample_aa, y, clear_sample
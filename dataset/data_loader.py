""" CIFAR-10 CIFAR-100, Tiny-ImageNet data loader """

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler

from dataset.ClassAwareSampler import get_sampler
from dataset.ImbalanceCIFAR import IMBALANCECIFAR10, IMBALANCECIFAR100


def fetch_dataloader(types, _params):
    """
    Fetch and return train/dev dataloader with hyperparameters (params.subset_percent = 1.)
    """

    # using random crops and horizontal flip for train set
    if _params.augmentation == "yes":
        train_transformer = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),  # randomly flip image horizontally
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343),
                                 (0.2673342858792401, 0.2564384629170883, 0.27615047132568404))])
        # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.240, 0.243, 0.261))

    # data augmentation can be turned off
    else:
        train_transformer = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343),
                                 (0.2673342858792401, 0.2564384629170883, 0.27615047132568404))])

    # transformer for dev set
    dev_transformer = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343),
                             (0.2673342858792401, 0.2564384629170883, 0.27615047132568404))])

    trainset = devset = None  # Predefine
    if _params.dataset == 'cifar10':
        trainset = torchvision.datasets.CIFAR10(root='~/Data/cifar10', train=True,
                                                download=False, transform=train_transformer)
        devset = torchvision.datasets.CIFAR10(root='~/Data/cifar10', train=False,
                                              download=False, transform=dev_transformer)

    elif _params.dataset == 'cifar100':
        trainset = torchvision.datasets.CIFAR100(root='~/Data/cifar100', train=True,
                                                 download=False, transform=train_transformer)
        devset = torchvision.datasets.CIFAR100(root='~/Data/cifar100', train=False,
                                               download=False, transform=dev_transformer)

    elif _params.dataset == 'tiny_imagenet':
        data_dir = '~/Data/tiny-imagenet-200/'
        data_transforms = {
            'train': transforms.Compose([
                transforms.RandomRotation(20),
                transforms.RandomHorizontalFlip(0.5),
                transforms.ToTensor(),
                transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]),
            ]),
            'val': transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]),
            ])
        }
        train_dir = data_dir + 'train/'
        test_dir = data_dir + 'val/images/'
        trainset = torchvision.datasets.ImageFolder(train_dir, data_transforms['train'])
        devset = torchvision.datasets.ImageFolder(test_dir, data_transforms['val'])

    elif _params.dataset == 'imbalance_cifar100':
        # print('Fetching Imbalance Cifar100...')
        trainset = IMBALANCECIFAR100(types, imbalance_ratio=_params.cifar_imb_ratio,
                                     root='~/Data/cifar100')

        devset = IMBALANCECIFAR100(types, imbalance_ratio=_params.cifar_imb_ratio,
                                   root='~/Data/cifar100')

    elif _params.dataset == 'imbalance_cifar10':
        # print('Fetching Imbalance Cifar10...')
        trainset = IMBALANCECIFAR10(types, imbalance_ratio=_params.cifar_imb_ratio,
                                    root='~/Data/cifar10')

        devset = IMBALANCECIFAR10(types, imbalance_ratio=_params.cifar_imb_ratio,
                                  root='~/Data/cifar10')

    if _params.resample == "yes":
        sampler = get_sampler()
        trainloader = torch.utils.data.DataLoader(trainset, _params.batch_size, shuffle=False,
                                                  sampler=sampler(trainset, 4), num_workers=_params.num_workers)

        testloader = torch.utils.data.DataLoader(devset, _params.batch_size, shuffle=False,
                                                 num_workers=_params.num_workers)
    else:
        trainloader = torch.utils.data.DataLoader(trainset, _params.batch_size, shuffle=True,
                                                  num_workers=_params.num_workers)

        testloader = torch.utils.data.DataLoader(devset, _params.batch_size, shuffle=False,
                                                 num_workers=_params.num_workers)

    if types == 'train':
        _dataloader = trainloader
    else:
        _dataloader = testloader

    return _dataloader


def fetch_subset_dataloader(types, _params):
    """
    Use only a subset of dataset for KD training, depending on params.subset_percent
    """

    # using random crops and horizontal flip for train set
    if _params.augmentation == "yes":
        train_transformer = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),  # randomly flip image horizontally
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])

    # data augmentation can be turned off
    else:
        train_transformer = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])

    # transformer for dev set
    dev_transformer = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])

    # Predefine
    trainset = None
    devset = None
    if _params.dataset == 'cifar10':
        trainset = torchvision.datasets.CIFAR10(root='~/Data/cifar10', train=True,
                                                download=False, transform=train_transformer)
        devset = torchvision.datasets.CIFAR10(root='~/Data/cifar10', train=False,
                                              download=False, transform=dev_transformer)
    elif _params.dataset == 'cifar100':
        trainset = torchvision.datasets.CIFAR10(root='~/Data/cifar10', train=True,
                                                download=False, transform=train_transformer)
        devset = torchvision.datasets.CIFAR10(root='~/Data/cifar10', train=False,
                                              download=False, transform=dev_transformer)
    elif _params.dataset == 'tiny_imagenet':
        data_dir = '~/Data/tiny-imagenet-200/'
        data_transforms = {
            'train': transforms.Compose([
                transforms.RandomRotation(20),
                transforms.RandomHorizontalFlip(0.5),
                transforms.ToTensor(),
                transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]),
            ]),
            'val': transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]),
            ])
        }
        train_dir = data_dir + 'train/'
        test_dir = data_dir + 'val/images/'
        trainset = torchvision.datasets.ImageFolder(train_dir, data_transforms['train'])
        devset = torchvision.datasets.ImageFolder(test_dir, data_transforms['val'])

    trainset_size = len(trainset)
    indices = list(range(trainset_size))
    split = int(np.floor(_params.subset_percent * trainset_size))
    np.random.seed(230)
    np.random.shuffle(indices)

    train_sampler = SubsetRandomSampler(indices[:split])

    trainloader = torch.utils.data.DataLoader(trainset, _params.batch_size, sampler=train_sampler,
                                              num_workers=_params.num_workers, pin_memory=_params.cuda)

    devloader = torch.utils.data.DataLoader(devset, _params.batch_size, shuffle=False,
                                            num_workers=_params.num_workers, pin_memory=_params.cuda)

    if types == 'train':
        _dataloader = trainloader
    else:
        _dataloader = devloader

    return _dataloader

# if __name__ == '__main__':
#     json_path = os.path.join('experiments/imbalance_experiments/resample_resnet18/', 'params.json')
#     import utils
#
#     params = utils.Params(json_path)
#     train_dtloader = fetch_dataloader('train', params)
#     labels = []
#     for (data, label) in train_dtloader:
#         labels.append(label)
#     labels = torch.cat(labels)
#     print(labels.shape)
#     print(torch.unique(labels, return_counts=True))
# for i in range(100):
#     print((labels==i))

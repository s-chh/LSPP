import torch
from torchvision import transforms
import os
import torch.utils.data as data
from torchvision import datasets
import numpy as np

def data_loaders(args):
    train_transform = []

    if args.randomresizecrop:
        train_transform.append(transforms.RandomResizedCrop(args.image_size))
    elif args.padding > 0:
        train_transform.append(transforms.Pad(args.padding, padding_mode='reflect'))
        train_transform.append(transforms.RandomCrop([args.image_size, args.image_size]))
    elif args.resizecrop > 0:
        train_transform.append(transforms.Resize(args.resizecrop))
        train_transform.append(transforms.RandomCrop(args.image_size))
    else:
        train_transform.append(transforms.Resize([args.image_size, args.image_size]))

    if args.n_channels==1:
        train_transform.append(transforms.Grayscale(1))

    if args.hflip:
        train_transform.append(transforms.RandomHorizontalFlip())

    train_transform.append(transforms.ToTensor())
    train_transform.append(transforms.Normalize(args.mean, args.std))

    train_transform = transforms.Compose(train_transform)

    test_transform = []
    if args.image_size == 224:
        test_transform.append(transforms.Resize(256))
        test_transform.append(transforms.CenterCrop(224))
    else:
        test_transform.append(transforms.Resize([args.image_size, args.image_size]))

    if args.n_channels==1:
        test_transform.append(transforms.Grayscale(1))

    test_transform.append(transforms.ToTensor())
    test_transform.append(transforms.Normalize(args.mean, args.std))
    test_transform = transforms.Compose(test_transform)

    if args.dataset == 'cifar10':
        train = datasets.CIFAR10(root=args.data_path, train=True, transform=train_transform, download=True)
        test = datasets.CIFAR10(root=args.data_path, train=False, transform=test_transform, download=True)
    elif args.dataset == 'cifar100':
        train = datasets.CIFAR100(root=args.data_path, train=True, transform=train_transform, download=True)
        test = datasets.CIFAR100(root=args.data_path, train=False, transform=test_transform, download=True)
    elif args.dataset == 'fashionmnist':
        train = datasets.FashionMNIST(root=args.data_path, train=True, transform=train_transform, download=True)
        test = datasets.FashionMNIST(root=args.data_path, train=False, transform=test_transform, download=True)
    elif args.dataset == 'svhn':
        train = datasets.SVHN(root=args.data_path, split='train', transform=train_transform, download=True)
        test = datasets.SVHN(root=args.data_path, split='test', transform=test_transform, download=True)
    elif args.dataset == 'flowers102':
        train = datasets.Flowers102(root=args.data_path, split='train', transform=train_transform, download=True)
        test = datasets.Flowers102(root=args.data_path, split='test', transform=test_transform, download=True)
        train._labels = np.array(train._labels)
        train._labels = train._labels - train._labels.min()
        test._labels = np.array(test._labels)
        test._labels = test._labels - test._labels.min()
    else:
        train = datasets.ImageFolder(root=os.path.join(args.data_path, 'train'), transform=train_transform)
        test = datasets.ImageFolder(root=os.path.join(args.data_path, 'test'), transform=test_transform)

    train_loader = torch.utils.data.DataLoader(dataset=train,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               num_workers=args.n_workers,
                                               drop_last=True)

    test_loader = torch.utils.data.DataLoader(dataset=test,
                                              batch_size=args.batch_size,
                                              shuffle=True,
                                              num_workers=args.n_workers,
                                              drop_last=False)

    return train_loader, test_loader

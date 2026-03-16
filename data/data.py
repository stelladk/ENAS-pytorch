import sys
import os
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision.datasets import CIFAR10
from torchvision import transforms

# Add experimental_grow to path for tools imports
_TOOLS_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', '..', 'InriaGitlab', 'experimental_grow')
)
if _TOOLS_PATH not in sys.path:
    sys.path.insert(0, _TOOLS_PATH)

from tools.augmentations import default_augmentations, get_transforms, npy_datasets
from tools.datasets import (
    AddNIST, MultNIST, CIFARTile, LanguageASPELL,
    Gutenberg, GeoClassing, Chesseract, GameOfLife,
)

_DATASET_REGISTRY = {
    'addnist': AddNIST,
    'multnist': MultNIST,
    'cifartile': CIFARTile,
    'language': LanguageASPELL,
    'gutenberg': Gutenberg,
    'geoclassing': GeoClassing,
    'chesseract': Chesseract,
    'gameoflife': GameOfLife,
}


def _build_transform(dataset_name, augment=True):
    aug_list = default_augmentations.get(dataset_name, []) if augment else None
    base, aug = get_transforms(dataset_name, aug_list)
    if dataset_name in npy_datasets:
        # npy: augment on tensors, after base
        return transforms.Compose(base + aug)
    else:
        # PIL: augment before ToTensor
        return transforms.Compose(aug + base)


def get_loaders(args):
    dataset_name = getattr(args, 'dataset', 'cifar10').lower()

    if dataset_name == 'cifar10':
        return _get_cifar10_loaders(args)

    if dataset_name not in _DATASET_REGISTRY:
        raise ValueError(
            f"Unknown dataset '{dataset_name}'. "
            f"Supported: cifar10, {', '.join(sorted(_DATASET_REGISTRY))}"
        )

    return _get_custom_loaders(args, dataset_name)


def _get_cifar10_loaders(args):
    augment = not getattr(args, 'no_augment', False)
    train_transform = _build_transform('cifar10', augment=augment)
    valid_transform = _build_transform('cifar10', augment=False)

    train_dataset = CIFAR10(
        root=args.data, train=True, download=True, transform=train_transform,
    )
    valid_dataset = CIFAR10(
        root=args.data, train=False, download=False, transform=valid_transform,
    )

    indices = list(range(len(train_dataset)))

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=SubsetRandomSampler(indices[:-5000]),
        pin_memory=True,
        num_workers=2,
    )
    reward_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=SubsetRandomSampler(indices[-5000:]),
        pin_memory=True,
        num_workers=2,
    )
    test_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=2,
    )

    return train_loader, RepeatedDataLoader(reward_loader), reward_loader, tesist_loader


def _get_custom_loaders(args, dataset_name):
    dataset_cls = _DATASET_REGISTRY[dataset_name]

    augment = not getattr(args, 'no_augment', False)
    train_transform = _build_transform(dataset_name, augment=augment)
    valid_transform = _build_transform(dataset_name, augment=False)

    train_dataset = dataset_cls(
        train=True, root=args.data, download=True, transform=train_transform,
    )
    valid_dataset = dataset_cls(
        train=False, root=args.data, download=True, transform=valid_transform,
    )

    indices = list(range(len(train_dataset)))
    reward_split = min(5000, max(1, len(indices) // 10))

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=SubsetRandomSampler(indices[:-reward_split]),
        pin_memory=True,
        num_workers=2,
    )
    reward_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=SubsetRandomSampler(indices[-reward_split:]),
        pin_memory=True,
        num_workers=2,
    )
    test_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=2,
    )

    return train_loader, RepeatedDataLoader(reward_loader), reward_loader, test_loader


class RepeatedDataLoader():
    def __init__(self, data_loader):
        self.data_loader = data_loader
        self.data_iter = self.data_loader.__iter__()

    def __len__(self):
        return len(self.data_loader)

    def next_batch(self):
        try:
            batch = self.data_iter.__next__()
        except StopIteration:
            self.data_iter = self.data_loader.__iter__()
            batch = self.data_iter.__next__()
        return batch

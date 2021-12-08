import torch
from torchvision import datasets, transforms
from six.moves import urllib
from omegaconf import DictConfig, OmegaConf

# prepare tools for downloading
opener = urllib.request.build_opener()
opener.addheaders = [('User-agent', 'Mozilla/5.0')]
urllib.request.install_opener(opener)

# select cuda if it is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def mnist_transform():
    """Get mnist data to tensor format and normalize"""
    return transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])

def get_mnist_dataset(partition):
    """Download MNIST dataset (if not already locally available)"""

    if partition not in ['train', 'test']:
        raise ValueError('argument must be one of either train or test')


    return datasets.MNIST(
        root='../data', 
        train= True if partition == 'train' else False, 
        download=True, 
        transform=mnist_transform()
    )

def get_mnist_dataloader(cfg: DictConfig, partition, shuffle=True):
    """Get dataloader of train or test set"""
    return torch.utils.data.DataLoader(
        get_mnist_dataset(partition),
        batch_size=cfg.data.batch_size_train if partition == 'train' else cfg.data.batch_size_test,
        shuffle=shuffle,
        num_workers=4
        )

def cifar10_transform():
    return transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

def get_cifar10_dataset(partition):
    """Download CIFAR10 dataset (if not already locally available)"""

    if partition not in ['train', 'test']:
        raise ValueError('argument must be one of either train or test')


    return datasets.CIFAR10(
        root='../data', 
        train= True if partition == 'train' else False, 
        download=True, 
        transform=cifar10_transform()
    )

def get_cifar10_dataloader(cfg: DictConfig, partition, shuffle=True):
    """Get dataloader of train or test set"""
    return torch.utils.data.DataLoader(
        get_cifar10_dataset(partition),
        batch_size=cfg.data.batch_size_train if partition == 'train' else cfg.data.batch_size_test,
        shuffle=shuffle,
        num_workers=4
        )
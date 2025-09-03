import torch
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor


def get_dataloaders(batch_size: int = 256, root: str = ".", num_workers: int | None = None):
    if num_workers is None:
        # Reasonable default; 0 on mac by default to avoid multiprocessing issues without fork
        num_workers = 0

    train_ds = MNIST(root=root, download=True, train=True, transform=ToTensor())
    test_ds  = MNIST(root=root, download=True, train=False, transform=ToTensor())

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=False)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=False)
    return train_loader, test_loader

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

def get_dataloaders(batch_size: int = 128, root: str = '.', download: bool = True):
    """Return train and test dataloaders for MNIST."""
    train_ds = MNIST(root=root, train=True, download=download, transform=ToTensor())
    test_ds = MNIST(root=root, train=False, download=download, transform=ToTensor())
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

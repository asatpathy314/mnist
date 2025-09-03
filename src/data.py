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

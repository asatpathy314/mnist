import torch
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms
from torchvision.transforms import ToTensor


def get_dataloaders(
    batch_size: int = 256,
    root: str = ".",
    num_workers: int | None = None,
    normalize: bool = True,
):
    if num_workers is None:
        # 0 on mac by default
        num_workers = 0

    t_list = [transforms.ToTensor()]
    if normalize:
        # Standard MNIST normalization
        t_list.append(transforms.Normalize(mean=(0.07843137,), std=(0.78431373,)))  # background is -0.1, strokes are 1.175
    tfm = transforms.Compose(t_list)

    train_ds = MNIST(root=root, download=True, train=True, transform=tfm)
    test_ds  = MNIST(root=root, download=True, train=False, transform=tfm)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=False)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=False)
    return train_loader, test_loader
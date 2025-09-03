from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from typing import Callable, Optional, Tuple, Dict, Any


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    device: str | torch.device,
    preprocess: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
) -> Tuple[float, float]:
    """
    Evaluate model on dataloader.

    Returns (avg_loss, accuracy_percent)
    """
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in dataloader:
        if preprocess is not None:
            images = preprocess(images)
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        loss = loss_fn(logits, labels)

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        _, preds = torch.max(logits, dim=1)
        total_correct += (preds == labels).sum().item()
        total_samples += batch_size

    avg_loss = total_loss / max(total_samples, 1)
    accuracy = 100.0 * total_correct / max(total_samples, 1)
    return avg_loss, accuracy


def compute_training_loss(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    device: str | torch.device,
    preprocess: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
) -> float:
    """Compute average loss over training set without gradient."""
    avg_loss, _ = evaluate(model, dataloader, loss_fn, device, preprocess)
    return avg_loss


def train(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str | torch.device,
    epochs: int = 1,
    preprocess: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
) -> list[float]:
    loss_history: list[float] = []
    model.train()

    for _ in range(epochs):
        total_loss = 0.0
        total_samples = 0

        for images, labels in dataloader:
            if preprocess is not None:
                images = preprocess(images)

            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

        epoch_loss = total_loss / max(total_samples, 1)
        loss_history.append(epoch_loss)

    return loss_history

import torch
from tqdm import tqdm
from typing import Callable, Optional


def train(model: torch.nn.Module,
          dataloader: torch.utils.data.DataLoader,
          loss_fn: torch.nn.Module,
          optimizer: torch.optim.Optimizer,
          device: torch.device,
          epochs: int = 1,
          preprocess: Optional[Callable[[torch.Tensor], torch.Tensor]] = None):
    """Generic training loop returning average loss per epoch."""
    history = []
    model.train()
    for _ in tqdm(range(epochs)):
        running_loss = 0.0
        for imgs, labels in dataloader:
            imgs, labels = imgs.to(device), labels.to(device)
            if preprocess:
                imgs = preprocess(imgs)
            logits = model(imgs)
            loss = loss_fn(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        history.append(running_loss / len(dataloader))
    return history


def evaluate(model: torch.nn.Module,
             dataloader: torch.utils.data.DataLoader,
             loss_fn: torch.nn.Module,
             device: torch.device,
             preprocess: Optional[Callable[[torch.Tensor], torch.Tensor]] = None):
    """Evaluate the model returning average loss and accuracy."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for imgs, labels in dataloader:
            imgs, labels = imgs.to(device), labels.to(device)
            if preprocess:
                imgs = preprocess(imgs)
            logits = model(imgs)
            loss = loss_fn(logits, labels)
            total_loss += loss.item()
            _, preds = torch.max(logits.data, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
    avg_loss = total_loss / len(dataloader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy

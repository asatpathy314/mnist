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

import argparse
import torch
import torch.nn.functional as F
import torch.nn as nn

from models import LeNet5, lenet5_init_
from data import get_dataloaders
from train import train, evaluate
from levenberg_marquadt_optim import DiagLM


def main():
    parser = argparse.ArgumentParser(description="Train/Eval harness for LeNet5 on MNIST")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--device", type=str, default="auto", help="cpu|mps|cuda|auto")
    parser.add_argument("--root", type=str, default=".")
    args = parser.parse_args()

    if args.device == "auto":
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
    else:
        device = args.device

    train_loader, test_loader = get_dataloaders(batch_size=args.batch_size, root=args.root)

    model = LeNet5().to(device)
    model.apply(lenet5_init_)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = DiagLM(model.parameters(), lr=args.lr)

    preprocess = lambda x: F.pad(x, (2, 2, 2, 2))

    print(f"Training on device: {device}")
    loss_history = train(model, train_loader, loss_fn, optimizer, device, epochs=args.epochs, preprocess=preprocess)
    print("Training losses per epoch:", [f"{l:.4f}" for l in loss_history])

    test_loss, test_acc = evaluate(model, test_loader, loss_fn, device, preprocess=preprocess)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.2f}%")


if __name__ == "__main__":
    main()



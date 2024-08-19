import torch
import torch.nn as nn
from torch.optim import Optimizer
from argparse import Namespace


def save_checkpoint(path: str, args: Namespace, epoch: int, model: nn.Module, optimizer: Optimizer, metrics: dict):
    checkpoint = {
        "epoch": epoch,
        "args": args,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "metrics": metrics,
    }
    torch.save(checkpoint, path)

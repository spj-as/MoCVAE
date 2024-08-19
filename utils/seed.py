import torch
import numpy as np
import random


def set_seed(seed: int = 42) -> torch.Generator:
    """
    Set seed for reproducibility and return a torch.Generator object.
    """
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    g = torch.Generator()
    g.manual_seed(seed)
    return g

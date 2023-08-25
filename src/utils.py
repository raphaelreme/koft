"""Some utils functions used mostly during experiments"""

import random

import numpy as np
import torch
import torch.backends.cudnn


def enforce_all_seeds(seed: int, strict=True):
    """Enforce all the seeds

    If strict you may have to define the following env variable:
        CUBLAS_WORKSPACE_CONFIG=:4096:8  (Increase a bit the memory foot print ~25Mo)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if strict:
        torch.backends.cudnn.benchmark = False  # By default should already be to False
        torch.use_deterministic_algorithms(True)


def create_seed_worker(seed: int, strict=True):
    """Create a callable that will seed the workers

    If used with a train data loader with random data augmentation, one should probably
    set the `persistent_workers` argument. (So that the random augmentations differs between epochs)
    """

    def seed_worker(worker_id):
        enforce_all_seeds(seed + worker_id, strict)

    return seed_worker

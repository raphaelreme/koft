"""Some utils functions used mostly during experiments"""

from __future__ import annotations

import os
import random
import subprocess
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.backends.cudnn

if TYPE_CHECKING:
    from collections.abc import Callable


def enforce_all_seeds(seed: int, strict=True) -> None:
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


def create_seed_worker(seed: int, strict=True) -> Callable:
    """Create a callable that will seed the workers

    If used with a train data loader with random data augmentation, one should probably
    set the `persistent_workers` argument. (So that the random augmentations differs between epochs)
    """

    def seed_worker(worker_id):
        enforce_all_seeds(seed + worker_id, strict)

    return seed_worker


def kill_java_in_our_pgrp_pkill():
    """Ugly kill Icy if timeout and still active"""
    pgid = os.getpgrp()
    # Kill by process group and name "java"
    subprocess.run(["pkill", "-TERM", "-g", str(pgid), "java"], check=False)  # noqa: S603, S607
    subprocess.run(["pkill", "-KILL", "-g", str(pgid), "java"], check=False)  # noqa: S603, S607

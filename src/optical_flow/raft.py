import argparse
import os
import sys

import numpy as np
import torch

CWD = os.environ.get("EXPYRUN_CWD", ".")

sys.path.append(f"{CWD}/RAFT/core/")
from raft import RAFT  # type: ignore


class Raft:
    """Wrapper around raft model to do cv like optical flow

    CPU only but could be easily update to work on cuda
    """

    def __init__(self, n_iters=32) -> None:
        self.n_iters = n_iters

        assert os.path.exists(
            f"{CWD}/RAFT/models/raft-small.pth"
        ), "To use RAFT optical flow you must first download models"
        parser = argparse.ArgumentParser()
        parser.add_argument("--model", help="restore checkpoint")
        parser.add_argument("--small", action="store_true", help="use small model")
        parser.add_argument("--mixed_precision", action="store_true", help="use mixed precision")
        parser.add_argument("--alternate_corr", action="store_true", help="use efficent correlation implementation")
        args = parser.parse_args(args=["--model", f"{CWD}/RAFT/models/raft-small.pth", "--small"])

        dp_model = torch.nn.DataParallel(RAFT(args))
        dp_model.load_state_dict(torch.load(args.model, map_location="cpu"))

        self.model = dp_model.module.cpu().eval()

    def __call__(self, source: np.ndarray, destination: np.ndarray) -> np.ndarray:
        source_t = torch.tensor(source).to(torch.float32)[None]
        destination_t = torch.tensor(destination).to(torch.float32)[None]
        source_t = torch.cat((source_t, source_t, source_t))[None]
        destination_t = torch.cat((destination_t, destination_t, destination_t))[None]
        with torch.no_grad():
            _, flow_up = self.model(source_t, destination_t, iters=self.n_iters, test_mode=True)
        return flow_up[0].permute(1, 2, 0).detach().cpu().numpy()

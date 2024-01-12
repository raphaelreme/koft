import os
import sys

import numpy as np
import torch

CWD = os.environ.get("EXPYRUN_CWD", ".")
sys.path.append(f"{CWD}/voxelmorph/")
os.environ["NEURITE_BACKEND"] = "pytorch"
os.environ["VXM_BACKEND"] = "pytorch"

import voxelmorph as vxm  # type: ignore


class Vxm:
    """Wrapper around vxm model to do cv like optical flow

    CPU only but could be easily update to work on cuda

    Note: The training has been done for a particular size of images. You may adjust in_shape and channels parameters.
    """

    def __init__(self, model_path=f"{CWD}/vxm.pt", in_shape=(256, 256), channels=1) -> None:
        assert os.path.exists(model_path), "To use vxm optical flow you must first train and save a model"

        self.model = vxm.networks.VxmDense(in_shape, src_feats=channels, trg_feats=channels, int_steps=0)
        self.model.flow.bias = None
        self.model.load_state_dict(torch.load(model_path, map_location="cpu"))
        self.model = self.model.cpu().eval()

    def __call__(self, source: np.ndarray, destination: np.ndarray) -> np.ndarray:
        source_t = torch.tensor(source).to(torch.float32)[None, None] / 255
        destination_t = torch.tensor(destination).to(torch.float32)[None, None] / 255
        with torch.no_grad():
            _, flow = self.model(source_t, destination_t, registration=True)
        return -flow[0].permute(1, 2, 0).cpu().numpy()[..., ::-1]

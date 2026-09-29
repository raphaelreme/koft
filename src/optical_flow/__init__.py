import byotrack.api.optical_flow.optical_flow
import cv2
import numpy as np
from byotrack.implementation.optical_flow.opencv import OpenCVOpticalFlow

from .optical_flow import OptFlow, show_flow_on_video  # noqa: F401

# Create some default optical flows [OLD, but still used in experiments/optical_flow.py]

_cv2_tvl1 = cv2.optflow.DualTVL1OpticalFlow_create(lambda_=0.05)  # type: ignore[attr-defined]
_cv2_farneback = cv2.FarnebackOpticalFlow_create(winSize=20)  # type: ignore[attr-defined]

tvl1 = OptFlow(lambda x, y: _cv2_tvl1.calc(x, y, None), (0.0, 1.0), 4, 1.0)
farneback = OptFlow(lambda x, y: _cv2_farneback.calc(x, y, None), (0.0, 1.0), 4, 1.0)
no_optical_flow = OptFlow(lambda x, _: np.zeros((*x.shape, 2)), scale=4)

# Raft can be created from raft sub module following
# raft = OptFlow(Raft())

# Vxm can be created from vxm submodule (Required a trained model)
# vxm = OptFlow(Vxm())


# ByoTrack compatible optical flows:
bt_tvl1 = OpenCVOpticalFlow(_cv2_tvl1, downscale=4)
bt_farneback = OpenCVOpticalFlow(_cv2_farneback, downscale=4)
bt_no_flow = byotrack.api.optical_flow.optical_flow.DummyOpticalFlow(4)

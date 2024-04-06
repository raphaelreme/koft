import cv2
import numpy as np

from .optical_flow import OptFlow, show_flow_on_video


# Create some default optical flows

_cv2_tvl1 = cv2.optflow.DualTVL1OpticalFlow_create(lambda_=0.05)  # type: ignore
_cv2_farneback = cv2.FarnebackOpticalFlow_create(winSize=20)  # type: ignore

tvl1 = OptFlow(lambda x, y: _cv2_tvl1.calc(x, y, None), (0.0, 1.0), 4, 1.0)
farneback = OptFlow(lambda x, y: _cv2_farneback.calc(x, y, None), (0.0, 1.0), 4, 1.0)
no_optical_flow = OptFlow(lambda x, y: np.zeros((*x.shape, 2)), scale=4)

# Raft can be created from raft sub module following
# raft = OptFlow(Raft())

# Vxm can be created from vxm submodule (Required a trained model)
# vxm = OptFlow(Vxm())

"""Measure optical flow performances"""

import dataclasses
import enum
import pathlib
from typing import Dict, List
import warnings

import cv2
import dacite
import numpy as np
import torch
import tqdm  # type: ignore
import yaml  # type: ignore

import byotrack
from byotrack.implementation.refiner.stitching import emc2

from .. import optical_flow
from ..data import dupre, simulation, stitching
from ..metrics import stitching as stitching_metrics
from ..optical_flow import raft, vxm, propagate
from ..utils import enforce_all_seeds


class OpticalFlow(enum.Enum):
    FARNEBACK = "farneback"
    TVL1 = "tvl1"
    RAFT = "raft"
    NONE = "none"
    VXM = "vxm"


@dataclasses.dataclass
class ExperimentConfig:
    seed: int
    run_dupre: bool
    run_stitching: bool
    run_simulation: bool
    of_simulation: pathlib.Path
    springs_simulation: pathlib.Path
    dupre_data: dupre.DupreDataConfig
    stitching_data: stitching.StitchingDataConfig
    flow: OpticalFlow
    scale: int
    blur: float

    def build_of(self) -> optical_flow.OptFlow:
        """Build an optical flow"""
        if self.flow is OpticalFlow.NONE:
            return optical_flow.OptFlow(lambda x, y: np.zeros((*x.shape, 2), dtype=np.float32), scale=8, blur=0.0)

        if self.flow is OpticalFlow.FARNEBACK:
            cv2_farneback = cv2.FarnebackOpticalFlow_create(winSize=20)  # type: ignore
            return optical_flow.OptFlow(lambda x, y: cv2_farneback.calc(x, y, None), scale=self.scale, blur=self.blur)

        if self.flow is OpticalFlow.TVL1:
            cv2_tvl1 = cv2.optflow.DualTVL1OpticalFlow_create(lambda_=0.05)  # type: ignore
            return optical_flow.OptFlow(lambda x, y: cv2_tvl1.calc(x, y, None), scale=self.scale, blur=self.blur)

        if self.flow is OpticalFlow.RAFT:
            return optical_flow.OptFlow(raft.Raft(), scale=self.scale, blur=self.blur)

        # VXM
        if self.scale != 4.0 or self.blur != 1.0:
            warnings.warn("Except you have retrained VXM model, it has been trained with scale=4 and blur=1.0")
        return optical_flow.OptFlow(vxm.Vxm(), scale=self.scale, blur=self.blur)


def frame2frame_dist(
    video: byotrack.Video, tracks: List[byotrack.Track], optflow: optical_flow.OptFlow
) -> torch.Tensor:
    """Build frame to frame distance matrix"""
    tracks_matrix = byotrack.Track.tensorize(tracks)
    frame = video[0][..., 0]
    src = optflow.prepare(frame)
    dist = np.zeros((tracks_matrix.shape[0] - 1, tracks_matrix.shape[1]))
    points = tracks_matrix[0].numpy()  # Initial points for tracks

    for new_frame_id in tqdm.trange(1, tracks_matrix.shape[0]):
        new_frame = video[new_frame_id][..., 0]
        dest = optflow.prepare(new_frame)

        flow = optflow.calc(src, dest)

        next_points = optflow.transform(flow, points)
        dist[new_frame_id - 1] = np.linalg.norm(next_points - tracks_matrix[new_frame_id].numpy(), axis=-1)

        frame = new_frame
        src = dest

        # In frame to frame, points is the true pos
        # points != next_points  # For propag
        points = tracks_matrix[new_frame_id].numpy()

    return torch.tensor(dist).to(torch.float32)


def tracklet2tracklet_dist(video: byotrack.Video, tracklets: List[byotrack.Track], optflow) -> np.ndarray:
    """Compute the tracklet to tracklet positional distance after optical flow correction"""
    stitcher = emc2.EMC2Stitcher()  # Default stitcher

    # Propagation of tracklets
    directed = propagate.DirectedFlowPropagation(optflow)
    propagation_matrix = emc2.propagation.forward_backward_propagation(
        byotrack.Track.tensorize(tracklets), video, directed
    )

    skip_mask = stitcher.skip_computation(tracklets, stitcher.max_overlap, stitcher.max_dist, stitcher.max_gap)
    ranges = np.array([(track.start, track.start + len(track)) for track in tracklets])

    return emc2._fast_emc2_dist(  # pylint: disable=protected-access
        propagation_matrix.numpy(), skip_mask.numpy(), ranges
    )


def hard_thresh(tracks: List[byotrack.Track]) -> float:
    tracks_tensor = byotrack.Track.tensorize(tracks)

    # Compute frame-to-frame errors without flow
    default_errors = (tracks_tensor[1:] - tracks_tensor[:-1]).norm(dim=-1)

    # 0.99 quantile
    return torch.quantile(default_errors, 0.99).item()


def main(name: str, cfg_data: dict) -> None:
    print("Running:", name)
    print(yaml.dump(cfg_data))
    cfg = dacite.from_dict(ExperimentConfig, cfg_data, dacite.Config(cast=[pathlib.Path, tuple, enum.Enum]))
    enforce_all_seeds(cfg.seed)

    metrics: Dict[str, Dict[str, float]] = {}

    # Load of
    optflow = cfg.build_of()

    ##  Frame to frame metric
    f2f_dist: Dict[str, torch.Tensor] = {}

    if cfg.run_dupre:
        # Dupre
        video = cfg.dupre_data.open()
        tracks = cfg.dupre_data.cleaned_tracks()
        f2f_dist["dupre"] = frame2frame_dist(video, tracks, optflow)
        metrics["dupre"] = {
            "RMSE": f2f_dist["dupre"].pow(2).mean().sqrt().item(),
            "n-hard": (f2f_dist["dupre"] > hard_thresh(tracks)).sum().item() / f2f_dist["dupre"].numel() * 10**4,
        }
        print(f"Dupre: RMSE:{metrics['dupre']['RMSE']:.4f}, hard-links:{metrics['dupre']['n-hard']:.2f}")

    if cfg.run_simulation:
        # Simu - OF
        video = simulation.open_video(cfg.of_simulation)
        tracks = simulation.load_tracks(cfg.of_simulation)
        f2f_dist["of"] = frame2frame_dist(video, tracks, optflow)
        metrics["of"] = {
            "RMSE": f2f_dist["of"].pow(2).mean().sqrt().item(),
            "n-hard": (f2f_dist["of"] > hard_thresh(tracks)).sum().item() / f2f_dist["of"].numel() * 10**4,
        }
        print(f"OF: RMSE:{metrics['of']['RMSE']:.4f}, hard-links:{metrics['of']['n-hard']:.2f}")

        # Simu - springs
        video = simulation.open_video(cfg.springs_simulation)
        tracks = simulation.load_tracks(cfg.springs_simulation)
        f2f_dist["springs"] = frame2frame_dist(video, tracks, optflow)
        metrics["springs"] = {
            "RMSE": f2f_dist["springs"].pow(2).mean().sqrt().item(),
            "n-hard": (f2f_dist["springs"] > hard_thresh(tracks)).sum().item() / f2f_dist["springs"].numel() * 10**4,
        }
        print(f"Springs: RMSE:{metrics['springs']['RMSE']:.4f}, hard-links:{metrics['springs']['n-hard']:.2f}")

    if cfg.run_stitching:
        ## Stitching metric
        video = cfg.stitching_data.open()
        tracks = cfg.stitching_data.load_tracklets()
        links = cfg.stitching_data.load_links()

        stitching_ap = (
            stitching_metrics.compute_metrics(tracklet2tracklet_dist(video, tracks, optflow), links)[
                "average_precision"
            ]
            * 100
        )
        metrics["stitching"] = {"ap": stitching_ap}
        print(f"Stitching Average precision: {stitching_ap:.2f}")

    with open("metrics.yml", "w", encoding="utf-8") as file:
        file.write(yaml.dump(metrics))

    torch.save(f2f_dist, "f2f_dist.pt")

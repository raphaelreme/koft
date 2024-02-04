"""Measure optical flow performances vs SNR (Only runs simulation exp)"""

import enum
import pathlib
from typing import Dict

import dacite
import torch
import yaml  # type: ignore


from ..data import simulation
from ..utils import enforce_all_seeds
from .optical_flow import ExperimentConfig, frame2frame_dist, hard_thresh


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

    with open("metrics.yml", "w", encoding="utf-8") as file:
        file.write(yaml.dump(metrics))

    torch.save(f2f_dist, "f2f_dist.pt")

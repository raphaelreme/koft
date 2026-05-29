"""For loading tracking simulation data"""

from __future__ import annotations

from typing import TYPE_CHECKING

import byotrack
import torch

if TYPE_CHECKING:
    import pathlib


def open_video(simulation_path: pathlib.Path) -> byotrack.Video:
    """Open the video in the simulation folder"""
    video = byotrack.Video(simulation_path / "video.tiff")
    video.set_transform(byotrack.VideoTransformConfig(aggregate=True, normalize=True, q_min=0.00, q_max=1.0))
    return video


def load_ground_truth(simulation_path: pathlib.Path) -> dict[str, torch.Tensor]:
    """Load the ground truth in a dict format"""
    return torch.load(simulation_path / "video_data.pt")


def load_tracks(simulation_path: pathlib.Path) -> list[byotrack.Track]:
    """Load ground truth as tracks (Keep only positional data)"""

    ground_truth = load_ground_truth(simulation_path)

    tracks = []
    for i in range(ground_truth["mu"].shape[1]):
        tracks.append(byotrack.Track(0, ground_truth["mu"][:, i], i))  # noqa: PERF401

    return tracks

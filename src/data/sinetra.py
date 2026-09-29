"""For loading tracking simulation data"""

from __future__ import annotations

import dataclasses
import pathlib  # noqa: TC003

import byotrack
import torch


@dataclasses.dataclass
class SinetraDataConfig:
    simulation_path: pathlib.Path

    def open_video(self) -> byotrack.Video:
        """Open the video in the simulation folder"""
        return byotrack.Video(self.simulation_path / "video.tiff").normalize()

    def load_ground_truth(self) -> dict[str, torch.Tensor]:
        """Load the ground truth in a dict format"""
        return torch.load(self.simulation_path / "video_data.pt")

    def load_tracks(self) -> list[byotrack.Track]:
        """Load ground truth as tracks (Keep only positional data)"""

        ground_truth = self.load_ground_truth()

        tracks = []
        for i in range(ground_truth["mu"].shape[1]):
            tracks.append(byotrack.Track(0, ground_truth["mu"][:, i], i))  # noqa: PERF401

        return tracks

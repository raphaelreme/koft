from __future__ import annotations

import dataclasses
from typing import Literal

import byotrack
import torch

from src.data.dupre import DupreDataConfig  # noqa: TC001
from src.data.sinetra import SinetraDataConfig  # noqa: TC001
from src.data.trasein import TraseInDataConfig  # noqa: TC001


@dataclasses.dataclass
class TrackingDataConfig:
    dupre: DupreDataConfig
    sinetra: SinetraDataConfig
    trasein: TraseInDataConfig
    data_type: Literal["dupre", "sinetra", "trasein"] = "sinetra"

    def video(self) -> byotrack.Video:
        if self.data_type == "trasein":
            return self.trasein.open_video()

        if self.data_type == "dupre":
            video = self.dupre.open_video()

            mu = byotrack.Track.tensorize(self.tracks())
            return video[: len(mu)]  # For dupre, we need to crop the video

        return self.sinetra.open_video()

    def tracks(self) -> list[byotrack.Track]:
        if self.data_type == "trasein":
            raise RuntimeError("TraseIN has no ground-truth tracks.")

        if self.data_type == "dupre":
            return self.dupre.raw_tracks()  # Now, we use the raw tracks

        return self.sinetra.load_tracks()

    def ground_truth(self) -> dict[str, torch.Tensor]:
        if self.data_type == "trasein":
            raise RuntimeError("TraseIN has no ground-truth tracks.")

        if self.data_type == "sinetra":
            return self.sinetra.load_ground_truth()

        tracks = self.dupre.raw_tracks()
        mu = byotrack.Track.tensorize(tracks)
        return {"mu": mu, "weight": torch.ones_like(mu).mean(dim=-1)}

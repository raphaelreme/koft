"""For loading dupre's track / video"""


import dataclasses
import functools
import pathlib
from typing import List, Tuple

import numpy as np
import pandas as pd  # type: ignore
import torch

import byotrack

from .. import rts_smoothing


@dataclasses.dataclass
class DupreDataConfig:
    """Load Dupre tracks and clean them

    Smooth tracks with RTS to reduce the staircase behavior in the annotations (a pos update every few frames)
    Drop weird tracks (many are wrongly annotated)
    """

    video: pathlib.Path
    tracks: pathlib.Path
    measurement_noise: float = 2.5
    process_noise: float = 0.5
    speed_thresh: float = 1.0

    def open(self) -> byotrack.Video:
        """Load and transform the video"""
        video = byotrack.Video(self.video)
        video.set_transform(
            byotrack.VideoTransformConfig(aggregate=True, normalize=True, q_min=0.02, q_max=0.999, smooth_clip=0.1)
        )
        return video

    def raw_tracks(self) -> List[byotrack.Track]:
        tracks_data = np.array(pd.read_csv(self.tracks, header=None))

        identifiers = np.unique(tracks_data[:, 0])

        tracks = []

        for identifier in identifiers:
            track_data = tracks_data[tracks_data[:, 0] == identifier]
            assert len(track_data) == 201 and (track_data[:, 1] == np.arange(201)).all(), identifier
            points = torch.tensor(track_data[:, 2:]).to(torch.float32).flip(dims=(-1,))
            tracks.append(byotrack.Track(0, points, identifier))

        return tracks

    def _smoothing(self) -> Tuple[List[byotrack.Track], np.ndarray]:
        raw_tracks = self.raw_tracks()

        measured_positions = byotrack.Track.tensorize(raw_tracks).permute(1, 0, 2)
        smoothed_state = rts_smoothing.rts_smoothing(
            measured_positions.numpy(),
            functools.partial(rts_smoothing.create_cvkf, self.measurement_noise, self.process_noise),
        )

        smoothed_tracks = []
        for i, points in enumerate(smoothed_state):
            smoothed_tracks.append(byotrack.Track(0, torch.tensor(points, dtype=torch.float32)[:, :2], i))

        return smoothed_tracks, smoothed_state

    def smoothed_tracks(self) -> List[byotrack.Track]:
        return self._smoothing()[0]

    def cleaned_tracks(self) -> List[byotrack.Track]:
        smoothed_tracks, smoothed_state = self._smoothing()
        num_neighbors = 10

        diff = smoothed_state[None, ..., :2] - smoothed_state[:, None, ..., :2]
        dist = np.abs(diff).sum(axis=-1).mean(axis=-1)
        neighbors = np.argsort(dist, axis=1)[:, 1 : num_neighbors + 1]  # The N closest tracks
        neighbors_speed = smoothed_state[neighbors, ..., 2:].mean(axis=1)  # Avg speed in neighbors
        speed = smoothed_state[..., 2:]  # Speed of tracks
        speed_diff = np.linalg.norm(speed - neighbors_speed, axis=-1)

        invalid = (speed_diff > self.speed_thresh).any(axis=-1)
        print(f"Dropping {invalid.sum()} invalid tracks")

        valid_tracks = []
        for i in np.arange(len(smoothed_tracks))[~invalid]:
            valid_tracks.append(smoothed_tracks[i])

        return valid_tracks

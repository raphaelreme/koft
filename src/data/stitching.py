"""For loading Tracklet Stitching data from Reme 'Tracking Intermittent Particles with Self-Learned Visual Features'"""

from __future__ import annotations

import dataclasses
import json
import pathlib  # noqa: TC003

import byotrack
import byotrack.icy


@dataclasses.dataclass
class StitchingDataConfig:
    video: pathlib.Path
    tracklets: pathlib.Path
    links: pathlib.Path

    def open(self) -> byotrack.Video:
        """Load and transform the video"""
        video = byotrack.Video(self.video)
        video.set_transform(
            byotrack.VideoTransformConfig(aggregate=True, normalize=True, q_min=0.02, q_max=0.999, smooth_clip=0.1)
        )
        return video

    def load_tracklets(self) -> list[byotrack.Track]:
        if self.tracklets.suffix == ".xml":  # From icy
            return list(byotrack.icy.io.load_tracks(self.tracklets))
        raise ValueError("In this config we only support the tracks provided by Lagache (done with icy software)")

    def load_links(self) -> list[tuple[int, int]]:
        return [(int(key), value) for key, value in json.loads(self.links.read_text(encoding="utf-8")).items()]

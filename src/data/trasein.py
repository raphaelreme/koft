"""For loading Trase-IN videos"""

from __future__ import annotations

import dataclasses
import pathlib  # noqa: TC003

import byotrack


@dataclasses.dataclass
class TraseInDataConfig:
    """Load a Trase-IN video."""

    video: pathlib.Path

    def open_video(self) -> byotrack.Video:
        """Load and normalize the video"""
        return byotrack.Video(self.video)[..., :1].normalize()

from __future__ import annotations

from typing import TYPE_CHECKING

import byotrack
import numpy as np
import torch
import tqdm.auto as tqdm

if TYPE_CHECKING:
    from .optical_flow import OptFlow

# XXX Pretty ugly code here


def warp_detections(
    video, optflow: OptFlow, detections_sequence: list[byotrack.Detections]
) -> list[byotrack.PointDetections]:
    """Warp Detections onto the last frame using optical flow

    # quadratic complexity

    Warnings: Assume that the detections are sorted and that there is one Detections by video frame
    Will only warp the positions and drop the rest
    """
    src = optflow.prepare(video[0])
    positions: list[np.ndarray] = []
    shape = detections_sequence[0].shape
    for i, frame in enumerate(tqdm.tqdm(video[1:])):
        dst = optflow.prepare(frame)
        flow = optflow.calc(src, dst)
        positions.append(detections_sequence[i].position.clone().numpy())
        for position in positions:
            position[:] = optflow.transform(flow, position)

        src = dst

    positions.append(detections_sequence[-1].position.clone().numpy())

    # Has to round positions for a compatible SKT/u-track/eMHT unwarping
    return [
        byotrack.PointDetections(
            torch.tensor(position.clip(0.0, np.array(shape) - 1).round(), dtype=torch.float32),
            shape=shape,
            confidence=detections_sequence[i].confidence,
            labels=detections_sequence[i].confidence,
        )
        for i, position in enumerate(positions)
    ]


def warp_detections_linear(
    video, optflow: OptFlow, detections_sequence: list[byotrack.Detections]
) -> list[byotrack.PointDetections]:
    """Warp Detections onto the last frame using optical flow (Linear complexity)

    Warnings: Assume that the detections are sorted and that there is one Detections by video frame
    Will only warp the positions and drop the rest
    """
    video = video[::-1]
    dst = optflow.prepare(video[0])
    cum_flow = np.zeros((*dst.shape, 2))
    shape = detections_sequence[0].shape
    points = np.indices(dst.shape, dtype=np.float64).transpose(1, 2, 0)

    warped_positions = [detections_sequence[-1].position.round()]
    for i, frame in enumerate(tqdm.tqdm(video[1:])):
        src = optflow.prepare(frame)
        flow = optflow.calc(src, dst)
        # Compute cum flow from n-i to n
        warped_points = points + flow[:, :, ::-1]
        warped_points = warped_points + optflow.flow_at(cum_flow, warped_points.reshape(-1, 2), 1).reshape(
            *dst.shape, 2
        )
        cum_flow = (warped_points - points)[:, :, ::-1]

        position = optflow.transform(cum_flow, detections_sequence[len(video) - i - 2].position.clone().numpy())

        warped_positions.append(torch.tensor(position.clip(0.0, np.array(shape) - 1).round(), dtype=torch.float32))
        dst = src

    # Has to round positions for a compatible SKT/u-track/eMHT unwarping
    return [
        byotrack.PointDetections(
            position,
            shape=shape,
            confidence=detections_sequence[i].confidence,
            labels=detections_sequence[i].confidence,
        )
        for i, position in enumerate(reversed(warped_positions))
    ]


def unwarp_tracks(video, optflow: OptFlow, tracks: list[byotrack.Track]) -> list[byotrack.Track]:
    """Unwarp tracks to evaluate (If detections were previously warped)

    Very expensive. If you know directly the detections id it is much better to just inverse detections
    by id.
    """
    mu = byotrack.Track.tensorize(tracks).numpy()

    src = optflow.prepare(video[-1])
    for i in tqdm.trange(len(video) - 2, -1, -1):  # Compute flow in the video backwards
        dst = optflow.prepare(video[i])
        flow = optflow.calc(src, dst)

        for t in range(i + 1):
            mu[t] = optflow.transform(flow, mu[t])

        src = dst

    unwarped_tracks = []
    for i in range(mu.shape[1]):
        unwarped_tracks.append(byotrack.Track(0, torch.tensor(mu[:, i]), i))  # noqa: PERF401

    return unwarped_tracks


def unwarp_tracks_from_id(
    tracks: list[byotrack.Track],
    true_detections: list[byotrack.Detections],
    warped_detections: list[byotrack.Detections],
) -> list[byotrack.Track]:
    """Unwarp tracks cleverly using the position of tracks to retrieve the detection id"""
    tracks_tensor = byotrack.Track.tensorize(tracks)
    real_tracks_tensor = torch.full_like(tracks_tensor, torch.nan)

    for t, detections in enumerate(warped_detections):
        # Compute dist between tracks and detections at time t
        dist = (tracks_tensor[t, None] - detections.position[:, None]).abs().sum(dim=-1)
        dist[torch.isnan(dist)] = 100  # Undefined tracks are not valid
        mini, argmin = torch.min(dist, dim=0)
        valid = mini < 1e-5  # Valid tracks are the ones matching with a det precisely
        real_tracks_tensor[t, valid] = true_detections[t].position[argmin[valid]]

    # Rebuild tracks
    real_tracks = []

    for i in range(real_tracks_tensor.shape[1]):
        real_tracks.append(  # noqa: PERF401
            byotrack.Track(
                tracks[i].start, real_tracks_tensor[tracks[i].start : tracks[i].start + len(tracks[i]), i], i
            )
        )

    return real_tracks


def warp_mu(video, optflow: OptFlow, mu: torch.Tensor) -> torch.Tensor:
    """Warp ground truth mu tensor onto the last frame using optical flow"""
    mu_np = mu.clone().numpy()
    src = optflow.prepare(video[0])
    for i, frame in enumerate(tqdm.tqdm(video[1:])):
        dst = optflow.prepare(frame)

        flow = optflow.calc(src, dst)
        for t in range(i + 1):
            mu_np[t] = optflow.transform(flow, mu_np[t])

        src = dst

    return torch.tensor(mu_np)

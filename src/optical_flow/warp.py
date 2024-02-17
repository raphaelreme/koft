from typing import List

import numpy as np
import torch
import tqdm.auto as tqdm

import byotrack

from .optical_flow import OptFlow

# XXX Pretty ugly code here


def warp_detections(
    video, optflow: OptFlow, detections_sequence: List[byotrack.Detections]
) -> List[byotrack.Detections]:
    """Warp Detections onto the last frame using optical flow

    Warnings: Assume that the detections are sorted and that there is one Detections by video frame
    Will only warp the positions and drop the rest
    """
    src = optflow.prepare(video[0])
    positions: List[np.ndarray] = []
    shape = torch.tensor(detections_sequence[0].shape)
    for i, frame in enumerate(tqdm.tqdm(video[1:])):
        dst = optflow.prepare(frame)
        flow = optflow.calc(src, dst)
        positions.append(detections_sequence[i].position.clone().numpy())
        for position in positions:
            position[:] = optflow.transform(flow, position)

        src = dst

    positions.append(detections_sequence[-1].position.clone().numpy())

    detections_extra_data = [
        {
            key: value
            for key, value in detections_sequence[i].data.items()
            if key not in ["shape", "position", "bbox", "segmentation"]
        }
        for i in range(len(detections_sequence))
    ]

    return [
        byotrack.Detections(
            {  # Has to round positions for a compatible SKT/u-track/eMHT unwarping
                "position": torch.tensor(position.clip(0.0, shape.numpy()).round(), dtype=torch.float32),
                "shape": shape,
                **detections_extra_data[i],
            },
            frame_id=i,
        )
        for i, position in enumerate(positions)
    ]


def unwarp_tracks(video, optflow: OptFlow, tracks: List[byotrack.Track]) -> List[byotrack.Track]:
    """Unwarp tracks to evaluate (If detections were previously warped)

    Very expensive. If you know directly the detections id it is much better to just inverse detections
    by id.
    """
    mu = byotrack.Track.tensorize(tracks).numpy()

    src = optflow.prepare(video[-1])
    for i in tqdm.trange(len(video) - 2, -1, -1):  # Compute flow in the video backwards
        dst = optflow.prepare(video[i])
        flow = optflow.calc(src, dst)

        for t in range(0, i + 1):
            mu[t] = optflow.transform(flow, mu[t])

        src = dst

    unwarped_tracks = []
    for i in range(mu.shape[1]):
        unwarped_tracks.append(byotrack.Track(0, torch.tensor(mu[:, i]), i))

    return unwarped_tracks


def unwarp_tracks_from_id(
    tracks: List[byotrack.Track],
    true_detections: List[byotrack.Detections],
    warped_detections: List[byotrack.Detections],
) -> List[byotrack.Track]:
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
        real_tracks.append(
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

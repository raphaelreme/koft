import os
import sys
from typing import Collection, Dict

import numpy as np
from scipy.spatial.distance import cdist  # type: ignore
import torch

import byotrack

sys.path.append(f"{os.environ.get('EXPYRUN_CWD', '.')}/TrackEval/")

import trackeval  # type: ignore


def simulator_to_eval(mu: torch.Tensor, weight: torch.Tensor, weight_limit=0.0, is_gt=True) -> Dict:
    """Convert simulator data (mu, weight) into compatible trackeval data

    Args:
        mu (torch.Tensor): Position of tracks
            Shape: (n_frames, n_particles, d)
        weight (torch.Tensor): Weight of particles
            Shape: (n_frames, n_particles)
        weight_limit (float): If particles has less than this weight, we consider that it
            cannot be detected.
            Default: 0.0 (Always detectable)
        is_gt (bool): Whether to consider the data as ground truth or predicted
            Default: True

    Returns:
        Dict: data compatible with trackeval
    """
    name = "gt" if is_gt else "tracker"
    data = {
        f"num_{name}_ids": 0,
        f"num_{name}_dets": 0,
        f"{name}_ids": [],
        f"{name}_dets": [],
    }

    n_frames = mu.shape[0]

    valid_tracks = (weight[:n_frames] >= weight_limit).sum(dim=0) >= 2  # Keep tracks with at least two valid position
    tracks_ids = np.arange(mu.shape[1])[valid_tracks.numpy()]

    data[f"num_{name}_ids"] = tracks_ids.shape[0]

    for t in range(n_frames):
        ids = []
        dets = []
        for i, id_ in enumerate(tracks_ids):
            if weight[t, id_] < weight_limit:
                continue
            ids.append(i)
            dets.append(mu[t, id_].tolist())
            data[f"num_{name}_dets"] += 1  # type: ignore

        data[f"{name}_ids"].append(np.array(ids))  # type: ignore
        data[f"{name}_dets"].append(np.array(dets))  # type: ignore

    return data


def tracks_to_eval(tracks: Collection[byotrack.Track], is_gt=False) -> Dict:
    """Convert tracks into compatible data for trackeval"""
    mu = byotrack.Track.tensorize(tracks)
    # Set weight to 0 for nans and set a weight_limit at 0.75 to drop them
    weight = torch.ones_like(mu)
    weight[torch.isnan(mu)] = 0
    weight = weight.mean(dim=-1)

    return simulator_to_eval(mu, weight, 0.75, is_gt)


def add_similarity(data: Dict):
    """Add similarity to data (modify data in place)"""
    similarity = []
    for gt_dets_t, tracker_dets_t in zip(data["gt_dets"], data["tracker_dets"]):
        dist = cdist(gt_dets_t, tracker_dets_t)
        similarity.append(np.maximum(0, 1 - dist / 5))
    data["similarity_scores"] = similarity


def compute_tracking_metrics(
    tracks: Collection[byotrack.Track], ground_truth: Dict[str, torch.Tensor]
) -> Dict[str, torch.Tensor]:
    """Compute HOTA at different thresholds

    Keys:
        HOTA: HOTA
        DetA: Jacquard of detections
        DetPr: Precision of detections
        DetRe: Recall of detections
        AssA: Jacquard of associations
        AssPr: Precision of associations
        AssRe: Recall of associations
        LocA: Localization errors
        Thresholds: Localization thresholds for true positive association
    """
    gt_data = simulator_to_eval(ground_truth["mu"], ground_truth["weight"])
    track_data = tracks_to_eval(tracks)
    data = {**gt_data, **track_data}
    add_similarity(data)

    metric = trackeval.metrics.hota.HOTA()
    metrics = metric.eval_sequence(data)

    metrics = {key: torch.tensor(value).to(torch.float32) for key, value in metrics.items() if "0" not in key}

    metrics["LocA"] = 5 - 5 * metrics["LocA"]
    metrics["Thresholds"] = 5 - 5 * torch.linspace(0.05, 0.95, 19)

    return metrics

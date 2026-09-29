from __future__ import annotations

from typing import TYPE_CHECKING

import byotrack
import cv2
import torch
import tqdm

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy as np


class FakeDetector(byotrack.Detector):  # TODO: include weight
    def __init__(self, mu: torch.Tensor, noise=1.0, fpr=0.1, fnr=0.2, generate_outside_particles=True):
        self.noise = noise
        self.fpr = fpr
        self.fnr = fnr
        self.mu = mu
        self.n_particles = mu.shape[1]
        self.generate_outside_particles = generate_outside_particles

    def run(self, video: Sequence[np.ndarray] | np.ndarray) -> list[byotrack.PointDetections]:
        detections_sequence = []

        for k, frame in enumerate(tqdm.tqdm(video)):
            frame = frame[..., 0]  # Drop channel  # noqa: PLW2901
            shape = torch.tensor(frame.shape)

            detected = torch.rand(self.n_particles) >= self.fnr  # Miss some particles (randomly)

            idx = torch.arange(self.n_particles)[detected]
            positions = self.mu[k, detected] + torch.randn((detected.sum(), 2)) * self.noise

            valid = torch.logical_and((positions > 0).all(dim=-1), (positions < shape - 1).all(dim=-1))
            positions = positions[valid]
            idx = idx[valid]

            # Create fake detections
            # 1- Quickly compute the background mask
            mask = torch.tensor(cv2.GaussianBlur(frame, (33, 33), 15) > 0.2)
            mask_proportion = mask.sum().item() / mask.numel()

            # 2- Scale fpr by the mask proportion
            n_fake = int(len(positions) * (self.fpr + torch.randn(1).item() * self.fpr / 10) / mask_proportion)
            false_alarm = torch.rand(n_fake, 2) * (shape - 1)

            if not self.generate_outside_particles:  # Filter fake detections outside the mask
                false_alarm = false_alarm[mask[false_alarm.long()[:, 0], false_alarm.long()[:, 1]]]

            positions = torch.cat((positions, false_alarm))
            idx = torch.cat((idx, -torch.ones_like(false_alarm)[:, 0])) + 1

            # bbox = torch.cat((positions - 1, torch.zeros_like(positions) + 3), dim=-1)
            detections_sequence.append(byotrack.PointDetections(positions, radius=2.0, shape=frame.shape, labels=idx))

        return detections_sequence

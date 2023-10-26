from typing import Collection, Iterable

import cv2
import numpy as np
import torch
import tqdm

import byotrack


class FakeDetector(byotrack.Detector):  # TODO: include weight
    def __init__(self, mu: torch.Tensor, noise=1.0, fpr=0.1, fnr=0.2, generate_outside_particles=True):
        self.noise = noise
        self.fpr = fpr
        self.fnr = fnr
        self.mu = mu
        self.n_particles = mu.shape[1]
        self.generate_outside_particles = generate_outside_particles

    def run(self, video: Iterable[np.ndarray]) -> Collection[byotrack.Detections]:
        detections_sequence = []

        for k, frame in enumerate(tqdm.tqdm(video)):
            frame = frame[..., 0]  # Drop channel
            shape = torch.tensor(frame.shape)

            detected = torch.rand(self.n_particles) >= self.fnr  # Miss some particles (randomly)
            positions = self.mu[k, detected] + torch.randn((detected.sum(), 2)) * self.noise
            positions = positions[(positions > 0).all(dim=-1)]
            positions = positions[(positions < shape - 1).all(dim=-1)]

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

            # bbox = torch.cat((positions - 1, torch.zeros_like(positions) + 3), dim=-1)
            detections_sequence.append(
                byotrack.Detections(
                    {
                        "position": positions,
                        # "bbox": bbox.round().to(torch.int32),
                        "shape": shape,
                    },
                    frame_id=k,
                )
            )

        return detections_sequence

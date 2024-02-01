"""Simulate a fake video"""

import dataclasses
import pathlib

import cv2
import dacite
import numpy as np
import torch
import tqdm  # type: ignore
import yaml  # type: ignore

from ..simulator.simulator import Simulator, SimulatorConfig
from ..simulator.motion import ElasticMotion
from ..utils import enforce_all_seeds


@dataclasses.dataclass
class ExperimentConfig:
    seed: int
    n_frames: int
    display: bool
    simulator: SimulatorConfig


class Saver:
    """Save a running simulation"""

    def __init__(self, simulator: Simulator):
        self.simulator = simulator
        self._mu = [simulator.particles.mu.clone()[None]]
        self._theta = [simulator.particles.theta.clone()[None]]
        self._std = [simulator.particles.std.clone()[None]]
        self._weight = [simulator.particles.weight.clone()[None]]

    @property
    def mu(self):
        return torch.cat(self._mu)

    @property
    def theta(self):
        return torch.cat(self._theta)

    @property
    def std(self):
        return torch.cat(self._std)

    @property
    def weight(self):
        return torch.cat(self._weight)

    def save(self):
        self._mu.append(self.simulator.particles.mu.clone()[None])
        self._theta.append(self.simulator.particles.theta.clone()[None])
        self._std.append(self.simulator.particles.std.clone()[None])
        self._weight.append(self.simulator.particles.weight.clone()[None])


def main(name: str, cfg_data: dict) -> None:
    print("Running:", name)
    print(yaml.dump(cfg_data))
    cfg = dacite.from_dict(ExperimentConfig, cfg_data, dacite.Config(cast=[pathlib.Path, tuple]))

    enforce_all_seeds(cfg.seed)

    simulator = Simulator.from_config(cfg.simulator)
    saver = Saver(simulator)

    # Lets print the alpha used for the simulation (mixture coef between background and particles)
    snr = 10 ** (cfg.simulator.imaging_config.psnr / 10)
    alpha = (snr - 1) / (snr - 1 + 1 / 0.6)  # From simulator
    print("Alpha:", alpha)

    # Find springs for display and save
    springs = None
    for motion in simulator.motion.motions:
        if isinstance(motion, ElasticMotion):
            springs = motion.spring

    # Create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore
    writer = cv2.VideoWriter("video.mp4", fourcc, 30, simulator.particles.size, isColor=False)

    frame_saved = False
    k = 0
    try:
        for k in tqdm.trange(cfg.n_frames):
            frame_saved = False
            frame = simulator.generate_image().numpy()
            writer.write((frame * 255).astype(np.uint8))
            frame_saved = True

            if cfg.display:
                if springs:  # Display also the springs
                    points = springs.points
                    if simulator.global_motion:
                        points = simulator.global_motion.apply_tensor(springs.points)
                    for i, j in points.round().to(torch.int32).tolist():
                        cv2.circle(frame, (j, i), 2, 255, -1)

                cv2.imshow("Frame", frame)
                cv2.setWindowTitle("Frame", f"Frame {k}/{cfg.n_frames}")
                cv2.waitKey(delay=1)

            simulator.update()
            saver.save()

    finally:
        torch.save(
            {
                "mu": saver.mu[: k + frame_saved],
                "theta": saver.theta[: k + frame_saved],
                "std": saver.std[: k + frame_saved],
                "weight": saver.weight[: k + frame_saved],
            },
            "video_data.pt",
        )

        cv2.destroyAllWindows()
        writer.release()

import dataclasses
from typing import Optional

import cv2
import numpy as np
import torch

import byotrack

from .motion import GlobalDriftAndRotation, GlobalMotionConfig, MotionConfig, MultipleMotion
from .particle import GaussianParticles, GaussianParticlesConfig
from .nam import NeuronalActivityModel, NamConfig


@dataclasses.dataclass
class ImagingConfig:
    dt: float = 100.0
    snr: float = 1.5
    noise: float = 0.1


@dataclasses.dataclass
class VideoConfig:
    path: str = ""
    start: int = 0
    stop: int = -1
    step: int = 1

    transform: byotrack.VideoTransformConfig = byotrack.VideoTransformConfig()

    def open(self) -> Optional[byotrack.Video]:
        if self.path == "":
            return None

        video = byotrack.Video(self.path)[slice(self.start, self.stop, self.step)]
        video.set_transform(self.transform)
        return video


@dataclasses.dataclass
class SimulatorConfig:
    base_video: VideoConfig
    imaging_config: ImagingConfig
    particle: GaussianParticlesConfig
    background: GaussianParticlesConfig
    nam: NamConfig
    motion: MotionConfig
    global_motion: GlobalMotionConfig
    warm_up: int = 500


def random_mask(size=1024, verbose=False) -> torch.Tensor:
    """Generate a random ellipse mask roughly centered in the middle"""
    thresh = 1 / np.sqrt(2 * torch.pi) ** 2 * np.exp(-1 / 2)  # Thresh at 1 sigma (1 mahalanohobis)
    mask_area = torch.rand(1).item() * 0.15 + 0.2  # [0.2, 0.35]
    ratio = 0.5 + 1.5 * torch.rand(1).item()  # [0.5, 2]

    mean = size / 2 + torch.randn(2) * size / 60

    std = torch.tensor([mask_area * ratio / torch.pi, mask_area / ratio / torch.pi]).sqrt() * size

    distribution = torch.distributions.Normal(mean, std)

    indices = torch.tensor(np.indices((size, size)), dtype=torch.float32).permute(1, 2, 0)
    prob = distribution.log_prob(indices).sum(dim=2).exp() * std.prod()

    mask = prob > thresh
    if verbose:
        print(mask.sum() / mask.numel(), mask_area)

    return mask


def mask_from_frame(frame: np.ndarray) -> torch.Tensor:
    """Find the animal mask using a simple thresholding"""
    # First blur the image
    frame = cv2.GaussianBlur(frame, (35, 35), 10, 10)

    # Set the threshold using the inflexion point of the hist
    # threshold = np.quantile(frame, 0.8)

    # Limit the search to threshold resulting in 10 to 40% of the image
    mini = int((np.quantile(frame, 0.6) * 100).round())
    maxi = int((np.quantile(frame, 0.9) * 100).round())

    bins = np.array([k / 100 for k in range(101)])
    hist, _ = np.histogram(frame.ravel(), bins=bins)

    # Smoothing of the histogram before inflexion extraction
    cumsum = np.cumsum(hist)
    cumsum_pad = np.pad(cumsum, 10 // 2, mode="edge")
    cumsum_smooth = np.convolve(cumsum_pad, np.ones(10) / 10, mode="valid")

    argmax = np.gradient(np.gradient(np.gradient(cumsum_smooth)))[mini : maxi + 1].argmax() + mini

    threshold = bins[argmax + 1]

    return torch.tensor(frame > threshold)


class Simulator:
    """Simulator object

    Handle the image generation and temporal evolution.
    """

    def __init__(
        self,
        particles: GaussianParticles,
        background: Optional[GaussianParticles],
        nam: NeuronalActivityModel,
        motion: MultipleMotion,
        global_motion: Optional[GlobalDriftAndRotation],
        imaging_config: ImagingConfig,
    ):
        self.background = background
        self.background_gain = background.draw_truth(scale=4).max().item() if background else 1.0
        self.particles = particles

        self.nam = nam
        self.motion = motion
        self.global_motion = global_motion
        self.imaging_config = imaging_config

    def generate_image(self):
        particles = self.particles.draw_truth()

        if self.background:
            background = self.background.draw_truth(scale=4)
            background /= self.background_gain
            background = (
                1 - self.imaging_config.noise
            ) * background + self.imaging_config.noise  # Add a noise baseline
        else:
            background = self.imaging_config.noise * torch.ones_like(particles)

        snr = 10 ** (self.imaging_config.snr / 10)
        alpha = (snr - 1) / (
            snr - 1 + 1 / 0.6
        )  # Uses E[B(z_p)] = 0.6 and assume that the Poisson Shot noise is negligeable in the SNR
        baseline = (1 - alpha) * background + alpha * particles

        # Poisson shot noise
        image = torch.distributions.Poisson(self.imaging_config.dt * baseline).sample((1,))[0] / self.imaging_config.dt

        image.clip_(0.0, 1.0)

        return image

    def update(self):
        self.nam.update()
        self.motion.update()

        if self.global_motion:
            self.global_motion.revert(self.particles)
            if self.background:
                self.global_motion.revert(self.background)

        self.motion.apply(self.particles)
        if self.background:
            self.motion.apply(self.background)

        if self.global_motion:
            self.global_motion.update()
            self.global_motion.apply(self.particles)
            if self.background:
                self.global_motion.apply(self.background)

        self.particles.build_distribution()
        if self.background:
            self.background.build_distribution()

    @staticmethod
    def from_config(config: SimulatorConfig) -> "Simulator":
        video = config.base_video.open()
        if video is None:
            mask = random_mask()
        else:
            mask = mask_from_frame(video[0])

        particles = GaussianParticles(config.particle.n, mask, config.particle.min_std, config.particle.max_std)
        if config.particle.min_dist > 0:
            particles.filter_close_particles(config.particle.min_dist)

        background = None
        if config.background.n > 0:
            background = GaussianParticles(
                config.background.n, mask, config.background.min_std, config.background.max_std
            )
            if config.background.min_dist > 0:
                background.filter_close_particles(config.background.min_dist)

        nam = NeuronalActivityModel(particles, config.nam.firing_rate, config.nam.decay)

        motion = config.motion.build(video=video, mask=mask, particles=particles, background=background)

        global_motion = None
        if config.global_motion.noise_position > 0 or config.global_motion.noise_theta > 0:
            global_motion = GlobalDriftAndRotation(
                particles.mu.mean(dim=0),
                config.global_motion.period,
                config.global_motion.noise_position,
                config.global_motion.noise_theta,
            )

        # Warmup
        motion.warm_up(config.warm_up, particles, background)  # Update and apply motion

        for _ in range(config.warm_up):
            nam.update()  # Update nam

            if global_motion:  # Update global motional motion as it is first reverted on simulator.update
                global_motion.update()

        if global_motion:  # UGLY: Apply glob
            global_motion.apply(particles)
            if background:
                global_motion.apply(background)

        simulator = Simulator(particles, background, nam, motion, global_motion, config.imaging_config)
        # True warm up, but expensive with Optical flow
        # for _ in tqdm.trange(config.warm_up, desc="Warming up"):
        #     simulator.update()

        return simulator

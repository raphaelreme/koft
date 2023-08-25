import dataclasses
from typing import Any, Dict, List, Iterable, Optional

import numpy as np
import torch

import byotrack
import torch_tps

from .. import optical_flow
from . import springs
from .particle import GaussianParticles


@dataclasses.dataclass
class ShapeVariationConfig:
    period: float = 50.0
    noise: float = 0.0

    def build(self, **kwargs) -> "ShapeVariation":
        return ShapeVariation(kwargs["particles"], self.period, self.noise)


@dataclasses.dataclass
class LocalRotationConfig:
    period: float = 50.0
    noise: float = 0.0

    def build(self, **kwargs) -> "LocalRotation":
        return LocalRotation(kwargs["particles"], self.period, self.noise)


@dataclasses.dataclass
class BrownianRotationConfig:
    noise: float = 0.0

    def build(self, **_) -> "BrownianRotation":
        return BrownianRotation(self.noise)


@dataclasses.dataclass
class FlowMotionConfig:
    optflow_name: str = "farneback"

    def build(self, **kwargs) -> "FlowMotion":
        assert kwargs.get("video"), "Unable to create a flow motion without a true video input"

        return FlowMotion(optical_flow.__dict__[self.optflow_name], kwargs["video"])


@dataclasses.dataclass
class ElasticNoiseConfig:
    builder_name: str = "RandomMuscles"
    kwargs: Dict[str, Any] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class ElasticMotionConfig:
    alpha: float = 10.0
    period: float = 50.0
    grid_step: int = 100
    noise: ElasticNoiseConfig = ElasticNoiseConfig()

    def build(self, **kwargs) -> "ElasticMotion":
        points, neighbors = springs.RandomRelationalSprings.grid_springs_from_mask(kwargs["mask"], self.grid_step)
        quality = torch.tensor([0.5])
        w0 = torch.tensor([2 * torch.pi / self.period])
        random_accelerator = springs.__dict__[self.noise.builder_name](**self.noise.kwargs)
        spring = springs.RandomRelationalSprings(points, neighbors, w0**2, w0 / quality, random_accelerator)
        return ElasticMotion(spring, self.alpha)


@dataclasses.dataclass
class MotionConfig:
    motions: List[str]
    ShapeVariation: ShapeVariationConfig = ShapeVariationConfig()
    LocalRotation: LocalRotationConfig = LocalRotationConfig()
    BrownianRotation: BrownianRotationConfig = BrownianRotationConfig()
    FlowMotion: FlowMotionConfig = FlowMotionConfig()
    ElasticMotion: ElasticMotionConfig = ElasticMotionConfig()

    def build(self, **kwargs) -> "MultipleMotion":
        motions = []
        for motion_name in self.motions:
            sub_cfg = getattr(self, motion_name)
            if motion_name in ("ShapeVariation", "LocalRotation"):
                motions.append(sub_cfg.build(particles=kwargs["particles"]))
                if kwargs.get("background"):
                    motions.append(sub_cfg.build(particles=kwargs["background"]))

            motions.append(sub_cfg.build(**kwargs))

        return MultipleMotion(motions)


class BaseMotion:
    """Base motion Class

    A motion can be updated then applied (to background/particles)
    """

    def update(self) -> None:
        """Notify the motion that we have move to next frame"""

    def apply(self, particles: GaussianParticles) -> None:
        """Apply the motion to the particles

        It can moves mu, std or theta
        """

    def warm_up(self, warm_up: int, particles: GaussianParticles, background: Optional[GaussianParticles]) -> None:
        for _ in range(warm_up):
            self.update()
            self.apply(particles)
            if background:
                self.apply(background)


class MultipleMotion(BaseMotion):
    """Handle multiple motions to apply to particles/backgrounds

    The current implementation of motions is not very robust. To prevent bugs, you should avoid
    having multiple motions handling the same parameter (mu, theta, std).
    """

    def __init__(self, motions: Iterable[BaseMotion]) -> None:
        super().__init__()
        self.motions = motions

    def update(self) -> None:
        for motion in self.motions:
            motion.update()

    def apply(self, particles: GaussianParticles) -> None:
        for motion in self.motions:
            motion.apply(particles)

    def warm_up(self, warm_up: int, particles: GaussianParticles, background: Optional[GaussianParticles]) -> None:
        for motion in self.motions:
            motion.warm_up(warm_up, particles, background)


## Shape


class ShapeVariation(BaseMotion):
    """Create variation in shape (std)

    Shape of particles changes following  std_k = s_k * std_0
    Where s_k is a spring of equilibrium 1.0

    Will only modify the std of its particles
    """

    def __init__(self, particles: GaussianParticles, period=50.0, noise=0.0) -> None:
        super().__init__()
        self.particles_id = id(particles)
        self.std_0 = particles.std.clone()
        self.spring = springs.RandomAcceleratedSpring.build(
            torch.ones_like(particles.std),
            torch.tensor([0.5]),
            torch.tensor([noise]),
            torch.tensor([2 * torch.pi / period]),
        )

    def update(self) -> None:
        self.spring.update()

    def apply(self, particles: GaussianParticles) -> None:
        if id(particles) != self.particles_id:
            return

        particles.std = self.std_0 * self.spring.value


## Rotation


class BrownianRotation(BaseMotion):
    """Random rotation of particles

    The particles rotation follow a brownian motion (uncorrelated with the other rotations)
    """

    def __init__(self, noise=0.0) -> None:
        self.noise = noise

    def apply(self, particles: GaussianParticles) -> None:
        particles.theta += torch.randn(particles.theta.shape) * self.noise


class LocalRotation(BaseMotion):
    """Local rotation of particles

    Each particle can rotate locally but not going to much further from the equilibrium point
    More rotation can be made by global rotation. (Thus rotation diff between particles is mostly kept)

    Will only modify the rotation of its particles
    """

    def __init__(self, particles: GaussianParticles, period=50.0, noise=0.0) -> None:
        self.particles_id = id(particles)
        self.spring = springs.RandomAcceleratedSpring.build(
            particles.theta,
            torch.tensor([0.5]),
            torch.tensor([noise]),
            torch.tensor([2 * torch.pi / period]),
        )

    def update(self) -> None:
        self.spring.update()

    def apply(self, particles: GaussianParticles) -> None:
        if id(particles) != self.particles_id:
            return

        particles.theta = self.spring.value


## Position


class BrownianPosition(BaseMotion):
    """Brownian motion of each particle"""

    def __init__(self, noise=0.0) -> None:
        self.noise = noise

    def apply(self, particles: GaussianParticles) -> None:
        particles.mu += torch.randn(particles.mu.shape) * self.noise


# class ConfinedBrownianPosition(BaseMotion):
#     """TODO"""


class FlowMotion(BaseMotion):
    """Motion based on real optical flow of a real animal"""

    def __init__(self, optflow: optical_flow.OptFlow, video: byotrack.Video) -> None:
        self.optflow = optflow
        self.video = video
        self.dir = 1
        self.cursor = 0
        self.source = self.optflow.prepare(self.video[self.cursor])
        self.flow = np.zeros((1, 1, 2))

    def update(self) -> None:
        if not 0 <= self.cursor + self.dir < len(self.video):
            self.dir = -self.dir  # If no more frame, let's go backward in the video

        self.cursor += self.dir
        destination = self.optflow.prepare(self.video[self.cursor])

        self.flow = self.optflow.calc(self.source, destination)
        self.source = destination

    def apply(self, particles: GaussianParticles) -> None:
        particles.mu = torch.tensor(self.optflow.transform(self.flow, particles.mu.numpy())).to(torch.float32)

    def warm_up(self, warm_up: int, particles: GaussianParticles, background: GaussianParticles | None) -> None:
        pass  # No warmup with optical flow


class ElasticMotion(BaseMotion):
    """Elastic motion induced by RandomRelationalSprings

    Motion of particles are computed as a TPS interpolation of the springs points.
    """

    def __init__(self, spring: springs.RandomRelationalSprings, alpha=10.0) -> None:
        self.spring = spring
        self.tps = torch_tps.ThinPlateSpline(alpha)
        self.tps.fit(self.spring.points - self.spring.speeds, self.spring.points)

    def update(self) -> None:
        self.spring.update()
        self.tps.fit(self.spring.points - self.spring.speeds, self.spring.points)

    def apply(self, particles: GaussianParticles) -> None:
        particles.mu = self.tps.transform(particles.mu)


@dataclasses.dataclass
class GlobalMotionConfig:
    period: float = 1000.0
    noise_position: float = 0.0
    noise_theta: float = 0.0


class GlobalDriftAndRotation:
    """Global drift and rotation for all particles (Rigid motion)

    Note: With the current bad implementation, i'm not able to make this fit with other motions.
    Lets not make it inherit Motion and be given independently to the simulator

    Drift follow a spring (prevent all the particles to go out of focus) with a large period (slow vs the local motion)

    Rotation is also a spring (more for continuous reason) with the same period
    """

    def __init__(self, mass_center: torch.Tensor, period=1000.0, noise_position=0.0, noise_theta=0.0) -> None:
        super().__init__()
        self.spring = springs.RandomAcceleratedSpring.build(
            torch.cat((mass_center, torch.ones(1))),
            torch.tensor([0.5]),
            torch.tensor([noise_position, noise_position, noise_theta]),
            torch.tensor(2 * torch.pi / period),
        )
        self.global_theta = torch.tensor(0.0)
        self.global_translation = torch.tensor([0.0, 0.0])

    @property
    def transformation(self) -> torch.Tensor:
        return torch.tensor(
            [
                [torch.cos(self.global_theta), -torch.sin(self.global_theta), self.global_translation[0]],
                [torch.sin(self.global_theta), torch.cos(self.global_theta), self.global_translation[1]],
                [0.0, 0.0, 1.0],
            ]
        )

    def update(self) -> None:
        old_state = self.spring.value.clone()
        self.spring.update()
        # Add to translation the new motion of the center of mass
        self.global_translation += self.spring.value[:2] - old_state[:2]

        # New rotation=
        theta = self.spring.value[2] - old_state[2]
        rotation = torch.tensor(
            [
                [torch.cos(theta), -torch.sin(theta)],
                [torch.sin(theta), torch.cos(theta)],
            ]
        )

        self.global_translation -= self.spring.value[:2]
        self.global_translation = rotation @ self.global_translation
        self.global_translation += self.spring.value[:2]
        self.global_theta += theta

    def apply(self, particles: GaussianParticles) -> None:
        particles.mu = self.apply_tensor(particles.mu)

    def apply_tensor(self, points: torch.Tensor) -> torch.Tensor:
        homogenous = torch.cat((points, torch.ones(points.shape[0])[:, None]), dim=-1)  # N, 3
        homogenous = homogenous @ self.transformation.T

        return homogenous[:, :2]

    def revert(self, particles: GaussianParticles) -> None:
        particles.mu = self.revert_tensor(particles.mu)

    def revert_tensor(self, points: torch.Tensor) -> torch.Tensor:
        transformation_inv = self.transformation
        transformation_inv[:2, :2] = transformation_inv[:2, :2].clone().T
        transformation_inv[:2, 2] = -transformation_inv[:2, :2] @ transformation_inv[:2, 2]

        homogenous = torch.cat((points, torch.ones(points.shape[0])[:, None]), dim=-1)  # N, 3
        homogenous = homogenous @ transformation_inv.T

        return homogenous[:, :2]

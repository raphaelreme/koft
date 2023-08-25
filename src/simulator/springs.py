import math
from typing import Callable, Tuple
import warnings

import cv2
import numpy as np
import torch


class RandomAcceleratedSpring:
    """Model a damped harmonic oscillator (spring) with random forces

    a = -lambda v - k (x - x_eq) + noise * randn()

    We discretize the equation in
    x(t+1) = x(t) + v(t) dt
    v(t+1) = v(t) - lambda v(t) dt - k(x(t+1) - x_eq) dt + noise * randn() dt

    For this to be true we suppose that the variation in speeds and position is small
    and thus k dt and lambda dt should be small (<1)

    Additional math:
    ----------------

    The equation is usually reframed as
    a + w0/Q v + w0^2 x = 0

    Where Q is the quality factor and w0 the angular frequency (frequency when undamped)

    w0 = sqrt(k)
    Q = sqrt(k) / lambda

    One can show that there are two modes:
    Underdamped (Q > 0.5): Oscillates with a slighlty smaller frequency than w0.
        The amplitude decreases to 0. (Pseudo-Periodic)
    Overdamped (Q <= 0.5): Returns to the steady state without oscillation (the lower Q the slower)

    The pseudo period/critical time of the system is T = 2 pi / w0.
    Another constraint is that T >> dt (one update is small vs the critical time of the system).

    With friction we have energy conservation leading to: 1/2 v_m^2 = 1/2 k dl_m^2.
    Thus the maximum displacement is dl_m = v_m/w0.


    Attributes:
        initial_value (torch.Tensor): Steady sate
            dtype: float32
        value (torch.Tensor): Current value
            dtype: float32
        speed (torch.Tensor): Current speed
            dtype: float32
        k (torch.Tensor): Spring constant(s)
            Shape: Broadcastable with value, dtype: float32
        lambda_ (torch.Tensor): Damping coefficient
            Shape: Broadcastable with value, dtype: float32
        noise (torch.Tensor): Noise std
            Shape: Broadcastable with value, dtype: float32
    """

    def __init__(self, value: torch.Tensor, k: torch.Tensor, lambda_: torch.Tensor, noise: torch.Tensor):
        # Check shapes
        torch.broadcast_shapes(value.shape, k.shape)
        torch.broadcast_shapes(value.shape, lambda_.shape)
        torch.broadcast_shapes(value.shape, noise.shape)

        self.initial_value = value.clone()
        self.value = value.clone()
        self.speed = torch.zeros(value.shape)
        self.k = k.clone()
        self.lambda_ = lambda_.clone()
        self.noise = noise.clone()

    @property
    def w0(self) -> torch.Tensor:
        """Angular frequency"""
        return torch.sqrt(self.k)

    @property
    def quality(self) -> torch.Tensor:
        """Quality factor Q"""
        return self.w0 / self.lambda_

    @property
    def critical_time(self) -> torch.Tensor:
        return 2 * math.pi / self.w0

    def update(self, dt=1.0):
        """Update the position and speed of each spring

        x += v dt
        a = -lambda v - k x + noise * randn()
        v += a dt

        Args:
            dt (float): Size of the interval

        """

        # Acceleration
        acc = -self.lambda_ * self.speed  # Friction
        acc -= self.k * (self.value - self.initial_value)  # Spring model
        acc += self.noise / math.sqrt(dt) * torch.randn(self.speed.shape)  # Random motion
        # NOTE: We divide by sqrt(dt) to compensate the noise gain of splitting in small steps
        # One can show this by comparing the noise added to the speed and value
        # on a period tau << T (we can then skip friction and spring effect)
        # and splitting tau in n dt.
        # We can show that the speed is gaussian with std: sigma sqrt(tau * dt)
        # And the value is gaussian with std = sigma sqrt(tau^3 * dt / 3)
        # With this correction, we compensate the dt part of std for both speed and value

        # Update speed
        self.speed += acc * dt

        # Update value
        self.value += self.speed * dt
        # NOTE: We use xk+1 = xk + vk+1 * dt (rather than xk + vk * dt)
        # This compensate integration errors. If you reverse the equation this can be rewritten
        # xk+1 = xk + vk & vk+1 = vk - k (xk+1 - x0)
        # One can show that using xk+1 instead of xk improves the approximation of \int_k^k+1 a(t)dt

    @staticmethod
    def build(
        value: torch.Tensor,
        quality: torch.Tensor,
        noise: torch.Tensor,
        w0: torch.Tensor = torch.full((1,), 2 * math.pi / 100),
    ) -> "RandomAcceleratedSpring":
        """Build spring given meaningful quantities

        Look at the math section, but all can be defined from the angular frequency w0 and the quality factor Q

        k = w0 **2
        lambda = w0 / Q

        The noise should be scaled with w0 to keep the same level of noise with different angular frequency.
        We can show that scaling needed is w0^(3/2). (Note that with a scaling of w0^(1/2) the noise on speed
        is kept).

        Also if you have a v0 or vmax to set, to keep the same behavior (same hmax), v0 or vmax should be scaled by
        w0.

        As all is stable by w0, changing its value only affects the time scale. Double w0 and
        half dt and you'll now generate the same values.

        Be careful that a large w0 implies to use smaller dt.
        Remember that we should have dt << T = 2 pi / w0.
        The default value set T = 100 which works well by default with dt=1.

        Args:
            value (torch.Tensor): Steady state and initial value
                dtype: float32
            quality (torch.Tensor): Quality factor (Q). 0.5 for critically damped, >> 1 for low friction
                Shape: Broadcastable with value, dtype: float32
            noise (torch.Tensor): Std of the noise. We have choosen to scale it so that with Q=1/2, whatever w0,
                the std of the generated position is equal to noise.
                Shape: Broadcastable with value, dtype: float32
            w0 (torch.Tensor): Angular frequency (sqrt(k))
                Shape: Broadcastable with value, dtype: float32
                Default: 2 pi / 100 (T = 100)

        Returns:
            RandomSpring: Spring with the good quality factor, critical time and noise
        """
        return RandomAcceleratedSpring(value, w0**2, w0 / quality, noise * w0 ** (3 / 2) * 2)


class SimpleBrownianSpring:
    """Simpler spring motion without time consistency

    The model is not really a spring but more a confined brownian motion:
    xt+1 = xt + randn() * noise - k (xt - x0)

    Can be simulated with the RandomAcceleratedSpring with lambda dt = 1.0 but simpler this way

    To create a process with the right amount of noise, you can use this equation:
    x_t ~ N(0, noise / scale(k)).
    Therefore, knowing k and the true noise that you want you can write:
    spring = SimpleBrownianSpring(value, k, true_noise * SimpleBrownianSpring.noise_scaling(k))

    Attributes:
        initial_value (torch.Tensor): Steady sate
            dtype: float32
        value (torch.Tensor): Current value
            dtype: float32
        k (torch.Tensor): Spring constant(s)
            Shape: Broadcastable with value, dtype: float32
        noise (torch.Tensor): Noise std (Added at each step)
            Shape: Broadcastable with value, dtype: float32
    """

    def __init__(self, value: torch.Tensor, k: torch.Tensor, noise: torch.Tensor):
        torch.broadcast_shapes(value.shape, k.shape)
        torch.broadcast_shapes(value.shape, noise.shape)

        self.initial_value = value.clone()
        self.value = value.clone()
        self.k = k.clone()
        self.noise = noise.clone()

    def update(self) -> None:
        """Update the values

        xt+1 = xt + randn() * noise - k (xt - x0)
        """
        self.value += torch.randn(self.value.shape) * self.noise - self.k * (self.value - self.initial_value)

    @staticmethod
    def noise_scaling(k: torch.Tensor) -> torch.Tensor:
        """Return the scaling coefficient of noise given k

        x_t follows a N(0, noise / scale) when t is large enough, where noise is the noise added at each step and
        scale the scaling coefficient computed by this method.

        Args:
            k (torch.Tensor): Spring constant(s)
                dtype: float32
        """
        return torch.sqrt(1 - (1 - k).pow(2))

    @property
    def true_noise(self) -> torch.Tensor:
        """Std of x_t"""
        return self.noise / self.noise_scaling(self.k)


def no_acceleration(model: "RandomRelationalSprings", _: float) -> torch.Tensor:
    """The points do not accelerate"""
    return torch.zeros_like(model.points)


class RandomNoise:
    """Adds a normal white noise robust to w0 and dt

    Follows the theory of the RandomAcceleratedSpring for noise (it seems also to work experimentally).

    (It seems that there is more noise with more springs which is not compensate here.
    You can expect a gain in noise of around 1.5 with many springs)

    Attributes:
        noise (float): True noise generated on the position
    """

    def __init__(self, noise: float) -> None:
        self.noise = noise

    def __call__(self, model: "RandomRelationalSprings", dt: float) -> torch.Tensor:
        return 2 / math.sqrt(dt) * self.noise * torch.randn(model.points.shape) * model.w0 ** (3 / 2)


class RandomMuscles(RandomNoise):
    """Mimic random contraction or elongation of muscles + Random Noise

    At each step we sample some points that will follow a contraction or elongation.
    We also follow the theory of the RandomAcceleratedSpring to be robust to w0 / dt.

    At each step, we randomly do contractions/elongations (or nothing). Each contraction/elongation
    is done on several points that either are accelerated to the center of mass or from it.

    We scale the acceleration amplitude so that the order of magnitude of resulting displacement is amplitude.

    Attributes:
        motion_rate (float): Expected number of contraction/elongation in a period T (critical time of the model)
        motion_size (int): We sample the number of moving particle from 2 to motion_size at each contraction
        amplitude (float): Max magnitude of the displacement at each motion (sampled from [amplitude / 2, amplitude])
        noise (float): Additional white noise handled by RandomNoise

    """

    def __init__(self, motion_rate=1.0, motion_size=10, amplitude=2, noise=0.0) -> None:
        super().__init__(noise)
        self.motion_rate = motion_rate
        self.motion_size = motion_size
        self.amplitude = amplitude

    def __call__(self, model: "RandomRelationalSprings", dt: float) -> torch.Tensor:
        acc = super().__call__(model, dt)
        k = int(torch.distributions.Poisson(self.motion_rate * dt / model.critical_time).sample((1,)).item())
        for _ in range(k):
            n = 2 + int(torch.rand(1) * (self.motion_size - 1))  # Sampling between 2 and motion_size points to move
            ids = torch.randperm(model.points.shape[0])[:n]
            n = ids.shape[0]

            # Direction mean to point (elongation of the set of points)
            directions = model.points[ids] - model.points[ids].mean(dim=0)
            directions /= directions.norm(dim=-1, keepdim=True) + 1e-10

            if torch.rand(1) < 0.4:  # Switch to contraction
                directions = -directions

            amplitude = self.amplitude / 2 / dt * (1 + torch.rand(n, 1)) * model.w0
            acc[ids] += amplitude * directions

        return acc


class RandomRelationalSprings:
    """Create a model with multiple springs between multiple points

    The acceleration of ith points is computed using
    a_i(t+1) = - lambda v_i(t) - \\sum_j k (dl_{ij}) d_{ij} + random_acceleration
    """

    def __init__(
        self,
        points: torch.Tensor,
        neighbors: torch.Tensor,
        k: torch.Tensor,
        lambda_: torch.Tensor,
        random_accelerator: Callable[["RandomRelationalSprings", float], torch.Tensor] = no_acceleration,
    ):
        # Check size
        n, _ = neighbors.shape
        assert len(points.shape) == 2
        assert n == points.shape[0]

        # XXX: More work needed to handle the broadcasting
        torch.broadcast_shapes(neighbors.shape, k.shape)
        torch.broadcast_shapes(points.shape, lambda_.shape)
        torch.broadcast_shapes(k.shape, lambda_.shape)

        self.points = points.clone()
        self.neighbors = neighbors.clone()
        # when there is no neighbors let's put itself as a neighbors (the dist with itself is always 0)
        for i in range(self.neighbors.shape[0]):
            self.neighbors[i][self.neighbors[i] < 0] = i

        self.num_neighbors = (self.neighbors != torch.arange(self.neighbors.shape[0])[:, None]).sum(dim=-1)
        if self.num_neighbors.max() != self.neighbors.shape[1]:
            warnings.warn("Using more neighbors than the true max of neighbors")

        self.steady_state = torch.cdist(self.points[self.neighbors], self.points[:, None])  # Shape (n, m, 1)

        self.k = k.clone()
        self.lambda_ = lambda_.clone()
        self.random_accelerator = random_accelerator
        self.speeds = torch.zeros_like(self.points)

    @property
    def w0(self) -> torch.Tensor:
        return torch.sqrt(self.k)

    @property
    def quality(self) -> torch.Tensor:
        """Quality factor Q"""
        return self.w0 / self.lambda_

    @property
    def critical_time(self) -> torch.Tensor:
        return 2 * math.pi / self.w0

    def update(self, dt=1.0):
        """Update the position of points"""
        # Acceleration

        ## Friction
        acc = -self.lambda_ * self.speeds

        ## Springs
        diffs = self.points[self.neighbors] - self.points[:, None]  # Shape n x m x d, diffs from points to neighbors
        state: torch.Tensor = diffs.norm(dim=-1, keepdim=True)
        directions = diffs / (state + (state == 0))
        acc += (self.k * (state - self.steady_state) * directions).sum(dim=1) / (
            self.num_neighbors[..., None] + (self.num_neighbors[..., None] == 0)
        )

        ## Random
        acc += self.random_accelerator(self, dt)

        # Speed
        self.speeds += acc * dt

        # Pos
        self.points += self.speeds * dt

    @staticmethod
    def grid_springs_from_mask(mask: torch.Tensor, grid_step: int) -> Tuple[torch.Tensor, torch.Tensor]:
        max_neighbors = 8
        max_dist = grid_step * 2 - 1  # Less than 2 grid step
        size = mask.shape[0]
        points = (
            torch.tensor(np.indices((1 + size // grid_step, 1 + size // grid_step))).permute(1, 2, 0).reshape(-1, 2)
            * grid_step
        )

        # Extend the mask and take only the points in the extended mask
        mask = torch.tensor(cv2.dilate(mask.numpy().astype(np.uint8), np.ones((grid_step // 3, grid_step // 3)))) > 0
        points = points[mask[points[:, 0], points[:, 1]]].to(torch.float32)

        cdist = torch.cdist(points, points, compute_mode="donot_use_mm_for_euclid_dist")
        neighbors = cdist.argsort(dim=1)[:, 1 : 1 + max_neighbors]
        neighbors[cdist[torch.arange(points.shape[0])[:, None], neighbors] > max_dist] = -1
        # neighbors = neighbors[:, ~(neighbors == -1).all(dim=0)]  # can be use to filter useless columns

        return points, neighbors

    # @staticmethod
    # def build(points: torch.Tensor, neighbors: torch.Tensor, quality: torch.Tensor, w0: torch.Tensor, random_accelerator: Callable[["RandomRelationalSprings", float], torch.Tensor]):
    #     """Quick build method from useful quantities"""
    #     return

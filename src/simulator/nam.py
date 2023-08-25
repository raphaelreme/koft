import dataclasses
import torch

from . import particle


@dataclasses.dataclass
class NamConfig:
    firing_rate: float = 0.01
    decay: float = 0.95


class NeuronalActivityModel:
    """Simple Neuronal Activity Model (nam)

    The neurons are supposed independent (untrue) following a simple generation process:
    i(t+1) = decay * i(t) + gain * firing

    We add some complexity to prevent some behaviors:
    - The gain of each firing depends on the actual value of the neurons
        (Large value, small gain to prevent reaching MAX_WEIGHT)
    - The added value is then sampled from N(gain, (gain / 5)**2)
    - The real weights retained for neurons intensity is an EMA of the computed weights
        This is done to mimic natural firings where the intensity does not jump in a single frame

    Attributes:
        particles (GaussianParticles): The particles to handle
        firing_rate (float): Firing rates of particles
        decay (float): Exponential decay of the weights
        immediate_weights (torch.Tensor): Weights without the EMA (Smoothing)

    """

    MAX_WEIGHT = 2
    MIN_WEIGHT = 0.1  # Minimal baseline for particles
    FIRING_GAIN = 1.0
    SMOOTHING_DECAY = 0.75  # EMA factor to prevent hard firings

    def __init__(self, particles: particle.GaussianParticles, firing_rate=0.01, decay=0.95):
        self.firing_rate = firing_rate
        self.decay = decay
        self.particles = particles
        self.immediate_weight = particles.weight.clone()

    def update(self):
        self.immediate_weight *= self.decay
        firing = torch.rand(self.immediate_weight.shape) < self.firing_rate
        gain = self.FIRING_GAIN - self.immediate_weight[firing] * (self.FIRING_GAIN / self.MAX_WEIGHT)
        self.immediate_weight[firing] += gain + torch.randn(firing.sum()) * gain * 0.2

        # Update the particles weights as an EMA with a gamma of SMOOTHING_DELAY
        self.particles.weight.sub_(
            (1.0 - self.SMOOTHING_DECAY) * (self.particles.weight - self.immediate_weight - self.MIN_WEIGHT)
        )

        # NOTE: We could add some gaussian noise to weights

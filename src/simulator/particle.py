import dataclasses
from typing import Tuple

import cv2
import numba  # type: ignore
import numpy as np
import torch


@numba.njit(parallel=True)
def _fast_mahalanobis_pdist(mu: np.ndarray, sigma_inv: np.ndarray, thresh: float) -> np.ndarray:
    """Compute the mahalanobis distance between mu[j] and N(mu[i], sigma[i]) for all i and j

    Note: The distance is not symmetric and dist[i, i] = inf for all i

    Args:
        mu (np.ndarray): Mean of the normal distributions
            Shape: (n, d), dtype: float32
        sigma_inv (np.ndarray): Inverse Covariance matrices (precision matrices)
            Shape: (n, d, d), dtype: float32
        thresh (float): Threshold on the l1 distance when to skip the computation of the
            mahalanobis distance and set it to inf instead

    Returns:
        np.ndarray: Mahalanobis distance between each distribution
            Shape: (n, n)
    """
    n = mu.shape[0]
    dist = np.full((n, n), np.inf, dtype=np.float32)
    for i in numba.prange(n):  # pylint: disable=not-an-iterable
        for j in numba.prange(i + 1, n):  # pylint: disable=not-an-iterable
            if np.abs(mu[i] - mu[j]).sum() > thresh:
                continue
            delta_mu = mu[i] - mu[j]
            dist[i, j] = np.sqrt(delta_mu @ sigma_inv[i] @ delta_mu)
            dist[j, i] = np.sqrt(delta_mu @ sigma_inv[j] @ delta_mu)
    return dist


@numba.njit()
def _fast_valid(dist: np.ndarray, min_dist: float) -> np.ndarray:
    """Compute a validity mask

    The validity mask is a non-unique boolean mask defining a subset of particles where
    all pairwise distances are greater than min_dist

    Args:
        dist (np.ndarray): Distance between particles
            Shape: (n, n), dtype: float32
        min_dist (float): Minimum distance to keep

    Returns:
        np.ndarray: Validity mask
            Shape: (N,), dtype: bool
    """
    # XXX: Could probably be improved to keep the subset of highest cardinal
    n = dist.shape[0]
    valid = np.full((n,), True)

    for i in range(n):
        if not valid[i]:
            continue
        for j in range(n):
            if not valid[j]:
                continue
            if dist[i, j] < min_dist:
                valid[j] = False

    return valid


@numba.njit
def _fast_draw(samples: np.ndarray, size: Tuple[int, int], weights: np.ndarray) -> np.ndarray:
    """Generate a black image where the samples are added with their weights

    Args:
        samples (np.ndarray): Samples for each particles.
            Shape (n, m, 2), dtype: float32
        size (Tuple[int, int]): Size of the image generated
        weights (np.ndarray): Weight for each particle

    Returns:
        np.ndarray: The generated image
    """
    image = np.zeros(size, dtype=np.float32)

    for particle in range(len(samples)):  # pylint: disable=consider-using-enumerate
        for sample in samples[particle]:
            if sample.min() < 0 or sample[0] >= size[0] or sample[1] >= size[1]:
                continue

            image[sample[0], sample[1]] += weights[particle]

    return image


def gmm_pdf(
    indices: torch.Tensor, mu: torch.Tensor, sigma_inv: torch.Tensor, weight: torch.Tensor, thresh: float
) -> torch.Tensor:
    """Fast computation of the GMM pdf given mu, sigma_inv and weights

    We do not compute a true pdf, but rather a weighted sum of gaussian pdfs
    (scaled so that the max is at one for each gaussian no matter the covariance)

    For our use case, torch version uses too much memory (it keeps for each indice and each component the prob)
    moreover it computes the true pdf which is more expensive.

    Args:
        indices (torch.Tensor): Indices where to compute the pdf
            Shape: (n, d), dtype: float32
        mu (torch.Tensor): Mean of the normal distributions
            Shape: (m, d), dtype: float32
        sigma_inv (torch.Tensor): Inverse Covariance matrices (precision matrices)
            Shape: (m, d, d), dtype: float32
        weight (torch.Tensor): Weight of each gaussian components
            Shape: (m,), dtype: float32
        thresh (float): Threshold on the l1 distance when to skip computation

    Returns:
        torch.Tensor: Gaussian pdf for the given indices
            Shape: (n,), dtype: float32
    """
    # Move to GPU if possible
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    indices = indices.to(device)
    mu = mu.to(device)
    sigma_inv = sigma_inv.to(device)
    weight = weight.to(device)

    n = indices.shape[0]
    m = mu.shape[0]
    pdf = torch.full((n,), 0, dtype=torch.float32, device=device)
    for i in range(m):  # pylint: disable=not-an-iterable
        delta = indices - mu[i]
        valid = delta.abs().sum(dim=1) < thresh
        delta = delta[valid]
        pdf[valid] += weight[i] * torch.exp(-0.5 * delta[:, None] @ sigma_inv[i] @ delta[..., None])[:, 0, 0]
    return pdf


@dataclasses.dataclass
class GaussianParticlesConfig:
    n: int
    min_std: float
    max_std: float
    min_dist: float = 0


class GaussianParticles:
    """Handle multiple gaussian particles

    Each particle is defined by a position, deviation and angle (mu, std, theta).

    mu is the 2d center of the particle.
    std is standart deviation along the axis of the ellipse.
    theta is the rotation angle from the horizontal axis.

    Attributes:
        size (Tuple[int, int]): Size of the image generated
        mu (torch.Tensor): Positions of the spots
            Shape: (n, 2), dtype: float32
        std (torch.Tensor): Uncorrelated stds
            Shape: (n, 2), dtype: float32
        theta (torch.Tensor): Rotation of each spot
            Shape: (n,), dtype: float32
        weight (torch.Tensor): Weight of each spot (proportional to intensity)
            Shape: (n,), dtype: float32
    """

    def __init__(self, n: int, mask: torch.Tensor, min_std: float, max_std: float):
        """Constructor

        Args:
            n (int): Number of particles to generate. Note that due to random masking, the true number
                of particles generated is not exactly n (and is stored in self._n)
            mask (torch.Tensor): Boolean mask where to generate particles in the image
                self.size is extracted from it.
                Shape: (H, W), dtype: bool
            min_std, max_std (float): Minimum/Maximum std. Stds are generated uniformly between these values

        """
        self.size = (mask.shape[0], mask.shape[1])

        mask_proportion = mask.sum() / mask.numel()
        self.mu = torch.rand(int(n / mask_proportion.item()), 2) * torch.tensor(self.size)
        self.mu = self.mu[mask[self.mu[:, 0].long(), self.mu[:, 1].long()]]

        self._n = self.mu.shape[0]

        self.weight = torch.ones(self._n)
        self.std = min_std + torch.rand(self._n, 2) * (max_std - min_std)
        self.theta = torch.rand(self._n) * torch.pi

        self.build_distribution()

    def filter_close_particles(self, min_dist: float) -> None:
        """Drop too close particles based on mahalanobis distance

        Args:
            min_dist (float): Minimum mahalanobis distance between two particles
        """
        dist = _fast_mahalanobis_pdist(
            self.mu.numpy(),
            self._distribution.precision_matrix.contiguous().numpy(),
            min_dist * self.std.max().item() * 2,
        )
        valid = _fast_valid(dist, min_dist)

        self.mu = self.mu[valid]

        self._n = self.mu.shape[0]
        self.weight = self.weight[valid]
        self.std = self.std[valid]
        self.theta = self.theta[valid]

        self.build_distribution()
        print(f"Filtered particles from {valid.shape[0]} to {self._n}")

    def build_distribution(self):
        """Rebuild the distributions

        To be called each time a modification is made to mu, std or theta
        """
        rot = torch.empty((self._n, 2, 2), dtype=torch.float32)
        rot[:, 0, 0] = torch.cos(self.theta)
        rot[:, 0, 1] = torch.sin(self.theta)
        rot[:, 1, 0] = -rot[:, 0, 1]
        rot[:, 1, 1] = rot[:, 0, 0]

        sigma = torch.zeros((self._n, 2, 2), dtype=torch.float32)
        sigma[:, 0, 0] = self.std[:, 0].pow(2)
        sigma[:, 1, 1] = self.std[:, 1].pow(2)
        sigma = rot @ sigma @ rot.permute(0, 2, 1)
        sigma = (sigma + sigma.permute(0, 2, 1)) / 2  # prevent some floating error leading to non inversible matrix

        self._distribution = torch.distributions.MultivariateNormal(self.mu, sigma)
        # NOTE: A gmm could be created with torch.distributions.MixtureSameFamily(
        #   torch.distributions.Categorical(self.weight * self.std.prod(dim=-1)),
        #   self._distribution
        # )
        # This is not faster nor easier to use
        # torch.distributions.MixtureSameFamily(
        #   torch.distributions.Categorical(self.weight * self.std.prod(dim=-1)),
        #   self._distribution
        # )

    def draw_sample(self, n=20000, blur=0.0) -> torch.Tensor:
        """Draw an image of the particles

        The generation starts from a black image where we add at each sample location the weights of its particle.
        A blurring process can be added to smooth the results (With smaller n).

        Args:
            n (int): Number of samples by particles
            blur (float): std of the blurring process
                Default: 0.0 (No blurring)

        Returns:
            torch.Tensor: Image of the particles
                Shape: (H, W), dtype: float32
        """
        samples = self._distribution.sample((n,))  # type: ignore
        samples = samples.round().long().permute(1, 0, 2)  # Shape: self._n, n, 2
        weight = self.weight * self.std.prod(dim=-1)  # By default, smaller gaussian spots have higher intensities.
        image = _fast_draw(samples.numpy(), self.size, weight.numpy())
        if blur > 0:
            image = cv2.GaussianBlur(image, (55, 55), blur, blur)
        return torch.tensor(image) / n

    def draw_poisson(self, dt=100.0, scale=1) -> torch.Tensor:
        """Draw from ground truth with Poisson Shot noise

        Args:
            dt (float): Integration interval
                Default: 100.0
            scale (int): Down scaling to compute true pdf
                Default: 1

        Returns:
            torch.Tensor: Image of the particles
                Shape: (H, W), dtype: float32
        """
        return torch.distributions.Poisson(dt * self.draw_truth(scale)).sample((1,))[0] / dt

    def draw_truth(self, scale=1) -> torch.Tensor:
        """Draw the ground truth image, where the intensities are the pdf of the mixture of gaussians

        I(x) = \\sum_i w_i N(x; \\mu_i, \\Sigma_i)

        Args:
            scale (int): Downscale the image to make the computation faster
                Default: 1

        Returns:
            torch.Tensor: Pdf of the particles
                Shape: (H // scale, W // scale), dtype: float32
        """
        indices = torch.tensor(np.indices((self.size[0] // scale, self.size[1] // scale)))
        indices = indices.permute(1, 2, 0).to(torch.float32) * scale

        truth = gmm_pdf(
            indices.reshape(-1, 2), self.mu, self._distribution.precision_matrix, self.weight, self.std.max().item() * 5
        ).cpu()  # Limit to 5 times the std

        return torch.nn.functional.interpolate(
            truth.reshape((1, 1, self.size[0] // scale, self.size[1] // scale)), size=self.size, mode="bilinear"
        )[0, 0]

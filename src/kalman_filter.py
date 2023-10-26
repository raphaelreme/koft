import dataclasses
from typing import Optional

import torch
import torch.linalg

# Note on runtime:
# Computations can be faster using cholesky decomposition and cholesky_solve
# But in real cases, dim_z is small (limiting the benefits of cholesky) and we usually need to
# compare each state to all measurements (Involve a cholesky solve for each measurements).
# We could try to implement a numba fast version with a threshold to compare only with few close measurements.
# But the simple approach of computing once the inverse covariance, and then only performing matmul works pretty well
# (And can be sent to GPU)


@dataclasses.dataclass
class GaussianState:
    """Gaussian state in Kalman Filter

    We emphasize that the mean is at least 2d (dim_x, 1).

    Attributes:
        mean (torch.Tensor): Mean of the distribution
            Shape: (*, dim, 1)
        covariance (torch.Tensor): Covariance of the distribution
            Shape: (*, dim, dim)
        precision (Optional[torch.Tensor]): Inverse covariance matrix
            Shape: (*, dim, dim)
    """

    mean: torch.Tensor
    covariance: torch.Tensor
    precision: Optional[torch.Tensor] = None


class KalmanFilter:
    """Kalman filter implementation in PyTorch

    Estimate the true state x_k ~ N(mu_k, P_k) with the following hidden markov model

    x_{k+1} = F x_k + N(0, Q)
    z_k = H x_k + N(0, R)

    This implementation is compatible with batch computations (computing several kalman filters at the same time)

    .. note:

        In order to allow full flexibility on batch computation, the user has to be precise on the shape of its tensors
        1d vector should always be 2 dimensional and vertical. Check the documentation

    Attributes:
        process_matrix (torch.Tensor): State transition matrix (F)
            Shape: (*, dim_x, dim_x)
        measurement_matrix (torch.Tensor): Projection matrix (H)
            Shape: (*, dim_z, dim_x)
        process_noise (torch.Tensor): Uncertainty on the process (Q)
            Shape: (*, dim_x, dim_x)
        measurement_noise (torch.Tensor): Uncertainty on the measure (R)
            Shape: (*, dim_z, dim_z)

    """

    def __init__(
        self,
        process_matrix: torch.Tensor,
        measurement_matrix: torch.Tensor,
        process_noise: torch.Tensor,
        measurement_noise: torch.Tensor,
    ) -> None:
        self.process_matrix = process_matrix
        self.measurement_matrix = measurement_matrix
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        self._alpha_sq = 1.0  # Memory fadding KF

    @property
    def state_dim(self) -> int:
        return self.process_matrix.shape[0]

    @property
    def measure_dim(self) -> int:
        return self.measurement_matrix.shape[0]

    def predict(
        self,
        state: GaussianState,
        process_matrix: Optional[torch.Tensor] = None,
        process_noise: Optional[torch.Tensor] = None,
    ) -> GaussianState:
        """Prediction from the given state

        Use the process model x_{k+1} = F x_k + N(0, Q) to compute the prior on the future state.
        Support batch computation: you can provide multiple models (F, Q) or/and multiple states.
        You just need to ensure that shapes are broadcastable.

        Args:
            state (GaussianState): Current state estimation
            process_matrix (Optional[torch.Tensor]): Overwrite the default transition matrix
                Shape: (*, dim_x, dim_x)
            process_noise (Optional[torch.Tensor]): Overwrite the default process noise)
                Shape: (*, dim_x, dim_x)

        Returns:
            GaussianState: Prior on the next state

        """
        if process_matrix is None:
            process_matrix = self.process_matrix
        if process_noise is None:
            process_noise = self.process_noise

        mean = process_matrix @ state.mean
        covariance = (
            self._alpha_sq * process_matrix @ state.covariance @ process_matrix.transpose(-1, -2) + process_noise
        )

        return GaussianState(mean, covariance)

    def project(
        self,
        state: GaussianState,
        measurement_matrix: Optional[torch.Tensor] = None,
        measurement_noise: Optional[torch.Tensor] = None,
        precompute_precision=True,
    ) -> GaussianState:
        """Project the current state (usually the prior) onto the measurement space

        Use the measurement equation: z_k = H x_k + N(0, R).
        Support batch computation: You can provide multiple measurements or/and multiple states.
        You just need to ensure that shapes are broadcastable.

        Args:
            state (GaussianState): Current state estimation (Usually the results of `predict`)
            measurement_matrix (Optional[torch.Tensor]): Overwrite the default projection matrix
                Shape: (*, dim_z, dim_x)
            measurement_noise (Optional[torch.Tensor]): Overwrite the default projection noise)
                Shape: (*, dim_z, dim_z)
            precompute_precision (bool): Precompute precision matrix (inverse covariance)
                Done once to prevent more computations
                Default: True

        Returns:
            GaussianState: Prior on the next state

        """
        if measurement_matrix is None:
            measurement_matrix = self.measurement_matrix
        if measurement_noise is None:
            measurement_noise = self.measurement_noise

        mean = measurement_matrix @ state.mean
        covariance = measurement_matrix @ state.covariance @ measurement_matrix.transpose(-1, -2) + measurement_noise

        return GaussianState(
            mean,
            covariance,
            covariance.inverse().contiguous()  # Cholesky inverse is usually slower with small dimensions
            if precompute_precision
            else None,
        )

    def update(
        self,
        state: GaussianState,
        measure: torch.Tensor,
        projection: Optional[GaussianState] = None,
        measurement_matrix: Optional[torch.Tensor] = None,
        measurement_noise: Optional[torch.Tensor] = None,
    ) -> GaussianState:
        """Compute the posterior estimation by integrating a new measure into the state

        Support batch computation: You can provide multiple measurements or/and multiple states.
        You just need to ensure that shapes are broadcastable.

        Args:
            state (GaussianState): Current state estimation (Usually the results of `predict`)
            measure (torch.Tensor): State measure (z_k)
                Shape: (*, dim_z, 1)
            projection (GaussianState): Precomputed projection if any
            measurement_matrix (Optional[torch.Tensor]): Overwrite the default projection matrix
                Shape: (*, dim_z, dim_x)
            measurement_noise (Optional[torch.Tensor]): Overwrite the default projection noise)
                Shape: (*, dim_z, dim_z)

        Returns:
            GaussianState: Prior on the next state

        """
        if measurement_matrix is None:
            measurement_matrix = self.measurement_matrix
        if measurement_noise is None:
            measurement_noise = self.measurement_noise
        if projection is None:
            projection = self.project(state, measurement_matrix, measurement_noise)

        residual = measure - projection.mean

        if projection.precision is None:  # Old version using cholesky and solve to prevent the inverse computation
            # Find K without inversing S but by solving the linear system SK^T = (PH^T)^T
            chol_decomposition, _ = torch.linalg.cholesky_ex(projection.covariance)
            kalman_gain = torch.cholesky_solve(
                measurement_matrix @ state.covariance.transpose(-1, -2), chol_decomposition
            ).transpose(-1, -2)
        else:
            kalman_gain = state.covariance @ measurement_matrix.transpose(-1, -2) @ projection.precision

        mean = state.mean + kalman_gain @ residual
        covariance = state.covariance - kalman_gain @ measurement_matrix @ state.covariance

        return GaussianState(mean, covariance)

import dataclasses
import enum
from typing import Collection, Iterable, List, Optional

import filterpy.common  # type: ignore
import numpy as np
import torch
import tqdm  # type: ignore

import byotrack
import pylapy

from .kalman_filter import KalmanFilter, GaussianState
from .greedy_lap import greedy_assignment_solver


class PartialTrack:
    """Partial track class

    Partial tracks are created for each unlinked detections, and then updated with following detections.
    It requires CONFIRMED_AT consecutive detections to confirm the tracks (INITIATED => CONFIRMED). If a miss detection
    occurs, it deletes it (INITIATED => DELETED).

    Once confirmed, it is resilient to miss detections, waiting MAX_NON_MEASURE frames before ending the track
    (CONFIRMED => ENDED)

    Will also store the kalman data for analysis.
    """

    MAX_NON_MEASURE = 3
    CONFIRMED_AT = 3

    class TrackState(enum.IntEnum):
        INITIATED = 0
        CONFIRMED = 1
        ENDED = 2
        DELETED = 3

    def __init__(
        self,
        track_id: int,
        start: int,
        mean: torch.Tensor,
        covariance: torch.Tensor,
        measure: torch.Tensor,
        points=(0, 1),
    ) -> None:
        self._points = points  # Points data in state
        self.track_id = track_id
        self.start = start
        self.track_state = PartialTrack.TrackState.INITIATED
        self.last_measurement = 0
        self._mean = [mean.clone()]
        self._covariance = [covariance.clone()]
        self._measure = [measure.clone()]

    def __len__(self) -> int:
        return len(self._mean) - self.last_measurement

    def is_active(self) -> bool:
        return self.track_state < 2

    def update(self, mean: torch.Tensor, covariance: torch.Tensor, measure: Optional[torch.Tensor]) -> None:
        """Should be called only if the track is active"""
        self._mean.append(mean.clone())
        self._covariance.append(covariance.clone())

        if measure is None:  # Not associated with a measure
            self._measure.append(torch.full_like(self._measure[-1], torch.nan))
            self.last_measurement += 1

            if self.track_state == PartialTrack.TrackState.INITIATED:
                self.track_state = PartialTrack.TrackState.DELETED

            elif self.last_measurement >= self.MAX_NON_MEASURE:  # Could also check the width of the state covariance
                self.track_state = PartialTrack.TrackState.ENDED

            return

        self._measure.append(measure.clone())
        self.last_measurement = 0

        if self.track_state == PartialTrack.TrackState.INITIATED:
            if len(self) >= self.CONFIRMED_AT:
                self.track_state = PartialTrack.TrackState.CONFIRMED

    @property
    def points(self) -> torch.Tensor:
        return torch.cat([mean[None, self._points, 0] for mean in self._mean[: len(self)]])


def constant_kalman_filter(measurement_std: torch.Tensor, process_std: torch.Tensor, dim=2, order=1) -> KalmanFilter:
    """Create a constant Velocity/Acceleration/Jerk Kalman Filter

    Create a kalman filter with a state containing the positions on each dimension (x, y, z, ...)
    with their derivatives up to `order`. The order-th derivatives are supposed constant.

    Let x be the positions for each dim and x^i the i-th derivatives of these positions
    Prediction follows:
    x^i_{t+1} = x^i_t + x^{i+1}_t, for i < order
    x^order_{t+1} = x^order_t

    Args:
        measurement_std (torch.Tensor): Std of the measurements
            99.7% of measurements should fall within 3 std of the true position
            Shape: Broadcastable to dim, dtype: float64
        process_std (torch.Tensor): Process noise, a typical value is maximum diff between two consecutive
            order-th derivative. (Eg: for constant velocity -> Maximum acceleration between two frames)
            Shape: Broadcastable to dim, dtype: float64
        dim (int): Dimension of the motion (1d, 2d, 3d, ...)
            Default: 2
        order (int): Order of the filer (The order-th derivatives are constants)
            Default: 1 (Constant velocity)

    """
    measurement_std = torch.broadcast_to(measurement_std, (dim,))
    process_std = torch.broadcast_to(process_std, (dim,))

    state_dim = (order + 1) * dim

    # Measurement model
    # We only measure the positions
    # Noise is independent and can have a different value in each direction
    measurement_matrix = torch.eye(dim, state_dim)
    measurement_noise = torch.eye(dim) * measurement_std**2

    # Process
    # Constant model
    # Noise in velocity estimation (which induce a noise in position estimation)
    process_matrix = torch.eye(state_dim) + torch.tensor(np.eye(state_dim, k=dim)).to(torch.float32)

    if order == 0:
        process_noise = torch.eye(state_dim) * process_std**2
    else:
        process_noise = torch.tensor(
            filterpy.common.Q_discrete_white_noise(order + 1, block_size=dim, order_by_dim=False)
        ).to(torch.float32) * torch.cat([process_std**2] * (order + 1))
    # process_noise = torch.tensor(
    #     filterpy.common.Q_discrete_white_noise(order + 1, block_size=dim, order_by_dim=False)
    # ).to(torch.float32) * torch.cat([process_std**2] * (order + 1))

    return KalmanFilter(process_matrix, measurement_matrix, process_noise, measurement_noise)


class Dist(enum.Enum):
    MAHALANOBIS = "mahalanobis"
    EUCLIDIAN = "euclidian"
    LIKELIHOOD = "likelihood"


class Method(enum.Enum):
    """Matching methods

    Opt: GDM with Jonker-volgenant algorithm (Linear assignement solver)
        Can be smooth thresholding or hard
    Greedy: Takes the best matches iteratively
    """

    OPT_SMOOTH = "opt_smooth"
    OPT_HARD = "opt_hard"
    GREEDY = "greedy"


@dataclasses.dataclass
class MatchingConfig:
    thresh: float
    dist: Dist = Dist.MAHALANOBIS
    method: Method = Method.OPT_SMOOTH


class SimpleKalmanTracker(byotrack.Linker):
    """Simple Kalman tracker (SKT)"""

    def __init__(self, kalman_filter: KalmanFilter, match_cfg: MatchingConfig) -> None:
        super().__init__()
        self.kalman_filter = kalman_filter
        self.tracks: List[PartialTrack] = []
        self.active_tracks: List[PartialTrack] = []
        self.state = GaussianState(  # Current state of active tracks
            torch.zeros((0, self.kalman_filter.state_dim, 1)),
            torch.zeros((0, self.kalman_filter.state_dim, self.kalman_filter.state_dim)),
        )

        self.match_cfg = match_cfg

    def run(
        self, video: Iterable[np.ndarray], detections_sequence: Collection[byotrack.Detections]
    ) -> Collection[byotrack.Track]:
        # Reset tracks and states
        self.tracks = []
        self.active_tracks = []
        self.state = GaussianState(
            torch.zeros((0, self.kalman_filter.state_dim, 1)),
            torch.zeros((0, self.kalman_filter.state_dim, self.kalman_filter.state_dim)),
        )  # The first iteration will predict and associate with 0 tracks, leading to no association
        # Thus creating tracks for all detections in the first frame

        for detections in tqdm.tqdm(detections_sequence):
            self.update(detections)

        tracks = []
        for track in self.tracks + self.active_tracks:
            if track.track_state in (track.TrackState.DELETED, track.TrackState.INITIATED):
                continue  # Ignore unconfirmed tracks
            tracks.append(
                byotrack.Track(
                    track.start,
                    track.points,
                    track.track_id,
                )
            )
        return tracks

    def match(self, projection: GaussianState, measures: torch.Tensor) -> torch.Tensor:
        """Match projection with measures using positions

        If velocity measure (KOFT) is available, we do not use it here (Even if it could be better)

        Args:
            projection (GaussianState): Projection for all tracks. Only supports 2D (dim_z = 2 or 4
                if velocities are included). Mean: (n, dim_z, 1), Cov: (n, dim_z, dim_z)
            measures (torch.Tensor): Measures to match with tracks. Only supports 2D. Measures can
                include velocities but it won't be used for matching. (Though could be an easy upgrade)
                Shape: (m, 2, 1) or (m, 4 ,1), dtype: float32

        Returns:
            torch.Tensor: Links between tracks and measures
                Shape: (L, 2), dtype: int32
        """
        dist: torch.Tensor
        thresh: float

        if self.match_cfg.dist in (Dist.MAHALANOBIS, Dist.LIKELIHOOD):
            if projection.precision is None:
                # Register in case someone needs it afterwards (like kf.update)
                projection.precision = projection.covariance.inverse().contiguous()

            # precision = projection.covariance[:, None, :2, :2].inverse()
            precision = projection.precision[:, None, :2, :2]  # Handle 4d projection with speed. (n, 1, 2, 2)
            # We noticed that it is more efficient to use inv(cov)[:2, :2] rather than inv(cov[:2, :2])...
            # Need more investigatation but: This solution is equivalent to consider than the speed prediction
            # is perfect and using covariance between speed and position to quantify the errors on positions
            # precision != torch.linalg.inv(projection.covariance[:, None, :2, :2])

            diff = projection.mean[:, None, :2] - measures[None, :, :2]  # Shape: (n, m, 2, 1)
            dist = diff.mT @ precision @ diff  # Shape: (n, m, 1, 1)
            if self.match_cfg.dist == Dist.MAHALANOBIS:
                dist = dist[..., 0, 0]
                thresh = self.match_cfg.thresh**2  # No need to take the sqrt, let's compare to the sq thresh
            else:  # likelihood
                log_det = -torch.log(torch.det(precision))  # Shape (N, 1)
                # Dist = - log likelihood
                dist = 0.5 * (diff.shape[2] * torch.log(2 * torch.tensor(torch.pi)) + log_det + dist[..., 0, 0])
                thresh = -torch.log(torch.tensor(self.match_cfg.thresh)).item()
        else:  # Euclidian
            dist = torch.cdist(projection.mean[:, :2, 0], measures[:, :2, 0])
            thresh = self.match_cfg.thresh

        if self.match_cfg.method == Method.GREEDY:
            links = greedy_assignment_solver(dist.numpy(), thresh)
        else:
            dist[dist > thresh] = torch.inf
            links = pylapy.LapSolver().solve(
                dist.numpy(),
                float("inf") if self.match_cfg.method == Method.OPT_HARD else thresh,
            )

        return torch.tensor(links.astype(np.int32))

    def update(self, detections: byotrack.Detections):
        prior = self.kalman_filter.predict(self.state)
        projection = self.kalman_filter.project(prior)
        positions = detections.position[..., None].clone()  # Shape m, d, 1

        # Association
        links = self.match(projection, positions)

        # Update linked kalman filter
        posterior = self.kalman_filter.update(
            GaussianState(prior.mean[links[:, 0]], prior.covariance[links[:, 0]]),
            positions[links[:, 1]],
            GaussianState(
                projection.mean[links[:, 0]],
                projection.covariance[links[:, 0]],
                projection.precision[links[:, 0]] if projection.precision is not None else None,
            ),
        )

        # Take prior by default if non-linked
        prior.mean[links[:, 0]] = posterior.mean
        prior.covariance[links[:, 0]] = posterior.covariance
        posterior = prior

        self._handle_tracks(posterior, positions, links, detections.frame_id)

    def _handle_tracks(
        self, posterior: GaussianState, measures: torch.Tensor, links: torch.Tensor, frame_id: int
    ) -> None:
        """Handle tracks to save track data, start new tracks and delete lost ones

        Args:
            posterior (GaussianState): Posterior for all active tracks.
                Mean: (n, dim_x, 1), Cov: (n, dim_x, dim_x)
            measures (torch.Tensor): Measures (Only supports 2D). Measures can include velocities (KOFT)
                Shape: (m, 2, 1) or (m, 4 ,1), dtype: float32
            links (torch.Tensor): Links between tracks and measures
                Shape: (L, 2), dtype: int32
            frame_id (int): Current frame id

        """

        # Save both state and measurement in partial tracks.
        i_to_j = torch.full((len(self.active_tracks),), -1, dtype=torch.int32)
        i_to_j[links[:, 0]] = links[:, 1]
        active_mask = torch.full((len(self.active_tracks),), False)
        still_active = []
        for i, track in enumerate(self.active_tracks):
            j = i_to_j[i]
            if j == -1:
                track.update(posterior.mean[i], posterior.covariance[i], None)
            else:
                track.update(posterior.mean[i], posterior.covariance[i], measures[j])

            if track.is_active():
                still_active.append(track)
                active_mask[i] = True
            else:
                self.tracks.append(track)

        # Restrict posterior states to active tracks
        posterior = GaussianState(posterior.mean[active_mask], posterior.covariance[active_mask])

        # Create new track for every unmatch detection
        measures[links[:, 1]] = torch.nan
        unmatched_measures = measures[~torch.isnan(measures).squeeze().any(dim=-1)]

        if not unmatched_measures.numel():
            self.state = posterior
            self.active_tracks = still_active
            return

        # Inital state:
        # Initialize with no prior on the position (set at 0 with an inf uncertainty)
        # Initialize with a 0 prior velocity/acceleration/jerk with 5x process_std uncertainty
        # Then update with the measured state (and ensure that the position uncertainty is correct)

        # Init covariance with uncorrelated states, with 5 process noise except on x,y
        initial_covariance = 5 * self.kalman_filter.process_noise * torch.eye(self.kalman_filter.state_dim)
        initial_covariance[:2, :2] = torch.eye(2) * 1e20  # Unknown position
        unmatched_state = GaussianState(
            torch.zeros((unmatched_measures.shape[0], self.kalman_filter.state_dim, 1)),
            torch.cat([initial_covariance[None]] * unmatched_measures.shape[0]),
        )
        # Update with measures
        unmatched_state = self.kalman_filter.update(unmatched_state, unmatched_measures)
        # Correct for floating points errors with inf uncertainty
        unmatched_state.covariance[:, :2, :2] = self.kalman_filter.measurement_noise[:2, :2]

        # Create a new active track for each new state created
        for i in range(unmatched_measures.shape[0]):
            still_active.append(
                PartialTrack(
                    len(self.tracks) + len(still_active),
                    frame_id,
                    unmatched_state.mean[i],
                    unmatched_state.covariance[i],
                    unmatched_measures[i],
                )
            )

        # State is the posterior for all active tracks (concatenation of new tracks with old kept ones)
        self.active_tracks = still_active
        self.state = GaussianState(
            torch.cat((posterior.mean, unmatched_state.mean)),
            torch.cat((posterior.covariance, unmatched_state.covariance)),
        )

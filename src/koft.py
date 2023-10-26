import enum
from typing import Collection, Iterable, Sequence

import filterpy.common  # type: ignore
import numpy as np
import torch
import tqdm  # type: ignore

import byotrack

from .kalman_filter import KalmanFilter, GaussianState
from .skt import MatchingConfig, SimpleKalmanTracker
from .optical_flow import OptFlow


def constant_koft_filter(
    pos_std: torch.Tensor, vel_std: torch.Tensor, process_std: torch.Tensor, dim=2, order=1
) -> KalmanFilter:
    """Create a constant Velocity/Acceleration/Jerk Kalman Filter with pos and velocity measurements

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

    assert order >= 1, "Velocity is measured and has to be set"

    measurement_std = torch.cat((torch.broadcast_to(pos_std, (dim,)), torch.broadcast_to(vel_std, (dim,))))
    process_std = torch.broadcast_to(process_std, (dim,))

    measure_dim = 2 * dim
    state_dim = (order + 1) * dim

    # Measurement model
    # We measure position and velocity
    # Noise is independent and can have a different value in each direction
    measurement_matrix = torch.eye(measure_dim, state_dim)
    measurement_noise = torch.eye(measure_dim) * measurement_std**2

    # Process
    # Constant model
    # Noise in velocity estimation (which induce a noise in position estimation)
    process_matrix = torch.eye(state_dim) + torch.tensor(np.eye(state_dim, k=dim)).to(torch.float32)
    process_noise = torch.tensor(
        filterpy.common.Q_discrete_white_noise(order + 1, block_size=dim, order_by_dim=False)
    ).to(torch.float32) * torch.cat([process_std**2] * (order + 1))

    return KalmanFilter(process_matrix, measurement_matrix, process_noise, measurement_noise)


class SingleUpdateKOFTracker(SimpleKalmanTracker):
    """Kalman and Optical Flow tracker with a single update

    Update velocities only for matched tracks and measyre velocity from detected positions
    """

    __ALWAYS_UPDATE_VEL = False

    def __init__(self, kalman_filter: KalmanFilter, opt_flow: OptFlow, match_cfg: MatchingConfig) -> None:
        super().__init__(kalman_filter, match_cfg)
        self.opt_flow = opt_flow
        self.flow = np.zeros((1, 1, 2))

    def run(
        self, video: Iterable[np.ndarray], detections_sequence: Collection[byotrack.Detections]
    ) -> Collection[byotrack.Track]:
        assert isinstance(video, Sequence), "Only indexable videos are supported"

        # Reset tracks and states
        self.tracks = []
        self.active_tracks = []
        self.state = GaussianState(
            torch.zeros((0, self.kalman_filter.state_dim, 1)),
            torch.zeros((0, self.kalman_filter.state_dim, self.kalman_filter.state_dim)),
        )

        # Extract initial frame and prepare for optflow
        frame = video[next(iter(detections_sequence)).frame_id][..., 0]
        src = self.opt_flow.prepare(frame)

        for detections in tqdm.tqdm(detections_sequence):
            try:
                # We could compute flow from t-1 to t, or t-1 to t+1
                # But it is much better to compute flow from
                # frame = video[max(detections.frame_id - 1, 0)]
                # src = self.opt_flow.prepare(frame)
                # frame = video[detections.frame_id][..., 0]
                frame = video[detections.frame_id + 1][..., 0]
            except IndexError:
                pass

            dest = self.opt_flow.prepare(frame)
            self.flow = self.opt_flow.calc(src, dest)  # / 2 if computed from t-1 to t+1

            self.update(detections)

            src = dest

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

    def update(self, detections: byotrack.Detections):
        prior = self.kalman_filter.predict(self.state)
        projection = self.kalman_filter.project(prior)
        positions = detections.position[..., None].clone()  # Shape m, d, 1

        # Measures = positions + velocities
        velocities = self.opt_flow.flow_at(self.flow, positions[..., 0].numpy().astype(np.float64), self.opt_flow.scale)
        measures = torch.cat([positions, torch.tensor(velocities[..., None]).to(torch.float32)], dim=1)

        # Association
        links = self.match(projection, measures)

        if self.__ALWAYS_UPDATE_VEL:  # Single update for everyone even unmatched tracks (updated with inf pos cov)
            # Add measures for unlinked state
            prior_velocities = self.opt_flow.flow_at(
                self.flow, prior.mean[:, :2, 0].numpy().astype(np.float64), self.opt_flow.scale
            )
            all_measures = torch.cat(
                [prior.mean[:, :2], torch.tensor(prior_velocities[..., None]).to(torch.float32)], dim=1
            )
            all_measures[links[:, 0]] = measures[links[:, 1]]

            # For unmatched tracks, uncertainty on measurements (which is the prior here) is set to inf
            # Note that dropping this helps => Future investigation here
            cov = projection.covariance.clone()
            projection.covariance[:, 0, 0] = torch.inf
            projection.covariance[:, 1, 1] = torch.inf
            projection.covariance[links[:, 0]] = cov[links[:, 0]]
            projection.precision = None

            # Update linked kalman filter
            posterior = self.kalman_filter.update(
                prior,
                all_measures,
                projection,
            )
        else:  # Classic single update
            # Update linked kalman filter
            posterior = self.kalman_filter.update(
                GaussianState(prior.mean[links[:, 0]], prior.covariance[links[:, 0]]),
                measures[links[:, 1]],
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

        self._handle_tracks(posterior, measures, links, detections.frame_id)


class OptFlowExtraction(enum.Enum):
    """Extraction of optical flow from different positions"""

    DETECTED = 0
    POSTERIOR = 1
    PRIOR = 2


class TwoUpdateKOFTracker(SingleUpdateKOFTracker):
    """Kalman and Optical Flow tracker"""

    def __init__(
        self,
        kalman_filter: KalmanFilter,
        opt_flow: OptFlow,
        match_cfg: MatchingConfig,
        opt_flow_at=OptFlowExtraction.POSTERIOR,
        always_update_vel=True,
    ) -> None:
        super().__init__(kalman_filter, opt_flow, match_cfg)
        self.opt_flow_at = opt_flow_at
        self.always_update_vel = always_update_vel

    def update(self, detections: byotrack.Detections):
        projection = self.kalman_filter.project(
            self.state,
            # self.kalman_filter.measurement_matrix[:2],  # Let's also project velocity (useful for matching)
            # self.kalman_filter.measurement_noise[:2, :2],
        )

        positions = detections.position[..., None].clone()  # Shape m, d, 1

        # Association
        links = self.match(projection, positions)

        # First update (Update with associate detections positions)
        posterior = self.kalman_filter.update(
            GaussianState(self.state.mean[links[:, 0]], self.state.covariance[links[:, 0]]),
            positions[links[:, 1]],
            GaussianState(
                projection.mean[links[:, 0], :2],
                projection.covariance[links[:, 0], :2, :2],
                None,  # /!\ inv(cov[:2,:2]) != inv(cov)[:2, :2]
            ),
            self.kalman_filter.measurement_matrix[:2],
            self.kalman_filter.measurement_noise[:2, :2],
        )

        # Compute velocities
        velocities_measured = torch.tensor(  # Measured velocities
            self.opt_flow.flow_at(self.flow, positions[..., 0].numpy().astype(np.float64), self.opt_flow.scale)
        )[..., None].to(torch.float32)

        if self.opt_flow_at == OptFlowExtraction.DETECTED:
            velocities = velocities_measured[links[:, 1]]
        elif self.opt_flow_at == OptFlowExtraction.POSTERIOR:
            velocities = torch.tensor(
                self.opt_flow.flow_at(
                    self.flow, posterior.mean[..., :2, 0].numpy().astype(np.float64), self.opt_flow.scale
                )
            )[..., None].to(torch.float32)
            velocities_measured[links[:, 1]] = velocities
        else:  # Prior
            velocities = torch.tensor(
                self.opt_flow.flow_at(
                    self.flow, projection.mean[links[:, 0], :2, 0].numpy().astype(np.float64), self.opt_flow.scale
                )
            )[..., None].to(torch.float32)
            velocities_measured[links[:, 1]] = velocities

        # Update matched tracks with velocities
        posterior = self.kalman_filter.update(
            posterior,
            velocities,
            None,
            self.kalman_filter.measurement_matrix[2:],
            self.kalman_filter.measurement_noise[2:, 2:],
        )

        measures = torch.cat([positions, velocities_measured], dim=1)

        if self.always_update_vel:
            velocities = torch.tensor(
                self.opt_flow.flow_at(
                    self.flow, projection.mean[:, :2, 0].numpy().astype(np.float64), self.opt_flow.scale
                )
            )[..., None].to(torch.float32)
            self.state = self.kalman_filter.update(  # Update unmatched tracks with velocities
                self.state,
                velocities,
                None,
                self.kalman_filter.measurement_matrix[2:],
                self.kalman_filter.measurement_noise[2:, 2:],
            )

        # Take prior by default if non-linked, else posterior
        self.state.mean[links[:, 0]] = posterior.mean
        self.state.covariance[links[:, 0]] = posterior.covariance

        self._handle_tracks(self.state, measures, links, detections.frame_id)

        self.state = self.kalman_filter.predict(self.state)

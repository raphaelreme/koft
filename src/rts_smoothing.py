import filterpy  # type: ignore
import filterpy.common  # type: ignore
import filterpy.kalman  # type: ignore
import numpy as np
import tqdm

# Old implem of KF with filterpy. Useful for RTS smoothing not yet implemented in our fast torch kf


def create_cvkf(R, Q):
    """Constant velocity motion model with R the measurement noise in pixels and Q the process noise

    R is the std on spatial measurement (99.7% of measurement should fall within 3R of the true position)
    Q is the std on speed (Usually set as how much the constant speed can move Q ~= max delta speed)
    """
    kf = filterpy.kalman.KalmanFilter(dim_x=4, dim_z=2)

    # Process constant velocity
    kf.H = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
    kf.F = np.array(
        [
            [1.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )

    # Measurement noise:
    # independent x and y

    kf.R *= R**2  # 2.5 pixels errors => 99.7% < 7.5 pixels

    # Process noise
    # Assuming discret process noise with a max delta on the derivative of 1.5 pixels/frame

    kf.Q = (
        np.array(
            [
                [0.25, 0.0, 0.5, 0.0],
                [0.0, 0.25, 0.0, 0.5],
                [0.5, 0.0, 1.0, 0.0],
                [0.0, 0.5, 0.0, 1.0],
            ]
        )
        * Q**2
    )

    # Initial belief -> middle of the video without speed
    kf.x = np.array([[512.0, 512.0, 0.0, 0.0]]).T
    kf.P = np.array(
        [
            [250**2, 0.0, 0.0, 0.0],
            [0.0, 250**2, 0.0, 0.0],
            [0.0, 0.0, 2.0**2, 0.0],
            [0.0, 0.0, 0.0, 2.0**2],
        ]
    )

    return kf


def rts_smoothing(measured_positions: np.ndarray, kf_builder):
    """Smooth a track matrix using RTS + CVKF

    Args:
        measured_position (np.ndarray): Positions for tracks (time axis is second)
            Shape: (N, T, 2)
        kf_builder (Callable): Function to initialize a new kalman filter
    """
    estimated_state = np.zeros((measured_positions.shape[0], measured_positions.shape[1], 4))

    for i, z in enumerate(tqdm.tqdm(measured_positions)):
        kf = kf_builder()
        mu, cov, _, _ = kf.batch_filter(z, update_first=True)
        M, _, _, _ = kf.rts_smoother(mu, cov)
        estimated_state[i] = M[:, :4, 0]

    return estimated_state

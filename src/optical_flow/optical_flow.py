from typing import Callable

import cv2
import matplotlib as mpl  # type: ignore
import numpy as np
import scipy.ndimage  # type: ignore


class OptFlow:
    """Optical flow wrapper

    Prepare the frames (scaling, blurring, thresholding), compute the flow and transform points using the flow.

    Attributes:
        method (Callable[[np.ndarray], np.ndarray]): Compute an optical flow given two frames (stored as uint8)
        threshs (Tuple[float, float]): Low and high thresholds (Can be useful to filter out outliers)
        scale (int): Scaling factor. Allows to accelerate and regularize computations
        blur (float): Std of the gaussian blur (To drop the focus on small variations)
    """

    def __init__(self, method: Callable[[np.ndarray, np.ndarray], np.ndarray], threshs=(0.0, 1.0), scale=2, blur=0.0):
        self.method = method
        self.threshs = threshs
        self.scale = scale
        self.blur = blur

    def prepare(self, frame: np.ndarray) -> np.ndarray:
        """Prepare a frame before the OF algorithm

        Args:
            frame (np.ndarray): The frame to prepare
                Shape: (H, W[, -1]), dtype: float64

        Returns:
            np.ndarray: The prepared frame
                Shape: (H / scale, W / scale[, -1]), dtype: float64
        """
        frame = frame.copy()
        frame[frame < self.threshs[0]] = 0  # Filter out low energy loss
        frame[frame > self.threshs[1]] = self.threshs[1]  # Clip neurons firings

        if self.blur > 0:
            frame = cv2.GaussianBlur(frame, (0, 0), self.blur, self.blur)

        return cv2.resize(frame, (0, 0), fx=1 / self.scale, fy=1 / self.scale, interpolation=cv2.INTER_LINEAR)

    def calc(self, source: np.ndarray, destination: np.ndarray) -> np.ndarray:
        """Compute the OF from source to destination

        Depending on the method, channels are accepted or not.

        Args:
            source (np.ndarray): Source frame
                Shape: (H, W[, -1]), dtype: float64
            destination (np.ndarray): Destination frame
                Shape: (H, W[, -1]), dtype: float64

        Returns:
            np.ndarray: Optical flow between the two frames (coordinates: x, y)
                Shape: (H, W, 2), dtype: float64
        """
        return self.method(np.round(source * 255).astype(np.uint8), np.round(destination * 255).astype(np.uint8))

    @staticmethod
    def flow_at(flow: np.ndarray, points: np.ndarray, scale: int) -> np.ndarray:
        """Compute the displacement from flow at points

        Args:
            flow (np.ndarray): Optical flow (coordinates (x,y))
                Shape: (H / scale, W / scale, 2), dtype: float64
            points (np.ndarray): Points where to extract the motion (coordinates (i,i))
                Shape: (N, 2), dtype: float64
            scale (int): Scale of the OF

        Returns:
            np.ndarray: The displacement at points (coordinates (i,j))
                Shape: (N, 2), dtype: float64
        """
        u = scipy.ndimage.map_coordinates(flow[..., 0], points.T / scale)
        v = scipy.ndimage.map_coordinates(flow[..., 1], points.T / scale)

        return np.concatenate((v[:, None], u[:, None]), axis=1) * scale

    def transform(self, flow: np.ndarray, points: np.ndarray) -> np.ndarray:
        """Apply the flow to the given points

        Args:
            flow (np.ndarray): Optical flow (coordinates (x,y))
                Shape: (H, W, 2), dtype: float64
            points (np.ndarray): Points to transform (coordinates (i, j))
                Shape: (N, 2), dtype: float64

        Returns:
            np.ndarray: The dnew points (coordinates (i,i))
                Shape: (N, 2), dtype: float64
        """
        return points + self.flow_at(flow, points, self.scale)


def show_flow_on_video(video, optflow: OptFlow):
    """Display optical flow on a video

    Display the video with control points above that moves following the flow
    """
    hsv = mpl.colormaps["hsv"]
    colors = list(map(lambda x: x[:3], map(hsv, [i / 200 for i in range(200)])))

    display = 0

    frame_id = 0
    frame = video[frame_id][..., 0]
    src = optflow.prepare(frame)
    dest = optflow.prepare(frame)

    points = np.indices(frame.shape)[:, ::20, ::20].reshape(2, -1).transpose(1, 0).astype(np.float32)

    while True:
        draw = np.concatenate((frame[..., None], frame[..., None], frame[..., None]), axis=2)
        for k, (i, j) in enumerate(points):
            cv2.circle(draw, (round(j), round(i)), 3, colors[k % 200])

        # frames[frame_id] = draw

        draw = np.concatenate((np.zeros_like(draw), draw), axis=1)
        if display:
            draw[: dest.shape[0], : dest.shape[1], :] = dest[..., None]
        else:
            draw[: dest.shape[0], : dest.shape[1], :] = src[..., None]

        cv2.imshow("Frame", draw)
        cv2.setWindowTitle("Frame", f"Frame {frame_id}")
        key = cv2.waitKey() & 0xFF

        if key == ord("q"):
            break

        if key == ord("c"):
            display = 1 - display
            continue

        if key == ord("x"):
            frame_id = (frame_id + 1) % len(video)

        if key == ord("w"):
            frame_id = (frame_id - 1) % len(video)

        new_frame = video[frame_id][..., 0]

        src = optflow.prepare(frame)
        dest = optflow.prepare(new_frame)

        flow = optflow.calc(src, dest)

        points = optflow.transform(flow, points)

        frame = new_frame

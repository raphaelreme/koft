from typing import Union, Sequence

import numpy as np
import torch
import tqdm


from .. import optical_flow


class DirectedFlowPropagation:
    """Callable to compute propagation of tracks (Useful for Tracklet Stitching methods (like EMC2))"""

    def __init__(self, optflow: optical_flow.OptFlow):
        self.optflow = optflow

    def __call__(
        self,
        tracks_matrix: torch.Tensor,
        video: Union[Sequence[np.ndarray], np.ndarray],
        forward=True,
        **kwargs,
    ) -> torch.Tensor:
        """Propagate tracks matrix using an optical flow in a single direction

        Args:
            tracks_matrix (torch.Tensor): Tracks data in a single tensor (See `Tracks.tensorize`)
                Shape: (T, N, D), dtype: float32
            video (Sequence[np.ndarray] | np.ndarray): Video
            forward (bool): Forward or backward propagation
                Default: True (Forward)

        Returns:
            torch.Tensor: Estimation of tracks point in a single direction
                Shape: (T, N, D), dtype: float32
        """
        tracks_matrix = tracks_matrix if forward else torch.flip(tracks_matrix, (0,))
        frame_id = lambda i: i if forward else len(tracks_matrix) - i - 1

        propagation_matrix = tracks_matrix.clone()  # (T, N, D)
        valid = ~torch.isnan(propagation_matrix).any(dim=-1)  # (T, N)

        src = video[frame_id(0)][..., 0]
        src = self.optflow.prepare(src)

        for i in tqdm.trange(1, tracks_matrix.shape[0], desc=f"{'Forward' if forward else 'Backward'} propagation"):
            # Compute propagation mask (not valid and has a past)
            propagation_mask = ~valid[i] & ~torch.isnan(propagation_matrix[i - 1]).any(dim=-1)

            dest = video[frame_id(i)][..., 0]
            dest = self.optflow.prepare(dest)

            if propagation_mask.sum() == 0:
                src = dest
                continue  # No propagation to do

            flow = self.optflow.calc(src, dest)
            src = dest

            # Propagate points
            propagation_matrix[i, propagation_mask] = torch.tensor(
                self.optflow.transform(flow, propagation_matrix[i - 1, propagation_mask].numpy())
            )

        return propagation_matrix if forward else torch.flip(propagation_matrix, (0,))

import dataclasses
import enum
import pathlib
from typing import Collection, List

import dacite
import torch
import tqdm  # type: ignore
import yaml  # type: ignore

import byotrack
from byotrack.implementation.detector.wavelet import WaveletDetector
from byotrack.implementation.linker.icy_emht import EMHTParameters, IcyEMHTLinker, Motion
from byotrack.implementation.linker.trackmate.trackmate import TrackMateLinker, TrackMateParameters
from byotrack.implementation.refiner.interpolater import ForwardBackwardInterpolater

from ..data import simulation, dupre
from ..detector import FakeDetector
from ..metrics.detections import DetectionMetric
from ..metrics.tracking import compute_tracking_metrics
from ..skt import constant_kalman_filter, Dist, Method, MatchingConfig, SimpleKalmanTracker, PartialTrack
from ..koft import constant_koft_filter, OptFlowExtraction, SingleUpdateKOFTracker, TwoUpdateKOFTracker
from ..optical_flow import farneback
from ..utils import enforce_all_seeds


class DetectionMethod(enum.Enum):
    WAVELET = "wavelet"
    FAKE = "fake"


@dataclasses.dataclass
class WaveletConfig:
    k: float = 3.0
    scale: int = 1
    min_area: float = 10.0


@dataclasses.dataclass
class FakeConfig:
    fpr: float = 0.1  # Bad detection rate
    fnr: float = 0.2  # Miss detection rate
    measurement_noise: float = 1.0


@dataclasses.dataclass
class DetectionConfig:
    detector: DetectionMethod
    wavelet: WaveletConfig
    fake: FakeConfig

    def create_detector(self, mu: torch.Tensor) -> byotrack.Detector:
        if self.detector == DetectionMethod.WAVELET:
            return WaveletDetector(self.wavelet.scale, self.wavelet.k, self.wavelet.min_area)

        return FakeDetector(mu, self.fake.measurement_noise, self.fake.fpr, self.fake.fnr, False)


@dataclasses.dataclass
class KalmanConfig:
    detection_noise: float
    of_noise: float
    process_noise: float  # Miss evaluation of the process
    dist: Dist
    matching_method: Method
    always_update_velocities: bool = True
    dim: int = 2
    order: int = 1


class TrackingMethod(enum.Enum):
    SKT = "skt"
    KOFT = "koft"
    KOFTmm = "koft--"
    KOFTpp = "koft++"
    TRACKMATE = "trackmate"
    TRACKMATE_KF = "trackmate-kf"
    EMHT = "emht"


@dataclasses.dataclass
class ExperimentConfig:
    seed: int
    real_data: bool
    simulation_path: pathlib.Path
    dupre_data: dupre.DupreDataConfig
    tracking_method: TrackingMethod
    detection: DetectionConfig
    kalman: KalmanConfig
    icy_path: pathlib.Path
    fiji_path: pathlib.Path

    def create_linker(self, thresh: float) -> byotrack.Linker:
        """Create a linker"""
        if self.tracking_method is TrackingMethod.EMHT:
            return IcyEMHTLinker(
                self.icy_path,
                EMHTParameters(
                    gate_factor=thresh,
                    motion=Motion.MULTI,
                    tree_depth=2,
                ),
                timeout=120,  # Ensure Icy goes out of infinite loops. (Adapt to your hardware)
            )

        if self.tracking_method in (TrackingMethod.TRACKMATE, TrackingMethod.TRACKMATE_KF):
            # As kalman tracking we let a gap of 2 consecutive miss detections
            # In that case, we allow 1.5 thresh
            return TrackMateLinker(
                self.fiji_path,
                TrackMateParameters(
                    max_frame_gap=PartialTrack.MAX_NON_MEASURE,
                    linking_max_distance=thresh,
                    gap_closing_max_distance=thresh * 1.5,
                    kalman_search_radius=thresh if self.tracking_method is TrackingMethod.TRACKMATE_KF else None,
                ),
            )

        if self.tracking_method is TrackingMethod.SKT:
            kalman_filter = constant_kalman_filter(
                torch.tensor(self.kalman.detection_noise),
                torch.tensor(self.kalman.process_noise),
                self.kalman.dim,
                self.kalman.order,
            )

            return SimpleKalmanTracker(
                kalman_filter, MatchingConfig(thresh, self.kalman.dist, self.kalman.matching_method)
            )

        # self.tracking_method is TrackingMethod.KOFT:
        kalman_filter = constant_koft_filter(
            torch.tensor(self.kalman.detection_noise),
            torch.tensor(self.kalman.of_noise),
            torch.tensor(self.kalman.process_noise),
            self.kalman.dim,
            self.kalman.order,
        )

        if self.tracking_method is TrackingMethod.KOFTmm:
            return SingleUpdateKOFTracker(
                kalman_filter, farneback, MatchingConfig(thresh, self.kalman.dist, self.kalman.matching_method)
            )
            # <=> two updates, without updating vel for all tracks and using OptFlowExtraction at Detected pos
            # return TwoUpdateKOFTracker(
            #     kalman_filter,
            #     farneback,
            #     MatchingConfig(thresh, self.kalman.dist, self.kalman.matching_method),
            #     OptFlowExtraction.DETECTED,
            #     False,
            # )

        PartialTrack.MAX_NON_MEASURE = 5 if self.tracking_method is TrackingMethod.KOFTpp else 3
        return TwoUpdateKOFTracker(
            kalman_filter,
            farneback,
            MatchingConfig(thresh, self.kalman.dist, self.kalman.matching_method),
            OptFlowExtraction.POSTERIOR,
            self.kalman.always_update_velocities,
        )

    def create_thresholds(self) -> List[float]:
        if self.tracking_method is TrackingMethod.EMHT:
            return [2.0, 3.0, 4.0, 5.0, 7.0, 10.0]  # MAHA

        if (
            self.tracking_method in (TrackingMethod.TRACKMATE, TrackingMethod.TRACKMATE_KF)
            or self.kalman.dist is Dist.EUCLIDIAN
        ):
            return [5.0, 7.0, 10.0, 15.0, 20.0]

        if self.kalman.dist is Dist.MAHALANOBIS:
            return [2.0, 3.0, 4.0, 5.0, 7.0, 10.0]

        # self.dist is Dist.LIKELIHOOD:
        return [5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2]


def main(name: str, cfg_data: dict) -> None:
    print("Running:", name)
    print(yaml.dump(cfg_data))
    cfg = dacite.from_dict(ExperimentConfig, cfg_data, dacite.Config(cast=[pathlib.Path, tuple, enum.Enum]))

    enforce_all_seeds(cfg.seed)

    # Read video and ground truth
    # As the code was designed for simulation initially, let's just patch things with dupre's data
    if cfg.real_data:
        video = cfg.dupre_data.open()
        mu = byotrack.Track.tensorize(cfg.dupre_data.cleaned_tracks())
        video = video[: len(mu)]
        ground_truth = {"mu": mu, "weight": torch.ones_like(mu).mean(dim=-1)}
    else:
        video = simulation.open_video(cfg.simulation_path)
        ground_truth = simulation.load_ground_truth(cfg.simulation_path)

    # Detections
    detector = cfg.detection.create_detector(ground_truth["mu"])
    detections_sequence = detector.run(video)

    # Evaluate detections step performances
    tp = 0.0
    n_pred = 0.0
    n_true = 0.0
    for detections in detections_sequence:
        det_metrics = DetectionMetric(1.5).compute_at(
            detections, ground_truth["mu"][detections.frame_id], ground_truth["weight"][detections.frame_id]
        )
        tp += det_metrics["tp"]
        n_pred += det_metrics["n_pred"]
        n_true += det_metrics["n_true"]

    print("=======Detection======")
    print("Recall", tp / n_true if n_true else 1.0)
    print("Precision", tp / n_pred if n_pred else 1.0)
    print("f1", 2 * tp / (n_true + n_pred) if n_pred + n_true else 1.0)

    refiner = ForwardBackwardInterpolater()
    metrics = {}
    best_thresh = 0.0
    best_hota = 0.0
    best_tracks: Collection[byotrack.Track] = []
    for thresh in tqdm.tqdm(cfg.create_thresholds()):
        linker = cfg.create_linker(thresh)
        try:
            tracks = linker.run(video, detections_sequence)
            tracks = refiner.run(video, tracks)
        except Exception:  # pylint: disable=broad-exception-caught
            tracks = []  # Tracking failed (For instance: timeout in EMHT)

        tqdm.tqdm.write(f"Built {len(tracks)} tracks")

        if len(tracks) == 0 or len(tracks) > ground_truth["mu"].shape[1] * 20:
            tqdm.tqdm.write(f"Threshold: {thresh} => Tracking failed (too few or too many tracks). Continuing...")
            continue

        metrics[thresh] = compute_tracking_metrics(tracks, ground_truth)

        tqdm.tqdm.write(f"Threshold: {thresh} => HOTA: {metrics[thresh]['HOTA']}")
        tqdm.tqdm.write(yaml.dump(metrics[thresh]))

        if metrics[thresh]["HOTA"] > best_hota:
            best_thresh = thresh
            best_tracks = tracks
            best_hota = metrics[thresh]["HOTA"]

    if best_thresh == 0.0:
        print("!==============! Tracking failed for all thresholds !==============!")
        return

    print(f"Best threshold: {best_thresh}")
    print(yaml.dump(metrics[best_thresh]))

    with open("best_metrics.yml", "w", encoding="utf-8") as file:
        file.write(yaml.dump(metrics[best_thresh]))

    byotrack.Track.save(best_tracks, "tracks.pt")

from __future__ import annotations

import copy
import dataclasses
import enum
import pathlib
from typing import TYPE_CHECKING

import byotrack
import dacite
import torch
import tqdm.auto as tqdm
import yaml
from byotrack.implementation.detector.wavelet import WaveletDetector
from byotrack.implementation.linker.frame_by_frame import kalman_linker, koft, trackonstra
from byotrack.implementation.linker.icy_emht import EMHTParameters, IcyEMHTLinker, Motion
from byotrack.implementation.linker.trackastra import TrackAstraLinker, TrackAstraParameters
from byotrack.implementation.linker.trackmate import TrackMateLinker, TrackMateParameters
from byotrack.implementation.refiner.interpolater import ForwardBackwardInterpolater
from trackastra.model import Trackastra  # type: ignore[import-untyped]

from src.data import TrackingDataConfig  # noqa: TC001
from src.detector import FakeDetector
from src.metrics.detections import DetectionMetric
from src.metrics.tracking import compute_tracking_metrics
from src.optical_flow import bt_farneback as farneback
from src.utils import enforce_all_seeds, kill_java_in_our_pgrp_pkill

if TYPE_CHECKING:
    from collections.abc import Sequence


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
    fpr: float = 0.2
    fnr: float = 0.2
    measurement_noise: float = 1.0


@dataclasses.dataclass
class DetectionConfig:
    detector: DetectionMethod
    wavelet: WaveletConfig
    fake: FakeConfig

    def create_detector(self, mu: torch.Tensor) -> byotrack.Detector:
        if self.detector == DetectionMethod.WAVELET:
            return WaveletDetector(self.wavelet.scale, self.wavelet.k, min_area=self.wavelet.min_area)

        return FakeDetector(mu, self.fake.measurement_noise, self.fake.fpr, self.fake.fnr, False)


class TrackingMethod(enum.Enum):
    SKT = "skt"
    KOFT = "koft"
    KOFTmm = "koft--"
    TRACKMATE = "trackmate"
    TRACKMATE_KF = "trackmate-kf"
    EMHT = "emht"
    TRACKASTRA = "trackastra"
    TRACKASTRA_LAP = "trackastra-lap"


@dataclasses.dataclass
class ExperimentConfig:
    seed: int
    data: TrackingDataConfig
    tracking_methods: list[TrackingMethod]
    detection: DetectionConfig
    koft: koft.KOFTLinkerParameters
    icy_path: pathlib.Path
    fiji_path: pathlib.Path
    trackastra_model: pathlib.Path
    warp: bool = False

    def linkers(
        self,
        detections_sequence: Sequence[byotrack.Detections],
        video: byotrack.Video,
        gt_tracks: list[byotrack.Track],
    ) -> list[byotrack.Linker]:
        # Parameters estimations (KOFT/SKT/eMHT/TrackMate)
        if self.koft.flow_std <= 0.0:
            self.koft.estimate_flow_std_from_tracks(video[:50], farneback, gt_tracks[::10])

        # if self.koft.process_std <= 0.0:
        #     self.koft.estimate_process_std_from_tracks(gt_tracks[::10])

        self.koft.estimate(detections_sequence)

        tqdm.tqdm.write("KOFT main parameters:")
        tqdm.tqdm.write(
            yaml.dump(
                {
                    "threshold": str(self.koft.association_threshold),
                    "det_std": str(self.koft.detection_std),
                    "flow_std": str(self.koft.flow_std),
                    "process_std": str(self.koft.process_std),
                }
            )
        )

        skt_specs = kalman_linker.KalmanLinkerParameters(
            -1,
            detection_std=self.koft.detection_std,
            process_std=self.koft.process_std,
            kalman_order=self.koft.kalman_order,
            n_valid=self.koft.n_valid,
            n_gap=self.koft.n_gap,
            association_method=self.koft.association_method,
            cost=self.koft.cost,
            track_building=self.koft.track_building,
            initial_std_factor=self.koft.initial_std_factor,
        )
        skt_specs.estimate_association_threshold(2, 3.0)
        tqdm.tqdm.write(f"SKT threshold: {skt_specs.association_threshold}")

        trackmate_specs = kalman_linker.KalmanLinkerParameters(
            -1,
            detection_std=self.koft.detection_std,
            process_std=self.koft.process_std,
            kalman_order=self.koft.kalman_order,
            n_valid=self.koft.n_valid,
            n_gap=self.koft.n_gap,
            cost=koft.Cost.EUCLIDEAN,
        )
        trackmate_specs.estimate_association_threshold(2, 3.0)
        tqdm.tqdm.write(f"TrackMate threshold: {trackmate_specs.association_threshold}")
        tqdm.tqdm.write("")

        # Create linkers
        linkers: list[byotrack.Linker] = []
        for method in self.tracking_methods:
            if method is TrackingMethod.EMHT:
                linkers.append(
                    IcyEMHTLinker(
                        self.icy_path,
                        EMHTParameters(gate_factor=4.0, motion=Motion.MULTI, tree_depth=2),
                        timeout=180,  # Ensure Icy goes out of infinite loops. (Adapt to your hardware)))
                    )
                )

            if method in (TrackingMethod.TRACKMATE, TrackingMethod.TRACKMATE_KF):
                # We allow for n_gap consecutive miss detections, for which we link up to 1.5 x max_dist.
                linkers.append(
                    TrackMateLinker(
                        self.fiji_path,
                        TrackMateParameters(
                            max_frame_gap=trackmate_specs.n_gap,
                            linking_max_distance=trackmate_specs.association_threshold,
                            gap_closing_max_distance=trackmate_specs.association_threshold * 1.5,
                            kalman_search_radius=trackmate_specs.association_threshold
                            if method is TrackingMethod.TRACKMATE_KF
                            else None,
                        ),
                    )
                )

            if method is TrackingMethod.SKT:
                linkers.append(kalman_linker.KalmanLinker(skt_specs))

            if method is TrackingMethod.KOFT:
                linkers.append(koft.KOFTLinker(self.koft, farneback))

            if method is TrackingMethod.KOFTmm:
                koft_specs = copy.deepcopy(self.koft)
                koft_specs.always_measure_velocity = False  # Disable velocity update for miss detected tracks
                linkers.append(koft.KOFTLinker(koft_specs, farneback))

            if method is TrackingMethod.TRACKASTRA:
                linkers.append(
                    TrackAstraLinker(
                        Trackastra.from_folder(self.trackastra_model),
                        TrackAstraParameters(max_distance=20, solver="ilp_nodiv"),
                    )
                )

            if method is TrackingMethod.TRACKASTRA_LAP:
                linkers.append(
                    trackonstra.TrackOnStraLinker(
                        trackonstra.TrackOnStraParameters(positional_cutoff=20.0, n_valid=3, n_gap=3),
                        trackonstra.TrackastraFlex.from_folder(self.trackastra_model),
                    ),
                )

        return linkers


def main(name: str, cfg_data: dict) -> None:
    print("Running:", name)
    print(yaml.dump(cfg_data))
    cfg = dacite.from_dict(ExperimentConfig, cfg_data, dacite.Config(cast=[pathlib.Path, tuple, enum.Enum]))

    enforce_all_seeds(cfg.seed)

    video = cfg.data.video()
    ground_truth = cfg.data.ground_truth()  # SINETRA like ground truth
    gt_tracks = cfg.data.tracks()

    # Detections
    detector = cfg.detection.create_detector(ground_truth["mu"])
    detections_sequence = detector.run(video)

    # Evaluate detections step performances
    tp = 0.0
    n_pred = 0.0
    n_true = 0.0
    for frame_id, detections in enumerate(detections_sequence):
        det_metrics = DetectionMetric(2.0).compute_at(
            detections, ground_truth["mu"][frame_id], ground_truth["weight"][frame_id]
        )
        tp += det_metrics["tp"]
        n_pred += det_metrics["n_pred"]
        n_true += det_metrics["n_true"]

    print("=======Detection======")
    print("Recall", tp / n_true if n_true else 1.0)
    print("Precision", tp / n_pred if n_pred else 1.0)
    print("f1", 2 * tp / (n_true + n_pred) if n_pred + n_true else 1.0)

    # if cfg.warp:  # WTT is out of scope of the paper now... (Though in the thesis)
    #     true_detections = detections_sequence
    #     detections_sequence = warp.warp_detections_linear(video, farneback, list(detections_sequence))
    #     # ground_truth["mu"] = warp_mu(video, farneback, ground_truth["mu"])  # Let's not warp mu but unwarp tracks

    refiner = ForwardBackwardInterpolater()
    metrics = {}
    for method, linker in zip(
        cfg.tracking_methods, tqdm.tqdm(cfg.linkers(detections_sequence, video, gt_tracks)), strict=True
    ):
        try:
            if hasattr(linker, "setup"):  # TrackOnStra
                linker.setup(video, detections_sequence)

            tracks = linker.run(video, detections_sequence)

            # Fill miss detected positions (if not done in the linker) and remove tracks of length one
            tracks = [track for track in refiner.run(video, tracks) if len(track) > 1]
        except BaseException as exc:  # noqa: BLE001
            kill_java_in_our_pgrp_pkill()  # Kill Java just in case it survives (ugly, needs to be fixed in ByoTrack)
            tqdm.tqdm.write(str(exc))
            tracks = []  # Tracking failed (For instance: timeout in EMHT)

        tqdm.tqdm.write(f"Built {len(tracks)} tracks")

        if len(tracks) == 0 or len(tracks) > ground_truth["mu"].shape[1] * 40:
            tqdm.tqdm.write(f"Method: {method.value} => Tracking failed (too few or too many tracks). Continuing...")
            continue

        hota = compute_tracking_metrics(tracks, ground_truth)

        # Hota @ 2 (-8 => Thresholds is 2)
        metrics[method.value] = {key: value[-8].item() for key, value in hota.items()}

        tqdm.tqdm.write(f"Method: {method.value} => HOTA@2.0: {metrics[method.value]['HOTA']}")
        tqdm.tqdm.write(yaml.dump(metrics[method.value]))
        torch.save(hota, f"hota_{method.value}.pt")
        byotrack.Track.save(tracks, f"tracks_{method.value}.pt")

    pathlib.Path("metrics.yml").write_text(yaml.dump(metrics))

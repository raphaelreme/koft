"""Same as track but test several of_noise and only with koft"""

import enum
import pathlib

import dacite
import torch
import tqdm  # type: ignore
import yaml  # type: ignore


from ..data import simulation
from ..metrics.detections import DetectionMetric
from ..metrics.tracking import compute_tracking_metrics
from ..skt import MatchingConfig
from ..koft import constant_koft_filter, OptFlowExtraction, TwoUpdateKOFTracker
from ..optical_flow import farneback
from ..utils import enforce_all_seeds
from .track import ExperimentConfig, TrackingMethod


def main(name: str, cfg_data: dict) -> None:
    print("Running:", name)
    print(yaml.dump(cfg_data))
    cfg = dacite.from_dict(ExperimentConfig, cfg_data, dacite.Config(cast=[pathlib.Path, tuple, enum.Enum]))

    enforce_all_seeds(cfg.seed)

    # Ensure we are with a correct config
    assert not cfg.real_data
    assert cfg.tracking_method == TrackingMethod.KOFT

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

    best_metrics = {}
    for of_noise in tqdm.tqdm([1.0, 2.0, 5.0]):  # Create a linker for each of_noise=
        kalman_filter = constant_koft_filter(
            torch.tensor(cfg.kalman.detection_noise),
            torch.tensor(of_noise),
            torch.tensor(cfg.kalman.process_noise),
            order=cfg.kalman.order,
        )

        metrics = {}
        best_thresh = 0.0
        best_hota = 0.0
        for thresh in tqdm.tqdm(cfg.create_thresholds()):
            linker = TwoUpdateKOFTracker(
                kalman_filter,
                farneback,
                MatchingConfig(thresh, cfg.kalman.dist, cfg.kalman.matching_method),
                OptFlowExtraction.POSTERIOR,
                cfg.kalman.always_update_velocities,
            )
            try:
                tracks = linker.run(video, detections_sequence)
            except Exception as exc:  # pylint: disable=broad-exception-caught
                tqdm.tqdm.write(str(exc))
                tracks = []  # Tracking failed (For instance: timeout in EMHT)

            tqdm.tqdm.write(f"Built {len(tracks)} tracks")

            if len(tracks) == 0 or len(tracks) > ground_truth["mu"].shape[1] * 10:
                tqdm.tqdm.write(f"Threshold: {thresh} => Tracking failed (too few or too many tracks). Continuing...")
                continue

            hota = compute_tracking_metrics(tracks, ground_truth)

            # Hota @ 2 (-8 => Thresholds is 2)
            metrics[thresh] = {key: value[-8].item() for key, value in hota.items()}

            tqdm.tqdm.write(f"Threshold: {thresh} => HOTA@2.0: {metrics[thresh]['HOTA']}")
            tqdm.tqdm.write(yaml.dump(metrics[thresh]))

            if metrics[thresh]["HOTA"] > best_hota:
                torch.save(hota, "hota.pt")  # Save full hota just in case
                best_thresh = thresh
                best_hota = metrics[thresh]["HOTA"]

        if best_thresh == 0.0:
            print("!==============! Tracking failed for all thresholds !==============!")
            continue

        print(f"Best threshold: {best_thresh}")
        print(yaml.dump(metrics[best_thresh]))
        metrics[best_thresh]["best_threshold"] = best_thresh

        best_metrics[of_noise] = metrics[best_thresh]

    with open("best_metrics.yml", "w", encoding="utf-8") as file:
        file.write(yaml.dump(best_metrics))

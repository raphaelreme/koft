from __future__ import annotations

import dataclasses
import enum
import pathlib
import time

import byotrack
import dacite
import tqdm.auto as tqdm
import yaml
from byotrack.implementation.refiner.interpolater import ForwardBackwardInterpolater

from src.experiments.track import ExperimentConfig as _ExperimentConfig
from src.metrics.detections import DetectionMetric
from src.metrics.tracking import compute_tracking_metrics
from src.utils import enforce_all_seeds, kill_java_in_our_pgrp_pkill

byotrack.ZSTD_SEG = True  # Let's compress segmentations to reduce RAM usage


@dataclasses.dataclass
class ExperimentConfig(_ExperimentConfig):
    eval_lengths: tuple[int, ...] = (50, 100, 200, 500, 1000, 2000, 4000)
    max_length: int = 4000


def main(name: str, cfg_data: dict) -> None:
    print("Running:", name)
    print(yaml.dump(cfg_data))
    cfg = dacite.from_dict(ExperimentConfig, cfg_data, dacite.Config(cast=[pathlib.Path, tuple, enum.Enum]))

    enforce_all_seeds(cfg.seed)

    video = cfg.data.video()
    ground_truth = cfg.data.ground_truth()  # SINETRA like ground truth
    gt_tracks = cfg.data.tracks()

    # Restrict video to max_l
    eval_lengths = tuple(length for length in cfg.eval_lengths if length <= cfg.max_length)
    video = video[: cfg.max_length]

    # Detections
    detector = cfg.detection.create_detector(ground_truth["mu"])
    detections_sequence = detector.run(video)

    # Evaluate detections step performances
    tp = 0.0
    n_pred = 0.0
    n_true = 0.0
    for frame_id, detections in enumerate(detections_sequence):
        if frame_id % 10 != 0:
            continue

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

    refiner = ForwardBackwardInterpolater()
    metrics: dict[str, dict[int, dict[str, float]]] = {}
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

        tqdm.tqdm.write(f"-------------- {method.value} --------------")
        tqdm.tqdm.write(f"Built {len(tracks)} tracks")

        metrics[method.value] = {}

        for length in eval_lengths:
            t = time.time()
            hota = compute_tracking_metrics(
                byotrack.Track.temporal_slice_filter(tracks, 0, length),
                {"mu": ground_truth["mu"][:length], "weight": ground_truth["weight"][:length]},
            )
            tqdm.tqdm.write(f"Length={length} => HOTA={hota['HOTA'][-8]} (eval time: {time.time() - t})")

            # Hota @ 2 (-8 => Thresholds is 2)
            metrics[method.value][length] = {key: value[-8].item() for key, value in hota.items()}

        byotrack.Track.save(tracks, f"tracks_{method.value}.pt")

    pathlib.Path("metrics.yml").write_text(yaml.dump(metrics))

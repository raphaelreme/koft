import glob
import os
import pathlib
from typing import Dict, List

import numpy as np
import yaml  # type: ignore


def main():
    print(f"{'':10}|{'Springs':62}|{'Flow':62}")
    print(
        f"{'method':15}|{'Fake@90%':20}|{'Fake@70%':20}|{'Wavelet':20}|{'Fake@90%':20}|{'Fake@70%':20}|{'Wavelet':20}"
    )
    detections = ["Fake@90%", "Fake@70%", "Wavelet"]
    for method in ["trackmate", "trackmate-kf", "emht", "skt", "koft", "koft--", "koft++"]:
        aggregated = []
        for motion in ["springs", "flow_20140829_1"]:
            paths = glob.glob(f"{os.environ['EXPERIMENT_DIR']}/tracking/{motion}/1000/1.5-50.0/*/exp.0/{method}/*/")
            results: Dict[str, List[float]] = {detection_name: [] for detection_name in detections}
            for path in paths:
                if not (pathlib.Path(path) / "best_metrics.yml").exists():
                    continue  # Run has not finished

                with open(pathlib.Path(path) / "config.yml", "r", encoding="utf-8") as file:
                    detection_cfg = yaml.safe_load(file)["detection"]

                detection_name = ""
                if detection_cfg["detector"] == "wavelet":
                    detection_name = "Wavelet"
                elif detection_cfg["detector"] == "fake":
                    if detection_cfg["fake"]["fpr"] == 0.1:
                        if detection_cfg["fake"]["fnr"] == 0.1:
                            detection_name = "Fake@90%"
                    elif detection_cfg["fake"]["fpr"] == 0.3:
                        if detection_cfg["fake"]["fnr"] == 0.3:
                            detection_name = "Fake@70%"

                if not detection_name:
                    continue

                with open(pathlib.Path(path) / "best_metrics.yml", "r", encoding="utf-8") as file:
                    hota = yaml.safe_load(file)["HOTA"]

                results[detection_name].append(hota)

            for detection_name in detections:
                scores = results[detection_name]
                if not scores:
                    scores = [-1]
                mean, std = np.mean(scores), np.std(scores)
                aggregated.append(f"{mean*100:.1f} +/- {std*100:0.1f}% ({len(scores)})")

        print(
            f"{method:15}|{aggregated[0]:20}|{aggregated[1]:20}|{aggregated[2]:20}|{aggregated[3]:20}|{aggregated[4]:20}|{aggregated[5]:20}"
        )


if __name__ == "__main__":
    main()

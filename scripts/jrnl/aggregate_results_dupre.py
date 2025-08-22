import glob
import os
import pathlib
from typing import Dict, List

import numpy as np
import yaml  # type: ignore


def main():
    print(f"{'method':15}|{'Fake@90%':20}|{'Fake@80%':20}|{'Fake@70%':20}|{'Fake@60%':20}")
    detections = ["Fake@90%", "Fake@80%", "Fake@70%", "Fake@60%"]
    for method in ["trackmate", "trackmate-kf", "emht", "skt", "koft--", "koft", "koft++"]:
        aggregated = []
        paths = glob.glob(f"{os.environ['EXPERIMENT_DIR']}/tracking/dupre/{method}/*/")
        results: Dict[str, List[float]] = {detection_name: [] for detection_name in detections}
        for path in paths:
            if not (pathlib.Path(path) / "best_metrics.yml").exists():
                continue  # Run has not finished

            with open(pathlib.Path(path) / "config.yml", "r", encoding="utf-8") as file:
                detection_cfg = yaml.safe_load(file)["detection"]

            detection_name = ""
            if detection_cfg["fake"]["fpr"] == 0.1:
                if detection_cfg["fake"]["fnr"] == 0.1:
                    detection_name = "Fake@90%"
            if detection_cfg["fake"]["fpr"] == 0.2:
                if detection_cfg["fake"]["fnr"] == 0.2:
                    detection_name = "Fake@80%"
            if detection_cfg["fake"]["fpr"] == 0.3:
                if detection_cfg["fake"]["fnr"] == 0.3:
                    detection_name = "Fake@70%"
            elif detection_cfg["fake"]["fpr"] == 0.4:
                if detection_cfg["fake"]["fnr"] == 0.4:
                    detection_name = "Fake@60%"

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

        print(f"{method:15}|{aggregated[0]:20}|{aggregated[1]:20}|{aggregated[2]:20}|{aggregated[3]:20}")


if __name__ == "__main__":
    main()

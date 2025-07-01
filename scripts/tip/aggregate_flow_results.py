import pathlib
from typing import Dict, List

import numpy as np
import yaml  # type: ignore


def main():
    print(f"{'method':20}|{'Springs':30}|{'Flow':30}|{'Hydra Vulgaris (Dupre)':30}")
    lines = []
    for method in ["none-4-1.0", "tvl1-4-1.0", "farneback-4-1.0", "raft-2-1.0", "vxm-4-1.0"]:
        data: Dict[str, List[float]] = {
            "springs": [],
            "of": [],
            "dupre": [],
        }
        for seed in [111, 222, 333, 444, 555]:
            path = pathlib.Path(f"experiment_folder/optical_flow/{method}/0.2-50.0/{seed}/exp.0/metrics.yml")

            if not path.exists():
                print(f"Ignoring missing {path}")
                continue

            metrics: Dict[str, Dict[str, float]] = yaml.safe_load(path.read_text(encoding="utf-8"))

            for key, val in metrics.items():
                data[key].append(val["RMSE"])

        line = []
        for key, scores in data.items():
            if not scores:
                scores = [-1]

            mean, std = np.mean(scores), np.std(scores)
            line.append(f"{mean:.2f} +/- {std:.2f} ({len(scores)})")

        lines.append(f"{method:20}|{line[0]:30}|{line[1]:30}|{line[2]:30}")

    print("\n".join(lines))


if __name__ == "__main__":
    main()

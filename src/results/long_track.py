import pathlib

import numpy as np
import yaml

from src.results.drawing_helpers import plot_hota_vs_length


def main(simulation_name="springs_2d"):  # noqa: C901
    lengths = [50, 100, 200, 500, 1000, 2000, 4000]
    methods = ["trackmate-kf", "emht", "trackastra", "trackastra-lap", "skt", "koft--", "koft"]

    col_size = 20
    lines = []
    lines.append(
        "=" * ((col_size * len(lengths)) // 2)
        + " "
        + simulation_name.upper()
        + " "
        + "=" * ((col_size * len(lengths)) // 2)
    )
    lines.append("")
    lines.append("|".join([f"{'Method':15}"] + [f"{f'T={length}':^{col_size}}" for length in lengths]))
    lines.append("-" * ((len(lengths) + 1) * col_size + len(lengths) - (col_size - 15)))

    results: dict[str, dict[int, list[float]]] = {method: {length: [] for length in lengths} for method in methods}

    paths = (pathlib.Path("experiment_folder") / "long_tracking" / f"{simulation_name}" / "0.2-50.0").glob("*/*")
    for path in paths:
        if not (path / "metrics.yml").exists():
            continue  # Run has not finished

        detection_cfg = yaml.safe_load((path / "config.yml").read_text())["detection"]

        if detection_cfg["detector"] != "wavelet":
            continue

        metrics = yaml.safe_load((path / "metrics.yml").read_text())
        for method in metrics:
            if method not in results:
                continue
            for length in metrics[method]:
                if length not in lengths:
                    continue

                results[method][length].append(metrics[method][length]["HOTA"])

    for method in methods:
        line = [f"{method:15}"]
        for length in lengths:
            scores = results[method][length]
            if not scores:
                scores = [-1]
            mean, std = np.mean(scores), np.std(scores)
            line.append(f"{f'{mean * 100:.1f} +/- {std * 100:0.1f}% ({len(scores)})':>{col_size}}")

        lines.append("|".join(line))

    print("\n".join(lines))

    # Restrict plotting for these methods only
    methods = ["trackmate-kf", "emht", "trackastra-lap", "skt", "koft"]

    plot_hota_vs_length(methods, lengths, results, "hota_vs_length.svg")


if __name__ == "__main__":
    main("springs_2d")

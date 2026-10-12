import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

_it = iter(plt.rcParams["axes.prop_cycle"]())
colors = [next(_it)["color"] for _ in range(10)]
styles = ["o", "^", "s", "X", "*", "d", "p", "H", "D", "P"]

plt.rcParams.update({"font.size": 50, "text.usetex": True})

method_display_name = {
    "trackmate": "TrackMate (no-kf)",
    "trackmate-kf": "TrackMate",
    "trackastra": "TrackAstra-ilp",
    "trackastra-lap": "TrackAstra",
    "emht": "eMHT",
    "skt": "SKT",
    "koft--": "KOFT{-}{-}",
    "koft": "KOFT",
}

method_id = {
    "trackmate": 6,
    "trackmate-kf": 5,
    "trackastra": 7,
    "trackastra-lap": 2,
    "emht": 4,
    "skt": 3,
    "koft--": 1,
    "koft": 0,
}


marker_size = 15
line_width = 3
alpha = 0.15


def plot_method_with_uncertainty(x: np.ndarray, y: list[np.ndarray], method: str, conf=0.95) -> None:
    """Plot the method scores with their uncertainty.

    Args:
        x (np.ndarray): First axis of the plot
            Shape: (n,)
        y (list[np.ndarray]): Score to be plotted for each x. Expect to have several realisation (seed) per x.
            Assumed to be of length n, containing 1D array of realisations of the score

    """
    mean = np.array([np.mean(values) for values in y])

    half_width = np.array(
        [
            stats.t.ppf((1 + conf) / 2, df=values.size - 1) * np.std(values, ddof=1) / np.sqrt(values.size)
            if values.size > 1
            else 0.0
            for values in y
        ]
    )

    i = method_id[method]

    plt.plot(
        x,
        mean,
        marker=styles[i],
        linewidth=line_width,
        ms=marker_size,
        color=colors[i],
        label=method_display_name[method],
    )
    plt.fill_between(
        x, np.maximum(0, mean - half_width), np.minimum(100, mean + half_width), color=colors[i], alpha=alpha
    )


def plot_hota_vs_detections(
    methods: list[str], detections: list[str], results: dict[str, dict[str, list[float]]], figure_name: str, conf=0.95
) -> None:
    # Keep only Fake@ for plotting
    detections = [detection_name for detection_name in detections if detection_name[:5] == "Fake@"]

    # Sort per detection F1
    detections = sorted(detections, key=lambda detection_name: int(detection_name[5:-1]), reverse=False)

    x = np.array([int(detection_name[5:-1]) for detection_name in detections])

    plt.figure(figsize=(24, 16))

    for method in methods:
        if any(len(results[method][detection_name]) == 0 for detection_name in detections):
            continue

        y = [np.array(results[method][detection_name]) * 100 for detection_name in detections]

        plot_method_with_uncertainty(x, y, method, conf)

    plt.xlim(59, 101)
    plt.ylim(-5, 105)

    plt.grid()

    plt.yticks(list(range(0, 101, 20)), [str(k) for k in range(0, 101, 20)])
    plt.xticks(x, [f"Fake$@${f1}\\%" for f1 in x])

    plt.legend(title=r"\underline{Method}")
    plt.xlabel("Detections", fontsize=80)
    plt.ylabel("HOTA", fontsize=80)
    plt.savefig(figure_name)
    plt.show()


def plot_hota_vs_length(
    methods: list[str], lengths: list[int], results: dict[str, dict[int, list[float]]], figure_name: str, conf=0.95
) -> None:
    lengths = sorted(lengths)

    x = np.array(lengths)

    plt.figure(figsize=(24, 16))

    for method in methods:
        y = []
        for length in lengths:
            if len(results[method][length]) == 0:
                break
            y.append(np.array(results[method][length]) * 100)

        if not y:
            continue

        plot_method_with_uncertainty(x[: len(y)], y, method, conf)

    plt.xlim(49, 4001)
    plt.xscale("log")
    plt.ylim(-5, 105)

    plt.grid()

    plt.yticks(list(range(0, 101, 20)), [str(k) for k in range(0, 101, 20)])
    plt.xticks(x, [str(length) for length in lengths])

    plt.legend(title=r"\underline{Method}")
    plt.xlabel("Video length", fontsize=80)
    plt.ylabel("HOTA", fontsize=80)
    plt.savefig(figure_name)
    plt.show()


def plot_snr_impact(
    alphas: list[float],
    deltas: list[float],
    results: dict[float, dict[float, list[float]]],
    lines: dict[str, list[float]],
    figure_name: str,
    conf=0.95,
) -> None:
    alphas = sorted(alphas)
    deltas = sorted(deltas)

    plt.figure(figsize=(24, 16))

    # First let's plot the lines
    x = np.linspace(-1, 11, 13) / 10
    for k, method in enumerate(lines):
        values = np.array(lines[method]) * 100
        mean = np.mean(values)
        if values.size <= 1:
            half_width = 0.0
        else:
            half_width = stats.t.ppf((1 + conf) / 2, df=values.size - 1) * np.std(values, ddof=1) / np.sqrt(values.size)

        i = method_id[method]

        plt.plot(
            x, [mean] * len(x), color=colors[i], linestyle=":", linewidth=line_width, marker=styles[i], ms=marker_size
        )
        plt.fill_between(x, [mean - half_width] * len(x), [mean + half_width] * len(x), color=colors[i], alpha=alpha)
        plt.text(0.71 + k / 10, mean, method_display_name[method], color=colors[i])

    # Plot one KOFT curve per delta
    x = np.array(alphas)
    for k, delta in enumerate(deltas):
        if any(len(results[delta][alpha]) == 0 for alpha in alphas):
            continue

        y = [np.array(results[delta][alpha]) * 100 for alpha in alphas]

        mean = np.array([np.mean(values) for values in y])

        half_width_ = np.array(
            [
                stats.t.ppf((1 + conf) / 2, df=values.size - 1) * np.std(values, ddof=1) / np.sqrt(values.size)
                if values.size > 1
                else 0.0
                for values in y
            ]
        )

        plt.plot(
            x,
            mean,
            marker=styles[k],
            linewidth=line_width,
            ms=marker_size,
            color=colors[k],
            label=f"$\\Delta = {delta}$",
        )
        plt.fill_between(
            x, np.maximum(0, mean - half_width_), np.minimum(100, mean + half_width_), color=colors[k], alpha=alpha
        )

    plt.xlim(-0.01, 1.01)
    plt.ylim(-5, 105)

    # plt.yticks(list(range(0, 101, 20)), [str(k) for k in range(0, 101, 20)])
    # x = np.linspace(0.0, 1.0, 6)
    # plt.xticks(x, [str(_x) for _x in x])

    plt.grid()

    plt.legend(title=r"\underline{Integration time}", loc="lower right")
    plt.xlabel(r"Visibility coefficient $\alpha$", fontsize=80)
    plt.ylabel("HOTA", fontsize=80)
    plt.savefig(figure_name)
    plt.show()

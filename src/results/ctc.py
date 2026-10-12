import json
import pathlib

import numpy as np


def main():
    results: dict[str, dict[str, dict[str, float]]] = json.loads(pathlib.Path("ctc_results.json").read_text())
    datasets = list(results.keys())
    methods = list(results[datasets[0]].keys())
    metrics = ["LNK", "BIO", "CLB", "N_TOP_LNK", "N_TOP_BIO", "N_TOP_CLB"]

    fcol_size = 30
    col_size = 20

    lines = []
    lines.append("")
    lines.append("|".join([f"{'':^{fcol_size}}"] + [f"{metric:^{col_size}}" for metric in metrics]))
    lines.append("-" * ((len(metrics) + 1) * col_size + len(metrics) - (col_size - fcol_size)))

    per_method_results: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    for method in methods:
        lnks = []
        bios = []

        for dataset in datasets:
            lnks.append(results[dataset][method]["LNK"])
            bios.append(results[dataset][method]["BIO"])

        per_method_results[method] = (np.asarray(lnks), np.asarray(bios), (np.asarray(lnks) + np.asarray(bios)) / 2)

    res = np.asarray([per_method_results[method] for method in methods])
    res = res.transpose(0, 2, 1)

    # basleine_method = methods[0]

    for method in methods:
        line = [f"{method:<{fcol_size}}"]

        lnks_, bios_, clb = per_method_results[method]

        # lnks_ = lnks_ - per_method_results[basleine_method][0]
        # bios_ = bios_ - per_method_results[basleine_method][1]

        mean = np.mean(lnks_)
        std = np.std(lnks_)

        line.append(f"{f'{mean:.1f} +/- {std:0.1f}% ({len(lnks_)})':>{col_size}}")

        mean = np.mean(bios_)
        std = np.std(bios_)

        line.append(f"{f'{mean:.1f} +/- {std:0.1f}% ({len(bios_)})':>{col_size}}")

        mean = np.mean(clb)
        std = np.std(clb)

        line.append(f"{f'{mean:.1f} +/- {std:0.1f}% ({len(clb)})':>{col_size}}")

        top = 0

        # Number of time beated per a method in each dataset
        n_beated = (lnks_[None] < res[..., 0]).sum(axis=0)
        n_top = (n_beated <= top).sum()  # Number of dataset it got beated less then k time
        line.append(f"{n_top:>{col_size}}")

        # Number of time beated per a method in each dataset
        n_beated = (bios_[None] < res[..., 1]).sum(axis=0)
        n_top = (n_beated <= top).sum()  # Number of dataset it got beated less then k time
        line.append(f"{n_top:>{col_size}}")

        # Number of time beated per a method in each dataset
        n_beated = (clb[None] < res[..., 2]).sum(axis=0)
        n_top = (n_beated <= top).sum()  # Number of dataset it got beated less then k time
        line.append(f"{n_top:>{col_size}}")

        lines.append("|".join(line))

    print("\n".join(lines))


if __name__ == "__main__":
    main()

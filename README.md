# Kalman and Optical Flow Tracking (KOFT)

![koft](images/koft.png)

Code for the [paper](https://ieeexplore.ieee.org/abstract/document/10635656): "Particle tracking in biological images with optical-flow enhanced Kalman filtering", published at IEEE ISBI2024.

Abstract:
*Single-particle-tracking is a fundamental pre-requisite for studying biological processes in time-lapse microscopy. However, it remains a challenging task in many applications where numerous particles are driven by fast and complex motion. To anticipate the  motion of particles most tracking algorithms usually assume near-constant position, velocity or acceleration of particles over consecutive frames. However, such assumptions are not robust to the large and sudden changes in velocity that typically occur in in vivo imaging. In this paper, we exploit optical flow to directly measure the velocity of particles in a Kalman filtering context. The resulting method shows improved robustness and correctly predicts particles positions, even with sudden motions. We validate our method on simulated data, in particular with high particle density and fast, elastic motions. We show that it divides tracking errors by two, when compared to other tracking algorithms, while preserving fast execution time.*

## Data

![simulation](images/simulation.gif)

We rely on the [SINETRA](https://github.com/raphaelreme/SINETRA) synthetic datasets. It produces representative tracking data of fluorescence imaging of cells in freely-behaving animals.

Targets move according to elastic motions. The motions are either extracted from true fluorescence videos using optical flow (on the left) or produced from a physical-based simulation with springs (on the right).

The dataset can be generated with the code provided here. It relies on a fluorescence video of Hydra Vulgaris from Dupre C, et. al Non-overlapping Neural Networks in Hydra vulgaris. Curr Biol. 2017 Apr 24;27(8):1085-1097. doi: 10.1016/j.cub.2017.02.049. Epub 2017 Mar 30. PMID: 28366745; PMCID: PMC5423359. This video should be downloaded (see below).

We also use annnotated tracking data from the same paper. The video and ground truth tracks should also be downloaded. We provide a small script to download the dupre data and save it in `dataset/dupre` folder:

```bash
$ bash scripts/download_hydra_data.sh
```

## Install

First clone the repository and submodules

```bash
$ git clone git@github.com:raphaelreme/koft.git
$ cd koft
$ git submodule init
$ git submodule update
```

Install requirements

```bash
$ pip install -r requirements.txt
```

Additional requirements (Icy, Fiji) are needed to reproduce some results. See the installation guidelines of [ByoTrack](https://github.com/raphaelreme/byotrack) for a complete installation.

The experiment configuration files are using environment variables that needs to be set:
- $EXPERIMENT_DIR: Output folder of tracking experiments
- $DATA_FOLDER: Output folder of the simulation experiments
- $ICY: path to icy.jar
- $FIJI: path to fiji executable
- $RUN_KOFT_ENV: Prefix command to run inside the python env (if any). In our case, we used a conda env and set this to `conda run -n koft --live-stream`. We acknowledge that this is probably not robust to other type of python envs. It can be unset and you can modify the scripts to correctly use the python environement of your choice.

## Reproduce (ISBI 2024)

We provide scripts to generate the same dataset that we used and run the same experiments

```bash
$ # Generate dataset for 5 differents seeds
$ bash scripts/isbi/generate_dataset.sh 111
$ bash scripts/isbi/generate_dataset.sh 222
$ bash scripts/isbi/generate_dataset.sh 333
$ bash scripts/isbi/generate_dataset.sh 444
$ bash scripts/isbi/generate_dataset.sh 555
```

Reproducing the results for a particular method (skt, koft--, koft, emht, trackmate, trackmate-kf):

```bash
$ bash scripts/isbi/eval.sh $method  # With method in skt, koft, etc..
```

Aggregating all the results (mean +- std) on the different seeds:

```bash
$ python scripts/isbi/aggregate_results.py
```

## Results (ISBI 2024)

![results](images/results.png)

Note: *u-track* in the paper corresponds to the results of *trackmate-kf* in the code.


## Reproduce (TIP)

We provide scripts to generate the same dataset that we used and run the same experiments

```bash
$ # Generate dataset for 5 differents seeds
$ bash scripts/tip/generate_dataset.sh 111
$ bash scripts/tip/generate_dataset.sh 222
$ bash scripts/tip/generate_dataset.sh 333
$ bash scripts/tip/generate_dataset.sh 444
$ bash scripts/tip/generate_dataset.sh 555
```

### Optical flow
We benchmarked optical flow algorithms with:
```bash
$ bash scripts/tip/flow.sh 111
$ bash scripts/tip/flow.sh 222
$ bash scripts/tip/flow.sh 333
$ bash scripts/tip/flow.sh 444
$ bash scripts/tip/flow.sh 555
```

Aggregating tge results (mean +- std (N)) on the different seeds:

```bash
$ python scripts/tip/aggregate_flow_results.py
```

### Tracking (SINETRA)
To reproduce our tracking results on SINETRA, run:
```bash
$ bash scripts/tip/track_simulation.sh $method  # With method in (skt, koft--, koft, emht, trackmate-kf)
```

Aggregating the results (mean +- std (N)) on the different seeds:

```bash
$ python scripts/tip/aggregate_results_simulation.py
```

### Tracking (Dupre's Hydra)
To reproduce our tracking results on Hydra vulgaris, run:
```bash
$ bash scripts/tip/track_dupre.sh $method  # With method in (skt, koft--, koft, emht, trackmate-kf)
```

Aggregating the results (mean +- std (N)) on the different seeds:

```bash
$ python scripts/tip/aggregate_results_dupre.py
```

## Cite us


If you use this work, please cite our [paper](https://ieeexplore.ieee.org/abstract/document/10635656):

```bibtex
@INPROCEEDINGS{10635656koft,
  author={Reme, Raphael and Newson, Alasdair and Angelini, Elsa and Olivo-Marin, Jean-Christophe and Lagache, Thibault},
  booktitle={2024 IEEE International Symposium on Biomedical Imaging (ISBI)},
  title={Particle Tracking in Biological Images with Optical-Flow Enhanced Kalman Filtering},
  year={2024},
  volume={},
  number={},
  pages={1-5},
  keywords={Tracking;Filtering;Prediction algorithms;Particle measurements;Robustness;Kalman filters;Velocity measurement;Single-Particle-Tracking;Optical Flow;Kalman Filtering},
  doi={10.1109/ISBI56570.2024.10635656}
}
```

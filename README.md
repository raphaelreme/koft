# Kalman and Optical Flow Tracking (KOFT)

![koft](images/koft.png)

Code for "Particle tracking in biological images with optical-flow enhanced Kalman filtering", accepted at ISBI2024.

Abstract:
*Single-particle-tracking is a fundamental pre-requisite for studying biological processes in time-lapse microscopy. However, it remains a challenging task in many applications where numerous particles are driven by fast and complex motion. To anticipate the  motion of particles most tracking algorithms usually assume near-constant position, velocity or acceleration of particles over consecutive frames. However, such assumptions are not robust to the large and sudden changes in velocity that typically occur in in vivo imaging. In this paper, we exploit optical flow to directly measure the velocity of particles in a Kalman filtering context. The resulting method shows improved robustness and correctly predicts particles positions, even with sudden motions. We validate our method on simulated data, in particular with high particle density and fast, elastic motions. We show that it divides tracking errors by two, when compared to other tracking algorithms, while preserving fast execution time.*

## Data

![simulation](images/simulation.gif)

We simulate tracking data to validate our method. The goal of our simulator is to produce representative tracking data of fluorescence imaging of cells in freely-behaving animals.

Particles moves according to elastic motions. The motions are either extracted from true fluorescence videos using optical flow (on the left) or produced from a physical-based simulation with springs (on the right). 

Videos and ground truths used in the paper can be downloaded from https://partage.imt.fr/index.php/s/MHTpefDJWHp2HRD or reproduced with the code provided here.

To extract optical flow, we used a fluorescence video of Hydra Vulgaris from Dupre C, Yuste R. Non-overlapping Neural Networks in Hydra vulgaris. Curr Biol. 2017 Apr 24;27(8):1085-1097. doi: 10.1016/j.cub.2017.02.049. Epub 2017 Mar 30. PMID: 28366745; PMCID: PMC5423359.

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
$ bash scripts/isbi/generate_paper_dataset.sh 111
$ bash scripts/isbi/generate_paper_dataset.sh 222
$ bash scripts/isbi/generate_paper_dataset.sh 333
$ bash scripts/isbi/generate_paper_dataset.sh 444
$ bash scripts/isbi/generate_paper_dataset.sh 555
```

Reproducing the results for a particular method (skt, koft--, koft, koft++, emht, trackmate, trackmate-kf):

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

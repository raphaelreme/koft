set -e

# To train TrackAstra, we use 2 new seeds: train on two videos (one for each motion) & evaluate on two videos (different seeds)

# Springs 2D
# $RUN_KOFT_ENV expyrun config/dataset/springs_2d.yml --simulator.imaging_config.alpha 0.2 --simulator.imaging_config.delta 50 --seed 11
# $RUN_KOFT_ENV expyrun config/dataset/springs_2d.yml --simulator.imaging_config.alpha 0.2 --simulator.imaging_config.delta 50 --seed 22


# Hydra Flow
$RUN_KOFT_ENV expyrun config/dataset/hydra_flow.yml --simulator.imaging_config.alpha 0.2 --simulator.imaging_config.delta 50 --seed 33 --simulator.base_video.randomise False
# $RUN_KOFT_ENV expyrun config/dataset/hydra_flow.yml --simulator.imaging_config.alpha 0.2 --simulator.imaging_config.delta 50 --seed 44

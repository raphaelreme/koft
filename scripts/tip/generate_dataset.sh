set -e

### DEFAULT (alpha=0.2, delta=50, 1000 particles (~ 800 real ones), flow and springs)
### IMAGE DEGRADATION (alpha from 0.02 to 0.85, delta from 5 to 500)

# Springs 2D
$RUN_KOFT_ENV expyrun config/dataset/springs_2d.yml --simulator.imaging_config.alpha 0.02 --simulator.imaging_config.delta 5 --seed $@
$RUN_KOFT_ENV expyrun config/dataset/springs_2d.yml --simulator.imaging_config.alpha 0.02 --simulator.imaging_config.delta 50 --seed $@
$RUN_KOFT_ENV expyrun config/dataset/springs_2d.yml --simulator.imaging_config.alpha 0.02 --simulator.imaging_config.delta 500 --seed $@

$RUN_KOFT_ENV expyrun config/dataset/springs_2d.yml --simulator.imaging_config.alpha 0.07 --simulator.imaging_config.delta 5 --seed $@
$RUN_KOFT_ENV expyrun config/dataset/springs_2d.yml --simulator.imaging_config.alpha 0.07 --simulator.imaging_config.delta 50 --seed $@
$RUN_KOFT_ENV expyrun config/dataset/springs_2d.yml --simulator.imaging_config.alpha 0.07 --simulator.imaging_config.delta 500 --seed $@

$RUN_KOFT_ENV expyrun config/dataset/springs_2d.yml --simulator.imaging_config.alpha 0.2 --simulator.imaging_config.delta 5 --seed $@
$RUN_KOFT_ENV expyrun config/dataset/springs_2d.yml --simulator.imaging_config.alpha 0.2 --simulator.imaging_config.delta 50 --seed $@
$RUN_KOFT_ENV expyrun config/dataset/springs_2d.yml --simulator.imaging_config.alpha 0.2 --simulator.imaging_config.delta 500 --seed $@

$RUN_KOFT_ENV expyrun config/dataset/springs_2d.yml --simulator.imaging_config.alpha 0.35 --simulator.imaging_config.delta 5 --seed $@
$RUN_KOFT_ENV expyrun config/dataset/springs_2d.yml --simulator.imaging_config.alpha 0.35 --simulator.imaging_config.delta 50 --seed $@
$RUN_KOFT_ENV expyrun config/dataset/springs_2d.yml --simulator.imaging_config.alpha 0.35 --simulator.imaging_config.delta 500 --seed $@

$RUN_KOFT_ENV expyrun config/dataset/springs_2d.yml --simulator.imaging_config.alpha 0.85 --simulator.imaging_config.delta 5 --seed $@
$RUN_KOFT_ENV expyrun config/dataset/springs_2d.yml --simulator.imaging_config.alpha 0.85 --simulator.imaging_config.delta 50 --seed $@
$RUN_KOFT_ENV expyrun config/dataset/springs_2d.yml --simulator.imaging_config.alpha 0.85 --simulator.imaging_config.delta 500 --seed $@


# Hydra Flow
$RUN_KOFT_ENV expyrun config/dataset/hydra_flow.yml --simulator.imaging_config.alpha 0.02 --simulator.imaging_config.delta 5 --seed $@
$RUN_KOFT_ENV expyrun config/dataset/hydra_flow.yml --simulator.imaging_config.alpha 0.02 --simulator.imaging_config.delta 50 --seed $@
$RUN_KOFT_ENV expyrun config/dataset/hydra_flow.yml --simulator.imaging_config.alpha 0.02 --simulator.imaging_config.delta 500 --seed $@

$RUN_KOFT_ENV expyrun config/dataset/hydra_flow.yml --simulator.imaging_config.alpha 0.07 --simulator.imaging_config.delta 5 --seed $@
$RUN_KOFT_ENV expyrun config/dataset/hydra_flow.yml --simulator.imaging_config.alpha 0.07 --simulator.imaging_config.delta 50 --seed $@
$RUN_KOFT_ENV expyrun config/dataset/hydra_flow.yml --simulator.imaging_config.alpha 0.07 --simulator.imaging_config.delta 500 --seed $@

$RUN_KOFT_ENV expyrun config/dataset/hydra_flow.yml --simulator.imaging_config.alpha 0.2 --simulator.imaging_config.delta 5 --seed $@
$RUN_KOFT_ENV expyrun config/dataset/hydra_flow.yml --simulator.imaging_config.alpha 0.2 --simulator.imaging_config.delta 50 --seed $@
$RUN_KOFT_ENV expyrun config/dataset/hydra_flow.yml --simulator.imaging_config.alpha 0.2 --simulator.imaging_config.delta 500 --seed $@

$RUN_KOFT_ENV expyrun config/dataset/hydra_flow.yml --simulator.imaging_config.alpha 0.35 --simulator.imaging_config.delta 5 --seed $@
$RUN_KOFT_ENV expyrun config/dataset/hydra_flow.yml --simulator.imaging_config.alpha 0.35 --simulator.imaging_config.delta 50 --seed $@
$RUN_KOFT_ENV expyrun config/dataset/hydra_flow.yml --simulator.imaging_config.alpha 0.35 --simulator.imaging_config.delta 500 --seed $@

$RUN_KOFT_ENV expyrun config/dataset/hydra_flow.yml --simulator.imaging_config.alpha 0.85 --simulator.imaging_config.delta 5 --seed $@
$RUN_KOFT_ENV expyrun config/dataset/hydra_flow.yml --simulator.imaging_config.alpha 0.85 --simulator.imaging_config.delta 50 --seed $@
$RUN_KOFT_ENV expyrun config/dataset/hydra_flow.yml --simulator.imaging_config.alpha 0.85 --simulator.imaging_config.delta 500 --seed $@